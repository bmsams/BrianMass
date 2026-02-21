"""Cache Manager for prompt caching with 5-minute and 1-hour durations.

Implements a 7-layer caching strategy that identifies cache-stable content layers,
injects cache_control blocks into API requests, and tracks cache hit rates.

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Cache Layer Definitions
# ---------------------------------------------------------------------------

class CacheDuration(Enum):
    """Supported cache TTL durations."""
    ONE_HOUR = 3600      # 1-hour for stable content
    FIVE_MINUTE = 300    # 5-minute for semi-stable content
    NEVER = 0            # Never cache (volatile)


class CacheLayerType(Enum):
    """The 7 cache layers in the caching strategy."""
    SYSTEM_PROMPT = "system_prompt"
    BRAINMASS_MD = "brainmass_md"
    TOOL_SCHEMAS = "tool_schemas"
    POLICY_DEFINITIONS = "policy_definitions"
    SKILL_INSTRUCTIONS = "skill_instructions"
    CONVERSATION_HISTORY = "conversation_history"
    CURRENT_DIFF = "current_diff"


# Mapping from layer type to cache duration (Req 11.1)
LAYER_DURATIONS: dict[CacheLayerType, CacheDuration] = {
    CacheLayerType.SYSTEM_PROMPT: CacheDuration.ONE_HOUR,
    CacheLayerType.BRAINMASS_MD: CacheDuration.ONE_HOUR,
    CacheLayerType.TOOL_SCHEMAS: CacheDuration.ONE_HOUR,
    CacheLayerType.POLICY_DEFINITIONS: CacheDuration.FIVE_MINUTE,
    CacheLayerType.SKILL_INSTRUCTIONS: CacheDuration.FIVE_MINUTE,
    CacheLayerType.CONVERSATION_HISTORY: CacheDuration.NEVER,
    CacheLayerType.CURRENT_DIFF: CacheDuration.NEVER,
}


@dataclass
class CacheLayer:
    """Describes a single cache layer with its type and duration."""
    layer_type: CacheLayerType
    duration: CacheDuration

    @property
    def is_cacheable(self) -> bool:
        return self.duration != CacheDuration.NEVER

    def to_cache_control(self) -> dict[str, Any] | None:
        """Build the cache_control block for the Anthropic API (Req 11.2).

        1-hour layers: {"type": "ephemeral", "ttl": 3600}
        5-minute layers: {"type": "ephemeral"}  (default 5-min TTL)
        Never-cache layers: None
        """
        if self.duration == CacheDuration.ONE_HOUR:
            return {"type": "ephemeral", "ttl": 3600}
        if self.duration == CacheDuration.FIVE_MINUTE:
            return {"type": "ephemeral"}
        return None


@dataclass
class CacheStats:
    """Running statistics for cache hit rate tracking."""
    total_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a float in [0.0, 1.0]."""
        if self.total_input_tokens == 0:
            return 0.0
        return self.cache_read_tokens / self.total_input_tokens


# ---------------------------------------------------------------------------
# Cache Manager
# ---------------------------------------------------------------------------

class CacheManager:
    """Manages prompt caching strategy with 5-minute and 1-hour durations.

    Responsibilities:
    - Identify cache-stable layers and assign durations (Req 11.1)
    - Inject cache_control blocks into API request content blocks (Req 11.2)
    - Track cache hit rates and alert when below threshold (Req 11.3)
    - Account for cache economics (Req 11.4)
    - Parse API response usage metrics (Req 11.5)
    """

    DEFAULT_ALERT_THRESHOLD = 0.70  # 70%

    def __init__(
        self,
        alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
        alert_callback: Callable[[float], None] | None = None,
    ) -> None:
        self._alert_threshold = alert_threshold
        self._alert_callback = alert_callback
        self._stats = CacheStats()

        # Pre-build the layer objects for quick lookup
        self._layers: dict[CacheLayerType, CacheLayer] = {
            lt: CacheLayer(layer_type=lt, duration=dur)
            for lt, dur in LAYER_DURATIONS.items()
        }

    # -- Public helpers -------------------------------------------------------

    @property
    def alert_threshold(self) -> float:
        return self._alert_threshold

    @alert_threshold.setter
    def alert_threshold(self, value: float) -> None:
        self._alert_threshold = value

    def get_layer(self, layer_type: CacheLayerType) -> CacheLayer:
        """Return the CacheLayer for a given type."""
        return self._layers[layer_type]

    def get_duration(self, layer_type: CacheLayerType) -> CacheDuration:
        """Return the CacheDuration for a given layer type."""
        return LAYER_DURATIONS[layer_type]

    # -- Core API -------------------------------------------------------------

    def inject_cache_controls(
        self, request_blocks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Add ``cache_control`` markers to content blocks (Req 11.2).

        Each block is expected to have a ``"cache_layer"`` key indicating its
        :class:`CacheLayerType` (as a string value matching the enum).  If the
        layer is cacheable, a ``cache_control`` dict is injected.  Blocks
        without a recognised ``cache_layer`` or with a NEVER-cache layer are
        returned unchanged.

        Returns a *new* list — the original blocks are not mutated.
        """
        result: list[dict[str, Any]] = []
        for block in request_blocks:
            new_block = dict(block)  # shallow copy
            layer_name = new_block.pop("cache_layer", None)
            if layer_name is not None:
                try:
                    layer_type = CacheLayerType(layer_name)
                except ValueError:
                    # Unknown layer — pass through unchanged
                    result.append(new_block)
                    continue
                cache_layer = self._layers[layer_type]
                cc = cache_layer.to_cache_control()
                if cc is not None:
                    new_block["cache_control"] = cc
            result.append(new_block)
        return result

    def track_hit_rate(self, usage_data: dict[str, Any]) -> float:
        """Update running stats from API response usage metrics (Req 11.5).

        Expected keys in *usage_data*:
        - ``input_tokens``: total input tokens for the request
        - ``cache_read_input_tokens``: tokens served from cache (90% discount)
        - ``cache_creation_input_tokens``: tokens written to cache

        Returns the updated cumulative hit rate and fires the alert callback
        if the rate drops below the configured threshold (Req 11.3).
        """
        input_tokens = usage_data.get("input_tokens", 0)
        cache_read = usage_data.get("cache_read_input_tokens", 0)
        cache_creation = usage_data.get("cache_creation_input_tokens", 0)

        self._stats.total_input_tokens += input_tokens
        self._stats.cache_read_tokens += cache_read
        self._stats.cache_creation_tokens += cache_creation

        rate = self._stats.hit_rate
        if (
            self._stats.total_input_tokens > 0
            and rate < self._alert_threshold
            and self._alert_callback is not None
        ):
            self._alert_callback(rate)

        return rate

    def get_hit_rate(self) -> float:
        """Return the current cumulative cache hit rate."""
        return self._stats.hit_rate

    def get_stats(self) -> CacheStats:
        """Return a copy of the current cache statistics."""
        return CacheStats(
            total_input_tokens=self._stats.total_input_tokens,
            cache_read_tokens=self._stats.cache_read_tokens,
            cache_creation_tokens=self._stats.cache_creation_tokens,
        )

    def reset_stats(self) -> None:
        """Reset all running statistics."""
        self._stats = CacheStats()
