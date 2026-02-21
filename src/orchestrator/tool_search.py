"""MCP Tool Search integration for lazy-loading tool definitions.

When tool definitions exceed a configurable context threshold, this module
switches from eager loading (full definitions in context) to lazy loading
using tool_reference blocks (name + description only). This can reduce
context usage by up to 95%.

- Sonnet 4+ and Opus 4+ support tool_reference blocks (lazy loading).
- Haiku always falls back to eager loading (no search support).
- Controlled via ENABLE_TOOL_SEARCH env var: "auto" | "auto:N" | "false".

Requirements: 9.6
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.types.core import ModelTier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_THRESHOLD = 10_000  # tokens
"""Default token threshold above which lazy loading kicks in."""

ENV_VAR_NAME = "ENABLE_TOOL_SEARCH"
"""Environment variable controlling tool search behaviour."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class LoadingMode(Enum):
    """How tool definitions are provided to the model."""
    EAGER = "eager"   # Full definitions in context
    LAZY = "lazy"     # tool_reference blocks (name + description only)


@dataclass(frozen=True)
class ToolDefinition:
    """A single tool definition with its estimated token cost."""
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0


@dataclass(frozen=True)
class ToolReference:
    """A lightweight reference replacing a full tool definition in lazy mode."""
    name: str
    description: str


@dataclass
class ToolSearchResult:
    """Result of processing tool definitions through the search module."""
    mode: LoadingMode
    tools: list[ToolDefinition | ToolReference]
    total_tokens_original: int
    total_tokens_after: int

    @property
    def tokens_saved(self) -> int:
        """Context tokens saved by lazy loading."""
        return self.total_tokens_original - self.total_tokens_after

    @property
    def savings_percent(self) -> float:
        """Percentage of tokens saved (0.0–100.0)."""
        if self.total_tokens_original == 0:
            return 0.0
        return (self.tokens_saved / self.total_tokens_original) * 100.0


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _parse_env_config() -> tuple[bool, int]:
    """Parse ENABLE_TOOL_SEARCH env var.

    Returns:
        (enabled, threshold) where *enabled* is False when the env var is
        ``"false"`` and *threshold* is the custom token threshold (or the
        default).
    """
    raw = os.environ.get(ENV_VAR_NAME, "auto").strip().lower()

    if raw == "false":
        return False, DEFAULT_CONTEXT_THRESHOLD

    if raw.startswith("auto:"):
        try:
            threshold = int(raw.split(":", 1)[1])
            if threshold <= 0:
                threshold = DEFAULT_CONTEXT_THRESHOLD
            return True, threshold
        except (ValueError, IndexError):
            return True, DEFAULT_CONTEXT_THRESHOLD

    # "auto" or any unrecognised value → enabled with default threshold
    return True, DEFAULT_CONTEXT_THRESHOLD


def _supports_tool_search(model_tier: ModelTier) -> bool:
    """Return True if the model tier supports tool_reference blocks."""
    return model_tier in (ModelTier.SONNET, ModelTier.OPUS)


# ---------------------------------------------------------------------------
# Estimated token cost for a tool reference (name + description only)
# ---------------------------------------------------------------------------

def _estimate_reference_tokens(tool: ToolDefinition) -> int:
    """Rough token estimate for a tool_reference block.

    A reference only carries the name and description, so it is much
    cheaper than the full definition which includes the input_schema.
    We approximate 1 token ≈ 4 characters.
    """
    text_len = len(tool.name) + len(tool.description)
    return max(1, text_len // 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_tools(
    tools: list[ToolDefinition],
    model_tier: ModelTier,
    *,
    threshold: int | None = None,
) -> ToolSearchResult:
    """Decide whether to use eager or lazy loading for *tools*.

    Parameters
    ----------
    tools:
        Full tool definitions with estimated token counts.
    model_tier:
        The model tier that will receive the tools.
    threshold:
        Optional override for the context-token threshold.  When ``None``
        the value is read from the ``ENABLE_TOOL_SEARCH`` env var (or the
        built-in default of 10 000 tokens).

    Returns
    -------
    ToolSearchResult with the chosen loading mode, the (possibly
    converted) tool list, and context-savings metrics.
    """
    enabled, env_threshold = _parse_env_config()
    effective_threshold = threshold if threshold is not None else env_threshold

    total_original = sum(t.token_count for t in tools)

    # Fast-path: tool search disabled via env var
    if not enabled:
        return ToolSearchResult(
            mode=LoadingMode.EAGER,
            tools=list(tools),
            total_tokens_original=total_original,
            total_tokens_after=total_original,
        )

    # Haiku does not support tool_reference blocks → always eager
    if not _supports_tool_search(model_tier):
        return ToolSearchResult(
            mode=LoadingMode.EAGER,
            tools=list(tools),
            total_tokens_original=total_original,
            total_tokens_after=total_original,
        )

    # Below threshold → eager loading is fine
    if total_original <= effective_threshold:
        return ToolSearchResult(
            mode=LoadingMode.EAGER,
            tools=list(tools),
            total_tokens_original=total_original,
            total_tokens_after=total_original,
        )

    # Above threshold + capable model → switch to lazy loading
    refs: list[ToolReference] = []
    total_after = 0
    for tool in tools:
        ref = ToolReference(name=tool.name, description=tool.description)
        refs.append(ref)
        total_after += _estimate_reference_tokens(tool)

    return ToolSearchResult(
        mode=LoadingMode.LAZY,
        tools=refs,
        total_tokens_original=total_original,
        total_tokens_after=total_after,
    )
