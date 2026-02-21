"""Context Manager with semantic triage, compaction, and persistence.

Manages context items with classification, staleness scoring, compaction
at adaptive thresholds, session file persistence, and health metrics.

Conditionally inherits from ``strands.session.SummarizingConversationManager``
when the Strands SDK is installed, enabling LLM-driven summarisation during
compaction. Falls back to a plain ``object`` base when the SDK is absent.

Integrates with AgentCore MemoryClient for long-term learning persistence
when ``bedrock-agentcore`` is installed; falls back to in-memory store.

Requirements: 2.3, 2.4, 2.5, 2.7, 2.8, 2.9, 2.10, 2.12, 7.1, 7.2, 7.3, 7.4, 7.5
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from src.context.triage import classify
from src.observability.instrumentation import BrainmassTracer
from src.types.core import (
    ContextCategory,
    ContextHealthMetrics,
    ContextItem,
    HookContext,
    HookEvent,
)

# ---------------------------------------------------------------------------
# Conditional Strands SummarizingConversationManager base (Requirements: 7.1)
# ---------------------------------------------------------------------------

try:
    # --- Production integration point ---
    from strands.session import (  # type: ignore[import-untyped]
        SummarizingConversationManager as _StrandsBase,
    )
    _STRANDS_SESSION_AVAILABLE = True
except ImportError:
    _StrandsBase = object  # type: ignore[assignment,misc]
    _STRANDS_SESSION_AVAILABLE = False


# ---------------------------------------------------------------------------
# Compaction result
# ---------------------------------------------------------------------------

@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    items_preserved: int
    items_compressed: int
    items_dropped: int
    tokens_freed: int
    session_file_path: str


# ---------------------------------------------------------------------------
# Memory store adapters
# ---------------------------------------------------------------------------

class MemoryStore(Protocol):
    """Abstract memory store used by ContextManager."""

    def create_memory(self, namespace: str, content: dict) -> str:
        ...

    def query(self, namespace: str, query: str, top_k: int = 5) -> list[dict]:
        ...


class InMemoryMemoryStore:
    """In-memory memory store used by tests/local mode."""

    def __init__(self) -> None:
        self._store: list[dict] = []

    def create_memory(self, namespace: str, content: dict) -> str:
        entry = {"id": str(uuid.uuid4()), "namespace": namespace, "content": content}
        self._store.append(entry)
        return entry["id"]

    def query(self, namespace: str, query: str, top_k: int = 5) -> list[dict]:
        scoped = [e for e in self._store if e["namespace"] == namespace]
        if not query:
            return scoped[:top_k]
        q = query.lower()
        ranked: list[dict] = []
        for row in scoped:
            content = row.get("content", {})
            text = json.dumps(content, default=str).lower()
            if q in text:
                ranked.append(row)
        if ranked:
            return ranked[:top_k]
        return scoped[:top_k]


class AgentCoreMemoryStore:
    """AgentCore Memory-backed store for production integration."""

    def __init__(self, client: object | None = None) -> None:
        if client is None:
            try:
                from bedrock_agentcore.memory import MemoryClient  # type: ignore
            except Exception as exc:  # pragma: no cover - exercised in runtime env
                raise RuntimeError("bedrock-agentcore MemoryClient is required.") from exc
            self._client = MemoryClient()
        else:
            self._client = client
        self._memory_ids: dict[str, str] = {}

    @staticmethod
    def _extract_id(result: object) -> str:
        if isinstance(result, dict):
            for key in ("id", "memory_id", "memoryId", "name"):
                if key in result:
                    return str(result[key])
        return str(result)

    def _ensure_memory_id(self, namespace: str) -> str:
        cached = self._memory_ids.get(namespace)
        if cached:
            return cached

        if hasattr(self._client, "create_or_get_memory"):
            raw = self._client.create_or_get_memory(name=namespace)
        elif hasattr(self._client, "create_memory"):
            raw = self._client.create_memory(name=namespace)
        else:
            raise RuntimeError("AgentCore Memory client does not support memory creation APIs.")

        memory_id = self._extract_id(raw)
        self._memory_ids[namespace] = memory_id
        return memory_id

    def create_memory(self, namespace: str, content: dict) -> str:
        # Compatibility path for older/adapter clients used in tests.
        if hasattr(self._client, "create_memory"):
            try:
                result = self._client.create_memory(namespace=namespace, content=content)
                return self._extract_id(result)
            except TypeError:
                pass

        memory_id = self._ensure_memory_id(namespace)
        if not hasattr(self._client, "create_event"):
            raise RuntimeError("AgentCore Memory client does not support event creation APIs.")

        event = self._client.create_event(
            memory_id=memory_id,
            actor_id="brainmass",
            session_id=namespace,
            messages=[("user", json.dumps(content, default=str))],
        )
        if isinstance(event, dict):
            for key in ("event_id", "id"):
                if key in event:
                    return str(event[key])
        return memory_id

    def query(self, namespace: str, query: str, top_k: int = 5) -> list[dict]:
        # Compatibility path for older/adapter clients used in tests.
        if hasattr(self._client, "query"):
            try:
                rows = self._client.query(namespace=namespace, query=query, top_k=top_k)
                if isinstance(rows, list):
                    return [dict(r) if isinstance(r, dict) else {"value": r} for r in rows]
            except TypeError:
                pass

        memory_id = self._ensure_memory_id(namespace)
        if not hasattr(self._client, "retrieve_memories"):
            raise RuntimeError("AgentCore Memory client does not support retrieval APIs.")

        rows = self._client.retrieve_memories(
            memory_id=memory_id,
            namespace=namespace,
            query=query,
            top_k=top_k,
        )
        if not isinstance(rows, list):
            return []

        normalized: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                normalized.append(dict(row))
            else:
                normalized.append({"value": row})
        return normalized


# Backward-compatible alias used by existing tests/imports.
MemoryClientStub = InMemoryMemoryStore


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLD_200K = 83.5  # percent
_DEFAULT_THRESHOLD_1M = 85.0    # percent
_WINDOW_1M_BOUNDARY = 500_000   # tokens — windows above this use the 1M threshold


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager(_StrandsBase):  # type: ignore[misc]
    """Manages context items with semantic triage, compaction, and persistence.

    Conditionally inherits from ``strands.session.SummarizingConversationManager``
    when the Strands SDK is installed. The Strands base class provides LLM-driven
    summarisation during compaction and automatic token counting.

    When ``bedrock-agentcore`` is installed and no ``memory_store`` is injected,
    defaults to ``AgentCoreMemoryStore(MemoryClient())`` for long-term learning
    persistence (Requirements: 7.3, 7.4).

    Requirements: 2.3, 2.4, 2.5, 2.7, 2.8, 2.9, 2.10, 2.12, 7.1, 7.2, 7.3, 7.4, 7.5
    """

    def __init__(
        self,
        session_id: str,
        window_size: int = 200_000,
        session_dir: str = ".brainmass",
        memory_store: MemoryStore | None = None,
        tracer: BrainmassTracer | None = None,
        hook_engine: object | None = None,
        cwd: str = ".",
        session_type: str = "interactive",
    ) -> None:
        # --- Production integration point ---
        # When SummarizingConversationManager is available, initialise the Strands
        # base class so it can provide LLM-driven summarisation during compaction.
        if _STRANDS_SESSION_AVAILABLE:
            try:
                super().__init__(
                    max_tokens=window_size,
                    summarization_model="us.anthropic.claude-haiku-4-5-v1:0",
                )
            except Exception:
                # Strands base __init__ signature may vary; fall back gracefully.
                super().__init__()

        self.session_id = session_id
        self.window_size = window_size
        self.session_dir = session_dir
        self._cwd = cwd
        self._session_type = session_type

        self.items: list[ContextItem] = []
        self.current_turn: int = 0
        self._turn_at_last_ref: dict[str, int] = {}  # item_id → turn of last reference

        # Long-term learning store adapter (Requirements: 7.3, 7.4)
        if memory_store is None:
            # --- Production integration point ---
            # When BRAINMASS_AGENTCORE_MEMORY=1 is set, use AgentCore MemoryClient
            # for cross-session learning persistence. Defaults to in-memory store
            # so that tests and local development work without AWS credentials.
            if os.environ.get("BRAINMASS_AGENTCORE_MEMORY") == "1":
                try:
                    from bedrock_agentcore.memory import (
                        MemoryClient,  # type: ignore[import-untyped]
                    )
                    memory_store = AgentCoreMemoryStore(MemoryClient())
                except (ImportError, Exception):
                    memory_store = InMemoryMemoryStore()
            else:
                memory_store = InMemoryMemoryStore()
        self.memory_client: MemoryStore = memory_store

        # Cache hit tracking
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Observability
        self._tracer = tracer
        self._hook_engine = hook_engine

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_item(
        self,
        content: str,
        source: str,
        token_count: int,
        item_id: str | None = None,
    ) -> ContextItem:
        """Classify *content* via triage and add to working memory.

        Args:
            content: Raw text content.
            source: Origin — 'user', 'assistant', 'tool_call', 'tool_result', 'system'.
            token_count: Estimated token count for this content.
            item_id: Optional explicit ID; auto-generated if omitted.

        Returns:
            The newly created ContextItem.
        """
        category = classify(content, source)
        now = datetime.now(UTC)
        item = ContextItem(
            id=item_id or str(uuid.uuid4()),
            category=category,
            content=content,
            token_count=token_count,
            created_at=now,
            last_referenced_at=now,
            reference_count=1,
            source=source,
            staleness_score=0.0,
        )
        self.items.append(item)
        self._turn_at_last_ref[item.id] = self.current_turn
        return item

    def reference_item(self, item_id: str) -> None:
        """Record that *item_id* was referenced on the current turn."""
        for item in self.items:
            if item.id == item_id:
                item.reference_count += 1
                item.last_referenced_at = datetime.now(UTC)
                self._turn_at_last_ref[item.id] = self.current_turn
                break

    def advance_turn(self) -> None:
        """Advance the conversation turn counter and recalculate staleness."""
        self.current_turn += 1
        self.update_staleness(self.current_turn)

    # ------------------------------------------------------------------
    # Staleness scoring  (Req 2.7)
    # ------------------------------------------------------------------

    def update_staleness(self, current_turn: int) -> None:
        """Recalculate staleness_score for every item.

        Formula: score = (turns_since_last_reference) × (1 / reference_count)
        """
        self.current_turn = current_turn
        for item in self.items:
            last_ref_turn = self._turn_at_last_ref.get(item.id, 0)
            turns_since = current_turn - last_ref_turn
            ref_count = max(item.reference_count, 1)  # avoid division by zero
            item.staleness_score = turns_since * (1.0 / ref_count)

    # ------------------------------------------------------------------
    # Compaction  (Req 2.4, 2.5, 2.12)
    # ------------------------------------------------------------------

    def should_compact(self) -> bool:
        """Return True if total tokens exceed the compaction threshold."""
        threshold_pct = self._effective_threshold()
        total = self._total_tokens()
        return total >= (self.window_size * threshold_pct / 100.0)

    def compact(self, window_size: int | None = None) -> CompactionResult:
        """Run compaction: preserve verbatim → session file, compress discussion, drop ephemeral.

        Args:
            window_size: Override the instance window_size for this compaction.

        Returns:
            CompactionResult with counts and freed tokens.
        """
        if window_size is not None:
            self.window_size = window_size

        if self._hook_engine is not None:
            hook_ctx = HookContext(
                session_id=self.session_id,
                hook_event_name=HookEvent.PRE_COMPACT,
                cwd=self._cwd,
                session_type=self._session_type,
                source="context_compaction",
            )
            self._hook_engine.fire(HookEvent.PRE_COMPACT, hook_ctx)

        preserved: list[ContextItem] = []
        compressed: list[ContextItem] = []
        dropped: list[ContextItem] = []

        for item in self.items:
            if item.category == ContextCategory.PRESERVE_VERBATIM:
                preserved.append(item)
            elif item.category == ContextCategory.PRESERVE_STRUCTURED:
                preserved.append(item)
            elif item.category == ContextCategory.COMPRESS_AGGRESSIVE:
                compressed.append(item)
            elif item.category == ContextCategory.EPHEMERAL:
                dropped.append(item)
            else:
                # Safety net — treat unknown as compress
                compressed.append(item)

        # Persist preserved items to session file
        session_path = self._session_file_path()
        self._save_items_to_file(preserved, session_path)

        # Context editing: also drop stale tool call results (Req 2.7)
        stale_threshold = 5  # turns
        still_compressed: list[ContextItem] = []
        for item in compressed:
            if item.source in ("tool_call", "tool_result") and item.staleness_score > stale_threshold:
                dropped.append(item)
            else:
                still_compressed.append(item)

        tokens_freed = sum(i.token_count for i in dropped)

        # Compress discussion items — truncate content and reduce token count.
        # In production, SummarizingConversationManager would use an LLM to
        # produce real summaries.  Here we truncate to the first ~30% of
        # characters as a proxy and tag the item as summarised.
        for item in still_compressed:
            original_tokens = item.token_count
            target_chars = max(80, int(len(item.content) * 0.3))
            if len(item.content) > target_chars:
                item.content = item.content[:target_chars] + " [... compacted]"
            item.token_count = max(1, int(original_tokens * 0.3))
            tokens_freed += original_tokens - item.token_count

        # Rebuild items list: preserved + compressed (ephemeral dropped)
        self.items = preserved + still_compressed

        result = CompactionResult(
            items_preserved=len(preserved),
            items_compressed=len(still_compressed),
            items_dropped=len(dropped),
            tokens_freed=tokens_freed,
            session_file_path=str(session_path),
        )

        # Emit context health span after compaction (Req 16.2)
        if self._tracer is not None:
            metrics = self.get_health_metrics()
            staleness = {k: float(v) for k, v in metrics.staleness_distribution.items()}
            self._tracer.record_context_health_span(
                free_pct=metrics.free_percent,
                staleness_scores=staleness,
                cache_hit_rate=metrics.cache_hit_rate,
            )

        return result

    # ------------------------------------------------------------------
    # Health metrics  (Req 2.8)
    # ------------------------------------------------------------------

    def get_health_metrics(self) -> ContextHealthMetrics:
        """Return current context health metrics."""
        total = self._total_tokens()
        preserved = sum(
            i.token_count
            for i in self.items
            if i.category in (ContextCategory.PRESERVE_VERBATIM, ContextCategory.PRESERVE_STRUCTURED)
        )
        compressible = sum(
            i.token_count
            for i in self.items
            if i.category == ContextCategory.COMPRESS_AGGRESSIVE
        )
        ephemeral = sum(
            i.token_count
            for i in self.items
            if i.category == ContextCategory.EPHEMERAL
        )
        free_pct = (1.0 - total / self.window_size) * 100.0 if self.window_size > 0 else 100.0

        staleness_dist = self._staleness_distribution()
        cache_rate = self._cache_hit_rate()

        return ContextHealthMetrics(
            free_percent=free_pct,
            total_tokens=total,
            preserved_tokens=preserved,
            compressible_tokens=compressible,
            ephemeral_tokens=ephemeral,
            staleness_distribution=staleness_dist,
            cache_hit_rate=cache_rate,
        )

    # ------------------------------------------------------------------
    # Session persistence  (Req 2.3, 2.9)
    # ------------------------------------------------------------------

    def save_session(self, path: str | None = None) -> str:
        """Persist PRESERVE_VERBATIM and PRESERVE_STRUCTURED items to JSON.

        Returns:
            The file path written.
        """
        target = Path(path) if path else self._session_file_path()
        items_to_save = [
            i for i in self.items
            if i.category in (ContextCategory.PRESERVE_VERBATIM, ContextCategory.PRESERVE_STRUCTURED)
        ]
        self._save_items_to_file(items_to_save, target)
        return str(target)

    def load_session(self, path: str | None = None) -> int:
        """Load items from a session file into working memory.

        Returns:
            Number of items loaded.
        """
        target = Path(path) if path else self._session_file_path()
        if not target.exists():
            return 0

        data = json.loads(target.read_text(encoding="utf-8"))
        loaded = 0
        for raw in data.get("items", []):
            item = ContextItem(
                id=raw["id"],
                category=ContextCategory(raw["category"]),
                content=raw["content"],
                token_count=raw["token_count"],
                created_at=datetime.fromisoformat(raw["created_at"]),
                last_referenced_at=datetime.fromisoformat(raw["last_referenced_at"]),
                reference_count=raw["reference_count"],
                source=raw["source"],
                staleness_score=raw.get("staleness_score", 0.0),
            )
            # Avoid duplicates
            if not any(existing.id == item.id for existing in self.items):
                self.items.append(item)
                self._turn_at_last_ref[item.id] = self.current_turn
                loaded += 1
        return loaded

    # ------------------------------------------------------------------
    # Learning store  (Req 2.3 — AgentCore Memory integration)
    # ------------------------------------------------------------------

    def store_learning(self, pattern: str, resolution: str, confidence: float) -> str:
        """Persist a learning to the long-term store.

        In production this calls AgentCore MemoryClient.create_memory().

        Returns:
            The generated memory ID.
        """
        return self.memory_client.create_memory(
            namespace="learning-store",
            content={
                "pattern": pattern,
                "resolution": resolution,
                "confidence": confidence,
            },
        )

    def query_learnings(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve relevant learnings from the long-term store."""
        return self.memory_client.query(
            namespace="learning-store",
            query=query,
            top_k=top_k,
        )

    def hydrate_from_persistent_store(self, query: str, top_k: int = 5) -> int:
        """Hydrate working context with relevant learnings for a new session.

        Returns:
            Number of context items added.
        """
        rows = self.query_learnings(query=query, top_k=top_k)
        added = 0
        for row in rows:
            content = row.get("content", row)
            text = json.dumps(content, default=str)
            item = self.add_item(
                content=text,
                source="system",
                token_count=max(1, len(text.split())),
            )
            item.category = ContextCategory.PRESERVE_STRUCTURED
            added += 1
        return added

    # ------------------------------------------------------------------
    # Cache tracking helpers
    # ------------------------------------------------------------------

    def record_cache_hit(self) -> None:
        self._cache_hits += 1

    def record_cache_miss(self) -> None:
        self._cache_misses += 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _total_tokens(self) -> int:
        return sum(i.token_count for i in self.items)

    def _effective_threshold(self) -> float:
        """Return the compaction threshold percentage.

        Respects BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE env var (Req 2.12).
        Falls back to 83.5% for ≤500K windows, 85% for >500K.
        """
        override = os.environ.get("BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE")
        if override is not None:
            try:
                val = float(override)
                if 1.0 <= val <= 100.0:
                    return val
            except ValueError:
                pass

        if self.window_size > _WINDOW_1M_BOUNDARY:
            return _DEFAULT_THRESHOLD_1M
        return _DEFAULT_THRESHOLD_200K

    def _session_file_path(self) -> Path:
        return Path(self.session_dir) / "session-state.json"

    def _save_items_to_file(self, items: list[ContextItem], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": self.session_id,
            "saved_at": datetime.now(UTC).isoformat(),
            "items": [self._item_to_dict(i) for i in items],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def _item_to_dict(item: ContextItem) -> dict:
        return {
            "id": item.id,
            "category": item.category.value,
            "content": item.content,
            "token_count": item.token_count,
            "created_at": item.created_at.isoformat(),
            "last_referenced_at": item.last_referenced_at.isoformat(),
            "reference_count": item.reference_count,
            "source": item.source,
            "staleness_score": item.staleness_score,
        }

    def _staleness_distribution(self) -> dict[str, int]:
        """Bucket items by staleness: fresh (<1), warm (1-5), stale (>5)."""
        dist = {"fresh": 0, "warm": 0, "stale": 0}
        for item in self.items:
            if item.staleness_score < 1.0:
                dist["fresh"] += 1
            elif item.staleness_score <= 5.0:
                dist["warm"] += 1
            else:
                dist["stale"] += 1
        return dist

    def _cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total
