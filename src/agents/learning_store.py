"""Learning Store — persistent cross-session learning with vector index.

Persists learnings in-memory and optionally to disk as JSON. Supports
semantic similarity search via pluggable embedding callbacks and falls
back to keyword matching when embeddings are not available.

Integrates with AgentCore Memory long-term storage via a pluggable memory
callback for the semantic strategy.

Two persistence modes:

1. **Single-file** — ``LearningStore(path="learnings.json")`` with explicit
   ``save()`` / ``load()`` calls.
2. **Directory** — ``LearningStore(learnings_dir=".brainmass/learnings/")``
   where each entry is saved as ``{id}.json`` automatically on ``add()``.

When neither ``path`` nor ``learnings_dir`` is provided the store operates
purely in-memory (useful for testing).

Requirements: 21.1, 21.2, 21.3, 21.4
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.types.core import Learning

logger = logging.getLogger(__name__)

DEFAULT_LEARNINGS_DIR = os.path.join(".brainmass", "learnings")

# ---------------------------------------------------------------------------
# Type aliases for pluggable callbacks
# ---------------------------------------------------------------------------

# Generates a vector embedding from a text string.
# Signature: (text) -> list[float]
EmbeddingCallback = Callable[[str], list[float]]

# Persists a learning entry to AgentCore Memory long-term storage.
# Signature: (learning_id, learning_dict) -> bool  (True on success)
MemoryCallback = Callable[[str, dict], bool]


def _default_embedding_callback(text: str) -> list[float]:
    """Default no-op embedding callback — returns an empty vector."""
    return []


# --- Production integration point ---
def _production_memory_callback(learning_id: str, learning_dict: dict) -> bool:
    """Persist a learning to AgentCore Memory long-term storage.

    Imports ``bedrock_agentcore.memory.MemoryClient`` lazily so the module
    can be imported without the SDK installed.  Raises ``RuntimeError`` when
    the package is absent and no callback has been injected.

    Requirements: 8.1, 8.2, 8.3, 8.4
    """
    try:
        from bedrock_agentcore.memory import MemoryClient  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "bedrock-agentcore is required for production memory persistence. "
            "Install with: pip install bedrock-agentcore  "
            "Or inject a memory_callback to bypass this requirement."
        ) from exc

    # Reuse AgentCoreMemoryStore from context_manager for consistent API handling.
    from src.context.context_manager import AgentCoreMemoryStore

    store = AgentCoreMemoryStore(MemoryClient())
    store.create_memory(namespace="learning-store", content=learning_dict)
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 when either vector is empty or has zero magnitude.
    """
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def _keyword_score(query: str, text: str) -> float:
    """Compute a simple keyword overlap score between *query* and *text*.

    Returns the fraction of query tokens that appear in the text (case-insensitive).
    Returns 0.0 when the query is empty.
    """
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for token in query_tokens if token in text_lower)
    return matches / len(query_tokens)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _learning_to_dict(learning: Learning) -> dict:
    """Convert a Learning dataclass to a JSON-serialisable dict."""
    return {
        "pattern": learning.pattern,
        "resolution": learning.resolution,
        "confidence": learning.confidence,
        "source_iteration": learning.source_iteration,
        "embedding": learning.embedding,
    }


def _dict_to_learning(data: dict) -> Learning:
    """Parse a dict back into a Learning dataclass."""
    return Learning(
        pattern=data["pattern"],
        resolution=data["resolution"],
        confidence=float(data["confidence"]),
        source_iteration=int(data["source_iteration"]),
        embedding=data.get("embedding"),
    )


# ---------------------------------------------------------------------------
# LearningStore
# ---------------------------------------------------------------------------

class LearningStore:
    """Persistent cross-session learning store with vector index.

    Supports in-memory operation, single-file persistence (via ``path``),
    and directory-based persistence (via ``learnings_dir``).

    Retrieval strategies:

    1. **Vector similarity** — when an ``embedding_callback`` is provided and
       learnings have embeddings, cosine similarity is used for top-K retrieval.
    2. **Keyword fallback** — when embeddings are unavailable, a simple keyword
       overlap score is used.

    Integration with AgentCore Memory long-term storage is achieved via the
    ``memory_callback``, which is invoked on every ``add`` call.

    Requirements: 21.1 (persist), 21.2 (embeddings), 21.3 (query), 21.4 (AgentCore)
    """

    def __init__(
        self,
        path: str | None = None,
        learnings_dir: str | None = None,
        embedding_callback: EmbeddingCallback | None = None,
        memory_callback: MemoryCallback | None = None,
    ) -> None:
        self._path = Path(path) if path else None
        self._learnings_dir = Path(learnings_dir) if learnings_dir else None
        self._embedding_callback = embedding_callback or _default_embedding_callback
        # Default to production AgentCore callback; tests must inject a stub.
        # --- Production integration point ---
        self._memory_callback = memory_callback or _production_memory_callback

        # In-memory store: list of Learning objects
        self._learnings: list[Learning] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, learning: Learning) -> str:
        """Add a learning and optionally compute its embedding.

        Steps:
        1. Generate a unique ID and timestamp.
        2. If an embedding callback is configured, embed the learning text
           (pattern + resolution concatenated) and attach to the entry.
        3. Store in memory and optionally persist to the learnings directory.
        4. Invoke the memory callback for AgentCore integration (Req 21.4).

        Returns:
            The generated learning ID.
        """
        learning_id = uuid.uuid4().hex[:16]

        # Compute embedding if callback is not the default no-op (Req 21.2)
        embedding = self._compute_embedding(learning)
        if embedding:
            learning.embedding = embedding

        # Store in memory
        self._learnings.append(learning)

        # Persist to directory if configured (Req 21.1)
        if self._learnings_dir is not None:
            self._save_entry_to_dir(learning_id, learning)

        # AgentCore Memory integration (Req 21.4)
        entry_dict = _learning_to_dict(learning)
        entry_dict["id"] = learning_id
        entry_dict["timestamp"] = datetime.now(UTC).isoformat()
        try:
            self._memory_callback(learning_id, entry_dict)
        except Exception as exc:
            logger.warning(
                "Memory callback failed for learning '%s': %s",
                learning_id, exc,
            )

        logger.info(
            "Added learning '%s': pattern='%s', confidence=%.2f",
            learning_id, learning.pattern[:60], learning.confidence,
        )
        return learning_id

    # Alias for the original spec name
    add_learning = add

    def query(self, task_description: str, top_k: int = 5) -> list[Learning]:
        """Retrieve the most relevant learnings for *task_description*.

        Uses cosine similarity when embeddings are available on both the
        query and the stored learnings.  Falls back to keyword matching
        when embeddings are unavailable (Req 21.3).

        Args:
            task_description: Free-text description of the current task.
            top_k: Maximum number of results to return.

        Returns:
            A list of :class:`Learning` objects sorted by relevance
            (highest first).
        """
        if not self._learnings:
            return []

        # Attempt vector search first (Req 21.2)
        query_embedding = self._compute_embedding_for_text(task_description)
        use_vector = bool(query_embedding) and any(
            l.embedding for l in self._learnings
        )

        scored: list[tuple[float, Learning]] = []
        for learning in self._learnings:
            if use_vector and learning.embedding:
                score = _cosine_similarity(query_embedding, learning.embedding)
            else:
                # Keyword fallback
                combined_text = f"{learning.pattern} {learning.resolution}"
                score = _keyword_score(task_description, combined_text)

            scored.append((score, learning))

        # Sort by score descending, take top_k
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [learning for _, learning in scored[:top_k]]

        logger.debug(
            "Query for '%s' returned %d results (vector=%s, top_k=%d)",
            task_description[:60], len(results), use_vector, top_k,
        )
        return results

    def get_all(self) -> list[Learning]:
        """Return all learnings currently in the store."""
        return list(self._learnings)

    def remove(self, learning: Learning) -> bool:
        """Remove a learning from the store by identity.

        Returns ``True`` if the learning was found and removed, ``False`` otherwise.
        """
        try:
            self._learnings.remove(learning)
            logger.info("Removed learning: pattern='%s'.", learning.pattern[:60])
            return True
        except ValueError:
            logger.debug("Learning not found for removal: pattern='%s'.", learning.pattern[:60])
            return False

    def remove_by_id(self, learning_id: str) -> bool:
        """Remove a learning entry from the directory store by its ID.

        Returns ``True`` if the file was deleted, ``False`` if not found.
        Only applicable when ``learnings_dir`` is set.
        """
        if self._learnings_dir is None:
            return False

        filepath = self._learnings_dir / f"{learning_id}.json"
        if not filepath.exists():
            logger.debug("Learning '%s' not found for removal.", learning_id)
            return False

        try:
            filepath.unlink()
            logger.info("Removed learning file '%s'.", learning_id)
            return True
        except OSError as exc:
            logger.warning("Failed to remove learning '%s': %s", learning_id, exc)
            return False

    # ------------------------------------------------------------------
    # Single-file persistence (save/load)
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist all in-memory learnings to the configured ``path``.

        Creates parent directories if needed. Uses the single-file JSON
        format: a JSON array of learning dicts.

        Raises ``RuntimeError`` if no ``path`` was configured.
        """
        if self._path is None:
            raise RuntimeError("Cannot save: no 'path' was configured.")

        self._path.parent.mkdir(parents=True, exist_ok=True)

        entries = [_learning_to_dict(l) for l in self._learnings]
        self._path.write_text(
            json.dumps(entries, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d learnings to %s", len(entries), self._path)

    def load(self) -> None:
        """Load learnings from the configured ``path`` into memory.

        Replaces any existing in-memory learnings.

        Raises ``RuntimeError`` if no ``path`` was configured.
        Raises ``FileNotFoundError`` if the file does not exist.
        Raises ``json.JSONDecodeError`` on malformed JSON.
        """
        if self._path is None:
            raise RuntimeError("Cannot load: no 'path' was configured.")

        raw = self._path.read_text(encoding="utf-8")
        data = json.loads(raw)

        if not isinstance(data, list):
            raise ValueError("Learning store file must contain a JSON array")

        self._learnings = [_dict_to_learning(entry) for entry in data]
        logger.info("Loaded %d learnings from %s", len(self._learnings), self._path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_embedding(self, learning: Learning) -> list[float]:
        """Compute an embedding for a learning's pattern + resolution."""
        text = f"{learning.pattern} {learning.resolution}"
        return self._compute_embedding_for_text(text)

    def _compute_embedding_for_text(self, text: str) -> list[float]:
        """Compute an embedding for arbitrary text via the callback.

        Returns an empty list if the callback returns nothing or raises.
        """
        try:
            result = self._embedding_callback(text)
            return result if result else []
        except Exception as exc:
            logger.warning("Embedding callback failed: %s", exc)
            return []

    def _save_entry_to_dir(
        self,
        learning_id: str,
        learning: Learning,
    ) -> None:
        """Write a learning entry to the directory as ``{id}.json``.

        Creates the learnings directory if it does not exist.
        """
        assert self._learnings_dir is not None
        self._learnings_dir.mkdir(parents=True, exist_ok=True)

        entry = _learning_to_dict(learning)
        entry["id"] = learning_id
        entry["timestamp"] = datetime.now(UTC).isoformat()

        filepath = self._learnings_dir / f"{learning_id}.json"
        filepath.write_text(
            json.dumps(entry, indent=2),
            encoding="utf-8",
        )
        logger.debug("Persisted learning '%s' to %s", learning_id, filepath)

    def _load_from_dir(self) -> list[dict]:
        """Load all learning entries from the learnings directory.

        Skips corrupt files with a warning.
        """
        if self._learnings_dir is None or not self._learnings_dir.exists():
            return []

        entries: list[dict] = []
        for filepath in self._learnings_dir.iterdir():
            if filepath.suffix != ".json":
                continue
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    logger.warning("Skipping non-dict file: %s", filepath)
                    continue
                entries.append(data)
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                logger.warning("Corrupt learning file '%s': %s", filepath, exc)
                continue
        return entries
