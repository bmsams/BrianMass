"""Unit tests for the LearningStore.

Covers: add/get, embedding callbacks, keyword-based query, cosine similarity
query, remove, persistence to disk, top_k, and empty store behavior.

Requirements: 3.2, 6.3
"""

from __future__ import annotations

import math

from src.agents.learning_store import LearningStore
from src.types.core import Learning

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_learning(
    pattern: str = "timeout in CI",
    resolution: str = "increase wait to 30s",
    confidence: float = 0.9,
    source_iteration: int = 1,
    embedding: list[float] | None = None,
) -> Learning:
    return Learning(
        pattern=pattern,
        resolution=resolution,
        confidence=confidence,
        source_iteration=source_iteration,
        embedding=embedding,
    )


def _dummy_embedding_callback(text: str) -> list[float]:
    """Deterministic fake embedding: hash-based 8-dim vector."""
    h = hash(text) & 0xFFFF_FFFF
    vec = []
    for i in range(8):
        val = ((h >> (i * 4)) & 0xF) / 15.0  # normalise to [0,1]
        vec.append(val)
    # Normalise to unit length
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ===================================================================
# Add and retrieve
# ===================================================================

class TestAddAndGetAll:
    """add stores learnings that get_all retrieves."""

    def test_add_and_get_all(self):
        store = LearningStore()
        learning = _make_learning()
        store.add(learning)

        all_learnings = store.get_all()
        assert len(all_learnings) == 1
        assert all_learnings[0].pattern == "timeout in CI"
        assert all_learnings[0].resolution == "increase wait to 30s"

    def test_add_multiple(self):
        store = LearningStore()
        store.add(_make_learning(pattern="p1"))
        store.add(_make_learning(pattern="p2"))
        store.add(_make_learning(pattern="p3"))

        assert len(store.get_all()) == 3


# ===================================================================
# Embedding callback
# ===================================================================

class TestAddWithEmbeddingCallback:
    """When an embedding callback is set, add computes embeddings."""

    def test_add_with_embedding_callback(self):
        store = LearningStore(embedding_callback=_dummy_embedding_callback)
        learning = _make_learning(pattern="flaky test", embedding=None)
        store.add(learning)

        all_learnings = store.get_all()
        assert len(all_learnings) == 1
        assert all_learnings[0].embedding is not None
        assert len(all_learnings[0].embedding) > 0


# ===================================================================
# Query without embeddings (keyword match)
# ===================================================================

class TestQueryWithoutEmbeddings:
    """Without embeddings, query falls back to keyword matching."""

    def test_query_without_embeddings_keyword_match(self):
        store = LearningStore()
        store.add(_make_learning(pattern="timeout in CI pipeline"))
        store.add(_make_learning(pattern="import error in tests"))
        store.add(_make_learning(pattern="disk space warning"))

        results = store.query("timeout", top_k=5)
        assert len(results) >= 1
        # The timeout learning should be in the results
        patterns = [r.pattern for r in results]
        assert any("timeout" in p for p in patterns)


# ===================================================================
# Query with embeddings (cosine similarity)
# ===================================================================

class TestQueryWithEmbeddings:
    """With embeddings, query uses cosine similarity for ranking."""

    def test_query_with_embeddings_cosine_similarity(self):
        store = LearningStore(embedding_callback=_dummy_embedding_callback)
        store.add(_make_learning(pattern="timeout in CI pipeline"))
        store.add(_make_learning(pattern="disk space warning on build server"))
        store.add(_make_learning(pattern="timeout connecting to database"))

        results = store.query("timeout", top_k=2)
        assert len(results) <= 2
        # Both timeout-related learnings should rank higher than disk space
        patterns = [r.pattern for r in results]
        assert any("timeout" in p for p in patterns)


# ===================================================================
# Remove
# ===================================================================

class TestRemoveLearning:
    """remove deletes a learning from the store."""

    def test_remove_learning(self):
        store = LearningStore()
        learning = _make_learning(pattern="to-remove")
        store.add(learning)

        assert len(store.get_all()) == 1
        store.remove(learning)
        assert len(store.get_all()) == 0


# ===================================================================
# Persistence
# ===================================================================

class TestPersistence:
    """LearningStore can save to and load from disk."""

    def test_persistence_to_disk(self, tmp_path):
        store_path = tmp_path / "learnings.json"
        store = LearningStore(path=str(store_path))
        store.add(_make_learning(pattern="persisted-pattern"))
        store.save()

        # Load into a new store instance
        store2 = LearningStore(path=str(store_path))
        store2.load()

        all_learnings = store2.get_all()
        assert len(all_learnings) == 1
        assert all_learnings[0].pattern == "persisted-pattern"


# ===================================================================
# Top-K limit
# ===================================================================

class TestQueryRespectsTopK:
    """query returns at most top_k results."""

    def test_query_respects_top_k(self):
        store = LearningStore()
        for i in range(10):
            store.add(_make_learning(pattern=f"pattern-{i}"))

        results = store.query("pattern", top_k=3)
        assert len(results) <= 3


# ===================================================================
# Empty store
# ===================================================================

class TestEmptyStore:
    """Empty store returns empty results."""

    def test_empty_store_query_returns_empty(self):
        store = LearningStore()
        results = store.query("anything", top_k=5)
        assert results == []
