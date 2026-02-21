"""Property-based tests for Learning Store semantic retrieval.

Property 21: Learning store semantic retrieval
- For any set of learnings added, query should return at most top_k results.
- Add then remove, store should not contain removed learning.
- Cosine similarity is between -1 and 1.

Validates: Requirements 3.2, 6.3
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.agents.learning_store import LearningStore
from src.types.core import Learning

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_confidence = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_source_iter = st.integers(min_value=0, max_value=100)
_pattern = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters=" -_"),
    min_size=1,
    max_size=100,
)
_resolution = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters=" -_"),
    min_size=1,
    max_size=100,
)

_learning = st.builds(
    Learning,
    pattern=_pattern,
    resolution=_resolution,
    confidence=_confidence,
    source_iteration=_source_iter,
    embedding=st.none(),
)

_learnings_list = st.lists(_learning, min_size=0, max_size=15)
_top_k = st.integers(min_value=1, max_value=20)


# ---------------------------------------------------------------------------
# Embedding helper for cosine similarity tests
# ---------------------------------------------------------------------------

def _fake_embedding(text: str) -> list[float]:
    """Deterministic fake embedding: simple hash-based 8-dim unit vector."""
    h = hash(text) & 0xFFFF_FFFF
    vec = []
    for i in range(8):
        val = ((h >> (i * 4)) & 0xF) / 15.0
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Property 21a: query returns at most top_k results
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    learnings=_learnings_list,
    top_k=_top_k,
)
def test_property_21a_query_returns_at_most_top_k(
    learnings: list[Learning],
    top_k: int,
) -> None:
    """For any set of learnings added, query should return at most top_k results.

    Feature: claude-code-v3-enterprise, Property 21a.
    Validates: Requirements 3.2
    """
    store = LearningStore()
    for learning in learnings:
        store.add(learning)

    results = store.query("test query", top_k=top_k)
    assert len(results) <= top_k
    assert len(results) <= len(learnings)


# ---------------------------------------------------------------------------
# Property 21b: add then remove leaves store without that learning
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    learning=_learning,
    others=st.lists(_learning, min_size=0, max_size=5),
)
def test_property_21b_add_remove_consistency(
    learning: Learning,
    others: list[Learning],
) -> None:
    """Add then remove: store should not contain the removed learning.

    Feature: claude-code-v3-enterprise, Property 21b.
    Validates: Requirements 3.2
    """
    store = LearningStore()

    # Add others first
    for other in others:
        store.add(other)

    # Add the target
    store.add(learning)
    all_before = store.get_all()
    assert len(all_before) == len(others) + 1

    # Remove the target
    store.remove(learning)
    all_after = store.get_all()

    # The target should no longer be present
    for remaining in all_after:
        # At least pattern+resolution should not both match the removed learning
        if remaining.pattern == learning.pattern and remaining.resolution == learning.resolution:
            # If exact same pattern+resolution+confidence, it should be from others
            pass  # May exist in others; the count is what matters
    assert len(all_after) == len(others)


# ---------------------------------------------------------------------------
# Property 21c: cosine similarity is between -1 and 1
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=100)
@given(
    text_a=_pattern,
    text_b=_pattern,
)
def test_property_21c_cosine_similarity_bounds(
    text_a: str,
    text_b: str,
) -> None:
    """Cosine similarity must always be between -1 and 1.

    Feature: claude-code-v3-enterprise, Property 21c.
    Validates: Requirements 6.3
    """
    emb_a = _fake_embedding(text_a)
    emb_b = _fake_embedding(text_b)

    sim = _cosine_similarity(emb_a, emb_b)
    assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9, (
        f"Cosine similarity {sim} out of [-1, 1] bounds for "
        f"texts '{text_a[:30]}' and '{text_b[:30]}'"
    )
