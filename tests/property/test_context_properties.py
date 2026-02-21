"""Property-based tests for context classification and context manager behavior.

Properties covered:
- Property 1: Context classification is exhaustive and deterministic
- Property 2: PRESERVE_VERBATIM items survive compaction with full fidelity
- Property 3: Staleness score follows the required formula
- Property 4: Context health metrics are internally consistent
"""

from __future__ import annotations

import json
import tempfile

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.context.context_manager import ContextManager
from src.context.triage import classify
from src.types.core import ContextCategory

SOURCES = ("user", "assistant", "tool_call", "tool_result", "system")
ASCII_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-./:",
    min_size=0,
    max_size=120,
)


@pytest.mark.property
@settings(max_examples=100)
@given(content=ASCII_TEXT, source=st.sampled_from(SOURCES))
def test_property_1_context_classification_is_exhaustive_and_deterministic(
    content: str,
    source: str,
) -> None:
    """Feature: claude-code-v3-enterprise, Property 1."""
    first = classify(content, source)
    second = classify(content, source)

    assert first == second
    assert isinstance(first, ContextCategory)
    assert first in set(ContextCategory)


@pytest.mark.property
@settings(max_examples=100)
@given(
    snippets=st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-",
            min_size=1,
            max_size=40,
        ),
        min_size=1,
        max_size=6,
    )
)
def test_property_2_preserve_verbatim_items_survive_compaction(
    snippets: list[str],
) -> None:
    """Feature: claude-code-v3-enterprise, Property 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = ContextManager(session_id="prop-2", window_size=200_000, session_dir=tmpdir)
        original_by_id: dict[str, str] = {}

        for idx, snippet in enumerate(snippets):
            content = f"Error: {snippet} at module.func"
            item_id = f"verbatim-{idx}"
            original_by_id[item_id] = content
            item = cm.add_item(content=content, source="assistant", token_count=max(1, len(content)), item_id=item_id)
            assert item.category == ContextCategory.PRESERVE_VERBATIM

        cm.add_item("Discussing implementation tradeoffs", "user", 200, item_id="compress")
        cm.add_item("raw tool response line", "tool_result", 50, item_id="ephemeral")

        result = cm.compact()

        in_memory = {item.id: item.content for item in cm.items}
        for item_id, expected_content in original_by_id.items():
            assert in_memory[item_id] == expected_content

        saved = json.loads(open(result.session_file_path, encoding="utf-8").read())
        saved_items = {item["id"]: item["content"] for item in saved["items"]}
        for item_id, expected_content in original_by_id.items():
            assert saved_items[item_id] == expected_content


@pytest.mark.property
@settings(max_examples=100)
@given(
    turns_since=st.integers(min_value=0, max_value=2_000),
    reference_count=st.integers(min_value=1, max_value=100),
)
def test_property_3_staleness_score_correctness(
    turns_since: int,
    reference_count: int,
) -> None:
    """Feature: claude-code-v3-enterprise, Property 3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = ContextManager(session_id="prop-3", window_size=200_000, session_dir=tmpdir)
        item = cm.add_item("General discussion", "user", 10, item_id="s1")
        item.reference_count = reference_count
        cm._turn_at_last_ref[item.id] = 0

        cm.update_staleness(current_turn=turns_since)
        expected = turns_since * (1.0 / reference_count)
        assert item.staleness_score == pytest.approx(expected)


@pytest.mark.property
@settings(max_examples=100)
@given(
    window_size=st.integers(min_value=100, max_value=50_000),
    items=st.lists(
        st.tuples(
            st.sampled_from(list(ContextCategory)),
            st.integers(min_value=1, max_value=2_000),
        ),
        min_size=1,
        max_size=40,
    ),
)
def test_property_4_context_health_metrics_consistency(
    window_size: int,
    items: list[tuple[ContextCategory, int]],
) -> None:
    """Feature: claude-code-v3-enterprise, Property 4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = ContextManager(session_id="prop-4", window_size=window_size, session_dir=tmpdir)

        for idx, (category, tokens) in enumerate(items):
            item = cm.add_item(
                content=f"item-{idx}",
                source="user",
                token_count=tokens,
                item_id=f"i-{idx}",
            )
            item.category = category

        metrics = cm.get_health_metrics()
        expected_total = sum(tokens for _, tokens in items)
        expected_free_percent = (1.0 - expected_total / window_size) * 100.0

        assert metrics.total_tokens == expected_total
        assert (
            metrics.preserved_tokens
            + metrics.compressible_tokens
            + metrics.ephemeral_tokens
            == metrics.total_tokens
        )
        assert metrics.free_percent == pytest.approx(expected_free_percent)

