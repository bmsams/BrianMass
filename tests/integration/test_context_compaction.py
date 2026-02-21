"""Integration tests for context compaction behavior."""

from __future__ import annotations

import json
from pathlib import Path

from src.context.context_manager import ContextManager
from src.types.core import HookEvent, HookResult


def test_compaction_threshold_preserve_drop_and_precompact_hook(tmp_path):
    precompact_state: dict[str, object] = {}

    class RecordingHookEngine:
        def fire(self, event, context):
            precompact_state["event"] = event.value
            precompact_state["ephemeral_present_before"] = any(i.id == "e1" for i in manager.items)
            precompact_state["item_count_before"] = len(manager.items)
            return HookResult()

    manager = ContextManager(
        session_id="sess-compact",
        window_size=200_000,
        session_dir=str(tmp_path / ".brainmass"),
        hook_engine=RecordingHookEngine(),
        cwd=str(tmp_path),
    )

    manager.add_item("Error: failure at module.py:22", "assistant", 90_000, item_id="v1")
    manager.add_item("DECISION: keep rollback support", "user", 50_000, item_id="s1")
    manager.add_item("Working through an implementation approach", "user", 25_000, item_id="c1")
    manager.add_item("Searching for symbols...", "tool_call", 5_000, item_id="e1")

    assert manager.should_compact() is True
    result = manager.compact()

    assert precompact_state["event"] == HookEvent.PRE_COMPACT.value
    assert precompact_state["ephemeral_present_before"] is True
    assert precompact_state["item_count_before"] == 4

    remaining_ids = {item.id for item in manager.items}
    assert "v1" in remaining_ids
    assert "s1" in remaining_ids
    assert "e1" not in remaining_ids

    session_file = Path(result.session_file_path)
    assert session_file.exists()
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    saved_ids = {row["id"] for row in payload["items"]}
    assert "v1" in saved_ids
    assert "s1" in saved_ids
    assert "e1" not in saved_ids

    assert result.items_preserved >= 2
    assert result.items_dropped >= 1
