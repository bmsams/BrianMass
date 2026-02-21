"""Tests for TASK_QA_MAP completeness against processed completed tasks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = ROOT / ".kiro" / "specs" / "claude-code-v3-enterprise"
MAP_PATH = SPEC_DIR / "TASK_QA_MAP.json"
STATE_PATH = SPEC_DIR / ".qa_state.json"


def test_task_qa_map_exists() -> None:
    assert MAP_PATH.exists(), f"Missing {MAP_PATH}"


def test_task_qa_map_covers_processed_completed() -> None:
    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    missing = [task_id for task_id in state.get("processed_completed", []) if task_id not in mapping]
    assert missing == [], f"Missing QA mappings for tasks: {missing}"


def test_task_qa_map_references_existing_paths() -> None:
    mapping = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    missing_paths: list[str] = []
    for task_id, paths in mapping.items():
        for raw in paths:
            p = ROOT / raw
            if not p.exists():
                missing_paths.append(f"{task_id}:{raw}")
    assert missing_paths == [], f"QA map references missing paths: {missing_paths}"

