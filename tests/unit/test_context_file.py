"""Unit tests for context file management (Requirement 7.2).

Tests cover:
- Serialization of LoopContext to dict
- Deserialization of dict to LoopContext
- Round-trip fidelity (serialize → deserialize)
- Validation of required fields and types
- save() with atomic writes and directory creation
- load() with missing file handling
- load() with corrupt / invalid data
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.agents.context_file import (
    DEFAULT_CONTEXT_PATH,
    _validate_raw,
    deserialize,
    load,
    save,
    serialize,
)
from src.types.core import LoopContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**overrides) -> LoopContext:
    defaults = {
        "current_task": "Implement auth module",
        "acceptance_criteria": ["All tests pass", "JWT tokens work"],
        "constraints": ["Use bcrypt"],
        "learnings": [
            {"pattern": "bcrypt slow", "resolution": "use rounds=10", "confidence": 0.9, "source_iteration": 1}
        ],
        "failed_approaches": [
            {"iteration": 1, "approach": "plain sha256", "why_failed": "insecure"}
        ],
        "iteration_count": 2,
        "max_iterations": 5,
    }
    defaults.update(overrides)
    return LoopContext(**defaults)


def _make_raw_dict(**overrides) -> dict:
    defaults = {
        "current_task": "Implement auth module",
        "acceptance_criteria": ["All tests pass"],
        "constraints": [],
        "learnings": [],
        "failed_approaches": [],
        "iteration_count": 0,
        "max_iterations": 5,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialize:
    def test_returns_dict_with_all_keys(self):
        ctx = _make_context()
        result = serialize(ctx)
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "current_task", "acceptance_criteria", "constraints",
            "learnings", "failed_approaches", "iteration_count", "max_iterations",
        }

    def test_values_match_context(self):
        ctx = _make_context()
        result = serialize(ctx)
        assert result["current_task"] == ctx.current_task
        assert result["acceptance_criteria"] == ctx.acceptance_criteria
        assert result["constraints"] == ctx.constraints
        assert result["iteration_count"] == ctx.iteration_count
        assert result["max_iterations"] == ctx.max_iterations

    def test_learnings_are_plain_dicts(self):
        ctx = _make_context()
        result = serialize(ctx)
        for l in result["learnings"]:
            assert isinstance(l, dict)

    def test_empty_lists_preserved(self):
        ctx = _make_context(learnings=[], failed_approaches=[], constraints=[])
        result = serialize(ctx)
        assert result["learnings"] == []
        assert result["failed_approaches"] == []
        assert result["constraints"] == []

    def test_result_is_json_serializable(self):
        ctx = _make_context()
        result = serialize(ctx)
        # Should not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------

class TestDeserialize:
    def test_returns_loop_context(self):
        data = _make_raw_dict()
        ctx = deserialize(data)
        assert isinstance(ctx, LoopContext)

    def test_fields_match_input(self):
        data = _make_raw_dict(current_task="Fix bug", iteration_count=3)
        ctx = deserialize(data)
        assert ctx.current_task == "Fix bug"
        assert ctx.iteration_count == 3

    def test_learnings_round_trip(self):
        learning = {"pattern": "p", "resolution": "r", "confidence": 0.5, "source_iteration": 1}
        data = _make_raw_dict(learnings=[learning])
        ctx = deserialize(data)
        assert len(ctx.learnings) == 1
        assert ctx.learnings[0]["pattern"] == "p"

    def test_rejects_missing_keys(self):
        data = {"current_task": "x"}  # missing many keys
        with pytest.raises(ValueError, match="Missing required keys"):
            deserialize(data)

    def test_rejects_wrong_type_current_task(self):
        data = _make_raw_dict(current_task=123)
        with pytest.raises(ValueError, match="current_task.*string"):
            deserialize(data)

    def test_rejects_wrong_type_list_field(self):
        data = _make_raw_dict(acceptance_criteria="not a list")
        with pytest.raises(ValueError, match="acceptance_criteria.*list"):
            deserialize(data)

    def test_rejects_wrong_type_int_field(self):
        data = _make_raw_dict(iteration_count="two")
        with pytest.raises(ValueError, match="iteration_count.*integer"):
            deserialize(data)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_serialize_then_deserialize(self):
        original = _make_context()
        restored = deserialize(serialize(original))
        assert restored.current_task == original.current_task
        assert restored.acceptance_criteria == original.acceptance_criteria
        assert restored.constraints == original.constraints
        assert restored.learnings == original.learnings
        assert restored.failed_approaches == original.failed_approaches
        assert restored.iteration_count == original.iteration_count
        assert restored.max_iterations == original.max_iterations

    def test_round_trip_with_empty_collections(self):
        original = _make_context(learnings=[], failed_approaches=[], constraints=[])
        restored = deserialize(serialize(original))
        assert restored.learnings == []
        assert restored.failed_approaches == []
        assert restored.constraints == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_valid_data_no_errors(self):
        assert _validate_raw(_make_raw_dict()) == []

    def test_missing_keys_reported(self):
        errors = _validate_raw({})
        assert len(errors) == 1
        assert "Missing required keys" in errors[0]

    def test_wrong_type_string(self):
        errors = _validate_raw(_make_raw_dict(current_task=42))
        assert any("current_task" in e for e in errors)

    def test_wrong_type_list(self):
        errors = _validate_raw(_make_raw_dict(constraints="nope"))
        assert any("constraints" in e for e in errors)

    def test_wrong_type_int(self):
        errors = _validate_raw(_make_raw_dict(max_iterations=3.14))
        assert any("max_iterations" in e for e in errors)


# ---------------------------------------------------------------------------
# save() and load()
# ---------------------------------------------------------------------------

class TestSave:
    def test_creates_file(self, tmp_path):
        ctx = _make_context()
        path = str(tmp_path / "ctx.json")
        save(ctx, path)
        assert os.path.exists(path)

    def test_creates_parent_directories(self, tmp_path):
        ctx = _make_context()
        path = str(tmp_path / "deep" / "nested" / "ctx.json")
        save(ctx, path)
        assert os.path.exists(path)

    def test_file_contains_valid_json(self, tmp_path):
        ctx = _make_context()
        path = str(tmp_path / "ctx.json")
        save(ctx, path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["current_task"] == ctx.current_task

    def test_overwrites_existing_file(self, tmp_path):
        path = str(tmp_path / "ctx.json")
        save(_make_context(iteration_count=1), path)
        save(_make_context(iteration_count=2), path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["iteration_count"] == 2

    def test_atomic_write_no_temp_files_left(self, tmp_path):
        ctx = _make_context()
        path = str(tmp_path / "ctx.json")
        save(ctx, path)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        assert files[0].name == "ctx.json"


class TestLoad:
    def test_returns_none_for_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        assert load(path) is None

    def test_loads_saved_context(self, tmp_path):
        ctx = _make_context()
        path = str(tmp_path / "ctx.json")
        save(ctx, path)
        loaded = load(path)
        assert loaded is not None
        assert loaded.current_task == ctx.current_task
        assert loaded.iteration_count == ctx.iteration_count

    def test_raises_on_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load(str(path))

    def test_raises_on_non_object_json(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load(str(path))

    def test_raises_on_missing_keys(self, tmp_path):
        path = tmp_path / "partial.json"
        path.write_text('{"current_task": "x"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required keys"):
            load(str(path))

    def test_full_round_trip_via_filesystem(self, tmp_path):
        original = _make_context()
        path = str(tmp_path / "ctx.json")
        save(original, path)
        restored = load(path)
        assert restored is not None
        assert restored.current_task == original.current_task
        assert restored.acceptance_criteria == original.acceptance_criteria
        assert restored.constraints == original.constraints
        assert restored.learnings == original.learnings
        assert restored.failed_approaches == original.failed_approaches
        assert restored.iteration_count == original.iteration_count
        assert restored.max_iterations == original.max_iterations


class TestDefaultPath:
    def test_default_path_is_brainmass_dir(self):
        assert DEFAULT_CONTEXT_PATH == os.path.join(".brainmass", "loop-context.json")
