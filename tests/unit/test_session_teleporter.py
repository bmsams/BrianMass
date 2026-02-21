"""Unit tests for Session Teleporter.

Tests serialization, surface transfer, permission inheritance, worker lifecycle,
audit continuity, cost attribution, and conflict resolution.

Requirements: 13.1–13.8
"""

from __future__ import annotations

import json
import uuid

import pytest

from src.session.teleporter import (
    AgentCoreSessionMemoryAdapter,
    ConflictRecord,
    SessionTeleporter,
)
from src.types.core import SessionState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(**overrides) -> SessionState:
    defaults = dict(
        conversation_history=[{"role": "user", "content": "hello"}],
        tool_permissions={"bash": True, "read_file": True},
        active_workers=["worker-1"],
        pending_approvals=[],
        compaction_state={"last_compacted_at": "2025-01-01T00:00:00Z"},
        context_manager_state={"total_tokens": 5000},
        cost_tracking={"total_usd": 0.05, "input_tokens": 1000},
        hook_registrations={"PreToolUse": []},
        trace_id=str(uuid.uuid4()),
    )
    defaults.update(overrides)
    return SessionState(**defaults)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_serialize_produces_bytes(self):
        t = SessionTeleporter()
        state = make_state()
        blob = t.serialize(state)
        assert isinstance(blob, bytes)
        assert len(blob) > 0

    def test_deserialize_round_trip(self):
        t = SessionTeleporter()
        state = make_state()
        blob = t.serialize(state)
        restored = t.deserialize(blob)

        assert restored.conversation_history == state.conversation_history
        assert restored.tool_permissions == state.tool_permissions
        assert restored.active_workers == state.active_workers
        assert restored.pending_approvals == state.pending_approvals
        assert restored.compaction_state == state.compaction_state
        assert restored.context_manager_state == state.context_manager_state
        assert restored.cost_tracking == state.cost_tracking
        assert restored.hook_registrations == state.hook_registrations
        assert restored.trace_id == state.trace_id

    def test_blob_contains_schema_version(self):
        t = SessionTeleporter()
        blob = t.serialize(make_state())
        payload = json.loads(blob)
        assert payload["schema_version"] == "1.0"

    def test_blob_contains_serialized_at(self):
        t = SessionTeleporter()
        blob = t.serialize(make_state())
        payload = json.loads(blob)
        assert "serialized_at" in payload

    def test_empty_conversation_history(self):
        t = SessionTeleporter()
        state = make_state(conversation_history=[])
        restored = t.deserialize(t.serialize(state))
        assert restored.conversation_history == []

    def test_empty_active_workers(self):
        t = SessionTeleporter()
        state = make_state(active_workers=[])
        restored = t.deserialize(t.serialize(state))
        assert restored.active_workers == []

    def test_trace_id_preserved(self):
        t = SessionTeleporter()
        trace_id = "fixed-trace-id-abc123"
        state = make_state(trace_id=trace_id)
        restored = t.deserialize(t.serialize(state))
        assert restored.trace_id == trace_id


# ---------------------------------------------------------------------------
# Teleport / surface transfer
# ---------------------------------------------------------------------------


class TestTeleport:
    def test_teleport_returns_session_key(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web", to_surface="cli")
        assert isinstance(key, str)
        assert state.trace_id in key

    def test_teleport_stores_blob(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web", to_surface="cli")
        # Should be loadable
        blob = t._load_backend(key)
        assert len(blob) > 0

    def test_resume_restores_state(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web", to_surface="cli")
        restored = t.resume(key, surface="cli")
        assert restored.trace_id == state.trace_id
        assert restored.active_workers == state.active_workers

    def test_desktop_handoff_returns_key(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.desktop_handoff(state, from_surface="cli")
        assert isinstance(key, str)

    def test_invalid_surface_raises(self):
        t = SessionTeleporter()
        state = make_state()
        with pytest.raises(ValueError, match="Unknown surface"):
            t.teleport(state, from_surface="fax", to_surface="cli")

    def test_invalid_target_surface_raises(self):
        t = SessionTeleporter()
        state = make_state()
        with pytest.raises(ValueError, match="Unknown surface"):
            t.teleport(state, from_surface="cli", to_surface="pager")

    def test_resume_invalid_surface_raises(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web")
        with pytest.raises(ValueError, match="Unknown surface"):
            t.resume(key, surface="telegraph")

    def test_resume_missing_key_raises(self):
        t = SessionTeleporter()
        with pytest.raises(KeyError):
            t.resume("nonexistent-key", surface="cli")


# ---------------------------------------------------------------------------
# Permission inheritance
# ---------------------------------------------------------------------------


class TestPermissionInheritance:
    def test_merge_preserves_approvals(self):
        t = SessionTeleporter()
        base = make_state(tool_permissions={"bash": True, "read_file": True})
        incoming = make_state(tool_permissions={"bash": True, "write_file": True})
        merged = t.merge_permissions(base, incoming)
        assert merged["bash"] is True
        assert merged["read_file"] is True
        assert merged["write_file"] is True

    def test_deny_beats_allow(self):
        """False (deny) from either state takes precedence — safety first."""
        t = SessionTeleporter()
        base = make_state(tool_permissions={"bash": True})
        incoming = make_state(tool_permissions={"bash": False})
        merged = t.merge_permissions(base, incoming)
        assert merged["bash"] is False

    def test_deny_in_base_beats_allow_in_incoming(self):
        t = SessionTeleporter()
        base = make_state(tool_permissions={"bash": False})
        incoming = make_state(tool_permissions={"bash": True})
        merged = t.merge_permissions(base, incoming)
        assert merged["bash"] is False

    def test_new_tool_from_incoming_added(self):
        t = SessionTeleporter()
        base = make_state(tool_permissions={})
        incoming = make_state(tool_permissions={"new_tool": True})
        merged = t.merge_permissions(base, incoming)
        assert merged["new_tool"] is True

    def test_empty_permissions_merge(self):
        t = SessionTeleporter()
        base = make_state(tool_permissions={})
        incoming = make_state(tool_permissions={})
        merged = t.merge_permissions(base, incoming)
        assert merged == {}


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_workers_survive_transfer(self):
        t = SessionTeleporter()
        state = make_state(active_workers=["worker-1", "worker-2"])
        transferred = t.transfer_workers(state, target_surface="desktop")
        assert transferred.active_workers == ["worker-1", "worker-2"]

    def test_empty_workers_transfer(self):
        t = SessionTeleporter()
        state = make_state(active_workers=[])
        transferred = t.transfer_workers(state, target_surface="cli")
        assert transferred.active_workers == []

    def test_workers_preserved_in_serialization(self):
        t = SessionTeleporter()
        state = make_state(active_workers=["subagent-a", "teammate-b"])
        restored = t.deserialize(t.serialize(state))
        assert restored.active_workers == ["subagent-a", "teammate-b"]


# ---------------------------------------------------------------------------
# Audit continuity (trace ID)
# ---------------------------------------------------------------------------


class TestAuditContinuity:
    def test_ensure_trace_id_preserves_existing(self):
        t = SessionTeleporter()
        original_id = "my-trace-id"
        state = make_state(trace_id=original_id)
        result = t.ensure_trace_id(state)
        assert result.trace_id == original_id

    def test_ensure_trace_id_assigns_when_empty(self):
        t = SessionTeleporter()
        state = make_state(trace_id="")
        result = t.ensure_trace_id(state)
        assert result.trace_id != ""
        assert len(result.trace_id) > 0

    def test_trace_id_stable_across_teleport(self):
        t = SessionTeleporter()
        state = make_state()
        original_trace = state.trace_id
        key = t.teleport(state, from_surface="web")
        restored = t.resume(key, surface="cli")
        assert restored.trace_id == original_trace


# ---------------------------------------------------------------------------
# Cost attribution
# ---------------------------------------------------------------------------


class TestCostAttribution:
    def test_merge_cost_tracking_sums_numeric(self):
        t = SessionTeleporter()
        base = make_state(cost_tracking={"total_usd": 0.05, "input_tokens": 1000})
        incoming = make_state(cost_tracking={"total_usd": 0.03, "input_tokens": 500})
        merged = t.merge_cost_tracking(base, incoming)
        assert merged["total_usd"] == pytest.approx(0.08)
        assert merged["input_tokens"] == 1500

    def test_merge_cost_tracking_adds_new_keys(self):
        t = SessionTeleporter()
        base = make_state(cost_tracking={"total_usd": 0.01})
        incoming = make_state(cost_tracking={"output_tokens": 200})
        merged = t.merge_cost_tracking(base, incoming)
        assert merged["total_usd"] == pytest.approx(0.01)
        assert merged["output_tokens"] == 200

    def test_merge_cost_tracking_non_numeric_overwritten(self):
        t = SessionTeleporter()
        base = make_state(cost_tracking={"surface": "web"})
        incoming = make_state(cost_tracking={"surface": "cli"})
        merged = t.merge_cost_tracking(base, incoming)
        assert merged["surface"] == "cli"

    def test_cost_tracking_preserved_in_serialization(self):
        t = SessionTeleporter()
        state = make_state(cost_tracking={"total_usd": 1.23, "input_tokens": 50000})
        restored = t.deserialize(t.serialize(state))
        assert restored.cost_tracking["total_usd"] == pytest.approx(1.23)
        assert restored.cost_tracking["input_tokens"] == 50000


# ---------------------------------------------------------------------------
# Conflict detection (last-write-wins)
# ---------------------------------------------------------------------------


class TestConflictDetection:
    def test_no_conflict_on_first_resume(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web")
        t.resume(key, surface="cli")
        # First resume — no prior write from another surface
        assert len(t.get_conflicts()) == 0

    def test_conflict_recorded_on_concurrent_resume(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web")
        # Both surfaces resume — second one detects conflict
        t.resume(key, surface="cli")
        t.resume(key, surface="desktop")
        assert len(t.get_conflicts()) >= 1

    def test_clear_conflicts(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web")
        t.resume(key, surface="cli")
        t.resume(key, surface="desktop")
        t.clear_conflicts()
        assert t.get_conflicts() == []

    def test_conflict_record_has_surface_and_timestamp(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web")
        t.resume(key, surface="cli")
        t.resume(key, surface="desktop")
        conflicts = t.get_conflicts()
        if conflicts:
            c = conflicts[0]
            assert isinstance(c, ConflictRecord)
            assert c.surface in VALID_SURFACES_FOR_TEST
            assert c.timestamp is not None

    def test_conflict_to_dict(self):
        t = SessionTeleporter()
        state = make_state()
        key = t.teleport(state, from_surface="web")
        t.resume(key, surface="cli")
        t.resume(key, surface="desktop")
        conflicts = t.get_conflicts()
        if conflicts:
            d = conflicts[0].to_dict()
            assert "surface" in d
            assert "timestamp" in d
            assert "field" in d


VALID_SURFACES_FOR_TEST = {"web", "mobile", "desktop", "cli"}


class TestAgentCoreSessionMemoryAdapter:
    def test_legacy_client_interface(self):
        class LegacyClient:
            def __init__(self):
                self._rows = []

            def create_memory(self, namespace, content):
                self._rows.append({"content": content})
                return {"id": "legacy-id"}

            def query(self, namespace, query, top_k=1):
                return list(self._rows)

        adapter = AgentCoreSessionMemoryAdapter(client=LegacyClient())
        adapter.write_session_metadata("session:abc", {"from_surface": "web"})
        metadata = adapter.read_session_metadata("session:abc")

        assert metadata["from_surface"] == "web"

    def test_current_client_interface(self):
        class NewClient:
            def __init__(self):
                self.events = []

            def create_or_get_memory(self, name):
                return {"id": "mem-1", "name": name}

            def create_event(self, memory_id, actor_id, session_id, messages):
                self.events.append(
                    {
                        "memory_id": memory_id,
                        "actor_id": actor_id,
                        "session_id": session_id,
                        "messages": messages,
                    }
                )
                return {"event_id": "evt-1"}

            def retrieve_memories(self, memory_id, namespace, query, top_k=3):
                return [{"text": '{"session_key":"session:abc","metadata":{"to_surface":"cli"}}'}]

        client = NewClient()
        adapter = AgentCoreSessionMemoryAdapter(client=client)
        adapter.write_session_metadata("session:abc", {"to_surface": "cli"})
        metadata = adapter.read_session_metadata("session:abc")

        assert client.events[0]["memory_id"] == "mem-1"
        assert metadata["to_surface"] == "cli"
