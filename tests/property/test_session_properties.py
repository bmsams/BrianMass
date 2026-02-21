"""Property-based tests for Session Teleporter.

Property 18: Session serialization round-trip
For any valid session state (conversation history, tool permissions, active workers,
pending approvals, compaction state, cost tracking, hook registrations, trace ID),
serializing to a portable blob and deserializing SHALL produce an equivalent session
state with all fields preserved.

Validates: Requirements 13.1
"""

from __future__ import annotations

import uuid

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.session.teleporter import SessionTeleporter
from src.types.core import SessionState

# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

# Conversation message strategy
_role = st.sampled_from(["user", "assistant"])
_message = st.fixed_dictionaries({"role": _role, "content": st.text(min_size=0, max_size=200)})
_conversation = st.lists(_message, min_size=0, max_size=10)

# Tool permissions: tool name → bool
_tool_name = st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_"), min_size=1, max_size=30)
_tool_permissions = st.dictionaries(_tool_name, st.booleans(), max_size=10)

# Active workers: list of worker IDs
_worker_id = st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-"), min_size=1, max_size=40)
_active_workers = st.lists(_worker_id, min_size=0, max_size=5)

# Pending approvals: list of dicts
_approval = st.fixed_dictionaries({"tool": _tool_name, "approved": st.booleans()})
_pending_approvals = st.lists(_approval, min_size=0, max_size=5)

# Compaction state dict
_compaction_state = st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.one_of(st.text(max_size=50), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
    max_size=5,
)

# Context manager state dict
_context_state = st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.one_of(st.integers(min_value=0, max_value=10_000_000), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    max_size=5,
)

# Cost tracking dict
_cost_tracking = st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.one_of(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.integers(min_value=0, max_value=10_000_000),
    ),
    max_size=5,
)

# Hook registrations dict
_hook_registrations = st.dictionaries(
    st.sampled_from(["PreToolUse", "PostToolUse", "Stop", "SessionStart", "SubagentStop"]),
    st.lists(st.text(max_size=50), max_size=3),
    max_size=5,
)

# Trace ID
_trace_id = st.one_of(
    st.uuids().map(str),
    st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-"), min_size=8, max_size=64),
)

# Full SessionState strategy
_session_state = st.builds(
    SessionState,
    conversation_history=_conversation,
    tool_permissions=_tool_permissions,
    active_workers=_active_workers,
    pending_approvals=_pending_approvals,
    compaction_state=_compaction_state,
    context_manager_state=_context_state,
    cost_tracking=_cost_tracking,
    hook_registrations=_hook_registrations,
    trace_id=_trace_id,
)


# ---------------------------------------------------------------------------
# Property 18: Session serialization round-trip
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(state=_session_state)
@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow])
def test_session_serialization_round_trip(state: SessionState) -> None:
    """**Property 18: Session serialization round-trip**

    For any valid session state, serializing to a portable blob and
    deserializing SHALL produce an equivalent session state with all
    fields preserved.

    Validates: Requirements 13.1
    """
    teleporter = SessionTeleporter()
    blob = teleporter.serialize(state)
    restored = teleporter.deserialize(blob)

    # All fields must be preserved exactly
    assert restored.conversation_history == state.conversation_history
    assert restored.tool_permissions == state.tool_permissions
    assert restored.active_workers == state.active_workers
    assert restored.pending_approvals == state.pending_approvals
    assert restored.compaction_state == state.compaction_state
    assert restored.context_manager_state == state.context_manager_state
    assert restored.cost_tracking == state.cost_tracking
    assert restored.hook_registrations == state.hook_registrations
    assert restored.trace_id == state.trace_id


@pytest.mark.property
@given(state=_session_state)
@settings(max_examples=25)
def test_serialization_is_deterministic(state: SessionState) -> None:
    """Serializing the same state twice produces equivalent blobs (same content).

    Validates: Requirements 13.1
    """
    teleporter = SessionTeleporter()
    blob1 = teleporter.serialize(state)
    blob2 = teleporter.serialize(state)

    # Both blobs must deserialize to the same state
    r1 = teleporter.deserialize(blob1)
    r2 = teleporter.deserialize(blob2)

    assert r1.trace_id == r2.trace_id
    assert r1.conversation_history == r2.conversation_history
    assert r1.tool_permissions == r2.tool_permissions
    assert r1.active_workers == r2.active_workers
    assert r1.cost_tracking == r2.cost_tracking


@pytest.mark.property
@given(state=_session_state)
@settings(max_examples=25)
def test_trace_id_invariant_across_teleport(state: SessionState) -> None:
    """The trace ID must never change during a teleport cycle.

    Validates: Requirements 13.6
    """
    teleporter = SessionTeleporter()
    original_trace = state.trace_id
    session_key = teleporter.teleport(state, from_surface="web", to_surface="cli")
    restored = teleporter.resume(session_key, surface="cli")
    assert restored.trace_id == original_trace


@pytest.mark.property
@given(
    state=_session_state,
    surface=st.sampled_from(["web", "mobile", "desktop", "cli"]),
)
@settings(max_examples=25)
def test_all_surfaces_can_resume(state: SessionState, surface: str) -> None:
    """Any valid surface can resume a teleported session.

    Validates: Requirements 13.2, 13.3
    """
    teleporter = SessionTeleporter()
    # Teleport from a different surface than we resume on
    from_surface = "web" if surface != "web" else "cli"
    session_key = teleporter.teleport(state, from_surface=from_surface, to_surface=surface)
    restored = teleporter.resume(session_key, surface=surface)
    assert restored.trace_id == state.trace_id


@pytest.mark.property
@given(
    base_permissions=_tool_permissions,
    incoming_permissions=_tool_permissions,
)
@settings(max_examples=25)
def test_permission_merge_deny_beats_allow(
    base_permissions: dict[str, bool],
    incoming_permissions: dict[str, bool],
) -> None:
    """For any shared tool, deny (False) from either state beats allow (True).

    Validates: Requirements 13.4
    """
    teleporter = SessionTeleporter()
    base = SessionState(
        conversation_history=[],
        tool_permissions=base_permissions,
        active_workers=[],
        pending_approvals=[],
        compaction_state={},
        context_manager_state={},
        cost_tracking={},
        hook_registrations={},
        trace_id=str(uuid.uuid4()),
    )
    incoming = SessionState(
        conversation_history=[],
        tool_permissions=incoming_permissions,
        active_workers=[],
        pending_approvals=[],
        compaction_state={},
        context_manager_state={},
        cost_tracking={},
        hook_registrations={},
        trace_id=str(uuid.uuid4()),
    )
    merged = teleporter.merge_permissions(base, incoming)

    for tool in set(base_permissions) & set(incoming_permissions):
        expected = base_permissions[tool] and incoming_permissions[tool]
        assert merged[tool] == expected, (
            f"Tool '{tool}': base={base_permissions[tool]}, "
            f"incoming={incoming_permissions[tool]}, merged={merged[tool]}, expected={expected}"
        )
