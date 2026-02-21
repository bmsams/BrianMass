"""Integration tests for full request lifecycle behavior."""

from __future__ import annotations

from src.context.context_manager import ContextManager
from src.cost.cost_governor import CostGovernor
from src.hooks.hook_engine import BrainmassHookEngine
from src.orchestrator.orchestrator import Orchestrator
from src.types.core import (
    ContextCategory,
    HookContext,
    HookDefinition,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
    ModelTier,
)


def test_request_lifecycle_hooks_cost_and_context(tmp_path):
    events: list[str] = []

    class RecordingHookEngine:
        def fire(self, event, context):
            events.append(event.value)
            return HookResult()

    cm = ContextManager(session_id="integration-req", session_dir=str(tmp_path))
    cg = CostGovernor()

    def model_cb(prompt, tier, effort):
        return (
            "Draft response. "
            '[TOOL:read_file {"path":"src/agents/agent_dispatcher.py"}] '
            '[TOOL:broken_tool {"action":"fail"}]'
        )

    def tool_exec(tool_name, tool_input):
        if tool_name == "broken_tool":
            raise RuntimeError("simulated tool failure")
        return f"{tool_name} ok"

    orch = Orchestrator(
        context_manager=cm,
        hook_engine=RecordingHookEngine(),
        cost_governor=cg,
        session_id="sess-lifecycle",
        cwd=str(tmp_path),
        model_callback=model_cb,
        tool_executor=tool_exec,
    )

    first = orch.process_request("Review and debug the dispatcher flow")
    expected_order = [
        HookEvent.SESSION_START.value,
        HookEvent.USER_PROMPT_SUBMIT.value,
        HookEvent.PRE_TOOL_USE.value,
        HookEvent.POST_TOOL_USE.value,
        HookEvent.PRE_TOOL_USE.value,
        HookEvent.POST_TOOL_USE_FAILURE.value,
        HookEvent.STOP.value,
    ]
    assert events == expected_order
    assert len(first.tool_calls) == 2
    assert first.total_cost_usd > 0.0
    assert first.model_tier in list(ModelTier)

    cost_after_first = cg.get_dashboard_data()["cost_per_agent"].get(orch.agent_id, 0.0)
    second = orch.process_request("Repeat request to validate usage accrual")
    assert second.total_cost_usd > 0.0
    cost_after_second = cg.get_dashboard_data()["cost_per_agent"].get(orch.agent_id, 0.0)
    assert cost_after_second > cost_after_first

    verbatim = cm.add_item("Error: failed at line 17", "assistant", 20)
    structured = cm.add_item("DECISION: keep retry logic", "user", 12)
    ephemeral = cm.add_item("Searching for symbols...", "tool_call", 8)
    assert verbatim.category == ContextCategory.PRESERVE_VERBATIM
    assert structured.category == ContextCategory.PRESERVE_STRUCTURED
    assert ephemeral.category == ContextCategory.EPHEMERAL
    assert len(cm.items) == 3


def test_all_12_hook_events_fire_in_defined_order():
    engine = BrainmassHookEngine()
    observed: list[str] = []

    def track_handler(handler, event, context):
        observed.append(event.value)
        return HookResult()

    engine.set_handler_callback(track_handler)
    for event in HookEvent:
        engine.register(
            event,
            HookDefinition(
                matcher="*",
                hooks=[HookHandler(type=HookHandlerType.COMMAND, command="echo noop")],
            ),
            scope="project_local",
        )

    tool_events = {
        HookEvent.PRE_TOOL_USE,
        HookEvent.POST_TOOL_USE,
        HookEvent.POST_TOOL_USE_FAILURE,
        HookEvent.PERMISSION_REQUEST,
    }
    for event in HookEvent:
        ctx = HookContext(
            session_id="sess-hooks",
            hook_event_name=event,
            cwd=".",
            session_type="interactive",
            source="new" if event == HookEvent.SESSION_START else None,
            tool_name="read_file" if event in tool_events else None,
        )
        engine.fire(event, ctx)

    assert observed == [event.value for event in HookEvent]
