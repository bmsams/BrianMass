"""Unit tests for the SubagentManager.

Covers:
- Agents-as-Tools wrapping (Req 5.1)
- Isolated context/budget per subagent (Req 5.2)
- SubagentStop hook firing and structured AgentResult (Req 5.3)
- Scoped hook registration with Stop→SubagentStop conversion (Req 5.4)
- Multi-subagent execution and result synthesis (Req 5.5)
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest

from src.agents.subagent_manager import (
    SubagentManager,
    _convert_stop_to_subagent_stop,
    _default_agent_callback,
    _default_as_tool_callback,
)
from src.cost.cost_governor import CostGovernor
from src.hooks.hook_engine import BrainmassHookEngine
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    HookDefinition,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_agent_def(**overrides) -> AgentDefinition:
    """Create a minimal AgentDefinition with sensible defaults."""
    defaults = {
        "name": "test-agent",
        "description": "A test agent for unit testing",
        "model": "sonnet",
        "system_prompt": "You are a test agent.",
    }
    defaults.update(overrides)
    return AgentDefinition(**defaults)


def _make_budget(**overrides) -> AgentBudget:
    defaults = {
        "input_budget_tokens": 100_000,
        "output_budget_tokens": 25_000,
        "session_budget_usd": 2.0,
    }
    defaults.update(overrides)
    return AgentBudget(**defaults)


# ---------------------------------------------------------------------------
# Tests: _convert_stop_to_subagent_stop
# ---------------------------------------------------------------------------

class TestConvertStopToSubagentStop:
    """Req 5.4 — Stop hooks in agent frontmatter become SubagentStop."""

    def test_stop_key_converted(self):
        hooks = {
            HookEvent.STOP: [HookDefinition(matcher="*")],
        }
        result = _convert_stop_to_subagent_stop(hooks)
        assert HookEvent.SUBAGENT_STOP.value in result
        assert HookEvent.STOP.value not in result

    def test_string_stop_key_converted(self):
        hooks = {
            "Stop": [HookDefinition(matcher="*")],
        }
        result = _convert_stop_to_subagent_stop(hooks)
        assert HookEvent.SUBAGENT_STOP.value in result
        assert "Stop" not in result

    def test_non_stop_keys_preserved(self):
        hooks = {
            HookEvent.PRE_TOOL_USE: [HookDefinition(matcher="bash")],
            HookEvent.STOP: [HookDefinition(matcher="*")],
        }
        result = _convert_stop_to_subagent_stop(hooks)
        assert HookEvent.PRE_TOOL_USE.value in result
        assert HookEvent.SUBAGENT_STOP.value in result

    def test_empty_hooks(self):
        result = _convert_stop_to_subagent_stop({})
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Default callbacks
# ---------------------------------------------------------------------------

class TestDefaultCallbacks:
    def test_default_agent_callback_returns_complete(self):
        agent_def = _make_agent_def()
        budget = _make_budget()
        result = _default_agent_callback(agent_def, "do something", budget)
        assert result["exit_reason"] == "complete"
        assert "test-agent" in result["summary"]
        assert result["turns_used"] == 0

    def test_default_as_tool_callback(self):
        agent_def = _make_agent_def(name="code-reviewer")
        tool = _default_as_tool_callback(agent_def)
        assert tool["name"] == "agent:code-reviewer"
        assert tool["description"] == agent_def.description


# ---------------------------------------------------------------------------
# Tests: SubagentManager.execute
# ---------------------------------------------------------------------------

class TestSubagentManagerExecute:
    """Core execution flow — Req 5.2, 5.3."""

    def test_execute_returns_agent_result(self):
        manager = SubagentManager()
        agent_def = _make_agent_def()
        result = manager.execute(agent_def, "review code")
        assert isinstance(result, AgentResult)
        assert result.agent_name == "test-agent"
        assert result.exit_reason == "complete"

    def test_execute_uses_custom_callback(self):
        called_with = {}

        def my_callback(agent_def, task, budget):
            called_with["agent"] = agent_def.name
            called_with["task"] = task
            return {
                "summary": "Custom result",
                "tokens_consumed": {"input": 500, "output": 200, "cache_read": 0},
                "tools_used": ["read_file", "grep"],
                "files_modified": ["src/main.py"],
                "exit_reason": "complete",
                "turns_used": 3,
            }

        manager = SubagentManager(agent_callback=my_callback)
        result = manager.execute(_make_agent_def(), "fix bug")

        assert called_with["agent"] == "test-agent"
        assert called_with["task"] == "fix bug"
        assert result.summary == "Custom result"
        assert result.turns_used == 3
        assert result.tools_used == ["read_file", "grep"]
        assert result.files_modified == ["src/main.py"]

    def test_execute_creates_isolated_budget(self):
        """Req 5.2 — each subagent gets its own budget."""
        budgets_seen = []

        def capture_budget(agent_def, task, budget):
            budgets_seen.append(budget)
            return _default_agent_callback(agent_def, task, budget)

        parent_budget = _make_budget(session_budget_usd=10.0)
        manager = SubagentManager(agent_callback=capture_budget)
        manager.execute(_make_agent_def(), "task1", parent_budget)

        assert len(budgets_seen) == 1
        sub_budget = budgets_seen[0]
        # Subagent gets 25% of parent session budget
        assert sub_budget.session_budget_usd == pytest.approx(2.5)
        # Fresh budget — no usage yet
        assert sub_budget.current_cost_usd == 0.0

    def test_execute_default_budget_when_no_parent(self):
        budgets_seen = []

        def capture_budget(agent_def, task, budget):
            budgets_seen.append(budget)
            return _default_agent_callback(agent_def, task, budget)

        manager = SubagentManager(agent_callback=capture_budget)
        manager.execute(_make_agent_def(), "task1")

        sub_budget = budgets_seen[0]
        assert sub_budget.session_budget_usd == 1.0
        assert sub_budget.input_budget_tokens == 200_000

    def test_execute_handles_callback_error(self):
        def failing_callback(agent_def, task, budget):
            raise RuntimeError("Agent crashed")

        manager = SubagentManager(agent_callback=failing_callback)
        result = manager.execute(_make_agent_def(), "crash task")

        assert result.exit_reason == "error"
        assert "Agent crashed" in result.summary
        assert result.turns_used == 0


# ---------------------------------------------------------------------------
# Tests: SubagentManager with CostGovernor integration
# ---------------------------------------------------------------------------

class TestSubagentManagerCostGovernor:
    """Req 5.2 — budget registration and usage recording."""

    def test_registers_agent_with_cost_governor(self):
        governor = CostGovernor()

        def callback(agent_def, task, budget):
            return {
                "summary": "done",
                "tokens_consumed": {"input": 1000, "output": 500, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
                "turns_used": 1,
            }

        manager = SubagentManager(cost_governor=governor, agent_callback=callback)
        manager.execute(_make_agent_def(), "task")

        # The dashboard should show the subagent's usage
        dashboard = governor.get_dashboard_data()
        assert dashboard["token_consumption"]["total_input_tokens"] == 1000
        assert dashboard["token_consumption"]["total_output_tokens"] == 500

    def test_no_usage_recorded_when_zero_tokens(self):
        governor = CostGovernor()
        manager = SubagentManager(cost_governor=governor)
        manager.execute(_make_agent_def(), "task")

        dashboard = governor.get_dashboard_data()
        assert dashboard["token_consumption"]["total_input_tokens"] == 0

    def test_tracer_cost_span_does_not_fail_successful_execution(self):
        governor = CostGovernor()
        spans = []

        class FakeTracer:
            def trace_agent_action(self, **kwargs):
                return nullcontext()

            def record_cost_span(self, **kwargs):
                spans.append(kwargs)

        def callback(agent_def, task, budget):
            return {
                "summary": "done",
                "tokens_consumed": {"input": 200, "output": 100, "cache_read": 20},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
                "turns_used": 1,
            }

        manager = SubagentManager(
            cost_governor=governor,
            agent_callback=callback,
            tracer=FakeTracer(),
        )
        result = manager.execute(_make_agent_def(), "task")

        assert result.exit_reason == "complete"
        assert len(spans) == 1


# ---------------------------------------------------------------------------
# Tests: SubagentManager with HookEngine integration
# ---------------------------------------------------------------------------

class TestSubagentManagerHooks:
    """Req 5.3, 5.4 — SubagentStop firing and scoped hook registration."""

    def test_fires_subagent_stop_hook(self):
        engine = BrainmassHookEngine()
        fired_events = []

        def track_handler(handler, event, context):
            fired_events.append(event)
            return HookResult()

        engine.set_handler_callback(track_handler)
        engine.register(
            HookEvent.SUBAGENT_STOP,
            HookDefinition(matcher="*", hooks=[
                HookHandler(type=HookHandlerType.COMMAND, command="echo done")
            ]),
            scope="project_local",
        )

        manager = SubagentManager(hook_engine=engine)
        manager.execute(_make_agent_def(), "task")

        assert HookEvent.SUBAGENT_STOP in fired_events

    def test_scoped_hooks_cleaned_up_after_execution(self):
        engine = BrainmassHookEngine()
        agent_def = _make_agent_def(
            hooks={
                HookEvent.PRE_TOOL_USE: [
                    HookDefinition(
                        matcher="bash",
                        hooks=[HookHandler(type=HookHandlerType.COMMAND, command="lint")]
                    )
                ],
            }
        )

        manager = SubagentManager(hook_engine=engine)
        manager.execute(agent_def, "task")

        # After execution, scoped hooks should be cleaned up
        regs = engine.get_registrations(HookEvent.PRE_TOOL_USE)
        assert len(regs) == 0

    def test_stop_hooks_converted_to_subagent_stop(self):
        engine = BrainmassHookEngine()
        fired_events = []

        def track_handler(handler, event, context):
            fired_events.append(event)
            return HookResult()

        engine.set_handler_callback(track_handler)

        agent_def = _make_agent_def(
            hooks={
                HookEvent.STOP: [
                    HookDefinition(
                        matcher="*",
                        hooks=[HookHandler(type=HookHandlerType.COMMAND, command="notify")]
                    )
                ],
            }
        )

        manager = SubagentManager(hook_engine=engine)
        manager.execute(agent_def, "task")

        # The Stop hook should have been converted to SubagentStop and fired
        assert HookEvent.SUBAGENT_STOP in fired_events

    def test_subagent_stop_context_includes_agent_info(self):
        engine = BrainmassHookEngine()
        contexts_seen = []

        def capture_context(handler, event, context):
            contexts_seen.append(context)
            return HookResult()

        engine.set_handler_callback(capture_context)
        engine.register(
            HookEvent.SUBAGENT_STOP,
            HookDefinition(matcher="*", hooks=[
                HookHandler(type=HookHandlerType.COMMAND, command="log")
            ]),
            scope="project_local",
        )

        manager = SubagentManager(
            hook_engine=engine,
            session_id="sess-123",
            cwd="/project",
            session_type="interactive",
        )
        manager.execute(_make_agent_def(name="reviewer", model="opus"), "review")

        assert len(contexts_seen) == 1
        ctx = contexts_seen[0]
        assert ctx.session_id == "sess-123"
        assert ctx.hook_event_name == HookEvent.SUBAGENT_STOP
        assert ctx.cwd == "/project"
        assert ctx.source == "reviewer"
        assert ctx.model == "opus"


# ---------------------------------------------------------------------------
# Tests: SubagentManager.as_tools
# ---------------------------------------------------------------------------

class TestSubagentManagerAsTools:
    """Req 5.1 — Agents-as-Tools pattern."""

    def test_as_tools_returns_tool_descriptors(self):
        manager = SubagentManager()
        agents = [
            _make_agent_def(name="reviewer", description="Reviews code"),
            _make_agent_def(name="implementer", description="Writes code"),
        ]
        tools = manager.as_tools(agents)

        assert len(tools) == 2
        assert tools[0]["name"] == "agent:reviewer"
        assert tools[0]["description"] == "Reviews code"
        assert tools[1]["name"] == "agent:implementer"

    def test_as_tools_custom_callback(self):
        def custom_tool(agent_def):
            return {"name": f"custom:{agent_def.name}", "description": "custom"}

        manager = SubagentManager(as_tool_callback=custom_tool)
        tools = manager.as_tools([_make_agent_def(name="x")])
        assert tools[0]["name"] == "custom:x"

    def test_as_tools_empty_list(self):
        manager = SubagentManager()
        assert manager.as_tools([]) == []


# ---------------------------------------------------------------------------
# Tests: SubagentManager.execute_multiple
# ---------------------------------------------------------------------------

class TestSubagentManagerExecuteMultiple:
    """Req 5.5 — synthesize results from multiple subagents."""

    def test_execute_multiple_returns_all_results(self):
        call_order = []

        def ordered_callback(agent_def, task, budget):
            call_order.append(agent_def.name)
            return {
                "summary": f"Done: {task}",
                "tokens_consumed": {"input": 100, "output": 50, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
                "turns_used": 1,
            }

        manager = SubagentManager(agent_callback=ordered_callback)
        tasks = [
            (_make_agent_def(name="agent-a"), "task A"),
            (_make_agent_def(name="agent-b"), "task B"),
            (_make_agent_def(name="agent-c"), "task C"),
        ]
        results = manager.execute_multiple(tasks)

        assert len(results) == 3
        assert [r.agent_name for r in results] == ["agent-a", "agent-b", "agent-c"]
        # Sequential execution preserves order
        assert call_order == ["agent-a", "agent-b", "agent-c"]

    def test_execute_multiple_with_parent_budget(self):
        budgets_seen = []

        def capture(agent_def, task, budget):
            budgets_seen.append(budget.session_budget_usd)
            return _default_agent_callback(agent_def, task, budget)

        parent = _make_budget(session_budget_usd=8.0)
        manager = SubagentManager(agent_callback=capture)
        manager.execute_multiple(
            [(_make_agent_def(name="a"), "t1"), (_make_agent_def(name="b"), "t2")],
            parent_budget=parent,
        )

        # Each subagent gets 25% of parent
        assert all(b == pytest.approx(2.0) for b in budgets_seen)

    def test_execute_multiple_partial_failure(self):
        """One failing subagent shouldn't prevent others from running."""
        call_count = [0]

        def sometimes_fail(agent_def, task, budget):
            call_count[0] += 1
            if agent_def.name == "bad-agent":
                raise RuntimeError("boom")
            return _default_agent_callback(agent_def, task, budget)

        manager = SubagentManager(agent_callback=sometimes_fail)
        results = manager.execute_multiple([
            (_make_agent_def(name="good-1"), "t1"),
            (_make_agent_def(name="bad-agent"), "t2"),
            (_make_agent_def(name="good-2"), "t3"),
        ])

        assert len(results) == 3
        assert results[0].exit_reason == "complete"
        assert results[1].exit_reason == "error"
        assert results[2].exit_reason == "complete"


# ---------------------------------------------------------------------------
# Tests: Model tier resolution
# ---------------------------------------------------------------------------

class TestModelTierResolution:
    def test_known_aliases(self):
        from src.types.core import ModelTier
        assert SubagentManager._resolve_model_tier("opus") == ModelTier.OPUS
        assert SubagentManager._resolve_model_tier("sonnet") == ModelTier.SONNET
        assert SubagentManager._resolve_model_tier("haiku") == ModelTier.HAIKU
        assert SubagentManager._resolve_model_tier("inherit") == ModelTier.SONNET

    def test_unknown_alias_defaults_to_sonnet(self):
        from src.types.core import ModelTier
        assert SubagentManager._resolve_model_tier("unknown") == ModelTier.SONNET
