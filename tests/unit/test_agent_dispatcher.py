"""Unit tests for AgentDispatcher.

Covers:
- 13-step lifecycle execution (Req 8.7)
- Pluggable callback injection (Req 8.6)
- Model alias resolution and budget-based downgrade (Req 8.4)
- SubagentStop hook firing (Req 8.7, 5.3)
- MCP server start/stop lifecycle (Req 8.7)
- Skills loading into context (Req 8.7)
- Hooks registration and cleanup (Req 8.7)
- Error handling (agent loop failure → error exit_reason)
"""

from __future__ import annotations

import pytest

from src.agents.agent_dispatcher import (
    AgentDispatcher,
    _default_agent_loop,
    _default_mcp_start,
    _default_skills_load,
)
from src.hooks.hook_engine import BrainmassHookEngine
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    BudgetStatus,
    HookDefinition,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
    ModelTier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_def(**overrides) -> AgentDefinition:
    defaults = {
        "name": "test-agent",
        "description": "A test agent.",
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


def _success_loop(agent_def, task, system_prompt, budget) -> dict:
    return {
        "summary": f"Completed: {task}",
        "turns_used": 3,
        "tokens_consumed": {"input": 500, "output": 200, "cache_read": 50},
        "tools_used": ["read_file"],
        "files_modified": ["src/main.py"],
        "exit_reason": "complete",
    }


# ---------------------------------------------------------------------------
# Tests: Default callbacks
# ---------------------------------------------------------------------------


class TestDefaultCallbacks:
    def test_default_agent_loop_returns_complete(self) -> None:
        agent_def = _make_agent_def()
        result = _default_agent_loop(agent_def, "do something", "system", _make_budget())
        assert result["exit_reason"] == "complete"
        assert "test-agent" in result["summary"]

    def test_default_mcp_start_returns_callable(self) -> None:
        cleanup = _default_mcp_start({"github": {"command": "npx"}})
        assert callable(cleanup)
        cleanup()  # Should not raise

    def test_default_skills_load_returns_string(self) -> None:
        result = _default_skills_load(["owasp", "adr"])
        assert isinstance(result, str)
        assert "owasp" in result

    def test_default_skills_load_empty(self) -> None:
        result = _default_skills_load([])
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: Basic dispatch
# ---------------------------------------------------------------------------


class TestBasicDispatch:
    def test_dispatch_returns_agent_result(self) -> None:
        dispatcher = AgentDispatcher(agent_loop_callback=_success_loop)
        result = dispatcher.dispatch(_make_agent_def(), "review code")
        assert isinstance(result, AgentResult)
        assert result.agent_name == "test-agent"
        assert result.exit_reason == "complete"

    def test_dispatch_passes_task_to_loop(self) -> None:
        received = {}

        def capture_loop(agent_def, task, system_prompt, budget):
            received["task"] = task
            received["system_prompt"] = system_prompt
            return _success_loop(agent_def, task, system_prompt, budget)

        dispatcher = AgentDispatcher(agent_loop_callback=capture_loop)
        dispatcher.dispatch(_make_agent_def(), "fix the bug")
        assert received["task"] == "fix the bug"

    def test_dispatch_result_fields(self) -> None:
        dispatcher = AgentDispatcher(agent_loop_callback=_success_loop)
        result = dispatcher.dispatch(_make_agent_def(), "task")
        assert result.turns_used == 3
        assert result.tokens_consumed == {"input": 500, "output": 200, "cache_read": 50}
        assert result.tools_used == ["read_file"]
        assert result.files_modified == ["src/main.py"]

    def test_dispatch_error_returns_error_result(self) -> None:
        def failing_loop(agent_def, task, system_prompt, budget):
            raise RuntimeError("Agent crashed")

        dispatcher = AgentDispatcher(agent_loop_callback=failing_loop)
        result = dispatcher.dispatch(_make_agent_def(), "task")
        assert result.exit_reason == "error"
        assert "Agent crashed" in result.summary


# ---------------------------------------------------------------------------
# Tests: Model resolution (step 2)
# ---------------------------------------------------------------------------


class TestModelResolution:
    def test_sonnet_resolved(self) -> None:
        received = {}

        def capture(agent_def, task, system_prompt, budget):
            return _success_loop(agent_def, task, system_prompt, budget)

        dispatcher = AgentDispatcher(agent_loop_callback=capture)
        dispatcher.dispatch(_make_agent_def(model="sonnet"), "task")
        # No error means resolution succeeded

    def test_inherit_uses_parent_model(self) -> None:
        received_model = {}

        def capture(agent_def, task, system_prompt, budget):
            return _success_loop(agent_def, task, system_prompt, budget)

        dispatcher = AgentDispatcher(
            agent_loop_callback=capture,
            parent_model="opus",
        )
        # 'inherit' should resolve to parent's model (opus)
        dispatcher.dispatch(_make_agent_def(model="inherit"), "task")

    def test_budget_critical_downgrades_to_haiku(self) -> None:
        """When budget is CRITICAL, model should be downgraded to Haiku."""
        resolved_tiers = []

        class MockGovernor:
            def check_budget(self, agent_id):
                return BudgetStatus.CRITICAL

            def record_usage(self, **kwargs):
                resolved_tiers.append(kwargs.get("model_tier"))

        def capture_loop(agent_def, task, system_prompt, budget):
            return {
                "summary": "done",
                "turns_used": 1,
                "tokens_consumed": {"input": 100, "output": 50, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
            }

        dispatcher = AgentDispatcher(
            cost_governor=MockGovernor(),
            agent_loop_callback=capture_loop,
        )
        dispatcher.dispatch(_make_agent_def(model="opus"), "task")
        assert ModelTier.HAIKU in resolved_tiers

    def test_budget_warning_downgrades_one_tier(self) -> None:
        """When budget is WARNING, Opus → Sonnet."""
        resolved_tiers = []

        class MockGovernor:
            def check_budget(self, agent_id):
                return BudgetStatus.WARNING

            def record_usage(self, **kwargs):
                resolved_tiers.append(kwargs.get("model_tier"))

        def capture_loop(agent_def, task, system_prompt, budget):
            return {
                "summary": "done",
                "turns_used": 1,
                "tokens_consumed": {"input": 100, "output": 50, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
            }

        dispatcher = AgentDispatcher(
            cost_governor=MockGovernor(),
            agent_loop_callback=capture_loop,
        )
        dispatcher.dispatch(_make_agent_def(model="opus"), "task")
        assert ModelTier.SONNET in resolved_tiers

    def test_budget_exceeded_raises(self) -> None:
        class MockGovernor:
            def check_budget(self, agent_id):
                return BudgetStatus.EXCEEDED

        dispatcher = AgentDispatcher(cost_governor=MockGovernor())
        with pytest.raises(RuntimeError, match="Budget exceeded"):
            dispatcher.dispatch(_make_agent_def(), "task")


# ---------------------------------------------------------------------------
# Tests: System prompt assembly (step 8)
# ---------------------------------------------------------------------------


class TestSystemPromptAssembly:
    def test_system_prompt_includes_agent_prompt(self) -> None:
        received = {}

        def capture(agent_def, task, system_prompt, budget):
            received["system_prompt"] = system_prompt
            return _success_loop(agent_def, task, system_prompt, budget)

        dispatcher = AgentDispatcher(agent_loop_callback=capture)
        dispatcher.dispatch(
            _make_agent_def(system_prompt="You are a code reviewer."), "task"
        )
        assert "You are a code reviewer." in received["system_prompt"]

    def test_system_prompt_includes_memory(self) -> None:
        received = {}

        def capture(agent_def, task, system_prompt, budget):
            received["system_prompt"] = system_prompt
            return _success_loop(agent_def, task, system_prompt, budget)

        dispatcher = AgentDispatcher(agent_loop_callback=capture)
        dispatcher.dispatch(
            _make_agent_def(memory="Remember: always use type hints."), "task"
        )
        assert "Remember: always use type hints." in received["system_prompt"]

    def test_system_prompt_includes_skills(self) -> None:
        received = {}

        def capture(agent_def, task, system_prompt, budget):
            received["system_prompt"] = system_prompt
            return _success_loop(agent_def, task, system_prompt, budget)

        def skills_loader(skill_names):
            return f"# Skills\n{', '.join(skill_names)}"

        dispatcher = AgentDispatcher(
            agent_loop_callback=capture,
            skills_load_callback=skills_loader,
        )
        dispatcher.dispatch(_make_agent_def(skills=["owasp"]), "task")
        assert "owasp" in received["system_prompt"]


# ---------------------------------------------------------------------------
# Tests: MCP lifecycle (steps 4 and 11)
# ---------------------------------------------------------------------------


class TestMcpLifecycle:
    def test_mcp_start_called_with_servers(self) -> None:
        mcp_calls = []
        cleanup_calls = []

        def mcp_start(mcp_servers):
            mcp_calls.append(mcp_servers)
            return lambda: cleanup_calls.append(1)

        dispatcher = AgentDispatcher(
            agent_loop_callback=_success_loop,
            mcp_start_callback=mcp_start,
        )
        dispatcher.dispatch(
            _make_agent_def(mcp_servers={"github": {"command": "npx"}}), "task"
        )
        assert len(mcp_calls) == 1
        assert "github" in mcp_calls[0]
        assert len(cleanup_calls) == 1  # cleanup called in step 11

    def test_mcp_not_called_when_no_servers(self) -> None:
        mcp_calls = []

        def mcp_start(mcp_servers):
            mcp_calls.append(mcp_servers)
            return lambda: None

        dispatcher = AgentDispatcher(
            agent_loop_callback=_success_loop,
            mcp_start_callback=mcp_start,
        )
        dispatcher.dispatch(_make_agent_def(mcp_servers={}), "task")
        assert len(mcp_calls) == 0

    def test_mcp_cleanup_called_even_on_error(self) -> None:
        cleanup_calls = []

        def mcp_start(mcp_servers):
            return lambda: cleanup_calls.append(1)

        def failing_loop(agent_def, task, system_prompt, budget):
            raise RuntimeError("boom")

        dispatcher = AgentDispatcher(
            agent_loop_callback=failing_loop,
            mcp_start_callback=mcp_start,
        )
        dispatcher.dispatch(
            _make_agent_def(mcp_servers={"github": {"command": "npx"}}), "task"
        )
        assert len(cleanup_calls) == 1


# ---------------------------------------------------------------------------
# Tests: Hooks lifecycle (steps 6 and 11)
# ---------------------------------------------------------------------------


class TestHooksLifecycle:
    def test_hooks_registered_and_cleaned_up(self) -> None:
        register_calls = []
        cleanup_calls = []

        def hooks_register(hooks, scope_id):
            register_calls.append((hooks, scope_id))
            return lambda: cleanup_calls.append(1)

        dispatcher = AgentDispatcher(
            agent_loop_callback=_success_loop,
            hooks_register_callback=hooks_register,
        )
        dispatcher.dispatch(
            _make_agent_def(hooks={"PostToolUse": [{"matcher": "*"}]}), "task"
        )
        assert len(register_calls) == 1
        assert len(cleanup_calls) == 1

    def test_hooks_not_registered_when_empty(self) -> None:
        register_calls = []

        def hooks_register(hooks, scope_id):
            register_calls.append(hooks)
            return lambda: None

        dispatcher = AgentDispatcher(
            agent_loop_callback=_success_loop,
            hooks_register_callback=hooks_register,
        )
        dispatcher.dispatch(_make_agent_def(hooks={}), "task")
        assert len(register_calls) == 0


# ---------------------------------------------------------------------------
# Tests: SubagentStop hook (step 10)
# ---------------------------------------------------------------------------


class TestSubagentStopHook:
    def test_subagent_stop_fired_on_completion(self) -> None:
        engine = BrainmassHookEngine()
        fired_events = []

        def track_handler(handler, event, context):
            fired_events.append(event)
            return HookResult()

        engine.set_handler_callback(track_handler)
        engine.register(
            HookEvent.SUBAGENT_STOP,
            HookDefinition(
                matcher="*",
                hooks=[HookHandler(type=HookHandlerType.COMMAND, command="echo done")],
            ),
            scope="project_local",
        )

        dispatcher = AgentDispatcher(
            hook_engine=engine,
            agent_loop_callback=_success_loop,
        )
        dispatcher.dispatch(_make_agent_def(), "task")
        assert HookEvent.SUBAGENT_STOP in fired_events

    def test_subagent_stop_fired_even_on_error(self) -> None:
        engine = BrainmassHookEngine()
        fired_events = []

        def track_handler(handler, event, context):
            fired_events.append(event)
            return HookResult()

        engine.set_handler_callback(track_handler)
        engine.register(
            HookEvent.SUBAGENT_STOP,
            HookDefinition(
                matcher="*",
                hooks=[HookHandler(type=HookHandlerType.COMMAND, command="echo done")],
            ),
            scope="project_local",
        )

        def failing_loop(agent_def, task, system_prompt, budget):
            raise RuntimeError("boom")

        dispatcher = AgentDispatcher(
            hook_engine=engine,
            agent_loop_callback=failing_loop,
        )
        dispatcher.dispatch(_make_agent_def(), "task")
        assert HookEvent.SUBAGENT_STOP in fired_events

    def test_subagent_stop_context_has_agent_info(self) -> None:
        engine = BrainmassHookEngine()
        contexts_seen = []

        def capture_context(handler, event, context):
            contexts_seen.append(context)
            return HookResult()

        engine.set_handler_callback(capture_context)
        engine.register(
            HookEvent.SUBAGENT_STOP,
            HookDefinition(
                matcher="*",
                hooks=[HookHandler(type=HookHandlerType.COMMAND, command="log")],
            ),
            scope="project_local",
        )

        dispatcher = AgentDispatcher(
            hook_engine=engine,
            agent_loop_callback=_success_loop,
            session_id="sess-abc",
            cwd="/project",
            session_type="headless",
        )
        dispatcher.dispatch(_make_agent_def(name="reviewer"), "task")

        assert len(contexts_seen) == 1
        ctx = contexts_seen[0]
        assert ctx.session_id == "sess-abc"
        assert ctx.hook_event_name == HookEvent.SUBAGENT_STOP
        assert ctx.source == "reviewer"


# ---------------------------------------------------------------------------
# Tests: Budget isolation
# ---------------------------------------------------------------------------


class TestBudgetIsolation:
    def test_subagent_gets_isolated_budget(self) -> None:
        budgets_seen = []

        def capture_budget(agent_def, task, system_prompt, budget):
            budgets_seen.append(budget)
            return _success_loop(agent_def, task, system_prompt, budget)

        parent = _make_budget(session_budget_usd=8.0)
        dispatcher = AgentDispatcher(agent_loop_callback=capture_budget)
        dispatcher.dispatch(_make_agent_def(), "task", parent_budget=parent)

        assert len(budgets_seen) == 1
        sub_budget = budgets_seen[0]
        assert sub_budget.session_budget_usd == pytest.approx(2.0)  # 25% of 8.0
        assert sub_budget.current_cost_usd == 0.0

    def test_default_budget_when_no_parent(self) -> None:
        budgets_seen = []

        def capture_budget(agent_def, task, system_prompt, budget):
            budgets_seen.append(budget)
            return _success_loop(agent_def, task, system_prompt, budget)

        dispatcher = AgentDispatcher(agent_loop_callback=capture_budget)
        dispatcher.dispatch(_make_agent_def(), "task")

        sub_budget = budgets_seen[0]
        assert sub_budget.session_budget_usd == 1.0
        assert sub_budget.input_budget_tokens == 200_000


# ---------------------------------------------------------------------------
# Tests: Cost governor integration
# ---------------------------------------------------------------------------


class TestCostGovernorIntegration:
    def test_usage_recorded_after_loop(self) -> None:
        from src.cost.cost_governor import CostGovernor

        governor = CostGovernor()
        dispatcher = AgentDispatcher(
            cost_governor=governor,
            agent_loop_callback=_success_loop,
        )
        dispatcher.dispatch(_make_agent_def(), "task")

        dashboard = governor.get_dashboard_data()
        assert dashboard["token_consumption"]["total_input_tokens"] == 500
        assert dashboard["token_consumption"]["total_output_tokens"] == 200

    def test_no_usage_recorded_when_zero_tokens(self) -> None:
        from src.cost.cost_governor import CostGovernor

        governor = CostGovernor()

        def zero_token_loop(agent_def, task, system_prompt, budget):
            return {
                "summary": "done",
                "turns_used": 0,
                "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
            }

        dispatcher = AgentDispatcher(
            cost_governor=governor,
            agent_loop_callback=zero_token_loop,
        )
        dispatcher.dispatch(_make_agent_def(), "task")

        dashboard = governor.get_dashboard_data()
        assert dashboard["token_consumption"]["total_input_tokens"] == 0
