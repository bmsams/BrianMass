"""Unit tests for the Orchestrator (Task 6.1).

Covers:
- 8-step process_request flow
- Topology selection
- Request classification
- Hook firing order
- Cost Governor wiring
- Effort Controller wiring
- PreToolUse deny / modify
- Stop hook blocking
"""

from __future__ import annotations

from src.context.context_manager import ContextManager
from src.cost.cost_governor import CostGovernor, TaskSignals
from src.hooks.hook_engine import BrainmassHookEngine
from src.orchestrator.orchestrator import (
    EffortResult,
    Orchestrator,
    OrchestratorResult,
    _DefaultEffortController,
)
from src.types.core import (
    AgentBudget,
    BudgetStatus,
    HookEvent,
    HookResult,
    ModelTier,
    TopologyType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator(**overrides) -> Orchestrator:
    """Build an Orchestrator with sensible defaults for testing."""
    cm = overrides.pop("context_manager", ContextManager(session_id="test"))
    he = overrides.pop("hook_engine", BrainmassHookEngine())
    cg = overrides.pop("cost_governor", CostGovernor())
    return Orchestrator(
        context_manager=cm,
        hook_engine=he,
        cost_governor=cg,
        session_id="test-session",
        cwd="/tmp/test",
        **overrides,
    )


# ---------------------------------------------------------------------------
# Topology selection  (Requirement 1.3)
# ---------------------------------------------------------------------------

class TestSelectTopology:
    def test_team_lead_returns_agent_teams(self):
        signals = TaskSignals(is_team_lead=True)
        assert Orchestrator.select_topology(signals) == TopologyType.AGENT_TEAMS

    def test_deep_dependency_returns_agent_teams(self):
        signals = TaskSignals(dependency_depth=5)
        assert Orchestrator.select_topology(signals) == TopologyType.AGENT_TEAMS

    def test_exploration_with_reasoning_returns_loop(self):
        signals = TaskSignals(is_exploration=True, requires_reasoning=True)
        assert Orchestrator.select_topology(signals) == TopologyType.SELF_IMPROVING_LOOP

    def test_exploration_without_reasoning_returns_hierarchical(self):
        signals = TaskSignals(is_exploration=True, requires_reasoning=False)
        assert Orchestrator.select_topology(signals) == TopologyType.HIERARCHICAL

    def test_default_returns_hierarchical(self):
        signals = TaskSignals()
        assert Orchestrator.select_topology(signals) == TopologyType.HIERARCHICAL

    def test_reasoning_without_exploration_returns_hierarchical(self):
        signals = TaskSignals(requires_reasoning=True)
        assert Orchestrator.select_topology(signals) == TopologyType.HIERARCHICAL


# ---------------------------------------------------------------------------
# Request classification
# ---------------------------------------------------------------------------

class TestClassifyRequest:
    def test_exploration_keywords(self):
        signals = Orchestrator.classify_request("search for all TODO comments")
        assert signals.is_exploration is True

    def test_reasoning_keywords(self):
        signals = Orchestrator.classify_request("refactor the auth module")
        assert signals.requires_reasoning is True

    def test_team_keyword(self):
        signals = Orchestrator.classify_request("coordinate the team effort")
        assert signals.is_team_lead is True

    def test_cross_service_sets_high_dependency(self):
        signals = Orchestrator.classify_request("cross-service refactoring")
        assert signals.dependency_depth == 4

    def test_simple_request(self):
        signals = Orchestrator.classify_request("fix typo in readme")
        assert signals.is_exploration is False
        assert signals.requires_reasoning is False
        assert signals.is_team_lead is False

    def test_file_count_heuristic(self):
        signals = Orchestrator.classify_request("update file A and file B")
        assert signals.files_affected >= 3  # 2 "file" mentions + 1 base


# ---------------------------------------------------------------------------
# Default Effort Controller stub
# ---------------------------------------------------------------------------

class TestDefaultEffortController:
    def test_returns_standard(self):
        ctrl = _DefaultEffortController()
        result = ctrl.select_effort(
            TaskSignals(), ModelTier.SONNET, AgentBudget(100_000, 50_000, 5.0)
        )
        assert result.level == "standard"
        assert result.budget_tokens == 10_000


# ---------------------------------------------------------------------------
# 8-step process_request flow  (Requirement 1.2)
# ---------------------------------------------------------------------------

class TestProcessRequest:
    def test_returns_orchestrator_result(self):
        orch = _make_orchestrator()
        result = orch.process_request("hello world")
        assert isinstance(result, OrchestratorResult)
        assert result.response
        assert result.request_id

    def test_fires_session_start_on_first_request(self):
        fired: list[str] = []

        def _track_fire(event, ctx):
            fired.append(event.value)
            return HookResult()

        he = BrainmassHookEngine()
        he.fire = _track_fire  # type: ignore[assignment]
        orch = _make_orchestrator(hook_engine=he)
        orch.process_request("first request")

        assert HookEvent.SESSION_START.value in fired

    def test_no_session_start_on_second_request(self):
        fired: list[str] = []

        def _track_fire(event, ctx):
            fired.append(event.value)
            return HookResult()

        he = BrainmassHookEngine()
        he.fire = _track_fire  # type: ignore[assignment]
        orch = _make_orchestrator(hook_engine=he)
        orch.process_request("first")
        fired.clear()
        orch.process_request("second")

        assert HookEvent.SESSION_START.value not in fired

    def test_fires_user_prompt_submit(self):
        fired: list[str] = []

        def _track_fire(event, ctx):
            fired.append(event.value)
            return HookResult()

        he = BrainmassHookEngine()
        he.fire = _track_fire  # type: ignore[assignment]
        orch = _make_orchestrator(hook_engine=he)
        orch.process_request("test")

        assert HookEvent.USER_PROMPT_SUBMIT.value in fired

    def test_fires_stop_hook(self):
        fired: list[str] = []

        def _track_fire(event, ctx):
            fired.append(event.value)
            return HookResult()

        he = BrainmassHookEngine()
        he.fire = _track_fire  # type: ignore[assignment]
        orch = _make_orchestrator(hook_engine=he)
        orch.process_request("test")

        assert HookEvent.STOP.value in fired

    def test_hooks_fired_list_populated(self):
        orch = _make_orchestrator()
        result = orch.process_request("test")
        assert HookEvent.SESSION_START.value in result.hooks_fired
        assert HookEvent.USER_PROMPT_SUBMIT.value in result.hooks_fired
        assert HookEvent.STOP.value in result.hooks_fired

    def test_model_tier_selected(self):
        orch = _make_orchestrator()
        result = orch.process_request("fix typo")
        # Simple request → Haiku (exploration-like or single file)
        assert result.model_tier in list(ModelTier)

    def test_effort_level_set(self):
        orch = _make_orchestrator()
        result = orch.process_request("test")
        assert result.effort_level in ("quick", "standard", "deep")

    def test_cost_tracked(self):
        orch = _make_orchestrator()
        result = orch.process_request("test")
        assert result.total_cost_usd >= 0
        # Cost Governor should have recorded usage
        status = orch.cost_governor.check_budget(orch.agent_id)
        assert status == BudgetStatus.OK

    def test_complex_request_decomposes(self):
        orch = _make_orchestrator()
        result = orch.process_request("refactor the auth module across multiple files")
        # "refactor" triggers requires_reasoning, "file" bumps files_affected
        assert len(result.tasks) >= 1
        assert result.topology is not None

    def test_simple_request_no_decomposition(self):
        orch = _make_orchestrator()
        result = orch.process_request("fix typo")
        assert result.tasks == []
        assert result.topology is None


# ---------------------------------------------------------------------------
# Hook integration — PreToolUse deny / modify, Stop block
# ---------------------------------------------------------------------------

class TestHookIntegration:
    def test_pre_tool_use_deny_blocks_tool(self):
        """When PreToolUse returns deny, the tool should not execute."""
        executed_tools: list[str] = []

        def _deny_fire(event, ctx):
            if event == HookEvent.PRE_TOOL_USE:
                return HookResult(permission_decision="deny", permission_decision_reason="blocked")
            return HookResult()

        def _model_cb(prompt, tier, effort):
            return "[TOOL:dangerous_cmd] do something"

        def _tool_exec(name, inp):
            executed_tools.append(name)
            return "done"

        he = BrainmassHookEngine()
        he.fire = _deny_fire  # type: ignore[assignment]
        orch = _make_orchestrator(
            hook_engine=he,
            model_callback=_model_cb,
            tool_executor=_tool_exec,
        )
        result = orch.process_request("run dangerous command")

        assert len(executed_tools) == 0
        assert any("[DENIED" in tc.tool_response for tc in result.tool_calls)

    def test_pre_tool_use_modify_input(self):
        """When PreToolUse returns updatedInput, the tool gets modified input."""
        received_inputs: list[dict] = []

        def _modify_fire(event, ctx):
            if event == HookEvent.PRE_TOOL_USE:
                return HookResult(updated_input={"safe": True})
            return HookResult()

        def _model_cb(prompt, tier, effort):
            return "[TOOL:write_file] write something"

        def _tool_exec(name, inp):
            received_inputs.append(inp)
            return "done"

        he = BrainmassHookEngine()
        he.fire = _modify_fire  # type: ignore[assignment]
        orch = _make_orchestrator(
            hook_engine=he,
            model_callback=_model_cb,
            tool_executor=_tool_exec,
        )
        orch.process_request("write a file")

        assert len(received_inputs) == 1
        assert received_inputs[0] == {"safe": True}

    def test_stop_hook_block_appends_message(self):
        """When Stop hook returns block, the response should note it."""

        def _block_fire(event, ctx):
            if event == HookEvent.STOP:
                return HookResult(decision="block", reason="not done yet")
            return HookResult()

        he = BrainmassHookEngine()
        he.fire = _block_fire  # type: ignore[assignment]
        orch = _make_orchestrator(hook_engine=he)
        result = orch.process_request("test")

        assert "block" in result.response.lower() or "additional work" in result.response.lower()

    def test_post_tool_use_failure_on_exception(self):
        """When a tool raises, PostToolUseFailure should fire."""
        fired_events: list[str] = []

        def _track_fire(event, ctx):
            fired_events.append(event.value)
            return HookResult()

        def _model_cb(prompt, tier, effort):
            return "[TOOL:broken] fail"

        def _tool_exec(name, inp):
            raise RuntimeError("tool crashed")

        he = BrainmassHookEngine()
        he.fire = _track_fire  # type: ignore[assignment]
        orch = _make_orchestrator(
            hook_engine=he,
            model_callback=_model_cb,
            tool_executor=_tool_exec,
        )
        result = orch.process_request("run broken tool")

        assert HookEvent.POST_TOOL_USE_FAILURE.value in fired_events
        assert any("ERROR" in tc.tool_response for tc in result.tool_calls)


# ---------------------------------------------------------------------------
# Custom model callback and tool executor
# ---------------------------------------------------------------------------

class TestPluggableCallbacks:
    def test_custom_model_callback(self):
        def _custom_model(prompt, tier, effort):
            return f"custom response for {tier.value}"

        orch = _make_orchestrator(model_callback=_custom_model)
        result = orch.process_request("hello")
        assert "custom response" in result.response

    def test_custom_tool_executor(self):
        calls: list[tuple[str, dict]] = []

        def _custom_tool(name, inp):
            calls.append((name, inp))
            return "tool result"

        def _model_cb(prompt, tier, effort):
            return "[TOOL:my_tool] do it"

        orch = _make_orchestrator(
            model_callback=_model_cb,
            tool_executor=_custom_tool,
        )
        result = orch.process_request("use my tool")
        assert len(calls) == 1
        assert calls[0][0] == "my_tool"


# ---------------------------------------------------------------------------
# UserPromptSubmit hook can modify request
# ---------------------------------------------------------------------------

class TestUserPromptSubmitModification:
    def test_modified_request_used(self):
        prompts_seen: list[str] = []

        def _modify_fire(event, ctx):
            if event == HookEvent.USER_PROMPT_SUBMIT:
                return HookResult(updated_input={"request": "modified prompt"})
            return HookResult()

        def _model_cb(prompt, tier, effort):
            prompts_seen.append(prompt)
            return f"response to: {prompt}"

        he = BrainmassHookEngine()
        he.fire = _modify_fire  # type: ignore[assignment]
        orch = _make_orchestrator(hook_engine=he, model_callback=_model_cb)
        result = orch.process_request("original prompt")

        assert prompts_seen[0] == "modified prompt"


# ---------------------------------------------------------------------------
# Effort Controller injection
# ---------------------------------------------------------------------------

class TestEffortControllerInjection:
    def test_custom_effort_controller(self):
        class _DeepEffort:
            def select_effort(self, signals, model, budget):
                return EffortResult(level="deep", budget_tokens=50_000)

        orch = _make_orchestrator(effort_controller=_DeepEffort())
        result = orch.process_request("complex task")
        assert result.effort_level == "deep"
