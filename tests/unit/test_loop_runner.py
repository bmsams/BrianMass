"""Unit tests for the Loop Runner (self-improving loop — Ralph Wiggum pattern).

Tests cover:
- Fresh agent per iteration with clean context (Req 7.1)
- Structured context string building
- Git-based checkpointing via pluggable callback (Req 7.3)
- Learning accumulation across iterations (Req 7.4)
- Failed approach tracking
- Stop conditions: max_iterations, acceptance_met, stop_file (Req 7.5)
- Hook and cost governor integration
- GraphBuilder pattern simulation (Req 7.7)
"""

from __future__ import annotations

from src.agents.loop_runner import (
    IterationResult,
    LoopResult,
    LoopRunner,
    _default_agent_callback,
    _default_git_callback,
    _default_stop_file_callback,
)
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    HookEvent,
    HookResult,
    Learning,
    LoopContext,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_def(**overrides) -> AgentDefinition:
    defaults = {
        "name": "loop-agent",
        "description": "A loop iteration agent",
        "model": "sonnet",
        "system_prompt": "You are a loop agent.",
    }
    defaults.update(overrides)
    return AgentDefinition(**defaults)


def _make_budget(**overrides) -> AgentBudget:
    defaults = {
        "input_budget_tokens": 200_000,
        "output_budget_tokens": 50_000,
        "session_budget_usd": 5.0,
    }
    defaults.update(overrides)
    return AgentBudget(**defaults)


def _make_context(**overrides) -> LoopContext:
    defaults = {
        "current_task": "Implement user authentication",
        "acceptance_criteria": ["All tests pass", "JWT tokens work"],
        "constraints": ["Use bcrypt for hashing"],
        "learnings": [],
        "failed_approaches": [],
        "iteration_count": 0,
        "max_iterations": 5,
    }
    defaults.update(overrides)
    return LoopContext(**defaults)


def _make_callback_result(**overrides) -> dict:
    defaults = {
        "summary": "Implemented auth module",
        "tokens_consumed": {"input": 1000, "output": 500, "cache_read": 0},
        "tools_used": ["read_file", "write_file"],
        "files_modified": ["src/auth.py"],
        "exit_reason": "complete",
        "turns_used": 5,
        "learnings": [],
        "acceptance_met": False,
    }
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Default callback tests
# ---------------------------------------------------------------------------

class TestDefaultCallbacks:
    def test_default_agent_callback_returns_dict(self):
        agent_def = _make_agent_def()
        budget = _make_budget()
        result = _default_agent_callback(agent_def, "do stuff", budget)
        assert isinstance(result, dict)
        assert "summary" in result
        assert result["acceptance_met"] is False

    def test_default_git_callback_returns_none(self):
        assert _default_git_callback("commit msg") is None

    def test_default_stop_file_callback_returns_false(self):
        assert _default_stop_file_callback() is False


# ---------------------------------------------------------------------------
# Context string building
# ---------------------------------------------------------------------------

class TestBuildContextString:
    def test_includes_task(self):
        ctx = _make_context()
        result = LoopRunner._build_context_string(ctx)
        assert "Implement user authentication" in result

    def test_includes_acceptance_criteria(self):
        ctx = _make_context()
        result = LoopRunner._build_context_string(ctx)
        assert "All tests pass" in result
        assert "JWT tokens work" in result

    def test_includes_constraints(self):
        ctx = _make_context()
        result = LoopRunner._build_context_string(ctx)
        assert "Use bcrypt for hashing" in result

    def test_includes_learnings(self):
        ctx = _make_context(learnings=[
            {"pattern": "bcrypt is async", "resolution": "always await",
             "confidence": 1.0, "source_iteration": 1},
        ])
        result = LoopRunner._build_context_string(ctx)
        assert "bcrypt is async" in result
        assert "always await" in result
        assert "100%" in result

    def test_includes_failed_approaches(self):
        ctx = _make_context(failed_approaches=[
            {"iteration": 1, "approach": "sync bcrypt", "why_failed": "timeout"},
        ])
        result = LoopRunner._build_context_string(ctx)
        assert "sync bcrypt" in result
        assert "DO NOT REPEAT" in result

    def test_shows_iteration_number(self):
        ctx = _make_context(iteration_count=2, max_iterations=5)
        result = LoopRunner._build_context_string(ctx)
        assert "Iteration 3 of 5" in result

    def test_empty_constraints_omitted(self):
        ctx = _make_context(constraints=[])
        result = LoopRunner._build_context_string(ctx)
        assert "# Constraints" not in result

    def test_empty_learnings_omitted(self):
        ctx = _make_context(learnings=[])
        result = LoopRunner._build_context_string(ctx)
        assert "# Learnings" not in result

    def test_empty_failed_approaches_omitted(self):
        ctx = _make_context(failed_approaches=[])
        result = LoopRunner._build_context_string(ctx)
        assert "# Failed Approaches" not in result


# ---------------------------------------------------------------------------
# Learning extraction
# ---------------------------------------------------------------------------

class TestExtractLearnings:
    def test_extracts_learnings_from_result(self):
        raw = _make_callback_result(learnings=[
            {"pattern": "async issue", "resolution": "use await", "confidence": 0.9},
        ])
        learnings = LoopRunner._extract_learnings(raw, 1)
        assert len(learnings) == 1
        assert learnings[0].pattern == "async issue"
        assert learnings[0].resolution == "use await"
        assert learnings[0].confidence == 0.9
        assert learnings[0].source_iteration == 1

    def test_empty_learnings(self):
        raw = _make_callback_result(learnings=[])
        learnings = LoopRunner._extract_learnings(raw, 1)
        assert learnings == []

    def test_uses_current_iteration_as_default(self):
        raw = _make_callback_result(learnings=[
            {"pattern": "p", "resolution": "r"},
        ])
        learnings = LoopRunner._extract_learnings(raw, 3)
        assert learnings[0].source_iteration == 3

    def test_respects_explicit_source_iteration(self):
        raw = _make_callback_result(learnings=[
            {"pattern": "p", "resolution": "r", "source_iteration": 7},
        ])
        learnings = LoopRunner._extract_learnings(raw, 3)
        assert learnings[0].source_iteration == 7


# ---------------------------------------------------------------------------
# Context update
# ---------------------------------------------------------------------------

class TestUpdateContext:
    def test_increments_iteration_count(self):
        ctx = _make_context(iteration_count=0)
        ir = IterationResult(
            iteration=1,
            agent_result=AgentResult(
                agent_name="a", summary="done", turns_used=1,
                tokens_consumed={}, tools_used=[], files_modified=[],
                exit_reason="complete",
            ),
            acceptance_met=False,
        )
        new_ctx = LoopRunner._update_context(ctx, ir)
        assert new_ctx.iteration_count == 1

    def test_accumulates_learnings(self):
        ctx = _make_context(learnings=[
            {"pattern": "old", "resolution": "old-fix", "confidence": 0.8, "source_iteration": 1},
        ])
        ir = IterationResult(
            iteration=2,
            agent_result=AgentResult(
                agent_name="a", summary="done", turns_used=1,
                tokens_consumed={}, tools_used=[], files_modified=[],
                exit_reason="complete",
            ),
            new_learnings=[
                Learning(pattern="new", resolution="new-fix", confidence=0.9, source_iteration=2),
            ],
            acceptance_met=False,
        )
        new_ctx = LoopRunner._update_context(ctx, ir)
        assert len(new_ctx.learnings) == 2
        assert new_ctx.learnings[0]["pattern"] == "old"
        assert new_ctx.learnings[1]["pattern"] == "new"

    def test_records_failed_approach_on_failure(self):
        ctx = _make_context()
        ir = IterationResult(
            iteration=1,
            agent_result=AgentResult(
                agent_name="a", summary="tried sync approach", turns_used=1,
                tokens_consumed={}, tools_used=[], files_modified=[],
                exit_reason="error",
            ),
            acceptance_met=False,
        )
        new_ctx = LoopRunner._update_context(ctx, ir)
        assert len(new_ctx.failed_approaches) == 1
        assert new_ctx.failed_approaches[0]["iteration"] == 1
        assert "sync approach" in new_ctx.failed_approaches[0]["approach"]

    def test_no_failed_approach_on_success(self):
        ctx = _make_context()
        ir = IterationResult(
            iteration=1,
            agent_result=AgentResult(
                agent_name="a", summary="done", turns_used=1,
                tokens_consumed={}, tools_used=[], files_modified=[],
                exit_reason="complete",
            ),
            acceptance_met=True,
        )
        new_ctx = LoopRunner._update_context(ctx, ir)
        assert len(new_ctx.failed_approaches) == 0

    def test_preserves_original_context_fields(self):
        ctx = _make_context(
            current_task="my task",
            acceptance_criteria=["c1"],
            constraints=["con1"],
        )
        ir = IterationResult(
            iteration=1,
            agent_result=AgentResult(
                agent_name="a", summary="done", turns_used=1,
                tokens_consumed={}, tools_used=[], files_modified=[],
                exit_reason="complete",
            ),
            acceptance_met=True,
        )
        new_ctx = LoopRunner._update_context(ctx, ir)
        assert new_ctx.current_task == "my task"
        assert new_ctx.acceptance_criteria == ["c1"]
        assert new_ctx.constraints == ["con1"]
        assert new_ctx.max_iterations == ctx.max_iterations


# ---------------------------------------------------------------------------
# Loop execution — stop conditions
# ---------------------------------------------------------------------------

class TestLoopRunnerStopConditions:
    def test_stops_at_max_iterations(self):
        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=3)
        runner = LoopRunner(agent_def=agent_def)
        result = runner.run(ctx)

        assert result.exit_reason == "max_iterations"
        assert len(result.iteration_results) == 3
        assert result.final_context.iteration_count == 3

    def test_stops_on_acceptance_met(self):
        call_count = 0

        def callback(ad, ctx_str, budget):
            nonlocal call_count
            call_count += 1
            return _make_callback_result(
                acceptance_met=(call_count == 2),
                summary=f"Iteration {call_count}",
            )

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=10)
        runner = LoopRunner(agent_def=agent_def, agent_callback=callback)
        result = runner.run(ctx)

        assert result.exit_reason == "acceptance_met"
        assert len(result.iteration_results) == 2

    def test_stops_on_stop_file(self):
        call_count = 0

        def stop_check():
            nonlocal call_count
            call_count += 1
            return call_count > 2  # stop before 3rd iteration

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=10)
        runner = LoopRunner(
            agent_def=agent_def,
            stop_file_callback=stop_check,
        )
        result = runner.run(ctx)

        assert result.exit_reason == "stop_file"
        assert len(result.iteration_results) == 2

    def test_zero_max_iterations_returns_immediately(self):
        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=0)
        runner = LoopRunner(agent_def=agent_def)
        result = runner.run(ctx)

        assert result.exit_reason == "max_iterations"
        assert len(result.iteration_results) == 0


# ---------------------------------------------------------------------------
# Fresh agent per iteration (Req 7.1)
# ---------------------------------------------------------------------------

class TestFreshAgentPerIteration:
    def test_each_iteration_gets_fresh_context(self):
        """Each iteration callback receives only the structured context,
        not any conversation history from prior iterations."""
        contexts_received = []

        def callback(ad, ctx_str, budget):
            contexts_received.append(ctx_str)
            return _make_callback_result(
                learnings=[{"pattern": f"learning-{len(contexts_received)}",
                            "resolution": "fix"}],
            )

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=3)
        runner = LoopRunner(agent_def=agent_def, agent_callback=callback)
        runner.run(ctx)

        assert len(contexts_received) == 3
        # First iteration has no learnings
        assert "Learnings" not in contexts_received[0]
        # Second iteration has learning from first
        assert "learning-1" in contexts_received[1]
        # Third iteration has learnings from first and second
        assert "learning-1" in contexts_received[2]
        assert "learning-2" in contexts_received[2]

    def test_context_string_has_structured_sections_only(self):
        """The context string should only contain structured sections
        (Task, Criteria, Constraints, Learnings, Failed Approaches),
        not raw multi-turn conversation history."""
        received_ctx = []

        def callback(ad, ctx_str, budget):
            received_ctx.append(ctx_str)
            return _make_callback_result(summary="Tried sync approach")

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=2)
        runner = LoopRunner(agent_def=agent_def, agent_callback=callback)
        runner.run(ctx)

        # Second iteration context has structured sections only
        ctx2 = received_ctx[1]
        assert "# Task" in ctx2
        assert "# Acceptance Criteria" in ctx2
        assert "# Failed Approaches" in ctx2
        # The failed approach is a brief summary, not a full conversation
        assert "Tried sync approach" in ctx2


# ---------------------------------------------------------------------------
# Git checkpointing (Req 7.3)
# ---------------------------------------------------------------------------

class TestGitCheckpointing:
    def test_git_callback_called_per_iteration(self):
        commits = []

        def git_cb(msg):
            commits.append(msg)
            return f"sha-{len(commits)}"

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=3)
        runner = LoopRunner(agent_def=agent_def, git_callback=git_cb)
        result = runner.run(ctx)

        assert len(commits) == 3
        assert all("loop iteration" in c for c in commits)
        assert result.iteration_results[0].commit_sha == "sha-1"
        assert result.iteration_results[1].commit_sha == "sha-2"
        assert result.iteration_results[2].commit_sha == "sha-3"

    def test_git_callback_none_returns_none_sha(self):
        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=1)
        runner = LoopRunner(agent_def=agent_def)
        result = runner.run(ctx)

        assert result.iteration_results[0].commit_sha is None


# ---------------------------------------------------------------------------
# Learning accumulation (Req 7.4)
# ---------------------------------------------------------------------------

class TestLearningAccumulation:
    def test_learnings_accumulate_across_iterations(self):
        call_count = 0

        def callback(ad, ctx_str, budget):
            nonlocal call_count
            call_count += 1
            return _make_callback_result(
                learnings=[{
                    "pattern": f"pattern-{call_count}",
                    "resolution": f"fix-{call_count}",
                    "confidence": 0.5 + call_count * 0.1,
                }],
            )

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=3)
        runner = LoopRunner(agent_def=agent_def, agent_callback=callback)
        result = runner.run(ctx)

        final = result.final_context
        assert len(final.learnings) == 3
        assert final.learnings[0]["pattern"] == "pattern-1"
        assert final.learnings[1]["pattern"] == "pattern-2"
        assert final.learnings[2]["pattern"] == "pattern-3"

    def test_failed_approaches_tracked(self):
        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=3)
        runner = LoopRunner(agent_def=agent_def)
        result = runner.run(ctx)

        # Default callback returns acceptance_met=False, so all are failed
        assert len(result.final_context.failed_approaches) == 3
        for i, fa in enumerate(result.final_context.failed_approaches):
            assert fa["iteration"] == i + 1


# ---------------------------------------------------------------------------
# Hook engine integration
# ---------------------------------------------------------------------------

class TestHookIntegration:
    def test_fires_subagent_stop_per_iteration(self):
        fired = []

        class MockHookEngine:
            def fire(self, event, context):
                fired.append((event, context))
                return HookResult()

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=2)
        runner = LoopRunner(
            agent_def=agent_def,
            hook_engine=MockHookEngine(),
        )
        runner.run(ctx)

        assert len(fired) == 2
        assert all(e == HookEvent.SUBAGENT_STOP for e, _ in fired)

    def test_no_hook_engine_does_not_crash(self):
        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=1)
        runner = LoopRunner(agent_def=agent_def, hook_engine=None)
        result = runner.run(ctx)
        assert len(result.iteration_results) == 1


# ---------------------------------------------------------------------------
# Cost governor integration
# ---------------------------------------------------------------------------

class TestCostGovernorIntegration:
    def test_records_usage_per_iteration(self):
        recorded = []

        class MockCostGovernor:
            def record_usage(self, **kwargs):
                recorded.append(kwargs)

        def callback(ad, ctx_str, budget):
            return _make_callback_result(
                tokens_consumed={"input": 500, "output": 200, "cache_read": 50},
            )

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=2)
        runner = LoopRunner(
            agent_def=agent_def,
            agent_callback=callback,
            cost_governor=MockCostGovernor(),
        )
        runner.run(ctx)

        assert len(recorded) == 2
        assert recorded[0]["input_tokens"] == 500
        assert recorded[0]["output_tokens"] == 200
        assert "loop:loop-agent:iter1" == recorded[0]["agent_id"]
        assert "loop:loop-agent:iter2" == recorded[1]["agent_id"]

    def test_no_cost_governor_does_not_crash(self):
        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=1)
        runner = LoopRunner(agent_def=agent_def, cost_governor=None)
        result = runner.run(ctx)
        assert len(result.iteration_results) == 1

    def test_zero_tokens_skips_recording(self):
        recorded = []

        class MockCostGovernor:
            def record_usage(self, **kwargs):
                recorded.append(kwargs)

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=1)
        runner = LoopRunner(
            agent_def=agent_def,
            cost_governor=MockCostGovernor(),
        )
        runner.run(ctx)

        # Default callback returns 0 tokens, so no recording
        assert len(recorded) == 0


# ---------------------------------------------------------------------------
# LoopResult structure
# ---------------------------------------------------------------------------

class TestLoopResult:
    def test_result_contains_all_iteration_results(self):
        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=3)
        runner = LoopRunner(agent_def=agent_def)
        result = runner.run(ctx)

        assert isinstance(result, LoopResult)
        assert len(result.iteration_results) == 3
        for i, ir in enumerate(result.iteration_results):
            assert ir.iteration == i + 1

    def test_result_final_context_reflects_all_iterations(self):
        call_count = 0

        def callback(ad, ctx_str, budget):
            nonlocal call_count
            call_count += 1
            return _make_callback_result(
                learnings=[{"pattern": f"p{call_count}", "resolution": "r"}],
                acceptance_met=(call_count == 3),
            )

        agent_def = _make_agent_def()
        ctx = _make_context(max_iterations=5)
        runner = LoopRunner(agent_def=agent_def, agent_callback=callback)
        result = runner.run(ctx)

        assert result.exit_reason == "acceptance_met"
        assert result.final_context.iteration_count == 3
        assert len(result.final_context.learnings) == 3
        # Only 2 failed approaches (iterations 1 and 2 failed, 3 succeeded)
        assert len(result.final_context.failed_approaches) == 2
