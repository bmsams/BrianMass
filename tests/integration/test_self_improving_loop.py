"""Integration tests for self-improving loop execution."""

from __future__ import annotations

import json

from src.agents.context_file import save as save_context
from src.agents.loop_runner import LoopRunner
from src.agents.safety_controls import ErrorMonitor
from src.types.core import AgentDefinition, LoopContext


def _agent_def() -> AgentDefinition:
    return AgentDefinition(
        name="loop-worker",
        description="self improving worker",
        model="sonnet",
        system_prompt="iterate and improve",
    )


def _base_context(max_iterations: int) -> LoopContext:
    return LoopContext(
        current_task="Stabilize request lifecycle tests",
        acceptance_criteria=["All tests pass"],
        constraints=["Do not change public API"],
        learnings=[],
        failed_approaches=[],
        iteration_count=0,
        max_iterations=max_iterations,
    )


def test_three_iterations_clean_context_and_loop_context_file(tmp_path):
    contexts_seen: list[str] = []

    def callback(agent_def, context_str, budget):
        contexts_seen.append(context_str)
        iteration = len(contexts_seen)
        return {
            "summary": f"Iteration {iteration} attempt",
            "tokens_consumed": {"input": 100, "output": 50, "cache_read": 0},
            "tools_used": ["read_file"],
            "files_modified": ["src/sample.py"],
            "exit_reason": "complete",
            "turns_used": 2,
            "learnings": [
                {
                    "pattern": f"pattern-{iteration}",
                    "resolution": f"resolution-{iteration}",
                    "confidence": 0.8,
                }
            ],
            "acceptance_met": False,
        }

    runner = LoopRunner(agent_def=_agent_def(), agent_callback=callback)
    result = runner.run(_base_context(max_iterations=3))

    assert result.exit_reason == "max_iterations"
    assert len(result.iteration_results) == 3
    assert len(contexts_seen) == 3

    assert "# Task" in contexts_seen[0]
    assert "Learnings from Previous Iterations" not in contexts_seen[0]
    assert "pattern-1" in contexts_seen[1]
    assert "pattern-1" in contexts_seen[2]
    assert "pattern-2" in contexts_seen[2]
    for context_str in contexts_seen:
        assert "User:" not in context_str
        assert "Assistant:" not in context_str

    context_path = tmp_path / ".brainmass" / "loop-context.json"
    save_context(result.final_context, str(context_path))
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert payload["iteration_count"] == 3
    assert len(payload["learnings"]) == 3
    assert payload["learnings"][0]["pattern"] == "pattern-1"
    assert payload["learnings"][2]["pattern"] == "pattern-3"


def test_repeated_errors_halt_loop_after_three_occurrences():
    monitor = ErrorMonitor(threshold=3)

    def callback(agent_def, context_str, budget):
        monitor.record_error("same failure signature")
        return {
            "summary": "iteration failed",
            "tokens_consumed": {"input": 40, "output": 20, "cache_read": 0},
            "tools_used": [],
            "files_modified": [],
            "exit_reason": "error",
            "turns_used": 1,
            "learnings": [],
            "acceptance_met": False,
        }

    runner = LoopRunner(
        agent_def=_agent_def(),
        agent_callback=callback,
        stop_file_callback=lambda: monitor.should_pause,
    )
    result = runner.run(_base_context(max_iterations=10))

    assert monitor.should_pause is True
    assert len(result.iteration_results) == 3
    assert result.exit_reason == "stop_file"
