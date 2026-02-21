"""Property-based tests for self-improving loop behavior.

Properties covered:
- Property 12: Loop iteration context isolation
- Property 13: Loop context file round-trip
"""

from __future__ import annotations

import tempfile

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.agents.context_file import load, save
from src.agents.loop_runner import LoopRunner
from src.types.core import AgentDefinition, LoopContext

ASCII_TEXT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-",
    min_size=1,
    max_size=24,
)

LEARNING_DICT = st.fixed_dictionaries(
    {
        "pattern": ASCII_TEXT,
        "resolution": ASCII_TEXT,
        "confidence": st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        "source_iteration": st.integers(min_value=0, max_value=50),
    }
)

FAILED_APPROACH_DICT = st.fixed_dictionaries(
    {
        "iteration": st.integers(min_value=0, max_value=50),
        "approach": ASCII_TEXT,
        "why_failed": ASCII_TEXT,
    }
)


@st.composite
def loop_contexts(draw: st.DrawFn) -> LoopContext:
    max_iterations = draw(st.integers(min_value=1, max_value=8))
    iteration_count = draw(st.integers(min_value=0, max_value=max_iterations))
    return LoopContext(
        current_task=draw(ASCII_TEXT),
        acceptance_criteria=draw(st.lists(ASCII_TEXT, min_size=0, max_size=4)),
        constraints=draw(st.lists(ASCII_TEXT, min_size=0, max_size=4)),
        learnings=draw(st.lists(LEARNING_DICT, min_size=0, max_size=4)),
        failed_approaches=draw(st.lists(FAILED_APPROACH_DICT, min_size=0, max_size=4)),
        iteration_count=iteration_count,
        max_iterations=max_iterations,
    )


def _make_agent_definition() -> AgentDefinition:
    return AgentDefinition(
        name="loop-worker",
        description="Loop iteration worker",
        model="sonnet",
        system_prompt="Run one iteration from context.",
    )


@pytest.mark.property
@settings(max_examples=100)
@given(max_iterations=st.integers(min_value=1, max_value=5))
def test_property_12_loop_iteration_context_isolation(max_iterations: int) -> None:
    """Feature: claude-code-v3-enterprise, Property 12."""
    received_contexts: list[str] = []
    call_index = {"value": 0}

    def callback(_agent_def, context_str, _budget):
        call_index["value"] += 1
        received_contexts.append(context_str)
        return {
            "summary": f"iteration-{call_index['value']}",
            "tokens_consumed": {"input": 10, "output": 5, "cache_read": 0},
            "tools_used": [],
            "files_modified": [],
            "exit_reason": "complete",
            "turns_used": 1,
            "learnings": [
                {
                    "pattern": f"p-{call_index['value']}",
                    "resolution": "fix",
                    "confidence": 0.9,
                    "source_iteration": call_index["value"],
                }
            ],
            "acceptance_met": False,
        }

    initial = LoopContext(
        current_task="Implement feature",
        acceptance_criteria=["Tests pass"],
        constraints=["No breaking changes"],
        learnings=[],
        failed_approaches=[],
        iteration_count=0,
        max_iterations=max_iterations,
    )

    runner = LoopRunner(agent_def=_make_agent_definition(), agent_callback=callback)
    result = runner.run(initial)

    simulated_context = initial
    for idx, received in enumerate(received_contexts):
        expected = LoopRunner._build_context_string(simulated_context)
        assert received == expected
        simulated_context = LoopRunner._update_context(
            simulated_context,
            result.iteration_results[idx],
        )


@pytest.mark.property
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(context=loop_contexts())
def test_property_13_loop_context_file_round_trip(context: LoopContext) -> None:
    """Feature: claude-code-v3-enterprise, Property 13."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/loop-context.json"
        save(context, path)
        restored = load(path)

        assert restored is not None
        assert restored == context
