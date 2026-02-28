"""Bridge module connecting Brainmass evaluators to Strands Evals SDK.

Converts Brainmass eval cases into ``strands_evals.Case`` objects and builds
``strands_evals.Experiment`` instances with appropriate evaluators.

This module requires ``strands-agents-evals`` to be installed.

Usage::

    from src.evals.strands_bridge import build_strands_experiment
    experiment = build_strands_experiment(
        phase="ears_spec",
        agent_fn=my_agent_function,
    )
    reports = experiment.run_evaluations(my_agent_function)
"""

from __future__ import annotations

import logging
from typing import Any

from src.evals.cases import (
    PHASE_CASES_MAP,
    EvalCase,
)
from src.evals.runner import _get_phase_rubric

logger = logging.getLogger(__name__)


def build_strands_experiment(
    phase: str,
    evaluator_types: list[str] | None = None,
    custom_rubric: str | None = None,
) -> Any:
    """Build a ``strands_evals.Experiment`` for a specific phase.

    Args:
        phase: Phase name (ears_spec, journey_map, design_doc, tdd, coder, traceability)
        evaluator_types: List of evaluator names to use. Options:
            - "output" (OutputEvaluator with custom rubric)
            - "helpfulness" (HelpfulnessEvaluator)
            - "faithfulness" (FaithfulnessEvaluator)
            - "coherence" (CoherenceEvaluator)
            - "conciseness" (ConcisenessEvaluator)
            Default: ["output", "helpfulness", "coherence"]
        custom_rubric: Override the default rubric for OutputEvaluator

    Returns:
        A configured ``strands_evals.Experiment`` ready for execution.

    Raises:
        ImportError: If strands-agents-evals is not installed.
    """
    from strands_evals import Case, Experiment  # type: ignore[import-untyped]
    from strands_evals.evaluators import (  # type: ignore[import-untyped]
        CoherenceEvaluator,
        ConcisenessEvaluator,
        FaithfulnessEvaluator,
        HelpfulnessEvaluator,
        OutputEvaluator,
    )

    # Build Strands Cases from our EvalCases
    cases = PHASE_CASES_MAP.get(phase, [])
    strands_cases = []
    for case in cases:
        strands_cases.append(
            Case(
                name=case.name,
                input=_build_case_input(case),
                expected_output="",
                metadata={
                    "phase": phase,
                    "category": case.category,
                    "difficulty": case.difficulty,
                    "brainmass_case": case.name,
                },
            )
        )

    # Build evaluators
    eval_names = evaluator_types or ["output", "helpfulness", "coherence"]
    evaluators = []
    rubric = custom_rubric or _get_phase_rubric(phase)

    evaluator_map = {
        "output": lambda: OutputEvaluator(rubric=rubric, include_inputs=True),
        "helpfulness": lambda: HelpfulnessEvaluator(),
        "faithfulness": lambda: FaithfulnessEvaluator(),
        "coherence": lambda: CoherenceEvaluator(),
        "conciseness": lambda: ConcisenessEvaluator(),
    }

    for name in eval_names:
        factory = evaluator_map.get(name)
        if factory:
            evaluators.append(factory())
        else:
            logger.warning("Unknown evaluator type: %s", name)

    return Experiment(cases=strands_cases, evaluators=evaluators)


def build_full_experiment(
    phases: list[str] | None = None,
    evaluator_types: list[str] | None = None,
) -> dict[str, Any]:
    """Build Strands Experiments for multiple phases.

    Returns a dict of phase_name → Experiment.
    """
    target_phases = phases or list(PHASE_CASES_MAP.keys())
    experiments = {}
    for phase in target_phases:
        if phase in PHASE_CASES_MAP:
            experiments[phase] = build_strands_experiment(
                phase=phase,
                evaluator_types=evaluator_types,
            )
    return experiments


def _build_case_input(case: EvalCase) -> str:
    """Build the full input string for a Strands Case, including context."""
    parts = [case.input]
    for key, value in case.context.items():
        if isinstance(value, str) and value.strip():
            parts.append(f"\n\n## Prior Artifact: {key}\n{value}")
    return "\n\n---\n\n".join(parts)
