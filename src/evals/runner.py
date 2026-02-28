"""Evaluation runner — executes eval suites against real Strands agents.

Runs every eval case through the actual SDLC agents via Strands SDK +
Amazon Bedrock, then scores the output with deterministic evaluators.

Usage::

    from src.evals.runner import run_all_evaluations
    report = run_all_evaluations()
    print(report.summary())

    # Run evals for a specific phase
    report = run_phase_evaluations("ears_spec")

Requires:
- ``strands-agents`` installed
- AWS credentials configured for Bedrock access
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.evals.cases import (
    CODER_CASES,
    DESIGN_DOC_CASES,
    EARS_SPEC_CASES,
    JOURNEY_MAP_CASES,
    TDD_CASES,
    TRACEABILITY_CASES,
    EvalCase,
)
from src.evals.evaluators import (
    CoderEvaluator,
    DesignDocEvaluator,
    EARSSpecEvaluator,
    EvalResult,
    EvalVerdict,
    JourneyMapEvaluator,
    TDDEvaluator,
    TraceabilityEvaluator,
    WorkflowEvaluator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase → evaluator mapping
# ---------------------------------------------------------------------------

PHASE_EVALUATOR_MAP: dict[str, type[WorkflowEvaluator]] = {
    "ears_spec": EARSSpecEvaluator,
    "journey_map": JourneyMapEvaluator,
    "design_doc": DesignDocEvaluator,
    "tdd": TDDEvaluator,
    "coder": CoderEvaluator,
    "traceability": TraceabilityEvaluator,
}

PHASE_CASES_MAP: dict[str, list[EvalCase]] = {
    "ears_spec": EARS_SPEC_CASES,
    "journey_map": JOURNEY_MAP_CASES,
    "design_doc": DESIGN_DOC_CASES,
    "tdd": TDD_CASES,
    "coder": CODER_CASES,
    "traceability": TRACEABILITY_CASES,
}


# ---------------------------------------------------------------------------
# Agent callback type
# ---------------------------------------------------------------------------

# Runs agent: (agent_name, task, context) → output string
AgentCallback = Callable[[str, str, dict[str, str]], str]


# ---------------------------------------------------------------------------
# Real Strands agent callback
# ---------------------------------------------------------------------------

# Phase → agent system prompt (loaded from .brainmass/agents/ templates)
_AGENT_PROMPTS: dict[str, str] = {}


def _load_agent_prompt(agent_name: str) -> str:
    """Load the system prompt from a .brainmass/agents/<name>.md file."""
    if agent_name in _AGENT_PROMPTS:
        return _AGENT_PROMPTS[agent_name]

    import frontmatter

    for search_dir in [
        Path(".brainmass/agents"),
        Path.home() / ".brainmass" / "agents",
    ]:
        md_path = search_dir / f"{agent_name}.md"
        if md_path.exists():
            post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
            _AGENT_PROMPTS[agent_name] = post.content.strip()
            return _AGENT_PROMPTS[agent_name]

    return f"You are the {agent_name} agent."


def _strands_agent_callback(agent_name: str, task: str, context: dict[str, str]) -> str:
    """Execute a real Strands Agent via Bedrock and return the output.

    This is the only agent callback — no stubs, no fakes.
    """
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result

    system_prompt = _load_agent_prompt(agent_name)

    # Build the full prompt with prior artifact context
    parts = [f"## Task\n{task}"]
    for key, content in context.items():
        if content.strip():
            parts.append(f"## Prior Artifact: {key}\n{content}")
    full_prompt = "\n\n---\n\n".join(parts)

    # Use Sonnet for evals (fast + capable)
    model_id = _BEDROCK_MODEL_IDS.get("sonnet", "us.anthropic.claude-sonnet-4-5-v1:0")
    model = BedrockModel(model_id=model_id)

    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        callback_handler=None,
    )

    raw = agent(full_prompt)
    result = _normalize_strands_result(raw)
    return result.get("summary", "")


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

@dataclass
class PhaseReport:
    """Eval report for a single phase."""
    phase: str
    results: list[EvalResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.overall_score for r in self.results) / len(self.results)

    @property
    def failed_cases(self) -> list[EvalResult]:
        return [r for r in self.results if not r.passed]


@dataclass
class EvalReport:
    """Full evaluation report across all phases."""
    phase_reports: dict[str, PhaseReport] = field(default_factory=dict)
    timestamp: str = ""
    execution_mode: str = "deterministic"

    @property
    def total_cases(self) -> int:
        return sum(len(pr.results) for pr in self.phase_reports.values())

    @property
    def total_passed(self) -> int:
        return sum(
            sum(1 for r in pr.results if r.passed)
            for pr in self.phase_reports.values()
        )

    @property
    def overall_pass_rate(self) -> float:
        total = self.total_cases
        return self.total_passed / total if total > 0 else 0.0

    @property
    def overall_avg_score(self) -> float:
        all_scores = [
            r.overall_score
            for pr in self.phase_reports.values()
            for r in pr.results
        ]
        return sum(all_scores) / len(all_scores) if all_scores else 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "  BRAINMASS WORKFLOW EVALUATION REPORT",
            f"  Mode: {self.execution_mode}  |  Timestamp: {self.timestamp}",
            "=" * 70,
            "",
            f"  Overall: {self.total_passed}/{self.total_cases} passed "
            f"({self.overall_pass_rate:.0%})  |  Avg score: {self.overall_avg_score:.2f}",
            "",
            "  Phase Breakdown:",
            "  " + "-" * 66,
        ]

        for phase, pr in sorted(self.phase_reports.items()):
            status = "PASS" if pr.pass_rate >= 0.75 else "FAIL"
            lines.append(
                f"  {phase:<20} {len(pr.results):>3} cases  "
                f"{pr.pass_rate:>5.0%} pass  "
                f"avg={pr.avg_score:.2f}  [{status}]"
            )

        # Failed cases detail
        all_failed = [
            (phase, r)
            for phase, pr in self.phase_reports.items()
            for r in pr.failed_cases
        ]
        if all_failed:
            lines.extend(["", "  Failed Cases:", "  " + "-" * 66])
            for phase, result in all_failed[:20]:  # Limit to 20
                lines.append(f"  [{phase}] {result.case_name}: {result.summary}")

        # Category analysis
        lines.extend(["", "  Category Analysis:", "  " + "-" * 66])
        category_scores: dict[str, list[float]] = {}
        for pr in self.phase_reports.values():
            for r in pr.results:
                for check in r.checks:
                    cat = check.category or "uncategorized"
                    category_scores.setdefault(cat, []).append(check.score)

        for cat, scores in sorted(category_scores.items()):
            avg = sum(scores) / len(scores)
            lines.append(f"  {cat:<20} avg={avg:.2f}  ({len(scores)} checks)")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize report to JSON."""
        data = {
            "timestamp": self.timestamp,
            "execution_mode": self.execution_mode,
            "total_cases": self.total_cases,
            "total_passed": self.total_passed,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_avg_score": self.overall_avg_score,
            "phases": {},
        }
        for phase, pr in self.phase_reports.items():
            data["phases"][phase] = {
                "total": len(pr.results),
                "passed": sum(1 for r in pr.results if r.passed),
                "pass_rate": pr.pass_rate,
                "avg_score": pr.avg_score,
                "results": [
                    {
                        "case": r.case_name,
                        "score": r.overall_score,
                        "passed": r.passed,
                        "checks": [
                            {
                                "name": c.name,
                                "verdict": c.verdict.value,
                                "score": c.score,
                                "reason": c.reason,
                                "category": c.category,
                            }
                            for c in r.checks
                        ],
                    }
                    for r in pr.results
                ],
            }
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Pattern-based output validation (works with both stub and real agents)
# ---------------------------------------------------------------------------

def validate_output_patterns(output: str, case: EvalCase) -> list[tuple[str, bool, str]]:
    """Check expected and forbidden patterns against output.

    Returns list of (pattern, matched, description) tuples.
    """
    results: list[tuple[str, bool, str]] = []

    for pattern in case.expected_output_patterns:
        matched = bool(re.search(pattern, output, re.IGNORECASE))
        results.append((pattern, matched, "expected" if matched else "MISSING"))

    for pattern in case.forbidden_patterns:
        matched = bool(re.search(pattern, output, re.IGNORECASE))
        results.append((pattern, not matched, "forbidden FOUND" if matched else "clean"))

    return results


# ---------------------------------------------------------------------------
# Runner functions
# ---------------------------------------------------------------------------

def run_phase_evaluations(
    phase: str,
    agent_callback: AgentCallback | None = None,
    cases: list[EvalCase] | None = None,
) -> PhaseReport:
    """Run evaluations for a single phase using real Strands agents.

    Args:
        phase: Phase name (ears_spec, journey_map, design_doc, tdd, coder, traceability)
        agent_callback: Function that executes the agent and returns output.
                        Defaults to ``_strands_agent_callback`` (real Bedrock agent).
        cases: Override cases list. If None, uses the default cases for the phase.

    Returns:
        PhaseReport with results for all cases.
    """
    evaluator_cls = PHASE_EVALUATOR_MAP.get(phase)
    if evaluator_cls is None:
        logger.warning("No evaluator for phase '%s'", phase)
        return PhaseReport(phase=phase)

    evaluator = evaluator_cls()
    case_list = cases or PHASE_CASES_MAP.get(phase, [])
    callback = agent_callback or _strands_agent_callback

    report = PhaseReport(phase=phase)

    for case in case_list:
        # Get agent output
        agent_name = f"sdlc-{phase.replace('_', '-')}"
        prior = {k: v for k, v in case.context.items() if isinstance(v, str)}
        output = callback(agent_name, case.input, prior)

        # Run evaluator
        eval_context = {
            "case_name": case.name,
            "category": case.category,
            "difficulty": case.difficulty,
            **case.context,
        }
        result = evaluator.evaluate(output, eval_context)

        # Also check pattern expectations
        pattern_results = validate_output_patterns(output, case)
        for pattern, matched, desc in pattern_results:
            if not matched and desc == "MISSING":
                result.checks.append(
                    _make_pattern_check(pattern, matched=False, expected=True)
                )
            elif not matched and desc.startswith("forbidden"):
                result.checks.append(
                    _make_pattern_check(pattern, matched=True, expected=False)
                )

        # Recalculate score with pattern checks included
        if result.checks:
            total = sum(c.score for c in result.checks)
            result.overall_score = total / len(result.checks)
            result.passed = result.overall_score >= result.pass_threshold

        report.results.append(result)

    return report


def run_all_evaluations(
    agent_callback: AgentCallback | None = None,
    phases: list[str] | None = None,
) -> EvalReport:
    """Run evaluations across all phases using real Strands agents.

    Args:
        agent_callback: Function that executes agents. Defaults to real
                        Strands agent via Bedrock.
        phases: Specific phases to run. None = all phases.

    Returns:
        Full EvalReport with breakdown by phase.
    """
    report = EvalReport(
        timestamp=datetime.now(UTC).isoformat(),
        execution_mode="strands_agent",
    )

    target_phases = phases or list(PHASE_EVALUATOR_MAP.keys())

    for phase in target_phases:
        phase_report = run_phase_evaluations(phase, agent_callback=agent_callback)
        report.phase_reports[phase] = phase_report
        logger.info(
            "Phase '%s': %d/%d passed (%.0f%%)",
            phase,
            sum(1 for r in phase_report.results if r.passed),
            len(phase_report.results),
            phase_report.pass_rate * 100,
        )

    return report


# ---------------------------------------------------------------------------
# Rubrics for LLM-as-Judge mode
# ---------------------------------------------------------------------------

def _get_phase_rubric(phase: str) -> str:
    """Return the scoring rubric for a phase (used by OutputEvaluator)."""
    rubrics = {
        "ears_spec": (
            "Score this EARS requirements specification on a scale of 0.0 to 1.0.\n\n"
            "Score 1.0 if:\n"
            "- Uses correct EARS templates (shall, when/shall, while/shall, where/shall, if/then/shall)\n"
            "- Has 5+ unique EARS-XXX requirements with sequential IDs\n"
            "- Each requirement has 2+ testable acceptance criteria with specific values\n"
            "- Includes MUST/SHOULD/COULD/WONT priorities\n"
            "- Has at least one UNWANTED (safety) requirement\n"
            "- Provides rationale for each requirement\n"
            "- No vague language ('appropriate', 'good enough')\n\n"
            "Score 0.5 if partially meets criteria.\n"
            "Score 0.0 if output is not an EARS spec or misses most criteria."
        ),
        "journey_map": (
            "Score this customer journey map on a scale of 0.0 to 1.0.\n\n"
            "Score 1.0 if:\n"
            "- Has JOURNEY-XXX and STEP-XXX identifiers\n"
            "- Maps at least 2 personas with distinct goals\n"
            "- Includes happy paths AND error paths\n"
            "- Every step cross-references EARS-XXX requirements\n"
            "- Identifies touchpoints (UI, API, etc.)\n"
            "- Has entry/exit points\n"
            "- Includes coverage table showing which requirements are covered\n"
            "- Identifies gaps where requirements have no journey coverage\n\n"
            "Score 0.5 if partially meets criteria.\n"
            "Score 0.0 if not a journey map or misses most criteria."
        ),
        "design_doc": (
            "Score this design document on a scale of 0.0 to 1.0.\n\n"
            "Score 1.0 if:\n"
            "- Defines component architecture with clear responsibilities\n"
            "- Has CP-XXX correctness properties (≥5)\n"
            "- Uses multiple property types (INVARIANT, PRECONDITION, POSTCONDITION, SAFETY, LIVENESS)\n"
            "- Cross-references EARS-XXX requirements and STEP-XXX journey steps\n"
            "- Defines interfaces with signatures\n"
            "- Specifies test strategy for each property\n"
            "- Describes data flow between components\n"
            "- Includes design decisions with rationale\n\n"
            "Score 0.5 if partially meets criteria.\n"
            "Score 0.0 if not a design doc or misses most criteria."
        ),
        "tdd": (
            "Score this TDD test suite on a scale of 0.0 to 1.0.\n\n"
            "Score 1.0 if:\n"
            "- Has test_*.py file patterns with test_ functions\n"
            "- Tests reference CP-XXX correctness properties\n"
            "- Tests reference EARS-XXX requirements\n"
            "- Includes UAT script referencing JOURNEY-XXX\n"
            "- Covers unit, integration, and edge cases\n"
            "- Has meaningful assertions (assert, pytest.raises)\n"
            "- Docstrings include traceability references\n"
            "- Acknowledges tests should be RED (failing) initially\n\n"
            "Score 0.5 if partially meets criteria.\n"
            "Score 0.0 if not a test suite or misses most criteria."
        ),
        "coder": (
            "Score this implementation task list on a scale of 0.0 to 1.0.\n\n"
            "Score 1.0 if:\n"
            "- Has TASK-XXX identifiers with [R]/[Y]/[G] status markers\n"
            "- Each task references test files, EARS-XXX, and CP-XXX\n"
            "- Shows progress summary with counts/percentages\n"
            "- Tasks are grouped by component\n"
            "- Dependencies between tasks are tracked\n"
            "- References implementation files\n\n"
            "Score 0.5 if partially meets criteria.\n"
            "Score 0.0 if not a task list or misses most criteria."
        ),
        "traceability": (
            "Score this traceability matrix on a scale of 0.0 to 1.0.\n\n"
            "Score 1.0 if:\n"
            "- Has forward traceability (EARS → Journey → Design → Tests → Tasks)\n"
            "- Includes status indicators (Full/Partial/Missing)\n"
            "- Detects orphan tasks (not linked to requirements)\n"
            "- Identifies coverage gaps explicitly\n"
            "- Cross-references all artifact types\n\n"
            "Score 0.5 if partially meets criteria.\n"
            "Score 0.0 if not a traceability matrix or misses most criteria."
        ),
    }
    return rubrics.get(phase, "Score the output quality on a scale of 0.0 to 1.0.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_pattern_check(pattern: str, matched: bool, expected: bool) -> Any:
    """Create an EvalCheck for a pattern match result."""
    from src.evals.evaluators import EvalCheck, EvalVerdict

    if expected and not matched:
        return EvalCheck(
            name=f"pattern_{pattern[:20]}",
            verdict=EvalVerdict.FAIL,
            score=0.0,
            reason=f"Expected pattern not found: {pattern}",
            category="pattern_match",
        )
    if not expected and matched:
        return EvalCheck(
            name=f"forbidden_{pattern[:20]}",
            verdict=EvalVerdict.FAIL,
            score=0.0,
            reason=f"Forbidden pattern found: {pattern}",
            category="pattern_match",
        )
    return EvalCheck(
        name=f"pattern_{pattern[:20]}",
        verdict=EvalVerdict.PASS,
        score=1.0,
        reason=f"Pattern check passed: {pattern}",
        category="pattern_match",
    )
