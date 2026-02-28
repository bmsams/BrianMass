"""Unit tests for src/evals/runner.py and src/evals/cases.py.

Tests cover:
- Case data integrity (all cases have required fields)
- Runner execution in deterministic mode
- Phase report aggregation
- Full report generation and summary formatting
- Pattern validation
- Case diversity verification
- Rubric completeness
"""

from __future__ import annotations

import json

import pytest

from src.evals.cases import (
    ALL_CASES,
    CODER_CASES,
    DESIGN_DOC_CASES,
    EARS_SPEC_CASES,
    JOURNEY_MAP_CASES,
    TDD_CASES,
    TRACEABILITY_CASES,
    WORKFLOW_E2E_CASES,
    EvalCase,
)
from src.evals.runner import (
    PHASE_EVALUATOR_MAP,
    PHASE_CASES_MAP,
    EvalReport,
    PhaseReport,
    _get_phase_rubric,
    run_all_evaluations,
    run_phase_evaluations,
    validate_output_patterns,
)


# ==========================================================================
# Case data integrity
# ==========================================================================

class TestCaseIntegrity:
    """Verify all eval cases have valid structure."""

    def test_all_cases_have_name(self):
        for case in ALL_CASES:
            assert case.name, f"Case missing name: {case}"

    def test_all_cases_have_input(self):
        for case in ALL_CASES:
            assert case.input, f"Case '{case.name}' missing input"

    def test_all_cases_have_phase(self):
        for case in ALL_CASES:
            assert case.phase, f"Case '{case.name}' missing phase"

    def test_all_case_names_unique(self):
        names = [c.name for c in ALL_CASES]
        duplicates = [n for n in names if names.count(n) > 1]
        assert not duplicates, f"Duplicate case names: {set(duplicates)}"

    def test_valid_categories(self):
        valid = {"happy_path", "ambiguous", "complex", "adversarial", "edge_case", "cross_cutting"}
        for case in ALL_CASES:
            assert case.category in valid, (
                f"Case '{case.name}' has invalid category '{case.category}'"
            )

    def test_valid_difficulties(self):
        valid = {"easy", "medium", "hard", "adversarial"}
        for case in ALL_CASES:
            assert case.difficulty in valid, (
                f"Case '{case.name}' has invalid difficulty '{case.difficulty}'"
            )

    def test_minimum_case_count(self):
        """Each phase should have at least 3 eval cases."""
        for phase_name, cases in PHASE_CASES_MAP.items():
            assert len(cases) >= 3, (
                f"Phase '{phase_name}' only has {len(cases)} cases (need ≥3)"
            )

    def test_total_case_count(self):
        """Should have a substantial number of total cases."""
        assert len(ALL_CASES) >= 30, (
            f"Only {len(ALL_CASES)} total cases — need more diversity"
        )


# ==========================================================================
# Case diversity
# ==========================================================================

class TestCaseDiversity:
    """Verify eval suite has diverse coverage."""

    def test_ears_has_adversarial_cases(self):
        adversarial = [c for c in EARS_SPEC_CASES if c.category == "adversarial"]
        assert len(adversarial) >= 2, "EARS needs ≥2 adversarial cases"

    def test_ears_has_edge_cases(self):
        edge = [c for c in EARS_SPEC_CASES if c.category == "edge_case"]
        assert len(edge) >= 2, "EARS needs ≥2 edge cases"

    def test_ears_has_happy_path(self):
        happy = [c for c in EARS_SPEC_CASES if c.category == "happy_path"]
        assert len(happy) >= 2, "EARS needs ≥2 happy path cases"

    def test_each_phase_has_category_spread(self):
        """Each major phase should have at least 2 different categories."""
        for phase_name in ["ears_spec", "journey_map", "design_doc", "tdd", "coder"]:
            cases = PHASE_CASES_MAP.get(phase_name, [])
            categories = set(c.category for c in cases)
            assert len(categories) >= 2, (
                f"Phase '{phase_name}' only has {len(categories)} categories: {categories}"
            )

    def test_difficulty_distribution(self):
        """Should have cases at multiple difficulty levels."""
        difficulties = set(c.difficulty for c in ALL_CASES)
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties
        assert "adversarial" in difficulties

    def test_prompt_injection_case_exists(self):
        """Must have at least one prompt injection adversarial test."""
        injection_cases = [
            c for c in ALL_CASES
            if "injection" in c.name.lower() or "ignore" in c.input.lower()
        ]
        assert len(injection_cases) >= 1, "Missing prompt injection adversarial case"


# ==========================================================================
# Runner execution
# ==========================================================================

class TestRunPhaseEvaluations:
    """Test the phase evaluation runner in deterministic mode."""

    def test_ears_phase_runs(self):
        """Runner should execute all EARS cases without errors."""
        report = run_phase_evaluations("ears_spec")
        assert isinstance(report, PhaseReport)
        assert len(report.results) == len(EARS_SPEC_CASES)

    def test_journey_phase_runs(self):
        report = run_phase_evaluations("journey_map")
        assert isinstance(report, PhaseReport)
        assert len(report.results) == len(JOURNEY_MAP_CASES)

    def test_design_phase_runs(self):
        report = run_phase_evaluations("design_doc")
        assert len(report.results) == len(DESIGN_DOC_CASES)

    def test_tdd_phase_runs(self):
        report = run_phase_evaluations("tdd")
        assert len(report.results) == len(TDD_CASES)

    def test_coder_phase_runs(self):
        report = run_phase_evaluations("coder")
        assert len(report.results) == len(CODER_CASES)

    def test_traceability_phase_runs(self):
        report = run_phase_evaluations("traceability")
        assert len(report.results) == len(TRACEABILITY_CASES)

    def test_unknown_phase_returns_empty(self):
        report = run_phase_evaluations("nonexistent")
        assert len(report.results) == 0

    def test_custom_agent_callback(self):
        """Agent callback should be called for each case."""
        calls = []

        def callback(agent_name: str, task: str, context: dict) -> str:
            calls.append(agent_name)
            return f"EARS-001 [MUST] The system shall do things. AC-001.1: test within 5 seconds"

        report = run_phase_evaluations("ears_spec", agent_callback=callback)
        assert len(calls) == len(EARS_SPEC_CASES)
        assert all("ears" in c for c in calls)

    def test_custom_cases_override(self):
        custom = [
            EvalCase(
                name="custom_1",
                input="Custom test",
                phase="ears_spec",
                expected_output_patterns=[r"custom"],
                category="happy_path",
                difficulty="easy",
            ),
        ]
        report = run_phase_evaluations("ears_spec", cases=custom)
        assert len(report.results) == 1
        assert report.results[0].case_name == "custom_1"


class TestRunAllEvaluations:
    """Test the full evaluation runner."""

    def test_runs_all_phases(self):
        report = run_all_evaluations()
        assert isinstance(report, EvalReport)
        assert len(report.phase_reports) == len(PHASE_EVALUATOR_MAP)

    def test_runs_specific_phases(self):
        report = run_all_evaluations(phases=["ears_spec", "tdd"])
        assert len(report.phase_reports) == 2
        assert "ears_spec" in report.phase_reports
        assert "tdd" in report.phase_reports

    def test_report_has_timestamp(self):
        report = run_all_evaluations(phases=["ears_spec"])
        assert report.timestamp


# ==========================================================================
# Report formatting
# ==========================================================================

class TestEvalReport:
    def test_overall_pass_rate(self):
        report = run_all_evaluations(phases=["ears_spec"])
        # In deterministic mode (case input = agent output), scores will vary
        assert 0.0 <= report.overall_pass_rate <= 1.0

    def test_overall_avg_score(self):
        report = run_all_evaluations(phases=["ears_spec"])
        assert 0.0 <= report.overall_avg_score <= 1.0

    def test_summary_output(self):
        report = run_all_evaluations(phases=["ears_spec"])
        summary = report.summary()
        assert "BRAINMASS WORKFLOW EVALUATION REPORT" in summary
        assert "ears_spec" in summary
        assert "Phase Breakdown" in summary

    def test_json_serialization(self):
        report = run_all_evaluations(phases=["ears_spec"])
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "phases" in data
        assert "ears_spec" in data["phases"]
        assert "total_cases" in data
        assert "overall_pass_rate" in data

    def test_total_cases_correct(self):
        report = run_all_evaluations(phases=["ears_spec", "tdd"])
        expected = len(EARS_SPEC_CASES) + len(TDD_CASES)
        assert report.total_cases == expected


class TestPhaseReport:
    def test_pass_rate_calculation(self):
        report = PhaseReport(phase="test")
        assert report.pass_rate == 0.0  # Empty report

    def test_avg_score_calculation(self):
        report = PhaseReport(phase="test")
        assert report.avg_score == 0.0  # Empty report

    def test_failed_cases_property(self):
        report = run_phase_evaluations("ears_spec")
        # All cases should have some failed checks in deterministic mode
        assert isinstance(report.failed_cases, list)


# ==========================================================================
# Pattern validation
# ==========================================================================

class TestPatternValidation:
    def test_expected_pattern_found(self):
        case = EvalCase(
            name="test",
            input="test",
            phase="ears_spec",
            expected_output_patterns=[r"EARS-\d+"],
        )
        results = validate_output_patterns("EARS-001 requirement", case)
        assert results[0][1] is True

    def test_expected_pattern_missing(self):
        case = EvalCase(
            name="test",
            input="test",
            phase="ears_spec",
            expected_output_patterns=[r"EARS-\d+"],
        )
        results = validate_output_patterns("no requirements here", case)
        assert results[0][1] is False

    def test_forbidden_pattern_absent(self):
        case = EvalCase(
            name="test",
            input="test",
            phase="ears_spec",
            forbidden_patterns=[r"HACKED"],
        )
        results = validate_output_patterns("EARS-001 valid output", case)
        assert results[0][1] is True  # Good: forbidden pattern NOT found

    def test_forbidden_pattern_present(self):
        case = EvalCase(
            name="test",
            input="test",
            phase="ears_spec",
            forbidden_patterns=[r"HACKED"],
        )
        results = validate_output_patterns("HACKED", case)
        assert results[0][1] is False  # Bad: forbidden pattern found


# ==========================================================================
# Rubric completeness
# ==========================================================================

class TestRubrics:
    def test_all_phases_have_rubrics(self):
        """Every eval phase should have an LLM-as-Judge rubric."""
        for phase in PHASE_EVALUATOR_MAP:
            rubric = _get_phase_rubric(phase)
            assert len(rubric) > 50, (
                f"Phase '{phase}' rubric too short ({len(rubric)} chars)"
            )
            assert "0.0" in rubric or "1.0" in rubric, (
                f"Phase '{phase}' rubric missing scoring scale"
            )

    def test_rubrics_mention_key_criteria(self):
        """EARS rubric should mention EARS-specific criteria."""
        ears_rubric = _get_phase_rubric("ears_spec")
        assert "EARS" in ears_rubric
        assert "shall" in ears_rubric.lower()
        assert "acceptance criteria" in ears_rubric.lower()


# ==========================================================================
# Evaluator map completeness
# ==========================================================================

class TestEvaluatorMapCompleteness:
    def test_all_phases_have_evaluators(self):
        for phase in ["ears_spec", "journey_map", "design_doc", "tdd", "coder", "traceability"]:
            assert phase in PHASE_EVALUATOR_MAP, f"Missing evaluator for '{phase}'"

    def test_all_phases_have_cases(self):
        for phase in PHASE_EVALUATOR_MAP:
            assert phase in PHASE_CASES_MAP, f"Missing cases for '{phase}'"
            assert len(PHASE_CASES_MAP[phase]) > 0, f"Empty cases for '{phase}'"
