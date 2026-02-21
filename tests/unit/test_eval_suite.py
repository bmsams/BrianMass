"""Unit tests for the Evaluation Suite.

Covers: task completion, safety, cost efficiency, coordination overhead,
latency P95, run_all, summary, and actor simulator.

Requirements: 17.1, 17.2, 17.3, 17.4
"""

from __future__ import annotations

import pytest

from evals.eval_suite import (
    ActorSimulator,
    BrainmassEvalSuite,
    EvalResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def suite() -> BrainmassEvalSuite:
    return BrainmassEvalSuite()


# ===================================================================
# Task Completion
# ===================================================================

class TestTaskCompletion:
    """Task completion evaluator checks whether tasks succeeded."""

    def test_passes_at_target(self, suite: BrainmassEvalSuite) -> None:
        results = [{"completed": True}] * 10
        result = suite.evaluate_task_completion(results)
        assert result.passed is True
        assert result.actual_value == 100.0

    def test_fails_below_target(self, suite: BrainmassEvalSuite) -> None:
        # 5 out of 10 = 50%, below 90% target
        results = [{"completed": True}] * 5 + [{"completed": False}] * 5
        result = suite.evaluate_task_completion(results)
        assert result.passed is False

    def test_empty_results_fails(self, suite: BrainmassEvalSuite) -> None:
        result = suite.evaluate_task_completion([])
        assert result.passed is False


# ===================================================================
# Safety
# ===================================================================

class TestSafety:
    """Safety evaluator fails on any critical violations."""

    def test_zero_critical_passes(self, suite: BrainmassEvalSuite) -> None:
        findings = [
            {"severity": "warning", "description": "minor issue"},
            {"severity": "info", "description": "note"},
        ]
        result = suite.evaluate_safety(findings)
        assert result.passed is True

    def test_with_critical_fails(self, suite: BrainmassEvalSuite) -> None:
        findings = [
            {"severity": "critical", "description": "SQL injection"},
        ]
        result = suite.evaluate_safety(findings)
        assert result.passed is False

    def test_empty_findings_passes(self, suite: BrainmassEvalSuite) -> None:
        result = suite.evaluate_safety([])
        assert result.passed is True


# ===================================================================
# Cost Efficiency
# ===================================================================

class TestCostEfficiency:
    """Cost efficiency evaluator checks actual vs expected within variance."""

    def test_within_variance(self, suite: BrainmassEvalSuite) -> None:
        # 10% over is within 20% target
        result = suite.evaluate_cost_efficiency(
            actual_cost=1.10, expected_cost=1.00,
        )
        assert result.passed is True

    def test_exceeds_variance(self, suite: BrainmassEvalSuite) -> None:
        # 100% over exceeds 20% target
        result = suite.evaluate_cost_efficiency(
            actual_cost=2.00, expected_cost=1.00,
        )
        assert result.passed is False

    def test_zero_expected_passes(self, suite: BrainmassEvalSuite) -> None:
        result = suite.evaluate_cost_efficiency(
            actual_cost=0.0, expected_cost=0.0,
        )
        assert result.passed is True


# ===================================================================
# Coordination Overhead
# ===================================================================

class TestCoordinationOverhead:
    """Coordination overhead evaluator checks multi-agent overhead."""

    def test_passes_within_limit(self, suite: BrainmassEvalSuite) -> None:
        # 20% overhead, target is 85%
        result = suite.evaluate_coordination(
            overhead_ms=2000.0, total_ms=10000.0,
        )
        assert result.passed is True

    def test_fails_over_limit(self, suite: BrainmassEvalSuite) -> None:
        # 90% overhead exceeds 85% target
        result = suite.evaluate_coordination(
            overhead_ms=9000.0, total_ms=10000.0,
        )
        assert result.passed is False


# ===================================================================
# Latency P95
# ===================================================================

class TestLatencyP95:
    """Latency P95 evaluator checks 95th percentile under target."""

    def test_passes_under_target(self, suite: BrainmassEvalSuite) -> None:
        latencies = [float(i * 100) for i in range(1, 21)]  # 100-2000ms
        result = suite.evaluate_latency(latencies)
        assert result.passed is True  # p95 ~1900ms << 30000ms

    def test_fails_over_target(self, suite: BrainmassEvalSuite) -> None:
        latencies = [float(i * 2000) for i in range(1, 21)]  # 2000-40000ms
        result = suite.evaluate_latency(latencies)
        assert result.passed is False  # p95 ~38000ms >> 30000ms

    def test_empty_durations_passes(self, suite: BrainmassEvalSuite) -> None:
        result = suite.evaluate_latency([])
        assert result.passed is True


# ===================================================================
# run_all
# ===================================================================

class TestRunAll:
    """BrainmassEvalSuite.run_all returns results for all dimensions."""

    def test_returns_all_dimensions(self, suite: BrainmassEvalSuite) -> None:
        test_data = {
            "task_results": [{"completed": True}],
            "code_samples": [{"score": 4.5}],
            "safety_findings": [],
            "context_before": [{"id": "1", "category": "PRESERVE_VERBATIM"}],
            "context_after": [{"id": "1", "category": "PRESERVE_VERBATIM"}],
            "actual_cost": 1.0,
            "expected_cost": 1.0,
            "overhead_ms": 100.0,
            "total_ms": 1000.0,
            "iterations": [{"score": 0.5}, {"score": 0.8}],
            "durations_ms": [100.0, 200.0, 300.0],
        }
        results = suite.run_all(test_data)
        assert len(results) == 8
        assert all(isinstance(r, EvalResult) for r in results)


# ===================================================================
# summary
# ===================================================================

class TestSummary:
    """BrainmassEvalSuite.summary reports overall pass/fail."""

    def test_overall_pass(self, suite: BrainmassEvalSuite) -> None:
        test_data = {
            "task_results": [{"completed": True}] * 10,
            "code_samples": [{"score": 4.5}],
            "safety_findings": [],
            "context_before": [{"id": "1", "category": "PRESERVE_VERBATIM"}],
            "context_after": [{"id": "1", "category": "PRESERVE_VERBATIM"}],
            "actual_cost": 1.0,
            "expected_cost": 1.0,
            "overhead_ms": 100.0,
            "total_ms": 1000.0,
            "iterations": [{"score": 0.5}, {"score": 0.8}],
            "durations_ms": [100.0, 200.0, 300.0],
        }
        results = suite.run_all(test_data)
        summary = suite.summary(results)
        assert "overall_pass" in summary
        assert "total_dimensions" in summary
        assert "pass_count" in summary


# ===================================================================
# ActorSimulator
# ===================================================================

class TestActorSimulator:
    """ActorSimulator replays conversation turns."""

    def test_replays_turns(self) -> None:
        sim = ActorSimulator()
        sim.add_turn("user", "Write a sort function")
        sim.add_turn("user", "Add type hints")

        def agent_fn(role: str, content: str) -> str:
            return f"Response to: {content}"

        results = sim.simulate(agent_fn)
        assert len(results) == 2
        assert results[0]["role"] == "user"
        assert "Response to:" in results[0]["response"]

    def test_get_transcript(self) -> None:
        sim = ActorSimulator()
        sim.add_turn("user", "Hello")
        transcript = sim.get_transcript()
        assert len(transcript) == 1
        assert transcript[0]["role"] == "user"
        assert transcript[0]["content"] == "Hello"

    def test_turn_count(self) -> None:
        sim = ActorSimulator()
        assert sim.turn_count == 0
        sim.add_turn("user", "msg1")
        sim.add_turn("user", "msg2")
        assert sim.turn_count == 2
