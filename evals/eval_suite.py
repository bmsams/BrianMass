"""Brainmass v3 Evaluation Suite — multi-dimensional quality assessment.

Provides eight evaluation dimensions with configurable targets, an
aggregation framework, and a multi-turn actor simulator for scenario
testing.

Evaluation dimensions:
1. Task completion    (>=90%)
2. Code quality       (>=4.0/5.0)
3. Safety             (zero critical findings)
4. Context preservation (>=95%)
5. Cost efficiency    (<=20% variance)
6. Coordination       (<=85% overhead)
7. Learning retention (improvement across iterations)
8. Latency            (p95 <= 30s)

All scoring logic uses pluggable callbacks for external dependencies
(code quality tools, etc.) so the suite is fully testable without the
Strands SDK or any third-party services.

Requirements: 17.1, 17.2, 17.3, 17.4
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric types
# ---------------------------------------------------------------------------

class MetricType(Enum):
    """Classification of how a metric is measured and compared."""
    PERCENTAGE = "percentage"      # 0-100, higher is better
    SCORE = "score"                # arbitrary scale, higher is better
    COUNT = "count"                # integer, lower is better (for safety)
    TIME = "time"                  # milliseconds, lower is better
    VARIANCE = "variance"          # percentage deviation, lower is better
    RATIO = "ratio"               # percentage, lower is better (overhead)


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EvalDimension:
    """Definition of a single evaluation dimension.

    Attributes
    ----------
    name:
        Human-readable dimension name (e.g. ``"Task Completion"``).
    target:
        The target value that must be met for the dimension to pass.
    metric_type:
        How the metric is measured (percentage, score, count, time, etc.).
    weight:
        Relative weight for aggregate scoring (all weights should sum
        to 1.0 across dimensions).
    higher_is_better:
        When ``True``, ``actual >= target`` is a pass.  When ``False``,
        ``actual <= target`` is a pass.
    """
    name: str
    target: float
    metric_type: MetricType
    weight: float
    higher_is_better: bool = True


@dataclass
class EvalResult:
    """Result of evaluating a single dimension.

    Attributes
    ----------
    dimension_name:
        Which dimension was evaluated.
    actual_value:
        The measured value.
    target_value:
        The target value from the dimension definition.
    passed:
        Whether the dimension target was met.
    details:
        Human-readable explanation of the result.
    """
    dimension_name: str
    actual_value: float
    target_value: float
    passed: bool
    details: str = ""


# ---------------------------------------------------------------------------
# Evaluation Suite
# ---------------------------------------------------------------------------

class BrainmassEvalSuite:
    """Multi-dimensional evaluation suite for Brainmass agents and sessions.

    Eight dimensions are defined as class-level constants.  Each
    ``evaluate_*`` method produces an :class:`EvalResult` for its
    dimension.  :meth:`run_all` executes all evaluations from a
    single ``test_data`` dict, and :meth:`summary` aggregates results.

    Usage::

        suite = BrainmassEvalSuite()
        results = suite.run_all(test_data)
        report = suite.summary(results)
    """

    # ------------------------------------------------------------------
    # Dimension definitions (class-level constants)
    # ------------------------------------------------------------------

    TASK_COMPLETION = EvalDimension(
        name="Task Completion",
        target=90.0,
        metric_type=MetricType.PERCENTAGE,
        weight=0.20,
        higher_is_better=True,
    )

    CODE_QUALITY = EvalDimension(
        name="Code Quality",
        target=4.0,
        metric_type=MetricType.SCORE,
        weight=0.15,
        higher_is_better=True,
    )

    SAFETY = EvalDimension(
        name="Safety",
        target=0.0,
        metric_type=MetricType.COUNT,
        weight=0.20,
        higher_is_better=False,
    )

    CONTEXT_PRESERVATION = EvalDimension(
        name="Context Preservation",
        target=95.0,
        metric_type=MetricType.PERCENTAGE,
        weight=0.10,
        higher_is_better=True,
    )

    COST_EFFICIENCY = EvalDimension(
        name="Cost Efficiency",
        target=20.0,
        metric_type=MetricType.VARIANCE,
        weight=0.10,
        higher_is_better=False,
    )

    COORDINATION = EvalDimension(
        name="Coordination Overhead",
        target=85.0,
        metric_type=MetricType.RATIO,
        weight=0.10,
        higher_is_better=False,
    )

    LEARNING_RETENTION = EvalDimension(
        name="Learning Retention",
        target=0.0,
        metric_type=MetricType.SCORE,
        weight=0.05,
        higher_is_better=True,
    )

    LATENCY = EvalDimension(
        name="Latency (p95)",
        target=30000.0,
        metric_type=MetricType.TIME,
        weight=0.10,
        higher_is_better=False,
    )

    DIMENSIONS: list[EvalDimension] = [
        TASK_COMPLETION,
        CODE_QUALITY,
        SAFETY,
        CONTEXT_PRESERVATION,
        COST_EFFICIENCY,
        COORDINATION,
        LEARNING_RETENTION,
        LATENCY,
    ]

    def __init__(
        self,
        quality_scorer: Optional[Callable[[dict], float]] = None,
    ) -> None:
        """Initialise the evaluation suite.

        Parameters
        ----------
        quality_scorer:
            Pluggable callback that scores a code sample dict and returns
            a float in [0.0, 5.0].  When ``None``, a default scorer is
            used that returns the sample's ``"score"`` key or 3.0.
        """
        self._quality_scorer = quality_scorer or self._default_quality_scorer

    # ------------------------------------------------------------------
    # 1. Task Completion (>=90%)
    # ------------------------------------------------------------------

    def evaluate_task_completion(self, results: list[dict]) -> EvalResult:
        """Evaluate task completion rate.

        Parameters
        ----------
        results:
            List of task result dicts, each with a ``"completed"`` bool key.

        Returns
        -------
        EvalResult
            Percentage of tasks completed vs target (90%).
        """
        dim = self.TASK_COMPLETION

        if not results:
            return EvalResult(
                dimension_name=dim.name,
                actual_value=0.0,
                target_value=dim.target,
                passed=False,
                details="No task results provided",
            )

        completed = sum(1 for r in results if r.get("completed", False))
        rate = (completed / len(results)) * 100.0

        passed = rate >= dim.target
        return EvalResult(
            dimension_name=dim.name,
            actual_value=round(rate, 2),
            target_value=dim.target,
            passed=passed,
            details=f"{completed}/{len(results)} tasks completed ({rate:.1f}%)",
        )

    # ------------------------------------------------------------------
    # 2. Code Quality (>=4.0/5.0)
    # ------------------------------------------------------------------

    def evaluate_code_quality(self, code_samples: list[dict]) -> EvalResult:
        """Evaluate code quality via pluggable scorer.

        Parameters
        ----------
        code_samples:
            List of code sample dicts passed to the quality scorer callback.

        Returns
        -------
        EvalResult
            Average quality score vs target (4.0/5.0).
        """
        dim = self.CODE_QUALITY

        if not code_samples:
            return EvalResult(
                dimension_name=dim.name,
                actual_value=0.0,
                target_value=dim.target,
                passed=False,
                details="No code samples provided",
            )

        scores = [self._quality_scorer(sample) for sample in code_samples]
        avg_score = statistics.mean(scores)

        passed = avg_score >= dim.target
        return EvalResult(
            dimension_name=dim.name,
            actual_value=round(avg_score, 2),
            target_value=dim.target,
            passed=passed,
            details=(
                f"Average quality score {avg_score:.2f}/5.0 "
                f"across {len(code_samples)} samples "
                f"(min={min(scores):.2f}, max={max(scores):.2f})"
            ),
        )

    # ------------------------------------------------------------------
    # 3. Safety (zero critical findings)
    # ------------------------------------------------------------------

    def evaluate_safety(self, findings: list[dict]) -> EvalResult:
        """Evaluate safety — zero critical findings required.

        Parameters
        ----------
        findings:
            List of finding dicts. Critical findings have
            ``"severity" == "critical"``.

        Returns
        -------
        EvalResult
            Count of critical findings vs target (0).
        """
        dim = self.SAFETY

        critical = [
            f for f in findings
            if f.get("severity", "").lower() == "critical"
        ]
        count = len(critical)

        passed = count <= dim.target
        details_parts = [f"{count} critical finding(s)"]
        if critical:
            # Include first few critical finding descriptions
            for finding in critical[:3]:
                desc = finding.get("description", "No description")
                details_parts.append(f"  - {desc[:120]}")
            if len(critical) > 3:
                details_parts.append(f"  ... and {len(critical) - 3} more")

        return EvalResult(
            dimension_name=dim.name,
            actual_value=float(count),
            target_value=dim.target,
            passed=passed,
            details="\n".join(details_parts),
        )

    # ------------------------------------------------------------------
    # 4. Context Preservation (>=95%)
    # ------------------------------------------------------------------

    def evaluate_context_preservation(
        self,
        before_items: list[dict],
        after_items: list[dict],
    ) -> EvalResult:
        """Evaluate context preservation across compaction.

        Measures what percentage of pre-compaction PRESERVE_VERBATIM
        items survive compaction intact.

        Parameters
        ----------
        before_items:
            Context items before compaction.  Each dict has at least
            ``"id"`` and ``"category"`` keys.
        after_items:
            Context items after compaction with the same key structure.

        Returns
        -------
        EvalResult
            Preservation percentage vs target (95%).
        """
        dim = self.CONTEXT_PRESERVATION

        # Only measure preservation of PRESERVE_VERBATIM items
        preserved_before = {
            item["id"]
            for item in before_items
            if item.get("category") == "PRESERVE_VERBATIM"
        }

        if not preserved_before:
            return EvalResult(
                dimension_name=dim.name,
                actual_value=100.0,
                target_value=dim.target,
                passed=True,
                details="No PRESERVE_VERBATIM items to track",
            )

        after_ids = {item["id"] for item in after_items}
        survived = preserved_before & after_ids
        rate = (len(survived) / len(preserved_before)) * 100.0

        lost = preserved_before - after_ids
        passed = rate >= dim.target

        details = f"{len(survived)}/{len(preserved_before)} PRESERVE_VERBATIM items preserved ({rate:.1f}%)"
        if lost:
            details += f"; lost IDs: {sorted(lost)[:5]}"

        return EvalResult(
            dimension_name=dim.name,
            actual_value=round(rate, 2),
            target_value=dim.target,
            passed=passed,
            details=details,
        )

    # ------------------------------------------------------------------
    # 5. Cost Efficiency (<=20% variance)
    # ------------------------------------------------------------------

    def evaluate_cost_efficiency(
        self,
        actual_cost: float,
        expected_cost: float,
    ) -> EvalResult:
        """Evaluate cost efficiency as percentage variance from expected.

        Parameters
        ----------
        actual_cost:
            The actual cost in USD.
        expected_cost:
            The expected/budgeted cost in USD.

        Returns
        -------
        EvalResult
            Cost variance percentage vs target (20%).
        """
        dim = self.COST_EFFICIENCY

        if expected_cost <= 0:
            return EvalResult(
                dimension_name=dim.name,
                actual_value=0.0,
                target_value=dim.target,
                passed=True,
                details="No expected cost provided (zero budget)",
            )

        variance_pct = abs(actual_cost - expected_cost) / expected_cost * 100.0
        passed = variance_pct <= dim.target

        direction = "over" if actual_cost > expected_cost else "under"
        return EvalResult(
            dimension_name=dim.name,
            actual_value=round(variance_pct, 2),
            target_value=dim.target,
            passed=passed,
            details=(
                f"${actual_cost:.4f} actual vs ${expected_cost:.4f} expected "
                f"({variance_pct:.1f}% {direction} budget)"
            ),
        )

    # ------------------------------------------------------------------
    # 6. Coordination Overhead (<=85%)
    # ------------------------------------------------------------------

    def evaluate_coordination(
        self,
        overhead_ms: float,
        total_ms: float,
    ) -> EvalResult:
        """Evaluate coordination overhead as a percentage of total time.

        Parameters
        ----------
        overhead_ms:
            Time spent on coordination (message passing, task assignment, etc.)
            in milliseconds.
        total_ms:
            Total execution time in milliseconds.

        Returns
        -------
        EvalResult
            Overhead percentage vs target (85%).
        """
        dim = self.COORDINATION

        if total_ms <= 0:
            return EvalResult(
                dimension_name=dim.name,
                actual_value=0.0,
                target_value=dim.target,
                passed=True,
                details="No execution time recorded",
            )

        overhead_pct = (overhead_ms / total_ms) * 100.0
        passed = overhead_pct <= dim.target

        productive_pct = 100.0 - overhead_pct
        return EvalResult(
            dimension_name=dim.name,
            actual_value=round(overhead_pct, 2),
            target_value=dim.target,
            passed=passed,
            details=(
                f"{overhead_ms:.0f}ms overhead / {total_ms:.0f}ms total "
                f"({overhead_pct:.1f}% overhead, {productive_pct:.1f}% productive)"
            ),
        )

    # ------------------------------------------------------------------
    # 7. Learning Retention
    # ------------------------------------------------------------------

    def evaluate_learning_retention(
        self,
        iterations: list[dict],
    ) -> EvalResult:
        """Evaluate whether the system improves across loop iterations.

        Measures learning retention by checking if later iterations
        produce better outcomes than earlier ones.  Each iteration dict
        should have a ``"score"`` (float) key representing iteration
        quality.

        Parameters
        ----------
        iterations:
            List of iteration result dicts in chronological order,
            each with a ``"score"`` key (float).

        Returns
        -------
        EvalResult
            Positive trend coefficient indicates learning.  The target
            is 0.0 (any positive value passes).
        """
        dim = self.LEARNING_RETENTION

        if len(iterations) < 2:
            return EvalResult(
                dimension_name=dim.name,
                actual_value=0.0,
                target_value=dim.target,
                passed=True,
                details="Fewer than 2 iterations — learning retention not applicable",
            )

        scores = [it.get("score", 0.0) for it in iterations]

        # Compute a simple linear trend: positive slope = improving
        trend = self._compute_trend(scores)

        # Also check if last half is better than first half
        midpoint = len(scores) // 2
        first_half_avg = statistics.mean(scores[:midpoint]) if midpoint > 0 else 0.0
        second_half_avg = statistics.mean(scores[midpoint:])
        improvement = second_half_avg - first_half_avg

        passed = trend >= dim.target
        return EvalResult(
            dimension_name=dim.name,
            actual_value=round(trend, 4),
            target_value=dim.target,
            passed=passed,
            details=(
                f"Trend coefficient: {trend:.4f} across {len(iterations)} iterations; "
                f"first-half avg={first_half_avg:.2f}, second-half avg={second_half_avg:.2f}, "
                f"improvement={improvement:+.2f}"
            ),
        )

    # ------------------------------------------------------------------
    # 8. Latency (p95 <= 30s)
    # ------------------------------------------------------------------

    def evaluate_latency(self, durations_ms: list[float]) -> EvalResult:
        """Evaluate p95 latency.

        Parameters
        ----------
        durations_ms:
            List of request/operation durations in milliseconds.

        Returns
        -------
        EvalResult
            p95 latency vs target (30000ms / 30s).
        """
        dim = self.LATENCY

        if not durations_ms:
            return EvalResult(
                dimension_name=dim.name,
                actual_value=0.0,
                target_value=dim.target,
                passed=True,
                details="No duration samples provided",
            )

        sorted_durations = sorted(durations_ms)
        p95_index = int(math.ceil(0.95 * len(sorted_durations))) - 1
        p95_index = max(0, min(p95_index, len(sorted_durations) - 1))
        p95 = sorted_durations[p95_index]

        p50_index = int(math.ceil(0.50 * len(sorted_durations))) - 1
        p50_index = max(0, min(p50_index, len(sorted_durations) - 1))
        p50 = sorted_durations[p50_index]

        passed = p95 <= dim.target
        return EvalResult(
            dimension_name=dim.name,
            actual_value=round(p95, 2),
            target_value=dim.target,
            passed=passed,
            details=(
                f"p50={p50:.0f}ms, p95={p95:.0f}ms across "
                f"{len(durations_ms)} samples "
                f"(min={min(durations_ms):.0f}ms, max={max(durations_ms):.0f}ms)"
            ),
        )

    # ------------------------------------------------------------------
    # Aggregate methods
    # ------------------------------------------------------------------

    def run_all(self, test_data: dict) -> list[EvalResult]:
        """Run all eight evaluation dimensions from a single test data dict.

        Parameters
        ----------
        test_data:
            Dict with keys corresponding to each dimension's input::

                {
                    "task_results": list[dict],          # for task completion
                    "code_samples": list[dict],          # for code quality
                    "safety_findings": list[dict],       # for safety
                    "context_before": list[dict],        # for context preservation
                    "context_after": list[dict],         # for context preservation
                    "actual_cost": float,                # for cost efficiency
                    "expected_cost": float,              # for cost efficiency
                    "overhead_ms": float,                # for coordination
                    "total_ms": float,                   # for coordination
                    "iterations": list[dict],            # for learning retention
                    "durations_ms": list[float],         # for latency
                }

        Returns
        -------
        list[EvalResult]
            One result per dimension.
        """
        results: list[EvalResult] = []

        results.append(
            self.evaluate_task_completion(
                test_data.get("task_results", [])
            )
        )

        results.append(
            self.evaluate_code_quality(
                test_data.get("code_samples", [])
            )
        )

        results.append(
            self.evaluate_safety(
                test_data.get("safety_findings", [])
            )
        )

        results.append(
            self.evaluate_context_preservation(
                test_data.get("context_before", []),
                test_data.get("context_after", []),
            )
        )

        results.append(
            self.evaluate_cost_efficiency(
                test_data.get("actual_cost", 0.0),
                test_data.get("expected_cost", 0.0),
            )
        )

        results.append(
            self.evaluate_coordination(
                test_data.get("overhead_ms", 0.0),
                test_data.get("total_ms", 0.0),
            )
        )

        results.append(
            self.evaluate_learning_retention(
                test_data.get("iterations", [])
            )
        )

        results.append(
            self.evaluate_latency(
                test_data.get("durations_ms", [])
            )
        )

        return results

    def summary(self, results: list[EvalResult]) -> dict:
        """Aggregate evaluation results into a summary report.

        Parameters
        ----------
        results:
            List of ``EvalResult`` objects (typically from :meth:`run_all`).

        Returns
        -------
        dict
            Summary report::

                {
                    "overall_pass": bool,
                    "pass_count": int,
                    "fail_count": int,
                    "total_dimensions": int,
                    "weighted_score": float,   # 0.0 - 1.0
                    "timestamp": str,          # ISO-8601
                    "dimensions": {
                        "Dimension Name": {
                            "actual": float,
                            "target": float,
                            "passed": bool,
                            "details": str,
                        },
                        ...
                    },
                }
        """
        pass_count = sum(1 for r in results if r.passed)
        fail_count = len(results) - pass_count

        # Compute weighted score using dimension weights
        dim_lookup = {d.name: d for d in self.DIMENSIONS}
        weighted_sum = 0.0
        total_weight = 0.0

        for r in results:
            dim = dim_lookup.get(r.dimension_name)
            weight = dim.weight if dim else (1.0 / max(len(results), 1))
            total_weight += weight
            if r.passed:
                weighted_sum += weight

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        dimensions: dict[str, dict] = {}
        for r in results:
            dimensions[r.dimension_name] = {
                "actual": r.actual_value,
                "target": r.target_value,
                "passed": r.passed,
                "details": r.details,
            }

        return {
            "overall_pass": fail_count == 0,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "total_dimensions": len(results),
            "weighted_score": round(weighted_score, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dimensions": dimensions,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_quality_scorer(sample: dict) -> float:
        """Default quality scorer — returns the sample's 'score' key or 3.0."""
        score = sample.get("score", 3.0)
        return max(0.0, min(5.0, float(score)))

    @staticmethod
    def _compute_trend(values: list[float]) -> float:
        """Compute a simple linear regression slope (trend coefficient).

        Positive slope indicates improvement over time.  Returns 0.0 for
        fewer than 2 data points.
        """
        n = len(values)
        if n < 2:
            return 0.0

        # Simple linear regression: y = a + b*x
        # x = 0, 1, 2, ... n-1
        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(values)

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return numerator / denominator


# ---------------------------------------------------------------------------
# Actor Simulator (multi-turn scenario testing)
# ---------------------------------------------------------------------------

@dataclass
class _Turn:
    """A single turn in a simulated multi-turn conversation."""
    role: str
    content: str
    response: Optional[str] = None
    timestamp: Optional[str] = None


class ActorSimulator:
    """Multi-turn scenario simulator for testing agent behaviour.

    Records a sequence of turns (user/assistant/system) and replays them
    through an agent callback, capturing responses for assertion.

    Usage::

        sim = ActorSimulator()
        sim.add_turn("user", "Write a function to sort a list")
        sim.add_turn("user", "Now add type hints")

        def agent_fn(role: str, content: str) -> str:
            return f"Response to: {content[:50]}"

        sim.simulate(agent_fn)
        transcript = sim.get_transcript()
    """

    def __init__(self) -> None:
        self._turns: list[_Turn] = []
        self._simulated: bool = False

    # -- public API --------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """Add a turn to the scenario.

        Parameters
        ----------
        role:
            The role for this turn (``"user"``, ``"assistant"``, ``"system"``).
        content:
            The content/message for this turn.
        """
        self._turns.append(_Turn(role=role, content=content))

    def simulate(
        self,
        agent_callback: Callable[[str, str], str],
    ) -> list[dict]:
        """Replay all turns through the agent callback.

        Parameters
        ----------
        agent_callback:
            Callable with signature ``(role: str, content: str) -> str``
            that simulates agent processing and returns a response string.

        Returns
        -------
        list[dict]
            List of turn result dicts with keys: ``role``, ``content``,
            ``response``, ``timestamp``.
        """
        results: list[dict] = []

        for turn in self._turns:
            timestamp = datetime.now(timezone.utc).isoformat()
            try:
                response = agent_callback(turn.role, turn.content)
            except Exception as exc:
                logger.warning(
                    "ActorSimulator: agent callback failed on turn '%s': %s",
                    turn.content[:50], exc,
                )
                response = f"ERROR: {exc}"

            turn.response = response
            turn.timestamp = timestamp

            results.append({
                "role": turn.role,
                "content": turn.content,
                "response": response,
                "timestamp": timestamp,
            })

        self._simulated = True
        return results

    def get_transcript(self) -> list[dict]:
        """Return the full transcript of all turns.

        Returns
        -------
        list[dict]
            Each dict has keys: ``role``, ``content``, ``response``
            (``None`` if :meth:`simulate` has not been called yet),
            and ``timestamp``.
        """
        return [
            {
                "role": turn.role,
                "content": turn.content,
                "response": turn.response,
                "timestamp": turn.timestamp,
            }
            for turn in self._turns
        ]

    @property
    def turn_count(self) -> int:
        """Return the number of turns in the scenario."""
        return len(self._turns)

    @property
    def is_simulated(self) -> bool:
        """Return whether :meth:`simulate` has been called."""
        return self._simulated

    def reset(self) -> None:
        """Clear all turns and reset simulation state."""
        self._turns.clear()
        self._simulated = False
