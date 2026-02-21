"""Effort Controller — adaptive thinking and effort level selection.

Selects Quick / Standard / Deep effort levels based on task complexity
signals, model capability, and budget constraints.  Maps effort levels
to the ``budget_tokens`` parameter used by the Anthropic extended
thinking API.

Supports Fast Mode for Opus 4.6 — when the model is Opus and the task
is simple, the controller selects Quick with a ``fast_mode`` flag and
an OTel speed attribute for observability.

Requirements: 12.1, 12.2, 12.3, 12.4
"""

from __future__ import annotations

from dataclasses import dataclass

from src.cost.cost_governor import TaskSignals
from src.observability.instrumentation import BrainmassTracer
from src.types.core import (
    AgentBudget,
    BudgetStatus,
    EffortLevel,
    ModelTier,
)

# ---------------------------------------------------------------------------
# Budget-token mapping
# ---------------------------------------------------------------------------

EFFORT_BUDGET_TOKENS: dict[EffortLevel, int] = {
    EffortLevel.QUICK: 2_000,
    EffortLevel.STANDARD: 10_000,
    EffortLevel.DEEP: 50_000,
}

# ---------------------------------------------------------------------------
# Model capability constraints
# ---------------------------------------------------------------------------
# Haiku: always Quick
# Sonnet: Quick or Standard
# Opus: any level (Quick / Standard / Deep)

_MODEL_MAX_EFFORT: dict[ModelTier, EffortLevel] = {
    ModelTier.HAIKU: EffortLevel.QUICK,
    ModelTier.SONNET: EffortLevel.STANDARD,
    ModelTier.OPUS: EffortLevel.DEEP,
}

# Ordered from lowest to highest for downgrade logic
_EFFORT_ORDER: list[EffortLevel] = [
    EffortLevel.QUICK,
    EffortLevel.STANDARD,
    EffortLevel.DEEP,
]


# ---------------------------------------------------------------------------
# Extended EffortResult (adds fast_mode over the orchestrator's base)
# ---------------------------------------------------------------------------

@dataclass
class EffortResult:
    """Result of effort-level selection.

    Compatible with ``EffortControllerProtocol`` from the Orchestrator.
    Adds ``fast_mode`` and ``otel_attributes`` for Opus Fast Mode
    (Requirement 12.4).
    """

    level: str  # 'quick' | 'standard' | 'deep'
    budget_tokens: int  # 2000 / 10000 / 50000+
    fast_mode: bool = False
    otel_attributes: dict[str, object] | None = None


# ---------------------------------------------------------------------------
# EffortController
# ---------------------------------------------------------------------------

class EffortController:
    """Selects the appropriate effort level for a task.

    Satisfies ``EffortControllerProtocol`` expected by the Orchestrator.

    Selection algorithm:
        1. Determine the *natural* effort from task signals.
        2. Cap to the model's maximum capability.
        3. Apply budget overrides (downgrade at WARNING, force Quick at CRITICAL).
        4. Apply explicit override if provided.
        5. Detect Fast Mode (Opus + Quick).
        6. Map to ``budget_tokens``.

    Parameters
    ----------
    fast_mode_enabled : bool
        Whether Fast Mode is enabled for Opus 4.6 (Requirement 12.4).
        Defaults to ``True``.
    """

    def __init__(self, fast_mode_enabled: bool = True, tracer: BrainmassTracer | None = None) -> None:
        self.fast_mode_enabled = fast_mode_enabled
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API  (matches EffortControllerProtocol)
    # ------------------------------------------------------------------

    def select_effort(
        self,
        signals: TaskSignals,
        model: ModelTier,
        budget: AgentBudget,
        override: EffortLevel | None = None,
    ) -> EffortResult:
        """Select effort level and return an ``EffortResult``.

        Parameters
        ----------
        signals : TaskSignals
            Complexity signals from the request classifier.
        model : ModelTier
            The model tier selected by the Cost Governor.
        budget : AgentBudget
            Current agent budget (used for status-based downgrades).
        override : EffortLevel | None
            Explicit developer override.  When provided, it still gets
            capped to the model's maximum capability and budget constraints.
        """
        # Step 1: natural effort from signals
        natural = self._natural_effort(signals)

        # Step 2: apply explicit override (if any)
        effort = override if override is not None else natural

        # Step 3: cap to model capability
        effort = self._cap_to_model(effort, model)

        # Step 4: budget overrides
        effort = self._apply_budget_override(effort, budget)

        # Step 5: map to budget_tokens
        budget_tokens = EFFORT_BUDGET_TOKENS[effort]

        # Step 6: detect Fast Mode (Opus + Quick + enabled)
        fast_mode = (
            self.fast_mode_enabled
            and model == ModelTier.OPUS
            and effort == EffortLevel.QUICK
        )

        otel_attributes: dict[str, object] | None = None
        if fast_mode:
            otel_attributes = {
                "effort.fast_mode": True,
                "effort.speed": "fast",
            }

        result = EffortResult(
            level=effort.value,
            budget_tokens=budget_tokens,
            fast_mode=fast_mode,
            otel_attributes=otel_attributes,
        )

        # Emit effort span (Req 16.2)
        if self._tracer is not None:
            self._tracer.record_effort_span(
                effort_level=effort.value,
                fast_mode=fast_mode,
                budget_tokens=budget_tokens,
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _natural_effort(signals: TaskSignals) -> EffortLevel:
        """Determine effort purely from task complexity signals.

        Quick indicators:
            - Exploration tasks (search, grep, list)
            - Single file affected AND no reasoning required

        Deep indicators:
            - Requires reasoning (architecture, security, complex debug)
            - High dependency depth (> 3, cross-service)
            - Many files affected (> 5)

        Everything else → Standard.
        """
        # Quick signals
        if signals.is_exploration:
            return EffortLevel.QUICK
        if signals.files_affected <= 1 and not signals.requires_reasoning:
            return EffortLevel.QUICK

        # Deep signals
        if signals.requires_reasoning and signals.dependency_depth > 3:
            return EffortLevel.DEEP
        if signals.files_affected > 5 and signals.requires_reasoning:
            return EffortLevel.DEEP

        # Default
        return EffortLevel.STANDARD

    @staticmethod
    def _cap_to_model(effort: EffortLevel, model: ModelTier) -> EffortLevel:
        """Cap effort to the model's maximum capability.

        Haiku  → always Quick
        Sonnet → max Standard
        Opus   → any level
        """
        max_effort = _MODEL_MAX_EFFORT[model]
        max_idx = _EFFORT_ORDER.index(max_effort)
        cur_idx = _EFFORT_ORDER.index(effort)
        if cur_idx > max_idx:
            return max_effort
        return effort

    @staticmethod
    def _apply_budget_override(
        effort: EffortLevel,
        budget: AgentBudget,
    ) -> EffortLevel:
        """Downgrade effort based on budget status.

        - CRITICAL / EXCEEDED → force Quick
        - WARNING → downgrade one level
        - OK → no change
        """
        status = budget.status
        if status in (BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED):
            return EffortLevel.QUICK
        if status == BudgetStatus.WARNING:
            idx = _EFFORT_ORDER.index(effort)
            return _EFFORT_ORDER[max(idx - 1, 0)]
        return effort
