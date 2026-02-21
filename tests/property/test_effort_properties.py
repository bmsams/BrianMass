"""Property-based tests for Effort Controller selection logic.

Properties covered:
- Property 19: Effort level selection
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.cost.cost_governor import TaskSignals
from src.orchestrator.effort_controller import EFFORT_BUDGET_TOKENS, EffortController
from src.types.core import AgentBudget, BudgetStatus, EffortLevel, ModelTier

EFFORT_ORDER = [EffortLevel.QUICK, EffortLevel.STANDARD, EffortLevel.DEEP]
MODEL_MAX = {
    ModelTier.HAIKU: EffortLevel.QUICK,
    ModelTier.SONNET: EffortLevel.STANDARD,
    ModelTier.OPUS: EffortLevel.DEEP,
}


def _budget_for_status(status: BudgetStatus) -> AgentBudget:
    base = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=100.0,
        current_cost_usd=0.0,
    )
    if status == BudgetStatus.WARNING:
        base.current_cost_usd = 80.0
    elif status == BudgetStatus.CRITICAL:
        base.current_cost_usd = 96.0
    elif status == BudgetStatus.EXCEEDED:
        base.current_cost_usd = 101.0
    return base


def _expected_natural_effort(signals: TaskSignals) -> EffortLevel:
    if signals.is_exploration:
        return EffortLevel.QUICK
    if signals.files_affected <= 1 and not signals.requires_reasoning:
        return EffortLevel.QUICK
    if signals.requires_reasoning and signals.dependency_depth > 3:
        return EffortLevel.DEEP
    if signals.files_affected > 5 and signals.requires_reasoning:
        return EffortLevel.DEEP
    return EffortLevel.STANDARD


def _cap_to_model(effort: EffortLevel, model: ModelTier) -> EffortLevel:
    max_effort = MODEL_MAX[model]
    return EFFORT_ORDER[min(EFFORT_ORDER.index(effort), EFFORT_ORDER.index(max_effort))]


def _apply_budget_override(effort: EffortLevel, status: BudgetStatus) -> EffortLevel:
    if status in (BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED):
        return EffortLevel.QUICK
    if status == BudgetStatus.WARNING:
        return EFFORT_ORDER[max(EFFORT_ORDER.index(effort) - 1, 0)]
    return effort


@pytest.mark.property
@settings(max_examples=100)
@given(
    signals=st.builds(
        TaskSignals,
        is_exploration=st.booleans(),
        files_affected=st.integers(min_value=0, max_value=20),
        requires_reasoning=st.booleans(),
        is_team_lead=st.booleans(),
        dependency_depth=st.integers(min_value=0, max_value=10),
    ),
    model=st.sampled_from(list(ModelTier)),
    budget_status=st.sampled_from(list(BudgetStatus)),
)
def test_property_19_effort_level_selection(
    signals: TaskSignals,
    model: ModelTier,
    budget_status: BudgetStatus,
) -> None:
    """Feature: claude-code-v3-enterprise, Property 19."""
    controller = EffortController()
    budget = _budget_for_status(budget_status)

    natural = _expected_natural_effort(signals)
    capped = _cap_to_model(natural, model)
    expected_effort = _apply_budget_override(capped, budget_status)

    result = controller.select_effort(signals, model, budget)
    assert result.level == expected_effort.value
    assert result.budget_tokens == EFFORT_BUDGET_TOKENS[expected_effort]

