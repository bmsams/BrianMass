"""Property-based tests for Cost Governor behavior.

Properties covered:
- Property 8: Cost calculation correctness
- Property 9: Budget status thresholds and override behavior
- Property 10: Model selection from TaskSignals
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.cost.cost_governor import CostGovernor, TaskSignals, _downgrade_one_tier
from src.types.core import PRICING, AgentBudget, BudgetStatus, ModelTier


def _expected_natural_tier(signals: TaskSignals) -> ModelTier:
    if signals.is_exploration:
        return ModelTier.HAIKU
    if signals.files_affected <= 1 and not signals.requires_reasoning:
        return ModelTier.HAIKU
    if signals.is_team_lead or signals.requires_reasoning or signals.dependency_depth > 3:
        return ModelTier.OPUS
    return ModelTier.SONNET


@pytest.mark.property
@settings(max_examples=100)
@given(
    input_tokens=st.integers(min_value=0, max_value=5_000_000),
    output_tokens=st.integers(min_value=0, max_value=5_000_000),
    cache_read_tokens=st.integers(min_value=0, max_value=5_000_000),
    cache_write_tokens=st.integers(min_value=0, max_value=5_000_000),
    model_tier=st.sampled_from(list(ModelTier)),
    is_batch=st.booleans(),
)
def test_property_8_cost_calculation_correctness(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    model_tier: ModelTier,
    is_batch: bool,
) -> None:
    """Feature: claude-code-v3-enterprise, Property 8."""
    pricing = PRICING[model_tier]
    regular_input = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)
    expected = (
        (regular_input / 1_000_000) * pricing.input_per_million
        + (output_tokens / 1_000_000) * pricing.output_per_million
        + (cache_read_tokens / 1_000_000) * pricing.cached_input_per_million
        + (cache_write_tokens / 1_000_000) * pricing.cache_write_per_million
    )
    if is_batch:
        expected *= 0.5

    actual = CostGovernor.calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        model_tier=model_tier,
        is_batch=is_batch,
    )
    assert actual == pytest.approx(expected)


@pytest.mark.property
@settings(max_examples=100)
@given(
    utilization=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    signals=st.builds(
        TaskSignals,
        is_exploration=st.booleans(),
        files_affected=st.integers(min_value=0, max_value=20),
        requires_reasoning=st.booleans(),
        is_team_lead=st.booleans(),
        dependency_depth=st.integers(min_value=0, max_value=10),
    ),
)
def test_property_9_budget_status_thresholds_and_overrides(
    utilization: float,
    signals: TaskSignals,
) -> None:
    """Feature: claude-code-v3-enterprise, Property 9."""
    gov = CostGovernor()
    budget = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=100.0,
        current_cost_usd=100.0 * utilization,
    )

    if utilization >= 1.0:
        expected_status = BudgetStatus.EXCEEDED
    elif utilization >= 0.95:
        expected_status = BudgetStatus.CRITICAL
    elif utilization >= 0.80:
        expected_status = BudgetStatus.WARNING
    else:
        expected_status = BudgetStatus.OK
    assert budget.status == expected_status

    natural = _expected_natural_tier(signals)
    chosen = gov.select_model(signals, budget)
    if expected_status in (BudgetStatus.CRITICAL, BudgetStatus.EXCEEDED):
        assert chosen == ModelTier.HAIKU
    elif expected_status == BudgetStatus.WARNING:
        assert chosen == _downgrade_one_tier(natural)
    else:
        assert chosen == natural


@pytest.mark.property
@settings(max_examples=100)
@given(
    is_exploration=st.booleans(),
    files_affected=st.integers(min_value=0, max_value=20),
    requires_reasoning=st.booleans(),
    is_team_lead=st.booleans(),
    dependency_depth=st.integers(min_value=0, max_value=10),
)
def test_property_10_model_selection_from_task_signals(
    is_exploration: bool,
    files_affected: int,
    requires_reasoning: bool,
    is_team_lead: bool,
    dependency_depth: int,
) -> None:
    """Feature: claude-code-v3-enterprise, Property 10."""
    gov = CostGovernor()
    signals = TaskSignals(
        is_exploration=is_exploration,
        files_affected=files_affected,
        requires_reasoning=requires_reasoning,
        is_team_lead=is_team_lead,
        dependency_depth=dependency_depth,
    )

    expected_natural = _expected_natural_tier(signals)
    ok_budget = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=100.0,
        current_cost_usd=0.0,
    )
    warning_budget = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=100.0,
        current_cost_usd=85.0,
    )
    critical_budget = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=100.0,
        current_cost_usd=96.0,
    )

    assert gov.select_model(signals, ok_budget) == expected_natural
    assert gov.select_model(signals, warning_budget) == _downgrade_one_tier(expected_natural)
    assert gov.select_model(signals, critical_budget) == ModelTier.HAIKU

