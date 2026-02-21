"""Unit tests for the Cost Governor."""

import pytest

from src.cost.cost_governor import CostGovernor, TaskSignals, _downgrade_one_tier
from src.types.core import AgentBudget, BudgetStatus, ModelTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_budget(
    session_budget_usd: float = 10.0,
    current_cost_usd: float = 0.0,
    input_budget_tokens: int = 100_000,
    output_budget_tokens: int = 50_000,
) -> AgentBudget:
    return AgentBudget(
        input_budget_tokens=input_budget_tokens,
        output_budget_tokens=output_budget_tokens,
        session_budget_usd=session_budget_usd,
        current_cost_usd=current_cost_usd,
    )


# ===================================================================
# Model selection tests  (Requirement 4.4, 4.5)
# ===================================================================

class TestSelectModel:
    """select_model routes to the cheapest capable tier."""

    def test_exploration_returns_haiku(self):
        gov = CostGovernor()
        signals = TaskSignals(is_exploration=True)
        assert gov.select_model(signals, _make_budget()) == ModelTier.HAIKU

    def test_single_file_no_reasoning_returns_haiku(self):
        gov = CostGovernor()
        signals = TaskSignals(files_affected=1, requires_reasoning=False)
        assert gov.select_model(signals, _make_budget()) == ModelTier.HAIKU

    def test_zero_files_no_reasoning_returns_haiku(self):
        gov = CostGovernor()
        signals = TaskSignals(files_affected=0, requires_reasoning=False)
        assert gov.select_model(signals, _make_budget()) == ModelTier.HAIKU

    def test_team_lead_returns_opus(self):
        gov = CostGovernor()
        signals = TaskSignals(is_team_lead=True, files_affected=5)
        assert gov.select_model(signals, _make_budget()) == ModelTier.OPUS

    def test_requires_reasoning_returns_opus(self):
        gov = CostGovernor()
        signals = TaskSignals(requires_reasoning=True, files_affected=5)
        assert gov.select_model(signals, _make_budget()) == ModelTier.OPUS

    def test_deep_dependency_returns_opus(self):
        gov = CostGovernor()
        signals = TaskSignals(dependency_depth=4, files_affected=5)
        assert gov.select_model(signals, _make_budget()) == ModelTier.OPUS

    def test_default_returns_sonnet(self):
        gov = CostGovernor()
        signals = TaskSignals(files_affected=3)
        assert gov.select_model(signals, _make_budget()) == ModelTier.SONNET

    def test_warning_budget_downgrades_one_tier(self):
        gov = CostGovernor()
        # Natural pick would be Opus (requires_reasoning + multi-file)
        signals = TaskSignals(requires_reasoning=True, files_affected=5)
        budget = _make_budget(session_budget_usd=10.0, current_cost_usd=8.5)
        assert budget.status == BudgetStatus.WARNING
        assert gov.select_model(signals, budget) == ModelTier.SONNET

    def test_warning_budget_downgrades_sonnet_to_haiku(self):
        gov = CostGovernor()
        signals = TaskSignals(files_affected=3)  # natural = Sonnet
        budget = _make_budget(session_budget_usd=10.0, current_cost_usd=8.5)
        assert gov.select_model(signals, budget) == ModelTier.HAIKU

    def test_critical_budget_forces_haiku(self):
        gov = CostGovernor()
        signals = TaskSignals(is_team_lead=True, files_affected=10)
        budget = _make_budget(session_budget_usd=10.0, current_cost_usd=9.6)
        assert budget.status == BudgetStatus.CRITICAL
        assert gov.select_model(signals, budget) == ModelTier.HAIKU

    def test_exceeded_budget_forces_haiku(self):
        gov = CostGovernor()
        signals = TaskSignals(is_team_lead=True, files_affected=10)
        budget = _make_budget(session_budget_usd=10.0, current_cost_usd=10.5)
        assert budget.status == BudgetStatus.EXCEEDED
        assert gov.select_model(signals, budget) == ModelTier.HAIKU


# ===================================================================
# Tier downgrade helper
# ===================================================================

class TestDowngradeOneTier:
    def test_opus_to_sonnet(self):
        assert _downgrade_one_tier(ModelTier.OPUS) == ModelTier.SONNET

    def test_sonnet_to_haiku(self):
        assert _downgrade_one_tier(ModelTier.SONNET) == ModelTier.HAIKU

    def test_haiku_stays_haiku(self):
        assert _downgrade_one_tier(ModelTier.HAIKU) == ModelTier.HAIKU


# ===================================================================
# Cost calculation tests  (Requirement 4.6)
# ===================================================================

class TestCalculateCost:
    """calculate_cost applies the pricing formula correctly."""

    def test_sonnet_basic(self):
        # 1M input, 500K output, no cache
        cost = CostGovernor.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=500_000,
            model_tier=ModelTier.SONNET,
        )
        expected = (1_000_000 / 1e6) * 3.0 + (500_000 / 1e6) * 15.0
        assert cost == pytest.approx(expected)

    def test_haiku_basic(self):
        cost = CostGovernor.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model_tier=ModelTier.HAIKU,
        )
        expected = 0.80 + 4.00
        assert cost == pytest.approx(expected)

    def test_opus_basic(self):
        cost = CostGovernor.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model_tier=ModelTier.OPUS,
        )
        expected = 5.00 + 25.00
        assert cost == pytest.approx(expected)

    def test_cache_read_reduces_regular_input(self):
        cost = CostGovernor.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read_tokens=500_000,
            model_tier=ModelTier.SONNET,
        )
        # regular = 1M - 500K = 500K
        expected = (500_000 / 1e6) * 3.0 + (500_000 / 1e6) * 0.30
        assert cost == pytest.approx(expected)

    def test_cache_write_tokens(self):
        cost = CostGovernor.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=0,
            cache_write_tokens=200_000,
            model_tier=ModelTier.SONNET,
        )
        # regular = 1M - 200K = 800K
        expected = (800_000 / 1e6) * 3.0 + (200_000 / 1e6) * 3.75
        assert cost == pytest.approx(expected)

    def test_batch_api_50_percent_discount(self):
        normal = CostGovernor.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=500_000,
            model_tier=ModelTier.SONNET,
            is_batch=False,
        )
        batch = CostGovernor.calculate_cost(
            input_tokens=1_000_000,
            output_tokens=500_000,
            model_tier=ModelTier.SONNET,
            is_batch=True,
        )
        assert batch == pytest.approx(normal * 0.5)

    def test_zero_tokens_zero_cost(self):
        cost = CostGovernor.calculate_cost(
            input_tokens=0,
            output_tokens=0,
            model_tier=ModelTier.HAIKU,
        )
        assert cost == 0.0

    def test_regular_input_never_negative(self):
        # cache tokens exceed input tokens — regular should clamp to 0
        cost = CostGovernor.calculate_cost(
            input_tokens=100,
            output_tokens=0,
            cache_read_tokens=200,
            model_tier=ModelTier.SONNET,
        )
        # regular = max(100 - 200, 0) = 0; only cache_read cost
        expected = (200 / 1e6) * 0.30
        assert cost == pytest.approx(expected)


# ===================================================================
# Budget status thresholds  (Requirement 4.1, 4.2, 4.3)
# ===================================================================

class TestBudgetStatus:
    """AgentBudget.status returns correct thresholds."""

    def test_ok_at_zero(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=0.0)
        assert b.status == BudgetStatus.OK

    def test_ok_below_80(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=7.99)
        assert b.status == BudgetStatus.OK

    def test_warning_at_80(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=8.0)
        assert b.status == BudgetStatus.WARNING

    def test_warning_between_80_and_95(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=9.0)
        assert b.status == BudgetStatus.WARNING

    def test_critical_at_95(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=9.5)
        assert b.status == BudgetStatus.CRITICAL

    def test_critical_between_95_and_100(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=9.9)
        assert b.status == BudgetStatus.CRITICAL

    def test_exceeded_at_100(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=10.0)
        assert b.status == BudgetStatus.EXCEEDED

    def test_exceeded_over_100(self):
        b = _make_budget(session_budget_usd=10.0, current_cost_usd=15.0)
        assert b.status == BudgetStatus.EXCEEDED

    def test_zero_budget_is_ok(self):
        b = _make_budget(session_budget_usd=0.0, current_cost_usd=5.0)
        assert b.status == BudgetStatus.OK


# ===================================================================
# record_usage + check_budget integration
# ===================================================================

class TestRecordUsageAndCheckBudget:
    """record_usage accumulates cost and check_budget reflects it."""

    def test_record_updates_budget(self):
        gov = CostGovernor()
        budget = _make_budget(session_budget_usd=1.0)
        gov.register_agent("a1", budget)

        gov.record_usage("a1", input_tokens=1_000_000, output_tokens=0,
                         model_tier=ModelTier.HAIKU)
        # cost = 1M / 1M * 0.80 = $0.80  → 80% → WARNING
        assert gov.check_budget("a1") == BudgetStatus.WARNING

    def test_multiple_records_accumulate(self):
        gov = CostGovernor()
        budget = _make_budget(session_budget_usd=2.0)
        gov.register_agent("a1", budget)

        gov.record_usage("a1", input_tokens=1_000_000, output_tokens=0,
                         model_tier=ModelTier.HAIKU)
        gov.record_usage("a1", input_tokens=1_000_000, output_tokens=0,
                         model_tier=ModelTier.HAIKU)
        # total cost = $1.60 → 80% of $2 → WARNING
        assert gov.check_budget("a1") == BudgetStatus.WARNING

    def test_cached_tokens_reduce_cost(self):
        gov = CostGovernor()
        budget = _make_budget(session_budget_usd=10.0)
        gov.register_agent("a1", budget)

        gov.record_usage("a1", input_tokens=1_000_000, output_tokens=0,
                         model_tier=ModelTier.SONNET, cached_tokens=900_000)
        # regular = 100K, cached = 900K
        # cost = (100K/1M)*3.0 + (900K/1M)*0.30 = 0.30 + 0.27 = 0.57
        assert gov.check_budget("a1") == BudgetStatus.OK


# ===================================================================
# Dashboard data  (Requirement 4.7)
# ===================================================================

class TestGetDashboardData:
    """get_dashboard_data returns correct aggregated metrics."""

    def test_empty_dashboard(self):
        gov = CostGovernor()
        data = gov.get_dashboard_data()
        assert data["token_consumption"]["total_input_tokens"] == 0
        assert data["cost_per_agent"] == {}
        assert data["cache_hit_rates"] == {}

    def test_single_agent_dashboard(self):
        gov = CostGovernor()
        gov.register_agent("a1", _make_budget())
        gov.record_usage("a1", input_tokens=1000, output_tokens=500,
                         model_tier=ModelTier.SONNET, cached_tokens=200)

        data = gov.get_dashboard_data()
        assert data["token_consumption"]["total_input_tokens"] == 1000
        assert data["token_consumption"]["total_output_tokens"] == 500
        assert data["token_consumption"]["total_cached_tokens"] == 200
        assert "a1" in data["cost_per_agent"]
        assert data["cost_per_agent"]["a1"] > 0
        assert data["cache_hit_rates"]["a1"] == pytest.approx(200 / 1000)

    def test_multi_agent_dashboard(self):
        gov = CostGovernor()
        gov.register_agent("a1", _make_budget())
        gov.register_agent("a2", _make_budget())
        gov.record_usage("a1", 500, 100, ModelTier.HAIKU)
        gov.record_usage("a2", 300, 200, ModelTier.OPUS)

        data = gov.get_dashboard_data()
        assert data["token_consumption"]["total_input_tokens"] == 800
        assert data["token_consumption"]["total_output_tokens"] == 300
        assert len(data["cost_per_agent"]) == 2
        assert data["model_tier_distribution"]["haiku"] == 1
        assert data["model_tier_distribution"]["opus"] == 1
        assert data["model_tier_distribution"]["sonnet"] == 0

    def test_cache_hit_rate_zero_when_no_input(self):
        gov = CostGovernor()
        gov.register_agent("a1", _make_budget())
        data = gov.get_dashboard_data()
        assert data["cache_hit_rates"]["a1"] == 0.0

    def test_tier_distribution_counts_calls(self):
        gov = CostGovernor()
        gov.register_agent("a1", _make_budget())
        gov.record_usage("a1", 100, 50, ModelTier.SONNET)
        gov.record_usage("a1", 100, 50, ModelTier.SONNET)
        gov.record_usage("a1", 100, 50, ModelTier.HAIKU)

        data = gov.get_dashboard_data()
        assert data["model_tier_distribution"]["sonnet"] == 2
        assert data["model_tier_distribution"]["haiku"] == 1
