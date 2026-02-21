"""Tests for Unified Quota Manager.

Validates Requirements 15.1-15.7:
- 15.1: Cross-surface usage tracking
- 15.2: Quota exhaustion prediction
- 15.3: Alert at 80% threshold
- 15.4: Usage breakdowns by surface and agent
- 15.5: Weekly rate limits
- 15.6: SDKRateLimitInfo event consumption
- 15.7: Overage purchase for Max subscribers
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import pytest

from src.cost.quota_manager import (
    VALID_SURFACES,
    QuotaAlert,
    QuotaConfig,
    QuotaManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> QuotaConfig:
    return QuotaConfig(
        total_token_quota=1_000_000,
        total_cost_quota_usd=50.0,
        alert_threshold=0.80,
    )


@pytest.fixture
def qm(config: QuotaConfig) -> QuotaManager:
    return QuotaManager(config=config)


# ---------------------------------------------------------------------------
# Req 15.1: Cross-surface usage tracking
# ---------------------------------------------------------------------------


class TestTrackUsage:
    """Req 15.1: Track consumption across all surfaces in real-time."""

    def test_track_single_surface(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=10_000, cost_usd=0.03)
        breakdown = qm.get_usage_breakdown()
        assert breakdown["total_tokens"] == 10_000
        assert breakdown["total_cost_usd"] == 0.03

    def test_track_multiple_surfaces(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=10_000, cost_usd=0.03)
        qm.track_usage("web", "agent-2", tokens=20_000, cost_usd=0.06)
        qm.track_usage("desktop", "agent-3", tokens=5_000, cost_usd=0.015)
        qm.track_usage("mobile", "agent-4", tokens=3_000, cost_usd=0.009)

        breakdown = qm.get_usage_breakdown()
        assert breakdown["total_tokens"] == 38_000
        assert len(breakdown["by_surface"]) == 4

    def test_all_valid_surfaces(self, qm: QuotaManager) -> None:
        for surface in VALID_SURFACES:
            qm.track_usage(surface, "test-agent", tokens=1_000, cost_usd=0.003)
        assert qm.get_usage_breakdown()["total_tokens"] == 4_000

    def test_invalid_surface_raises(self, qm: QuotaManager) -> None:
        with pytest.raises(ValueError, match="Invalid surface"):
            qm.track_usage("invalid", "agent-1", tokens=100, cost_usd=0.0003)

    def test_cumulative_tracking(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=5_000, cost_usd=0.015)
        qm.track_usage("cli", "agent-1", tokens=7_000, cost_usd=0.021)
        breakdown = qm.get_usage_breakdown()
        assert breakdown["by_agent"]["agent-1"]["tokens"] == 12_000

    def test_thread_safety(self, config: QuotaConfig) -> None:
        """Verify concurrent track_usage calls don't corrupt state."""
        qm = QuotaManager(config=config)
        errors: list[Exception] = []

        def worker(surface: str, n: int) -> None:
            try:
                for _ in range(n):
                    qm.track_usage(surface, f"agent-{surface}", tokens=100, cost_usd=0.0003)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("cli", 100)),
            threading.Thread(target=worker, args=("web", 100)),
            threading.Thread(target=worker, args=("desktop", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert qm.get_usage_breakdown()["total_tokens"] == 30_000


# ---------------------------------------------------------------------------
# Req 15.2: Predict quota exhaustion
# ---------------------------------------------------------------------------


class TestPredictExhaustion:
    """Req 15.2: Predict quota exhaustion based on current run rate."""

    def test_no_records_returns_none(self, qm: QuotaManager) -> None:
        assert qm.predict_exhaustion() is None

    def test_single_record_returns_none(self, qm: QuotaManager) -> None:
        """Need at least 2 records to compute rate."""
        qm.track_usage("cli", "agent-1", tokens=100, cost_usd=0.0003)
        assert qm.predict_exhaustion() is None

    def test_prediction_returns_future_datetime(self, config: QuotaConfig) -> None:
        """With usage, prediction should be in the future."""
        qm = QuotaManager(config=config)

        # Manually add records with known timestamps for predictable rate
        import time
        qm.track_usage("cli", "agent-1", tokens=100_000, cost_usd=0.3)
        time.sleep(0.05)  # Small delay to ensure time passes
        qm.track_usage("cli", "agent-1", tokens=100_000, cost_usd=0.3)

        prediction = qm.predict_exhaustion()
        assert prediction is not None
        assert prediction > datetime.now(UTC)

    def test_quota_already_exceeded_returns_now(self, qm: QuotaManager) -> None:
        """If already exceeded, prediction is approximately now."""
        import time
        qm.track_usage("cli", "agent-1", tokens=500_000, cost_usd=1.5)
        time.sleep(0.01)
        qm.track_usage("cli", "agent-1", tokens=600_000, cost_usd=1.8)

        prediction = qm.predict_exhaustion()
        assert prediction is not None
        # Should be very close to now since tokens > quota
        diff = abs((prediction - datetime.now(UTC)).total_seconds())
        assert diff < 5  # within 5 seconds

    def test_zero_quota_returns_none(self) -> None:
        config = QuotaConfig(total_token_quota=0)
        qm = QuotaManager(config=config)
        qm.track_usage("cli", "agent-1", tokens=100, cost_usd=0.0003)
        assert qm.predict_exhaustion() is None


# ---------------------------------------------------------------------------
# Req 15.3: Alert at 80% threshold
# ---------------------------------------------------------------------------


class TestAlertThreshold:
    """Req 15.3: Alert when consumption approaches 80% of quota."""

    def test_no_alert_below_threshold(self, config: QuotaConfig) -> None:
        alerts: list[QuotaAlert] = []
        qm = QuotaManager(config=config, alert_callback=alerts.append)

        qm.track_usage("cli", "agent-1", tokens=700_000, cost_usd=2.1)
        assert len(alerts) == 0
        assert not qm.has_alerted()

    def test_alert_at_threshold(self, config: QuotaConfig) -> None:
        alerts: list[QuotaAlert] = []
        qm = QuotaManager(config=config, alert_callback=alerts.append)

        # Cross 80% threshold
        qm.track_usage("cli", "agent-1", tokens=800_000, cost_usd=2.4)
        assert len(alerts) == 1
        assert alerts[0].threshold == 0.80
        assert alerts[0].current_usage_pct >= 0.80
        assert qm.has_alerted()

    def test_alert_fires_once(self, config: QuotaConfig) -> None:
        alerts: list[QuotaAlert] = []
        qm = QuotaManager(config=config, alert_callback=alerts.append)

        qm.track_usage("cli", "agent-1", tokens=800_000, cost_usd=2.4)
        qm.track_usage("cli", "agent-1", tokens=100_000, cost_usd=0.3)
        assert len(alerts) == 1  # Only one alert

    def test_alert_message_contains_usage(self, config: QuotaConfig) -> None:
        alerts: list[QuotaAlert] = []
        qm = QuotaManager(config=config, alert_callback=alerts.append)

        qm.track_usage("cli", "agent-1", tokens=850_000, cost_usd=2.55)
        assert "85%" in alerts[0].message or "850,000" in alerts[0].message

    def test_custom_threshold(self) -> None:
        config = QuotaConfig(total_token_quota=1_000_000, alert_threshold=0.50)
        alerts: list[QuotaAlert] = []
        qm = QuotaManager(config=config, alert_callback=alerts.append)

        qm.track_usage("cli", "agent-1", tokens=500_000, cost_usd=1.5)
        assert len(alerts) == 1
        assert alerts[0].threshold == 0.50

    def test_get_usage_percentage(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=250_000, cost_usd=0.75)
        assert qm.get_usage_percentage() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Req 15.4: Usage breakdowns by surface and agent
# ---------------------------------------------------------------------------


class TestUsageBreakdown:
    """Req 15.4: Provide usage breakdowns by surface and by agent."""

    def test_breakdown_by_surface(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=10_000, cost_usd=0.03)
        qm.track_usage("web", "agent-2", tokens=20_000, cost_usd=0.06)

        breakdown = qm.get_usage_breakdown()
        assert breakdown["by_surface"]["cli"]["tokens"] == 10_000
        assert breakdown["by_surface"]["web"]["tokens"] == 20_000

    def test_breakdown_by_agent(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "code-reviewer", tokens=15_000, cost_usd=0.045)
        qm.track_usage("cli", "implementer", tokens=25_000, cost_usd=0.075)

        breakdown = qm.get_usage_breakdown()
        assert breakdown["by_agent"]["code-reviewer"]["tokens"] == 15_000
        assert breakdown["by_agent"]["implementer"]["tokens"] == 25_000

    def test_breakdown_includes_usage_pct(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=500_000, cost_usd=1.5)

        breakdown = qm.get_usage_breakdown()
        assert breakdown["usage_pct"] == pytest.approx(0.5)

    def test_breakdown_empty_when_no_usage(self, qm: QuotaManager) -> None:
        breakdown = qm.get_usage_breakdown()
        assert breakdown["total_tokens"] == 0
        assert breakdown["total_cost_usd"] == 0.0
        assert breakdown["by_surface"] == {}
        assert breakdown["by_agent"] == {}

    def test_same_agent_different_surfaces(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=10_000, cost_usd=0.03)
        qm.track_usage("web", "agent-1", tokens=5_000, cost_usd=0.015)

        breakdown = qm.get_usage_breakdown()
        # Agent total should be combined
        assert breakdown["by_agent"]["agent-1"]["tokens"] == 15_000
        # Surface totals should be separate
        assert breakdown["by_surface"]["cli"]["tokens"] == 10_000
        assert breakdown["by_surface"]["web"]["tokens"] == 5_000


# ---------------------------------------------------------------------------
# Req 15.5: Weekly rate limits
# ---------------------------------------------------------------------------


class TestWeeklyRateLimits:
    """Req 15.5: Support weekly rate limits for 24/7 agent users."""

    def test_no_limit_by_default(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=999_999_999, cost_usd=3000.0)
        assert not qm.is_weekly_rate_limited()

    def test_weekly_limit_exceeded(self) -> None:
        config = QuotaConfig(
            total_token_quota=100_000_000,
            weekly_rate_limit_tokens=500_000,
        )
        qm = QuotaManager(config=config)

        qm.track_usage("cli", "agent-1", tokens=500_000, cost_usd=1.5)
        assert qm.is_weekly_rate_limited()

    def test_weekly_limit_not_exceeded(self) -> None:
        config = QuotaConfig(
            total_token_quota=100_000_000,
            weekly_rate_limit_tokens=500_000,
        )
        qm = QuotaManager(config=config)

        qm.track_usage("cli", "agent-1", tokens=400_000, cost_usd=1.2)
        assert not qm.is_weekly_rate_limited()

    def test_weekly_usage_stats(self) -> None:
        config = QuotaConfig(
            total_token_quota=100_000_000,
            weekly_rate_limit_tokens=1_000_000,
        )
        qm = QuotaManager(config=config)

        qm.track_usage("cli", "agent-1", tokens=300_000, cost_usd=0.9)
        stats = qm.get_weekly_usage()
        assert stats["weekly_tokens"] == 300_000
        assert stats["weekly_limit"] == 1_000_000
        assert stats["usage_pct"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Req 15.6: Consume SDKRateLimitInfo events
# ---------------------------------------------------------------------------


class TestSDKRateLimitInfo:
    """Req 15.6: Consume SDKRateLimitInfo events."""

    def test_consume_basic_event(self, qm: QuotaManager) -> None:
        event = {
            "utilization": 0.75,
            "remaining_tokens": 250_000,
            "limit_tokens": 1_000_000,
        }
        qm.consume_rate_limit_info(event)

        breakdown = qm.get_usage_breakdown()
        info = breakdown["rate_limit_info"]
        assert info is not None
        assert info["utilization"] == 0.75
        assert info["remaining_tokens"] == 250_000
        assert info["limit_tokens"] == 1_000_000

    def test_consume_event_with_reset_time(self, qm: QuotaManager) -> None:
        reset_time = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        event = {
            "utilization": 0.95,
            "reset_at": reset_time,
            "remaining_tokens": 50_000,
            "limit_tokens": 1_000_000,
        }
        qm.consume_rate_limit_info(event)

        breakdown = qm.get_usage_breakdown()
        info = breakdown["rate_limit_info"]
        assert info["reset_at"] is not None

    def test_consume_event_with_overage(self, qm: QuotaManager) -> None:
        event = {
            "utilization": 1.0,
            "overage_available": True,
            "overage_rate_per_million": 15.0,
        }
        qm.consume_rate_limit_info(event)

        breakdown = qm.get_usage_breakdown()
        info = breakdown["rate_limit_info"]
        assert info["overage_available"] is True

    def test_no_rate_limit_info_before_consumption(self, qm: QuotaManager) -> None:
        breakdown = qm.get_usage_breakdown()
        assert breakdown["rate_limit_info"] is None

    def test_invalid_reset_at_handled_gracefully(self, qm: QuotaManager) -> None:
        event = {
            "utilization": 0.5,
            "reset_at": "not-a-date",
        }
        # Should not raise
        qm.consume_rate_limit_info(event)
        breakdown = qm.get_usage_breakdown()
        info = breakdown["rate_limit_info"]
        assert info["reset_at"] is None


# ---------------------------------------------------------------------------
# Req 15.7: Overage purchase for Max subscribers
# ---------------------------------------------------------------------------


class TestOveragePurchase:
    """Req 15.7: Support overage purchase at API rates for Max subscribers."""

    def test_non_max_subscriber_cannot_purchase(self, qm: QuotaManager) -> None:
        assert not qm.can_purchase_overage()

    def test_max_subscriber_can_purchase_when_exceeded(self) -> None:
        config = QuotaConfig(
            total_token_quota=100_000,
            max_subscriber=True,
        )
        qm = QuotaManager(config=config)
        qm.track_usage("cli", "agent-1", tokens=100_000, cost_usd=0.3)
        assert qm.can_purchase_overage()

    def test_max_subscriber_with_overage_event(self) -> None:
        config = QuotaConfig(
            total_token_quota=1_000_000,
            max_subscriber=True,
        )
        qm = QuotaManager(config=config)
        qm.consume_rate_limit_info({
            "utilization": 1.0,
            "overage_available": True,
            "overage_rate_per_million": 15.0,
        })
        assert qm.can_purchase_overage()

    def test_overage_rate_from_config(self) -> None:
        config = QuotaConfig(
            total_token_quota=100_000,
            max_subscriber=True,
            overage_rate_per_million_usd=20.0,
        )
        qm = QuotaManager(config=config)
        assert qm.get_overage_rate() == 20.0

    def test_overage_rate_from_event_overrides(self) -> None:
        config = QuotaConfig(
            total_token_quota=100_000,
            max_subscriber=True,
            overage_rate_per_million_usd=20.0,
        )
        qm = QuotaManager(config=config)
        qm.consume_rate_limit_info({
            "utilization": 1.0,
            "overage_available": True,
            "overage_rate_per_million": 12.0,
        })
        assert qm.get_overage_rate() == 12.0

    def test_no_overage_rate_for_non_max(self, qm: QuotaManager) -> None:
        assert qm.get_overage_rate() is None


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Reset clears all tracking state."""

    def test_reset_clears_everything(self, qm: QuotaManager) -> None:
        qm.track_usage("cli", "agent-1", tokens=500_000, cost_usd=1.5)
        qm.consume_rate_limit_info({"utilization": 0.5})

        qm.reset()

        breakdown = qm.get_usage_breakdown()
        assert breakdown["total_tokens"] == 0
        assert breakdown["total_cost_usd"] == 0.0
        assert breakdown["by_surface"] == {}
        assert breakdown["by_agent"] == {}
        assert breakdown["rate_limit_info"] is None
        assert not qm.has_alerted()

    def test_alert_resets_after_reset(self, config: QuotaConfig) -> None:
        alerts: list[QuotaAlert] = []
        qm = QuotaManager(config=config, alert_callback=alerts.append)

        qm.track_usage("cli", "agent-1", tokens=800_000, cost_usd=2.4)
        assert qm.has_alerted()

        qm.reset()
        assert not qm.has_alerted()

        # Can alert again after reset
        qm.track_usage("cli", "agent-1", tokens=900_000, cost_usd=2.7)
        assert qm.has_alerted()
        assert len(alerts) == 2
