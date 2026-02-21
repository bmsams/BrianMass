"""Unified Quota Manager — cross-surface consumption tracking and prediction.

Tracks usage across all Brainmass surfaces (web, mobile, desktop, CLI),
predicts quota exhaustion, alerts at thresholds, and supports rate limits.

Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SURFACES = {"web", "mobile", "desktop", "cli"}

# Default alert threshold (Req 15.3: alert at 80%)
DEFAULT_ALERT_THRESHOLD = 0.80

# Default weekly rate limit (tokens per week, Req 15.5)
DEFAULT_WEEKLY_RATE_LIMIT: int | None = None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """A single usage record from any surface."""

    surface: str
    agent_id: str
    tokens: int
    cost_usd: float
    timestamp: datetime
    model_tier: str = "sonnet"


@dataclass
class RateLimitInfo:
    """Parsed SDKRateLimitInfo event data (Req 15.6)."""

    utilization: float  # 0.0-1.0
    reset_at: datetime | None = None
    remaining_tokens: int | None = None
    limit_tokens: int | None = None
    overage_available: bool = False
    overage_rate_per_million: float | None = None


@dataclass
class QuotaAlert:
    """An alert generated when consumption crosses a threshold."""

    threshold: float
    current_usage_pct: float
    predicted_exhaustion: datetime | None
    message: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class QuotaConfig:
    """Configuration for quota limits and alerting."""

    # Total token quota (across all surfaces)
    total_token_quota: int = 10_000_000

    # Total cost quota in USD
    total_cost_quota_usd: float = 100.0

    # Alert threshold (fraction, Req 15.3)
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD

    # Weekly rate limit in tokens (Req 15.5, None = unlimited)
    weekly_rate_limit_tokens: int | None = DEFAULT_WEEKLY_RATE_LIMIT

    # Max subscriber overage support (Req 15.7)
    max_subscriber: bool = False
    overage_rate_per_million_usd: float = 15.0  # API rate for overages


# ---------------------------------------------------------------------------
# QuotaManager
# ---------------------------------------------------------------------------


class QuotaManager:
    """Tracks consumption across all Brainmass surfaces with quota prediction.

    Thread-safe: all mutations are guarded by a lock so concurrent agents
    on different surfaces can safely call ``track_usage``.

    Usage::

        qm = QuotaManager(config=QuotaConfig(total_token_quota=5_000_000))
        qm.track_usage("cli", "code-reviewer", tokens=50_000, cost_usd=0.15)
        prediction = qm.predict_exhaustion()
        breakdown = qm.get_usage_breakdown()
    """

    def __init__(
        self,
        config: QuotaConfig | None = None,
        alert_callback: Callable[[QuotaAlert], None] | None = None,
    ) -> None:
        """
        Args:
            config: Quota limits and thresholds.
            alert_callback: Called when usage crosses alert threshold.
        """
        self._config = config or QuotaConfig()
        self._alert_callback = alert_callback
        self._lock = threading.Lock()

        # All usage records
        self._records: list[UsageRecord] = []

        # Aggregated counters (for fast lookups without scanning records)
        self._total_tokens: int = 0
        self._total_cost_usd: float = 0.0
        self._by_surface: dict[str, dict] = {}  # surface → {tokens, cost_usd}
        self._by_agent: dict[str, dict] = {}  # agent_id → {tokens, cost_usd}

        # Rate limit state from SDKRateLimitInfo (Req 15.6)
        self._rate_limit_info: RateLimitInfo | None = None

        # Alert tracking (avoid duplicate alerts for same threshold crossing)
        self._alerted: bool = False

        # Weekly rate limit tracking (Req 15.5)
        self._week_start: datetime = _start_of_week(datetime.now(UTC))
        self._weekly_tokens: int = 0

    # ------------------------------------------------------------------
    # Public API: Track usage (Req 15.1)
    # ------------------------------------------------------------------

    def track_usage(
        self,
        surface: str,
        agent_id: str,
        tokens: int,
        cost_usd: float,
        model_tier: str = "sonnet",
    ) -> None:
        """Record usage from any surface.

        Args:
            surface: One of 'web', 'mobile', 'desktop', 'cli'.
            agent_id: Identifier of the agent that consumed tokens.
            tokens: Number of tokens consumed.
            cost_usd: Cost in USD for this usage.
            model_tier: Model tier used (for breakdowns).

        Raises:
            ValueError: If surface is not a valid Brainmass surface.
        """
        if surface not in VALID_SURFACES:
            raise ValueError(
                f"Invalid surface '{surface}'. Must be one of: {sorted(VALID_SURFACES)}"
            )

        now = datetime.now(UTC)
        record = UsageRecord(
            surface=surface,
            agent_id=agent_id,
            tokens=tokens,
            cost_usd=cost_usd,
            timestamp=now,
            model_tier=model_tier,
        )

        with self._lock:
            self._records.append(record)

            # Update aggregated counters
            self._total_tokens += tokens
            self._total_cost_usd += cost_usd

            # By surface
            surf = self._by_surface.setdefault(surface, {"tokens": 0, "cost_usd": 0.0})
            surf["tokens"] += tokens
            surf["cost_usd"] += cost_usd

            # By agent
            agt = self._by_agent.setdefault(agent_id, {"tokens": 0, "cost_usd": 0.0})
            agt["tokens"] += tokens
            agt["cost_usd"] += cost_usd

            # Weekly rate limit tracking (Req 15.5)
            current_week = _start_of_week(now)
            if current_week > self._week_start:
                self._week_start = current_week
                self._weekly_tokens = 0
            self._weekly_tokens += tokens

            # Check alert threshold (Req 15.3)
            self._check_alert()

    # ------------------------------------------------------------------
    # Public API: Predict exhaustion (Req 15.2)
    # ------------------------------------------------------------------

    def predict_exhaustion(self) -> datetime | None:
        """Predict when the quota will be exhausted based on current run rate.

        Returns:
            Predicted exhaustion datetime, or None if rate is zero or
            quota is unlimited.
        """
        with self._lock:
            if self._config.total_token_quota <= 0:
                return None

            remaining = self._config.total_token_quota - self._total_tokens
            if remaining <= 0:
                return datetime.now(UTC)

            rate = self._compute_run_rate()
            if rate <= 0:
                return None

            seconds_remaining = remaining / rate
            return datetime.now(UTC) + timedelta(seconds=seconds_remaining)

    # ------------------------------------------------------------------
    # Public API: Usage breakdown (Req 15.4)
    # ------------------------------------------------------------------

    def get_usage_breakdown(self) -> dict:
        """Return usage breakdowns by surface and by agent.

        Returns:
            Dict with keys:
            - ``total_tokens``: int
            - ``total_cost_usd``: float
            - ``usage_pct``: float (0.0-1.0)
            - ``by_surface``: {surface: {tokens, cost_usd}}
            - ``by_agent``: {agent_id: {tokens, cost_usd}}
            - ``weekly_tokens``: int
            - ``weekly_rate_limit``: int | None
            - ``rate_limit_info``: dict | None
        """
        with self._lock:
            quota = self._config.total_token_quota
            usage_pct = self._total_tokens / quota if quota > 0 else 0.0

            return {
                "total_tokens": self._total_tokens,
                "total_cost_usd": round(self._total_cost_usd, 6),
                "usage_pct": round(usage_pct, 4),
                "by_surface": {
                    s: dict(d) for s, d in self._by_surface.items()
                },
                "by_agent": {
                    a: dict(d) for a, d in self._by_agent.items()
                },
                "weekly_tokens": self._weekly_tokens,
                "weekly_rate_limit": self._config.weekly_rate_limit_tokens,
                "rate_limit_info": self._format_rate_limit_info(),
            }

    # ------------------------------------------------------------------
    # Public API: Weekly rate limit check (Req 15.5)
    # ------------------------------------------------------------------

    def is_weekly_rate_limited(self) -> bool:
        """Check if the weekly rate limit has been exceeded.

        Returns True if a weekly rate limit is configured and the current
        week's token usage exceeds it.
        """
        with self._lock:
            limit = self._config.weekly_rate_limit_tokens
            if limit is None:
                return False
            return self._weekly_tokens >= limit

    def get_weekly_usage(self) -> dict:
        """Return weekly usage stats.

        Returns:
            Dict with ``weekly_tokens``, ``weekly_limit``, ``week_start``,
            ``usage_pct``.
        """
        with self._lock:
            limit = self._config.weekly_rate_limit_tokens
            pct = (
                self._weekly_tokens / limit
                if limit is not None and limit > 0
                else 0.0
            )
            return {
                "weekly_tokens": self._weekly_tokens,
                "weekly_limit": limit,
                "week_start": self._week_start.isoformat(),
                "usage_pct": round(pct, 4),
            }

    # ------------------------------------------------------------------
    # Public API: Consume SDKRateLimitInfo (Req 15.6)
    # ------------------------------------------------------------------

    def consume_rate_limit_info(self, event: dict) -> None:
        """Process an SDKRateLimitInfo event.

        Expected event keys:
        - ``utilization``: float (0.0-1.0)
        - ``reset_at``: ISO-8601 datetime string (optional)
        - ``remaining_tokens``: int (optional)
        - ``limit_tokens``: int (optional)
        - ``overage_available``: bool (optional)
        - ``overage_rate_per_million``: float (optional)

        Args:
            event: Raw SDKRateLimitInfo event dict.
        """
        with self._lock:
            reset_at = None
            if "reset_at" in event and event["reset_at"]:
                try:
                    reset_at = datetime.fromisoformat(event["reset_at"])
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid reset_at in SDKRateLimitInfo: %s", event["reset_at"]
                    )

            self._rate_limit_info = RateLimitInfo(
                utilization=float(event.get("utilization", 0.0)),
                reset_at=reset_at,
                remaining_tokens=event.get("remaining_tokens"),
                limit_tokens=event.get("limit_tokens"),
                overage_available=event.get("overage_available", False),
                overage_rate_per_million=event.get("overage_rate_per_million"),
            )

            logger.debug(
                "SDKRateLimitInfo consumed: utilization=%.2f, remaining=%s",
                self._rate_limit_info.utilization,
                self._rate_limit_info.remaining_tokens,
            )

    # ------------------------------------------------------------------
    # Public API: Overage purchase (Req 15.7)
    # ------------------------------------------------------------------

    def can_purchase_overage(self) -> bool:
        """Check if the current subscriber can purchase overages.

        Overage purchase is available for Max subscribers when the quota is
        exceeded and the rate limit info indicates overage is available.
        """
        with self._lock:
            if not self._config.max_subscriber:
                return False
            if self._rate_limit_info and self._rate_limit_info.overage_available:
                return True
            # Also available when quota is exceeded for Max subscribers
            return self._total_tokens >= self._config.total_token_quota

    def get_overage_rate(self) -> float | None:
        """Return the overage rate per million tokens, or None if unavailable."""
        with self._lock:
            if self._rate_limit_info and self._rate_limit_info.overage_rate_per_million:
                return self._rate_limit_info.overage_rate_per_million
            if self._config.max_subscriber:
                return self._config.overage_rate_per_million_usd
            return None

    # ------------------------------------------------------------------
    # Public API: Alert threshold status (Req 15.3)
    # ------------------------------------------------------------------

    def get_usage_percentage(self) -> float:
        """Return current usage as a fraction of the total quota."""
        with self._lock:
            quota = self._config.total_token_quota
            if quota <= 0:
                return 0.0
            return self._total_tokens / quota

    def has_alerted(self) -> bool:
        """Return True if the alert threshold has been crossed."""
        with self._lock:
            return self._alerted

    # ------------------------------------------------------------------
    # Public API: Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all usage tracking. Useful for testing or new billing cycles."""
        with self._lock:
            self._records.clear()
            self._total_tokens = 0
            self._total_cost_usd = 0.0
            self._by_surface.clear()
            self._by_agent.clear()
            self._weekly_tokens = 0
            self._week_start = _start_of_week(datetime.now(UTC))
            self._alerted = False
            self._rate_limit_info = None

    # ------------------------------------------------------------------
    # Internal: run rate computation
    # ------------------------------------------------------------------

    def _compute_run_rate(self) -> float:
        """Compute tokens per second based on recorded usage.

        Uses the time span between the first and last record.
        Returns 0 if fewer than 2 records exist.

        Must be called under lock.
        """
        if len(self._records) < 2:
            return 0.0

        first_ts = self._records[0].timestamp
        last_ts = self._records[-1].timestamp
        span = (last_ts - first_ts).total_seconds()
        if span <= 0:
            return 0.0

        return self._total_tokens / span

    # ------------------------------------------------------------------
    # Internal: alert check
    # ------------------------------------------------------------------

    def _check_alert(self) -> None:
        """Check if usage crosses the alert threshold and fire callback.

        Must be called under lock.
        """
        if self._alerted:
            return

        quota = self._config.total_token_quota
        if quota <= 0:
            return

        usage_pct = self._total_tokens / quota
        if usage_pct >= self._config.alert_threshold:
            self._alerted = True
            alert = QuotaAlert(
                threshold=self._config.alert_threshold,
                current_usage_pct=usage_pct,
                predicted_exhaustion=None,  # Computed without lock below
                message=(
                    f"Usage at {usage_pct:.0%} of quota "
                    f"({self._total_tokens:,} / {quota:,} tokens). "
                    f"Threshold: {self._config.alert_threshold:.0%}."
                ),
            )
            logger.warning("Quota alert: %s", alert.message)
            if self._alert_callback is not None:
                try:
                    self._alert_callback(alert)
                except Exception as exc:
                    logger.error("Alert callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal: format rate limit info for breakdown
    # ------------------------------------------------------------------

    def _format_rate_limit_info(self) -> dict | None:
        """Format current rate limit info as a dict.

        Must be called under lock.
        """
        if self._rate_limit_info is None:
            return None
        return {
            "utilization": self._rate_limit_info.utilization,
            "reset_at": (
                self._rate_limit_info.reset_at.isoformat()
                if self._rate_limit_info.reset_at
                else None
            ),
            "remaining_tokens": self._rate_limit_info.remaining_tokens,
            "limit_tokens": self._rate_limit_info.limit_tokens,
            "overage_available": self._rate_limit_info.overage_available,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_of_week(dt: datetime) -> datetime:
    """Return midnight UTC of the Monday starting the week containing *dt*."""
    # Monday = 0 in isoweekday()-1
    days_since_monday = dt.weekday()  # Monday=0, Sunday=6
    monday = dt - timedelta(days=days_since_monday)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)
