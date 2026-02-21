"""Unit tests for the Effort Controller.

Tests effort level selection based on task signals, model capability,
budget constraints, Fast Mode, and explicit overrides.
"""


from src.cost.cost_governor import TaskSignals
from src.orchestrator.effort_controller import (
    EFFORT_BUDGET_TOKENS,
    EffortController,
    EffortResult,
)
from src.types.core import AgentBudget, BudgetStatus, EffortLevel, ModelTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_budget() -> AgentBudget:
    """Budget at 0% utilisation (OK status)."""
    return AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=10.0,
    )


def _warning_budget() -> AgentBudget:
    """Budget at ~85% utilisation (WARNING status)."""
    b = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=10.0,
        current_cost_usd=8.5,
    )
    assert b.status == BudgetStatus.WARNING
    return b


def _critical_budget() -> AgentBudget:
    """Budget at ~96% utilisation (CRITICAL status)."""
    b = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=10.0,
        current_cost_usd=9.6,
    )
    assert b.status == BudgetStatus.CRITICAL
    return b


def _exceeded_budget() -> AgentBudget:
    """Budget at 100%+ utilisation (EXCEEDED status)."""
    b = AgentBudget(
        input_budget_tokens=1_000_000,
        output_budget_tokens=200_000,
        session_budget_usd=10.0,
        current_cost_usd=10.5,
    )
    assert b.status == BudgetStatus.EXCEEDED
    return b


# ---------------------------------------------------------------------------
# Budget-token mapping
# ---------------------------------------------------------------------------

class TestBudgetTokenMapping:
    """Verify the effort → budget_tokens constants."""

    def test_quick_maps_to_2000(self):
        assert EFFORT_BUDGET_TOKENS[EffortLevel.QUICK] == 2_000

    def test_standard_maps_to_10000(self):
        assert EFFORT_BUDGET_TOKENS[EffortLevel.STANDARD] == 10_000

    def test_deep_maps_to_50000(self):
        assert EFFORT_BUDGET_TOKENS[EffortLevel.DEEP] == 50_000


# ---------------------------------------------------------------------------
# Natural effort selection from signals
# ---------------------------------------------------------------------------

class TestNaturalEffort:
    """Test effort selection purely from task signals (no budget/model caps)."""

    def setup_method(self):
        self.ctrl = EffortController()

    def test_exploration_returns_quick(self):
        signals = TaskSignals(is_exploration=True)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.level == "quick"
        assert result.budget_tokens == 2_000

    def test_single_file_no_reasoning_returns_quick(self):
        signals = TaskSignals(files_affected=1, requires_reasoning=False)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.level == "quick"

    def test_default_task_returns_standard(self):
        signals = TaskSignals(files_affected=3, requires_reasoning=False)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.level == "standard"
        assert result.budget_tokens == 10_000

    def test_reasoning_with_high_deps_returns_deep(self):
        signals = TaskSignals(requires_reasoning=True, dependency_depth=4)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.level == "deep"
        assert result.budget_tokens == 50_000

    def test_many_files_with_reasoning_returns_deep(self):
        signals = TaskSignals(files_affected=6, requires_reasoning=True)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.level == "deep"

    def test_reasoning_low_deps_returns_standard(self):
        """Reasoning alone with low dependency depth → Standard, not Deep."""
        signals = TaskSignals(requires_reasoning=True, dependency_depth=1, files_affected=2)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.level == "standard"


# ---------------------------------------------------------------------------
# Model capability capping
# ---------------------------------------------------------------------------

class TestModelCapability:
    """Test that effort is capped to the model's maximum."""

    def setup_method(self):
        self.ctrl = EffortController()

    def test_haiku_always_quick(self):
        """Haiku can only do Quick, even for complex tasks."""
        signals = TaskSignals(requires_reasoning=True, dependency_depth=5)
        result = self.ctrl.select_effort(signals, ModelTier.HAIKU, _ok_budget())
        assert result.level == "quick"
        assert result.budget_tokens == 2_000

    def test_sonnet_caps_at_standard(self):
        """Sonnet caps at Standard, even for Deep-worthy tasks."""
        signals = TaskSignals(requires_reasoning=True, dependency_depth=5)
        result = self.ctrl.select_effort(signals, ModelTier.SONNET, _ok_budget())
        assert result.level == "standard"
        assert result.budget_tokens == 10_000

    def test_sonnet_allows_quick(self):
        signals = TaskSignals(is_exploration=True)
        result = self.ctrl.select_effort(signals, ModelTier.SONNET, _ok_budget())
        assert result.level == "quick"

    def test_opus_allows_deep(self):
        signals = TaskSignals(requires_reasoning=True, dependency_depth=5)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.level == "deep"


# ---------------------------------------------------------------------------
# Budget overrides
# ---------------------------------------------------------------------------

class TestBudgetOverrides:
    """Test budget-based effort downgrades."""

    def setup_method(self):
        self.ctrl = EffortController()

    def test_warning_downgrades_deep_to_standard(self):
        signals = TaskSignals(requires_reasoning=True, dependency_depth=5)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _warning_budget())
        assert result.level == "standard"

    def test_warning_downgrades_standard_to_quick(self):
        signals = TaskSignals(files_affected=3)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _warning_budget())
        assert result.level == "quick"

    def test_warning_keeps_quick_as_quick(self):
        """Quick can't be downgraded further."""
        signals = TaskSignals(is_exploration=True)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _warning_budget())
        assert result.level == "quick"

    def test_critical_forces_quick(self):
        signals = TaskSignals(requires_reasoning=True, dependency_depth=5)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _critical_budget())
        assert result.level == "quick"
        assert result.budget_tokens == 2_000

    def test_exceeded_forces_quick(self):
        signals = TaskSignals(requires_reasoning=True, dependency_depth=5)
        result = self.ctrl.select_effort(signals, ModelTier.OPUS, _exceeded_budget())
        assert result.level == "quick"


# ---------------------------------------------------------------------------
# Explicit override
# ---------------------------------------------------------------------------

class TestExplicitOverride:
    """Test developer-provided effort override."""

    def setup_method(self):
        self.ctrl = EffortController()

    def test_override_to_deep(self):
        signals = TaskSignals(is_exploration=True)  # would naturally be Quick
        result = self.ctrl.select_effort(
            signals, ModelTier.OPUS, _ok_budget(), override=EffortLevel.DEEP,
        )
        assert result.level == "deep"

    def test_override_capped_by_model(self):
        """Override to Deep on Haiku still caps to Quick."""
        signals = TaskSignals()
        result = self.ctrl.select_effort(
            signals, ModelTier.HAIKU, _ok_budget(), override=EffortLevel.DEEP,
        )
        assert result.level == "quick"

    def test_override_capped_by_budget(self):
        """Override to Deep at CRITICAL budget still forces Quick."""
        signals = TaskSignals()
        result = self.ctrl.select_effort(
            signals, ModelTier.OPUS, _critical_budget(), override=EffortLevel.DEEP,
        )
        assert result.level == "quick"


# ---------------------------------------------------------------------------
# Fast Mode (Requirement 12.4)
# ---------------------------------------------------------------------------

class TestFastMode:
    """Test Opus Fast Mode detection and OTel attributes."""

    def test_opus_quick_triggers_fast_mode(self):
        ctrl = EffortController(fast_mode_enabled=True)
        signals = TaskSignals(is_exploration=True)
        result = ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.fast_mode is True
        assert result.otel_attributes is not None
        assert result.otel_attributes["effort.fast_mode"] is True
        assert result.otel_attributes["effort.speed"] == "fast"

    def test_opus_standard_no_fast_mode(self):
        ctrl = EffortController(fast_mode_enabled=True)
        signals = TaskSignals(files_affected=3)
        result = ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.fast_mode is False
        assert result.otel_attributes is None

    def test_sonnet_quick_no_fast_mode(self):
        """Fast Mode is Opus-only."""
        ctrl = EffortController(fast_mode_enabled=True)
        signals = TaskSignals(is_exploration=True)
        result = ctrl.select_effort(signals, ModelTier.SONNET, _ok_budget())
        assert result.fast_mode is False

    def test_fast_mode_disabled(self):
        ctrl = EffortController(fast_mode_enabled=False)
        signals = TaskSignals(is_exploration=True)
        result = ctrl.select_effort(signals, ModelTier.OPUS, _ok_budget())
        assert result.fast_mode is False
        assert result.otel_attributes is None


# ---------------------------------------------------------------------------
# EffortResult dataclass
# ---------------------------------------------------------------------------

class TestEffortResult:
    """Test EffortResult construction and defaults."""

    def test_defaults(self):
        r = EffortResult(level="standard", budget_tokens=10_000)
        assert r.fast_mode is False
        assert r.otel_attributes is None

    def test_with_fast_mode(self):
        r = EffortResult(
            level="quick",
            budget_tokens=2_000,
            fast_mode=True,
            otel_attributes={"effort.speed": "fast"},
        )
        assert r.fast_mode is True
        assert r.otel_attributes["effort.speed"] == "fast"
