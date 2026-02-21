"""Cost Governor — model selection, cost calculation, budget tracking, and dashboard.

Wraps Strands BedrockModel instances for each tier and enforces per-agent
token budgets with cost-aware model routing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from src.types.core import (
    PRICING,
    AgentBudget,
    BudgetStatus,
    ModelPricing,
    ModelTier,
)

# ---------------------------------------------------------------------------
# TaskSignals — input to model selection
# ---------------------------------------------------------------------------

@dataclass
class TaskSignals:
    """Signals used by the Cost Governor to select the cheapest capable model."""
    is_exploration: bool = False
    files_affected: int = 1
    requires_reasoning: bool = False
    is_team_lead: bool = False
    dependency_depth: int = 0


# ---------------------------------------------------------------------------
# Per-agent usage record
# ---------------------------------------------------------------------------

@dataclass
class _AgentUsage:
    """Internal mutable record tracking an agent's cumulative usage."""
    budget: AgentBudget
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0
    tier_counts: dict[str, int] = field(default_factory=lambda: {
        ModelTier.OPUS.value: 0,
        ModelTier.SONNET.value: 0,
        ModelTier.HAIKU.value: 0,
    })


# ---------------------------------------------------------------------------
# Tier ordering helpers
# ---------------------------------------------------------------------------

_TIER_ORDER: list[ModelTier] = [ModelTier.OPUS, ModelTier.SONNET, ModelTier.HAIKU]


def _downgrade_one_tier(current: ModelTier) -> ModelTier:
    """Return the next cheaper tier, or Haiku if already at the bottom."""
    idx = _TIER_ORDER.index(current)
    return _TIER_ORDER[min(idx + 1, len(_TIER_ORDER) - 1)]


# ---------------------------------------------------------------------------
# CostGovernor
# ---------------------------------------------------------------------------

class CostGovernor:
    """Enforces budgets, routes to cheapest capable model, tracks costs.

    Thread-safe: all mutations are guarded by a lock so that concurrent
    agents can safely call ``record_usage`` / ``check_budget``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # agent_id → _AgentUsage
        self._agents: dict[str, _AgentUsage] = {}
        # Lazily-initialised BedrockModel instances per tier (Requirements: 12.1, 12.2, 12.3)
        self._bedrock_models: dict[ModelTier, object] = {}
        self._models_initialized: bool = False

    # ------------------------------------------------------------------
    # BedrockModel lazy initialisation (Requirements: 12.1, 12.2, 12.3)
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        """Lazily initialise one BedrockModel per ModelTier.

        Silently skips if ``strands-agents`` is not installed so that the
        CostGovernor remains usable in test environments without the SDK.
        """
        if self._models_initialized:
            return
        try:
            # --- Production integration point ---
            from strands.models.bedrock import BedrockModel  # type: ignore[import-untyped]

            _ids: dict[ModelTier, str] = {
                ModelTier.OPUS: "us.anthropic.claude-opus-4-6-v1:0",
                ModelTier.SONNET: "us.anthropic.claude-sonnet-4-5-v1:0",
                ModelTier.HAIKU: "us.anthropic.claude-haiku-4-5-v1:0",
            }
            for tier, model_id in _ids.items():
                self._bedrock_models[tier] = BedrockModel(model_id=model_id)
        except ImportError:
            pass  # SDK absent — models stay empty; callers receive None
        finally:
            self._models_initialized = True

    def get_bedrock_model(self, tier: ModelTier) -> object | None:
        """Return the ``BedrockModel`` for *tier*, or ``None`` when SDK is absent.

        Requirements: 12.1, 12.2, 12.3
        """
        self._ensure_models()
        return self._bedrock_models.get(tier)

    # ------------------------------------------------------------------
    # Agent registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, budget: AgentBudget) -> None:
        """Register an agent with its budget.  Must be called before any
        other method that takes *agent_id*."""
        with self._lock:
            self._agents[agent_id] = _AgentUsage(budget=budget)

    def has_agent(self, agent_id: str) -> bool:
        """Return True if an agent is registered."""
        with self._lock:
            return agent_id in self._agents

    def get_budget(self, agent_id: str) -> AgentBudget | None:
        """Return an agent budget if registered, else None."""
        with self._lock:
            usage = self._agents.get(agent_id)
            return usage.budget if usage is not None else None

    def ensure_agent(self, agent_id: str, budget: AgentBudget) -> AgentBudget:
        """Ensure an agent registration exists and return its budget."""
        with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = _AgentUsage(budget=budget)
            return self._agents[agent_id].budget

    # ------------------------------------------------------------------
    # Model selection  (Requirement 4.4, 4.5)
    # ------------------------------------------------------------------

    def select_model(
        self,
        signals: TaskSignals,
        budget: AgentBudget,
    ) -> ModelTier:
        """Pick the cheapest capable model tier given task signals and budget.

        Selection rules (evaluated top-to-bottom, first match wins):
        1. Budget CRITICAL  → Haiku
        2. Budget WARNING   → downgrade one tier from the "natural" pick
        3. is_exploration OR (files_affected <= 1 AND NOT requires_reasoning) → Haiku
        4. is_team_lead OR requires_reasoning OR dependency_depth > 3 → Opus
        5. Default → Sonnet
        """
        # Determine the "natural" tier ignoring budget pressure
        natural = self._natural_tier(signals)

        status = budget.status
        if status == BudgetStatus.CRITICAL or status == BudgetStatus.EXCEEDED:
            return ModelTier.HAIKU
        if status == BudgetStatus.WARNING:
            return _downgrade_one_tier(natural)
        return natural

    @staticmethod
    def _natural_tier(signals: TaskSignals) -> ModelTier:
        """Determine the model tier purely from task signals."""
        if signals.is_exploration:
            return ModelTier.HAIKU
        if signals.files_affected <= 1 and not signals.requires_reasoning:
            return ModelTier.HAIKU
        if signals.is_team_lead or signals.requires_reasoning or signals.dependency_depth > 3:
            return ModelTier.OPUS
        return ModelTier.SONNET

    # ------------------------------------------------------------------
    # Cost calculation  (Requirement 4.6, 1.5)
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_cost(
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        model_tier: ModelTier = ModelTier.SONNET,
        is_batch: bool = False,
    ) -> float:
        """Return the USD cost for a single API call.

        Formula:
            (regularInput / 1M × inputRate)
          + (output / 1M × outputRate)
          + (cacheRead / 1M × cachedRate)
          + (cacheWrite / 1M × cacheWriteRate)

        Batch API applies a 50 % discount on all rates.
        """
        pricing: ModelPricing = PRICING[model_tier]
        discount = 0.5 if is_batch else 1.0

        # Regular input = total input minus cached tokens
        regular_input = max(input_tokens - cache_read_tokens - cache_write_tokens, 0)

        cost = (
            (regular_input / 1_000_000) * pricing.input_per_million
            + (output_tokens / 1_000_000) * pricing.output_per_million
            + (cache_read_tokens / 1_000_000) * pricing.cached_input_per_million
            + (cache_write_tokens / 1_000_000) * pricing.cache_write_per_million
        )
        return cost * discount

    # ------------------------------------------------------------------
    # Usage recording  (Requirement 4.8)
    # ------------------------------------------------------------------

    def record_usage(
        self,
        agent_id: str,
        input_tokens: int,
        output_tokens: int,
        model_tier: ModelTier,
        cached_tokens: int = 0,
    ) -> None:
        """Record token usage for *agent_id* and update its running cost.

        Auto-registers the agent with a default budget if it has not been
        explicitly registered — this handles dispatched agents whose IDs are
        generated at runtime.
        """
        cost = self.calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cached_tokens,
            model_tier=model_tier,
        )
        with self._lock:
            if agent_id not in self._agents:
                self._agents[agent_id] = _AgentUsage(
                    budget=AgentBudget(
                        input_budget_tokens=200_000,
                        output_budget_tokens=50_000,
                        session_budget_usd=1.0,
                    )
                )
            usage = self._agents[agent_id]
            usage.total_input_tokens += input_tokens
            usage.total_output_tokens += output_tokens
            usage.total_cached_tokens += cached_tokens
            usage.total_cost_usd += cost
            usage.tier_counts[model_tier.value] = (
                usage.tier_counts.get(model_tier.value, 0) + 1
            )
            # Mirror back into the budget so that ``budget.status`` stays current
            usage.budget.current_input_tokens += input_tokens
            usage.budget.current_output_tokens += output_tokens
            usage.budget.current_cost_usd += cost

    # ------------------------------------------------------------------
    # Budget checking  (Requirement 4.1, 4.2, 4.3)
    # ------------------------------------------------------------------

    def check_budget(self, agent_id: str) -> BudgetStatus:
        """Return the current budget status for *agent_id*.

        Returns BudgetStatus.OK for unknown agent IDs — the agent has not yet
        consumed any budget so it is safe to proceed.
        """
        with self._lock:
            usage = self._agents.get(agent_id)
            if usage is None:
                return BudgetStatus.OK
            return usage.budget.status

    # ------------------------------------------------------------------
    # Dashboard data  (Requirement 4.7)
    # ------------------------------------------------------------------

    def get_dashboard_data(self) -> dict:
        """Return a real-time snapshot for the cost dashboard.

        Keys:
        - ``token_consumption``: total input / output / cached across all agents
        - ``cost_per_agent``: {agent_id: usd}
        - ``model_tier_distribution``: {tier: call_count}
        - ``cache_hit_rates``: {agent_id: rate}  (0.0 when no input tokens)
        """
        with self._lock:
            total_input = 0
            total_output = 0
            total_cached = 0
            cost_per_agent: dict[str, float] = {}
            tier_dist: dict[str, int] = {t.value: 0 for t in ModelTier}
            cache_hit_rates: dict[str, float] = {}

            for aid, u in self._agents.items():
                total_input += u.total_input_tokens
                total_output += u.total_output_tokens
                total_cached += u.total_cached_tokens
                cost_per_agent[aid] = u.total_cost_usd
                for tier_val, cnt in u.tier_counts.items():
                    tier_dist[tier_val] = tier_dist.get(tier_val, 0) + cnt
                if u.total_input_tokens > 0:
                    cache_hit_rates[aid] = u.total_cached_tokens / u.total_input_tokens
                else:
                    cache_hit_rates[aid] = 0.0

            return {
                "token_consumption": {
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "total_cached_tokens": total_cached,
                },
                "cost_per_agent": cost_per_agent,
                "model_tier_distribution": tier_dist,
                "cache_hit_rates": cache_hit_rates,
            }
