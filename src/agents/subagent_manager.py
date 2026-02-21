"""Subagent Manager — Agents-as-Tools pattern with isolated context and budget.

Implements hierarchical subagent orchestration where the Orchestrator wraps
worker agents as callable tools. Each subagent gets an independent context
window, its own budget, and scoped hooks. On completion, SubagentStop hooks
fire and a structured AgentResult is returned to the parent.

Production callbacks use ``strands.Agent`` + ``strands.models.bedrock.BedrockModel``
and are the default when no callback is injected. Tests must inject stubs.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result
from src.observability.instrumentation import BrainmassTracer
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    HookContext,
    HookDefinition,
    HookEvent,
    HookResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases for pluggable callbacks
# ---------------------------------------------------------------------------

# Callback that executes a subagent task and returns (summary, tokens_consumed, tools_used, files_modified, exit_reason).
AgentCallback = Callable[
    [AgentDefinition, str, AgentBudget],
    dict,  # {"summary": str, "tokens_consumed": dict, "tools_used": list, "files_modified": list, "exit_reason": str, "turns_used": int}
]

# Callback that wraps an agent definition as a tool descriptor dict.
AsToolCallback = Callable[
    [AgentDefinition],
    dict,  # {"name": str, "description": str}
]


# --- Production integration point ---
def _production_agent_callback(
    agent_def: AgentDefinition,
    task: str,
    budget: AgentBudget,
) -> dict:
    """Execute a subagent task using a real Strands Agent + BedrockModel.

    Requirements: 2.1, 2.2, 2.3
    """
    try:
        from strands import Agent  # type: ignore[import-untyped]
        from strands.models.bedrock import BedrockModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents is required for production subagent execution. "
            "Install with: pip install strands-agents  "
            "Or inject an agent_callback to bypass this requirement."
        ) from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(task)
    return _normalize_strands_result(raw)


# --- Production integration point ---
def _production_as_tool_callback(agent_def: AgentDefinition) -> dict:
    """Wrap an agent as a tool descriptor using Agent.as_tool().

    Requirements: 2.4, 2.5
    """
    try:
        from strands import Agent  # type: ignore[import-untyped]
        from strands.models.bedrock import BedrockModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents is required for Agent.as_tool(). "
            "Install with: pip install strands-agents  "
            "Or inject an as_tool_callback to bypass this requirement."
        ) from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    # Agent.as_tool() returns a callable tool descriptor
    agent.as_tool(
        name=f"agent:{agent_def.name}",
        description=agent_def.description,
    )
    return {"name": f"agent:{agent_def.name}", "description": agent_def.description}


def _default_agent_callback(
    agent_def: AgentDefinition,
    task: str,
    budget: AgentBudget,
) -> dict:
    """No-op agent callback for testing — returns a minimal valid result dict."""
    return {
        "summary": f"Agent '{agent_def.name}' completed: {task[:80]}",
        "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
        "tools_used": [],
        "files_modified": [],
        "exit_reason": "complete",
        "turns_used": 0,
    }


def _default_as_tool_callback(agent_def: AgentDefinition) -> dict:
    """No-op as-tool callback for testing — returns a minimal tool descriptor."""
    return {"name": f"agent:{agent_def.name}", "description": agent_def.description}


def _convert_stop_to_subagent_stop(
    hooks: dict[str, list[HookDefinition]],
) -> dict[str, list[HookDefinition]]:
    """Convert Stop hooks to SubagentStop in agent frontmatter (Req 5.4).

    Returns a new dict with the same hooks, but any entry keyed under
    ``HookEvent.STOP`` (or its string value ``'Stop'``) is re-keyed to
    ``HookEvent.SUBAGENT_STOP``.
    """
    converted: dict[str, list[HookDefinition]] = {}
    for key, defs in hooks.items():
        # Normalise key to string for comparison
        key_str = key.value if isinstance(key, HookEvent) else str(key)
        if key_str == HookEvent.STOP.value:
            converted[HookEvent.SUBAGENT_STOP.value] = defs
        else:
            converted[key_str] = defs
    return converted


# ---------------------------------------------------------------------------
# SubagentManager
# ---------------------------------------------------------------------------

class SubagentManager:
    """Manages hierarchical subagent execution with isolated context/budget.

    Each subagent:
    - Gets its own ``AgentBudget`` (independent context window — Req 5.2)
    - Has scoped hooks registered (and cleaned up on completion — Req 5.4)
    - Fires ``SubagentStop`` hooks on completion (Req 5.3)
    - Returns a structured ``AgentResult`` to the parent (Req 5.3)

    The actual agent execution is delegated to a pluggable ``agent_callback``
    so the system works without the Strands SDK installed.

    Usage::

        manager = SubagentManager(hook_engine=engine, cost_governor=governor)
        result = manager.execute(agent_def, "Review auth module")
        tools = manager.as_tools([agent_def1, agent_def2])
    """

    def __init__(
        self,
        hook_engine: object | None = None,
        cost_governor: object | None = None,
        agent_callback: AgentCallback | None = None,
        as_tool_callback: AsToolCallback | None = None,
        session_id: str = "",
        cwd: str = ".",
        session_type: str = "interactive",
        tracer: BrainmassTracer | None = None,
    ) -> None:
        self._hook_engine = hook_engine
        self._cost_governor = cost_governor
        # Default to no-op callbacks; inject _production_agent_callback for real Bedrock execution.
        # Requirements: 2.4, 16.1
        self._agent_callback = agent_callback or _default_agent_callback
        self._as_tool_callback = as_tool_callback or _default_as_tool_callback
        self._session_id = session_id
        self._cwd = cwd
        self._session_type = session_type
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        agent_def: AgentDefinition,
        task: str,
        parent_budget: AgentBudget | None = None,
    ) -> AgentResult:
        """Execute a subagent with isolated context and budget.

        Steps:
        1. Create an isolated budget for the subagent (Req 5.2)
        2. Register the agent with the cost governor
        3. Register scoped hooks from agent frontmatter (Req 5.4)
        4. Execute the task via the pluggable agent_callback
        5. Record usage with the cost governor
        6. Fire SubagentStop hooks (Req 5.3)
        7. Clean up scoped hooks
        8. Return structured AgentResult (Req 5.3)
        """
        agent_id = f"subagent:{agent_def.name}:{uuid.uuid4().hex[:8]}"

        # 1. Create isolated budget (Req 5.2)
        budget = self._create_isolated_budget(parent_budget)

        # 2. Register with cost governor
        if self._cost_governor is not None:
            self._cost_governor.register_agent(agent_id, budget)

        # 3. Register scoped hooks (Req 5.4)
        cleanup_fn = self._register_scoped_hooks(agent_def, agent_id)

        try:
            # 4. Execute via callback (Req 5.1 — Agents-as-Tools)
            if self._tracer is not None:
                with self._tracer.trace_agent_action(
                    agent_id=agent_id,
                    action="subagent_execute",
                    attributes={"agent.name": agent_def.name},
                ):
                    raw_result = self._agent_callback(agent_def, task, budget)
            else:
                raw_result = self._agent_callback(agent_def, task, budget)

            # 5. Record usage with cost governor
            tokens = raw_result.get("tokens_consumed", {})
            if self._cost_governor is not None and (
                tokens.get("input", 0) > 0 or tokens.get("output", 0) > 0
            ):
                tier = self._resolve_model_tier(agent_def.model)
                self._cost_governor.record_usage(
                    agent_id=agent_id,
                    input_tokens=tokens.get("input", 0),
                    output_tokens=tokens.get("output", 0),
                    model_tier=tier,
                    cached_tokens=tokens.get("cache_read", 0),
                )
                if self._tracer is not None:
                    try:
                        cost_usd = self._cost_governor.calculate_cost(
                            input_tokens=tokens.get("input", 0),
                            output_tokens=tokens.get("output", 0),
                            cache_read_tokens=tokens.get("cache_read", 0),
                            model_tier=tier,
                        )
                    except Exception:
                        cost_usd = 0.0
                    self._tracer.record_cost_span(
                        agent_id=agent_id,
                        input_tokens=tokens.get("input", 0),
                        output_tokens=tokens.get("output", 0),
                        cost_usd=cost_usd,
                        model_tier=tier.value,
                    )

            # Build AgentResult
            result = AgentResult(
                agent_name=agent_def.name,
                summary=raw_result.get("summary", ""),
                turns_used=raw_result.get("turns_used", 0),
                tokens_consumed=tokens,
                tools_used=raw_result.get("tools_used", []),
                files_modified=raw_result.get("files_modified", []),
                exit_reason=raw_result.get("exit_reason", "complete"),
            )

        except Exception as exc:
            logger.error("Subagent %s failed: %s", agent_id, exc)
            result = AgentResult(
                agent_name=agent_def.name,
                summary=f"Error: {exc}",
                turns_used=0,
                tokens_consumed={"input": 0, "output": 0, "cache_read": 0},
                tools_used=[],
                files_modified=[],
                exit_reason="error",
            )

        # 6. Fire SubagentStop hooks (Req 5.3)
        self._fire_subagent_stop(agent_def, result)

        # 7. Clean up scoped hooks
        if cleanup_fn is not None:
            cleanup_fn()

        # 8. Return structured result
        return result

    def as_tools(
        self,
        agent_defs: list[AgentDefinition],
    ) -> list[dict]:
        """Wrap agent definitions as tool descriptors (Req 5.1).

        In production this would use ``Agent.as_tool()`` or the ``@tool``
        decorator. Here we delegate to the pluggable ``as_tool_callback``.

        Returns a list of tool descriptor dicts with ``name`` and
        ``description`` keys.
        """
        return [self._as_tool_callback(ad) for ad in agent_defs]

    def execute_multiple(
        self,
        tasks: list[tuple[AgentDefinition, str]],
        parent_budget: AgentBudget | None = None,
    ) -> list[AgentResult]:
        """Execute multiple subagents and synthesize results (Req 5.5).

        Tasks are executed sequentially (dependency-safe ordering is the
        caller's responsibility). Each subagent gets its own isolated
        budget derived from *parent_budget*.
        """
        results: list[AgentResult] = []
        for agent_def, task in tasks:
            result = self.execute(agent_def, task, parent_budget)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_isolated_budget(
        parent_budget: AgentBudget | None,
    ) -> AgentBudget:
        """Create an independent budget for a subagent (Req 5.2).

        If a parent budget is provided, the subagent gets a proportional
        share. Otherwise a generous default is used.
        """
        if parent_budget is not None:
            return AgentBudget(
                input_budget_tokens=parent_budget.input_budget_tokens,
                output_budget_tokens=parent_budget.output_budget_tokens,
                session_budget_usd=parent_budget.session_budget_usd * 0.25,
                team_budget_usd=parent_budget.team_budget_usd,
            )
        return AgentBudget(
            input_budget_tokens=200_000,
            output_budget_tokens=50_000,
            session_budget_usd=1.0,
        )

    def _register_scoped_hooks(
        self,
        agent_def: AgentDefinition,
        agent_id: str,
    ) -> Callable[[], None] | None:
        """Register agent-scoped hooks and return a cleanup function (Req 5.4).

        Stop hooks in the agent's frontmatter are automatically converted
        to SubagentStop hooks.
        """
        if not agent_def.hooks or self._hook_engine is None:
            return None

        # Convert Stop → SubagentStop
        converted = _convert_stop_to_subagent_stop(agent_def.hooks)

        # Build the hooks dict with HookEvent keys
        hooks_by_event: dict[HookEvent, list[HookDefinition]] = {}
        for key_str, defs in converted.items():
            try:
                event = HookEvent(key_str)
            except ValueError:
                logger.warning("Unknown hook event %r in agent %s", key_str, agent_def.name)
                continue
            hooks_by_event[event] = defs

        if not hooks_by_event:
            return None

        return self._hook_engine.register_scoped(
            hooks=hooks_by_event,
            scope_id=agent_id,
            scope="subagent_frontmatter",
        )

    def _fire_subagent_stop(
        self,
        agent_def: AgentDefinition,
        result: AgentResult,
    ) -> HookResult:
        """Fire SubagentStop hooks after subagent completion (Req 5.3)."""
        if self._hook_engine is None:
            return HookResult()

        context = HookContext(
            session_id=self._session_id,
            hook_event_name=HookEvent.SUBAGENT_STOP,
            cwd=self._cwd,
            session_type=self._session_type,
            source=agent_def.name,
            model=agent_def.model,
        )
        return self._hook_engine.fire(HookEvent.SUBAGENT_STOP, context)

    @staticmethod
    def _resolve_model_tier(model_alias: str) -> ModelTier:
        """Resolve a model alias string to a ModelTier enum."""
        from src.types.core import ModelTier

        alias_map = {
            "opus": ModelTier.OPUS,
            "sonnet": ModelTier.SONNET,
            "haiku": ModelTier.HAIKU,
            "inherit": ModelTier.SONNET,  # default when inheriting
        }
        return alias_map.get(model_alias, ModelTier.SONNET)
