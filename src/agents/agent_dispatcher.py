"""Agent Dispatcher — 13-step agent lifecycle execution.

Dispatches tasks to agents based on explicit invocation or auto-dispatch.
Executes the full 13-step agent lifecycle:
  1. dispatch   — receive task + agent name
  2. resolve    — model alias + budget check via CostGovernor
  3. tools      — build effective tool list
  4. MCP        — start agent-scoped MCP servers (pluggable callback)
  5. skills     — load referenced skills into context (pluggable callback)
  6. hooks      — register agent-scoped hooks (pluggable callback)
  7. context    — create isolated context window
  8. system     — inject memory + skills + systemPrompt
  9. loop       — execute agentic loop (pluggable agent_callback)
  10. stop      — fire SubagentStop hooks
  11. cleanup   — deregister hooks, stop MCP servers
  12. release   — release context
  13. return    — return AgentResult

All external calls (Strands Agent execution, MCP, hooks) are injected as
pluggable callbacks so the system runs and tests without the SDK installed.

Requirements: 8.6, 8.7, 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from src.observability.instrumentation import BrainmassTracer
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    BudgetStatus,
    HookContext,
    HookEvent,
    ModelTier,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model alias → canonical ID map (mirrors AgentLoader.MODEL_MAP)
# ---------------------------------------------------------------------------

_MODEL_ALIAS_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "inherit": "claude-sonnet-4-5-20250929",  # default when inheriting
}

_MODEL_TIER_MAP: dict[str, ModelTier] = {
    "sonnet": ModelTier.SONNET,
    "opus": ModelTier.OPUS,
    "haiku": ModelTier.HAIKU,
    "inherit": ModelTier.SONNET,
    "claude-sonnet-4-5-20250929": ModelTier.SONNET,
    "claude-opus-4-6": ModelTier.OPUS,
    "claude-haiku-4-5-20251001": ModelTier.HAIKU,
}

# ---------------------------------------------------------------------------
# Pluggable callback type aliases
# ---------------------------------------------------------------------------

# Executes the agentic loop; returns raw result dict
AgentLoopCallback = Callable[
    [AgentDefinition, str, str, AgentBudget],  # (agent_def, task, system_prompt, budget)
    dict,  # {summary, turns_used, tokens_consumed, tools_used, files_modified, exit_reason}
]

# Starts agent-scoped MCP servers; returns a cleanup callable
McpStartCallback = Callable[
    [dict],  # mcp_servers config dict
    Callable[[], None],  # cleanup function
]

# Loads skills into context; returns combined skill context string
SkillsLoadCallback = Callable[
    [list[str]],  # skill names
    str,  # skill context text
]

# Registers agent-scoped hooks; returns cleanup callable
HooksRegisterCallback = Callable[
    [dict, str],  # (hooks dict, scope_id)
    Callable[[], None],  # cleanup function
]


# ---------------------------------------------------------------------------
# Default no-op callbacks
# ---------------------------------------------------------------------------


def _default_agent_loop(
    agent_def: AgentDefinition,
    task: str,
    system_prompt: str,
    budget: AgentBudget,
) -> dict:
    """Default no-op agent loop callback."""
    return {
        "summary": f"Agent '{agent_def.name}' completed task: {task}",
        "turns_used": 0,
        "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
        "tools_used": [],
        "files_modified": [],
        "exit_reason": "complete",
    }


def _default_mcp_start(mcp_servers: dict) -> Callable[[], None]:
    """Default no-op MCP start callback."""
    # --- Production integration point ---
    # from strands_tools import MCPClient
    # clients = {name: MCPClient(**cfg) for name, cfg in mcp_servers.items()}
    # for client in clients.values():
    #     client.start()
    # def cleanup():
    #     for client in clients.values():
    #         client.stop()
    # return cleanup
    return lambda: None


def _default_skills_load(skill_names: list[str]) -> str:
    """Default no-op skills load callback."""
    # --- Production integration point ---
    # from src.skills.skill_registry import SkillRegistry
    # registry = SkillRegistry()
    # return "\n\n".join(registry.get_skill_content(name) for name in skill_names)
    if not skill_names:
        return ""
    return f"# Skills\nLoaded skills: {', '.join(skill_names)}"


def _default_hooks_register(hooks: dict, scope_id: str) -> Callable[[], None]:
    """Default no-op hooks register callback."""
    return lambda: None


# ---------------------------------------------------------------------------
# Production callbacks (injected in production; no-ops used by default so
# existing tests continue to pass without the Strands SDK installed)
# ---------------------------------------------------------------------------


def _production_agent_loop(
    agent_def: AgentDefinition,
    task: str,
    system_prompt: str,
    budget: AgentBudget,
) -> dict:
    """Production agent loop using Strands Agent + BedrockModel.

    --- Production integration point ---
    # from strands import Agent
    # from strands.models.bedrock import BedrockModel
    # from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result
    #
    # model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, agent_def.model)
    # model = BedrockModel(model_id=model_id)
    # agent = Agent(
    #     name=agent_def.name,
    #     model=model,
    #     system_prompt=system_prompt,
    #     max_tokens=budget.output_budget_tokens,
    # )
    # raw = agent(task)
    # return _normalize_strands_result(raw)
    """
    try:
        from strands import Agent  # type: ignore
        from strands.models.bedrock import BedrockModel  # type: ignore

        from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result
    except ImportError as exc:
        raise RuntimeError(
            "strands package is required for production agent loop. "
            "Install it with: pip install strands-agents"
        ) from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, agent_def.model)
    model = BedrockModel(model_id=model_id)
    agent = Agent(
        name=agent_def.name,
        model=model,
        system_prompt=system_prompt,
        max_tokens=budget.output_budget_tokens,
    )
    raw = agent(task)
    return _normalize_strands_result(raw)


def _production_mcp_start(mcp_servers: dict) -> Callable[[], None]:
    """Production MCP start using strands_agents_tools MCPClient.

    --- Production integration point ---
    # from strands_tools import MCPClient
    # clients = {}
    # for name, cfg in mcp_servers.items():
    #     client = MCPClient(**cfg)
    #     client.start()
    #     clients[name] = client
    # def cleanup():
    #     for client in clients.values():
    #         try:
    #             client.stop()
    #         except Exception:
    #             pass
    # return cleanup
    """
    try:
        from strands_tools import MCPClient  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents-tools package is required for MCP server support. "
            "Install it with: pip install strands-agents-tools"
        ) from exc

    clients: dict[str, object] = {}
    for name, cfg in mcp_servers.items():
        client = MCPClient(**cfg)
        client.start()
        clients[name] = client

    def cleanup() -> None:
        for client in clients.values():
            try:
                client.stop()  # type: ignore[union-attr]
            except Exception:
                pass

    return cleanup


def _production_skills_load(skill_names: list[str]) -> str:
    """Production skills load using SkillRegistry.

    --- Production integration point ---
    # from src.skills.skill_registry import SkillRegistry
    # registry = SkillRegistry()
    # return "\n\n".join(registry.get_skill_content(name) for name in skill_names)
    """
    if not skill_names:
        return ""
    from src.skills.skill_registry import SkillRegistry  # type: ignore

    registry = SkillRegistry()
    parts: list[str] = []
    for name in skill_names:
        try:
            content = registry.get_skill_content(name)
            if content:
                parts.append(content)
        except Exception:
            pass
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# DispatchContext — internal state for one dispatch lifecycle
# ---------------------------------------------------------------------------


@dataclass
class DispatchContext:
    """Internal state threaded through the 13-step lifecycle."""

    agent_def: AgentDefinition
    task: str
    agent_id: str
    budget: AgentBudget
    resolved_model: str = ""
    model_tier: ModelTier = ModelTier.SONNET
    effective_tools: list[str] = field(default_factory=list)
    system_prompt: str = ""
    mcp_cleanup: Callable[[], None] | None = None
    hooks_cleanup: Callable[[], None] | None = None
    skill_context: str = ""


# ---------------------------------------------------------------------------
# AgentDispatcher
# ---------------------------------------------------------------------------


class AgentDispatcher:
    """Executes the 13-step agent lifecycle for dispatched agents.

    All external integrations (Strands SDK, MCP, skills, hooks) are injected
    as pluggable callbacks so the dispatcher is fully testable without any
    external dependencies.

    Usage::

        dispatcher = AgentDispatcher(
            cost_governor=governor,
            hook_engine=engine,
        )
        result = dispatcher.dispatch(agent_def, "Review the auth module")
    """

    def __init__(
        self,
        cost_governor: object | None = None,
        hook_engine: object | None = None,
        agent_loop_callback: AgentLoopCallback | None = None,
        mcp_start_callback: McpStartCallback | None = None,
        skills_load_callback: SkillsLoadCallback | None = None,
        hooks_register_callback: HooksRegisterCallback | None = None,
        session_id: str = "",
        cwd: str = ".",
        session_type: str = "interactive",
        parent_model: str = "sonnet",
        tracer: BrainmassTracer | None = None,
    ) -> None:
        self._cost_governor = cost_governor
        self._hook_engine = hook_engine
        # Default to production agent loop; tests always inject their own callback.
        self._agent_loop = agent_loop_callback or _production_agent_loop
        self._mcp_start = mcp_start_callback or _production_mcp_start
        self._skills_load = skills_load_callback or _production_skills_load
        # hooks_register_callback: None means wire through hook_engine.register_scoped()
        # in _step6_hooks when hook_engine is available.
        self._hooks_register_callback = hooks_register_callback
        self._session_id = session_id
        self._cwd = cwd
        self._session_type = session_type
        self._parent_model = parent_model
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(
        self,
        agent_def: AgentDefinition,
        task: str,
        parent_budget: AgentBudget | None = None,
    ) -> AgentResult:
        """Execute the 13-step agent lifecycle.

        Args:
            agent_def: The agent definition to execute.
            task: The task description to pass to the agent.
            parent_budget: Optional parent budget for budget inheritance.

        Returns:
            Structured AgentResult.
        """
        ctx = DispatchContext(
            agent_def=agent_def,
            task=task,
            agent_id=f"agent:{agent_def.name}:{uuid.uuid4().hex[:8]}",
            budget=self._create_budget(parent_budget),
        )

        if self._tracer is not None:
            with self._tracer.trace_agent_action(
                agent_id=ctx.agent_id,
                action="dispatch",
                attributes={"agent.name": agent_def.name},
            ):
                return self._dispatch_inner(ctx)
        return self._dispatch_inner(ctx)

    def _dispatch_inner(self, ctx: DispatchContext) -> AgentResult:
        """Run lifecycle steps for a prepared dispatch context."""
        self._step1_dispatch(ctx)
        self._step2_resolve(ctx)
        self._step3_tools(ctx)
        self._step4_mcp(ctx)
        self._step5_skills(ctx)
        self._step6_hooks(ctx)
        self._step7_context(ctx)
        self._step8_system(ctx)

        result = self._step9_loop(ctx)

        self._step10_stop(ctx, result)
        self._step11_cleanup(ctx)
        self._step12_release(ctx)
        return self._step13_return(ctx, result)

    # ------------------------------------------------------------------
    # 13-step lifecycle
    # ------------------------------------------------------------------

    def _step1_dispatch(self, ctx: DispatchContext) -> None:
        """Step 1: Receive task + agent name."""
        logger.info(
            "Dispatching agent '%s' (id=%s) for task: %s",
            ctx.agent_def.name,
            ctx.agent_id,
            ctx.task[:80],
        )

    def _step2_resolve(self, ctx: DispatchContext) -> None:
        """Step 2: Resolve model alias + check budget via CostGovernor."""
        raw_model = ctx.agent_def.model

        # Resolve 'inherit' to parent model
        if raw_model == "inherit":
            raw_model = self._parent_model

        ctx.resolved_model = _MODEL_ALIAS_MAP.get(raw_model, raw_model)
        ctx.model_tier = _MODEL_TIER_MAP.get(raw_model, ModelTier.SONNET)

        # Check budget — downgrade model if needed
        if self._cost_governor is not None:
            budget_status = self._cost_governor.check_budget(ctx.agent_id)
            if budget_status == BudgetStatus.EXCEEDED:
                logger.warning(
                    "Agent '%s' budget exceeded — aborting dispatch.", ctx.agent_def.name
                )
                raise RuntimeError(
                    f"Budget exceeded for agent '{ctx.agent_def.name}'. "
                    "Cannot dispatch."
                )
            elif budget_status == BudgetStatus.CRITICAL:
                logger.warning(
                    "Agent '%s' budget critical — downgrading to Haiku.", ctx.agent_def.name
                )
                ctx.resolved_model = _MODEL_ALIAS_MAP["haiku"]
                ctx.model_tier = ModelTier.HAIKU
            elif budget_status == BudgetStatus.WARNING:
                # Downgrade one tier
                ctx.model_tier, ctx.resolved_model = self._downgrade_one_tier(
                    ctx.model_tier
                )

        logger.debug(
            "Agent '%s' resolved model: %s (tier=%s)",
            ctx.agent_def.name,
            ctx.resolved_model,
            ctx.model_tier.value,
        )

    def _step3_tools(self, ctx: DispatchContext) -> None:
        """Step 3: Build effective tool list."""
        if ctx.agent_def.tools is not None:
            ctx.effective_tools = list(ctx.agent_def.tools)
        else:
            # Default: all tools (disallowedTools handled by executor)
            ctx.effective_tools = []

        logger.debug(
            "Agent '%s' effective tools: %s",
            ctx.agent_def.name,
            ctx.effective_tools or "<all>",
        )

    def _step4_mcp(self, ctx: DispatchContext) -> None:
        """Step 4: Start agent-scoped MCP servers.

        --- Production integration point ---
        In production, this starts MCP server processes for each server
        defined in agent_def.mcp_servers and returns a cleanup function.
        """
        if ctx.agent_def.mcp_servers:
            ctx.mcp_cleanup = self._mcp_start(ctx.agent_def.mcp_servers)
            logger.debug(
                "Agent '%s' started %d MCP server(s).",
                ctx.agent_def.name,
                len(ctx.agent_def.mcp_servers),
            )

    def _step5_skills(self, ctx: DispatchContext) -> None:
        """Step 5: Load referenced skills into context.

        --- Production integration point ---
        In production, this loads SKILL.md content for each referenced skill
        and injects it into the agent's context.
        """
        if ctx.agent_def.skills:
            ctx.skill_context = self._skills_load(ctx.agent_def.skills)
            logger.debug(
                "Agent '%s' loaded %d skill(s).",
                ctx.agent_def.name,
                len(ctx.agent_def.skills),
            )

    def _step6_hooks(self, ctx: DispatchContext) -> None:
        """Step 6: Register agent-scoped hooks (additive).

        When a ``hooks_register_callback`` was injected at construction time,
        it is used directly.  Otherwise, if a ``hook_engine`` is available,
        hooks are registered via ``hook_engine.register_scoped()`` so that
        agent-frontmatter hooks participate in the full scope-precedence chain.

        --- Production integration point ---
        # self._hook_engine.register_scoped(
        #     scope_id=ctx.agent_id,
        #     scope="subagent_frontmatter",
        #     hooks=ctx.agent_def.hooks,
        # )
        """
        if not ctx.agent_def.hooks:
            return

        if self._hooks_register_callback is not None:
            ctx.hooks_cleanup = self._hooks_register_callback(
                ctx.agent_def.hooks, ctx.agent_id
            )
        elif self._hook_engine is not None and hasattr(self._hook_engine, "register_scoped"):
            # --- Production integration point ---
            scope_id = ctx.agent_id
            self._hook_engine.register_scoped(  # type: ignore[union-attr]
                scope_id=scope_id,
                scope="subagent_frontmatter",
                hooks=ctx.agent_def.hooks,
            )

            def _cleanup() -> None:
                try:
                    self._hook_engine.deregister_scoped(scope_id)  # type: ignore[union-attr]
                except Exception:
                    pass

            ctx.hooks_cleanup = _cleanup
        else:
            ctx.hooks_cleanup = _default_hooks_register(ctx.agent_def.hooks, ctx.agent_id)

        logger.debug("Agent '%s' registered scoped hooks.", ctx.agent_def.name)

    def _step7_context(self, ctx: DispatchContext) -> None:
        """Step 7: Create isolated context window.

        --- Production integration point ---
        In production, this creates a new Strands ConversationManager
        instance for the agent, providing full context isolation.
        """
        logger.debug("Agent '%s' context window created.", ctx.agent_def.name)

    def _step8_system(self, ctx: DispatchContext) -> None:
        """Step 8: Inject memory + skills + systemPrompt."""
        parts: list[str] = []

        if ctx.agent_def.memory:
            parts.append(f"# Memory\n{ctx.agent_def.memory}")

        if ctx.skill_context:
            parts.append(ctx.skill_context)

        if ctx.agent_def.system_prompt:
            parts.append(ctx.agent_def.system_prompt)

        ctx.system_prompt = "\n\n".join(parts).strip()
        logger.debug(
            "Agent '%s' system prompt assembled (%d chars).",
            ctx.agent_def.name,
            len(ctx.system_prompt),
        )

    def _step9_loop(self, ctx: DispatchContext) -> dict:
        """Step 9: Execute agentic loop via pluggable callback.

        --- Production integration point ---
        In production, this creates a Strands Agent with the resolved model,
        tools, hooks, and system prompt, then calls agent(task).
        Example:
            from strands import Agent
            from strands.models.bedrock import BedrockModel
            model = BedrockModel(model_id=ctx.resolved_model)
            agent = Agent(
                name=ctx.agent_def.name,
                model=model,
                system_prompt=ctx.system_prompt,
                tools=self._resolve_tool_objects(ctx.effective_tools),
            )
            raw = agent(ctx.task)
        """
        logger.info(
            "Agent '%s' executing loop (model=%s).",
            ctx.agent_def.name,
            ctx.resolved_model,
        )
        try:
            if self._tracer is not None:
                with self._tracer.trace_model_interaction(
                    model_tier=ctx.model_tier.value,
                    agent_id=ctx.agent_id,
                ):
                    raw = self._agent_loop(
                        ctx.agent_def,
                        ctx.task,
                        ctx.system_prompt,
                        ctx.budget,
                    )
            else:
                raw = self._agent_loop(
                    ctx.agent_def,
                    ctx.task,
                    ctx.system_prompt,
                    ctx.budget,
                )
        except Exception as exc:
            logger.error("Agent '%s' loop failed: %s", ctx.agent_def.name, exc)
            raw = {
                "summary": f"Error: {exc}",
                "turns_used": 0,
                "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "error",
            }

        # Record usage with cost governor
        tokens = raw.get("tokens_consumed", {})
        if self._cost_governor is not None and (
            tokens.get("input", 0) > 0 or tokens.get("output", 0) > 0
        ):
            self._cost_governor.record_usage(
                agent_id=ctx.agent_id,
                input_tokens=tokens.get("input", 0),
                output_tokens=tokens.get("output", 0),
                model_tier=ctx.model_tier,
                cached_tokens=tokens.get("cache_read", 0),
            )
            if self._tracer is not None:
                try:
                    cost_usd = self._cost_governor.calculate_cost(
                        input_tokens=tokens.get("input", 0),
                        output_tokens=tokens.get("output", 0),
                        cache_read_tokens=tokens.get("cache_read", 0),
                        model_tier=ctx.model_tier,
                    )
                except Exception:
                    cost_usd = 0.0
                self._tracer.record_cost_span(
                    agent_id=ctx.agent_id,
                    input_tokens=tokens.get("input", 0),
                    output_tokens=tokens.get("output", 0),
                    cost_usd=cost_usd,
                    model_tier=ctx.model_tier.value,
                )

        return raw

    def _step10_stop(self, ctx: DispatchContext, raw_result: dict) -> None:
        """Step 10: Fire SubagentStop hooks."""
        if self._hook_engine is None:
            return

        hook_ctx = HookContext(
            session_id=self._session_id,
            hook_event_name=HookEvent.SUBAGENT_STOP,
            cwd=self._cwd,
            session_type=self._session_type,
            source=ctx.agent_def.name,
            model=ctx.resolved_model,
        )
        try:
            self._hook_engine.fire(HookEvent.SUBAGENT_STOP, hook_ctx)
        except Exception as exc:
            logger.error(
                "Agent '%s' SubagentStop hook failed: %s", ctx.agent_def.name, exc
            )

    def _step11_cleanup(self, ctx: DispatchContext) -> None:
        """Step 11: Deregister scoped hooks and stop MCP servers."""
        if ctx.hooks_cleanup is not None:
            try:
                ctx.hooks_cleanup()
            except Exception as exc:
                logger.error(
                    "Agent '%s' hooks cleanup failed: %s", ctx.agent_def.name, exc
                )

        if ctx.mcp_cleanup is not None:
            try:
                ctx.mcp_cleanup()
            except Exception as exc:
                logger.error(
                    "Agent '%s' MCP cleanup failed: %s", ctx.agent_def.name, exc
                )

    def _step12_release(self, ctx: DispatchContext) -> None:
        """Step 12: Release isolated context window.

        --- Production integration point ---
        In production, this releases the Strands ConversationManager
        instance and frees associated memory.
        """
        logger.debug("Agent '%s' context window released.", ctx.agent_def.name)

    def _step13_return(self, ctx: DispatchContext, raw_result: dict) -> AgentResult:
        """Step 13: Build and return AgentResult to parent."""
        result = AgentResult(
            agent_name=ctx.agent_def.name,
            summary=raw_result.get("summary", ""),
            turns_used=raw_result.get("turns_used", 0),
            tokens_consumed=raw_result.get(
                "tokens_consumed", {"input": 0, "output": 0, "cache_read": 0}
            ),
            tools_used=raw_result.get("tools_used", []),
            files_modified=raw_result.get("files_modified", []),
            exit_reason=raw_result.get("exit_reason", "complete"),
        )
        logger.info(
            "Agent '%s' completed: exit_reason=%s, turns=%d.",
            ctx.agent_def.name,
            result.exit_reason,
            result.turns_used,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_budget(parent_budget: AgentBudget | None) -> AgentBudget:
        """Create an isolated budget for the dispatched agent."""
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

    @staticmethod
    def _downgrade_one_tier(
        current_tier: ModelTier,
    ) -> tuple[ModelTier, str]:
        """Downgrade model tier by one level (Opus→Sonnet→Haiku)."""
        if current_tier == ModelTier.OPUS:
            return ModelTier.SONNET, _MODEL_ALIAS_MAP["sonnet"]
        # Sonnet or Haiku → Haiku
        return ModelTier.HAIKU, _MODEL_ALIAS_MAP["haiku"]
