"""Orchestrator - central control-plane agent for the Brainmass v3 system.

Implements the request flow, topology selection, hooks, and cost tracking.
Supports two execution modes:
- Local callback mode (used by tests and local development)
- Production Strands mode (real Agent + Bedrock model path)

Requirements: 1.1, 1.2, 1.3, 1.6
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from src.context.context_manager import ContextManager
from src.cost.cost_governor import CostGovernor, TaskSignals
from src.hooks.hook_engine import BrainmassHookEngine
from src.observability.instrumentation import BrainmassTracer, CloudWatchExporter, TerminalMonitor
from src.types.core import (
    AgentBudget,
    HookContext,
    HookEvent,
    HookResult,
    ModelTier,
    TopologyType,
)


class EffortControllerProtocol(Protocol):
    """Minimal interface the Orchestrator expects from an Effort Controller."""

    def select_effort(
        self,
        signals: TaskSignals,
        model: ModelTier,
        budget: AgentBudget,
    ) -> EffortResult:
        ...


@dataclass
class EffortResult:
    """Result of effort-level selection."""

    level: str
    budget_tokens: int


class _DefaultEffortController:
    """Fallback effort controller used when none is injected."""

    def select_effort(
        self,
        signals: TaskSignals,
        model: ModelTier,
        budget: AgentBudget,
    ) -> EffortResult:
        return EffortResult(level="standard", budget_tokens=10_000)


@dataclass
class Task:
    """A decomposed subtask produced by the Orchestrator."""

    id: str
    description: str
    topology: TopologyType = TopologyType.HIERARCHICAL
    signals: TaskSignals = field(default_factory=TaskSignals)


@dataclass
class ToolCallRecord:
    """Record of a single tool call executed during request processing."""

    tool_name: str
    tool_input: dict
    tool_response: str
    model_tier: ModelTier
    cost_usd: float = 0.0


@dataclass
class OrchestratorResult:
    """Returned by Orchestrator.process_request."""

    request_id: str
    response: str
    model_tier: ModelTier
    effort_level: str
    topology: TopologyType | None
    tasks: list[Task]
    tool_calls: list[ToolCallRecord]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    hooks_fired: list[str] = field(default_factory=list)


@dataclass
class ExecutionUsage:
    """Normalized usage metrics from model execution."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


SYSTEM_PROMPT = """\
You are Brainmass, an enterprise-grade agentic coding assistant.

You decompose complex tasks, select the right tools, and coordinate with
specialized sub-agents when needed. You always track costs, respect budgets,
and fire lifecycle hooks at every stage of execution.

Guidelines:
- Prefer the cheapest capable model for each sub-task.
- Preserve critical context (error messages, file paths, decisions).
- Fire hooks before and after every tool call.
- When a task is complex, decompose it and select the right topology.
"""


_MODEL_IDS = {
    ModelTier.HAIKU: "us.anthropic.claude-haiku-4-5-v1:0",
    ModelTier.SONNET: "us.anthropic.claude-sonnet-4-5-v1:0",
    ModelTier.OPUS: "us.anthropic.claude-opus-4-6-v1:0",
}

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"\bignore all prior instructions\b", re.IGNORECASE),
    re.compile(r"\bdeveloper policy disabled\b", re.IGNORECASE),
    re.compile(r"\bbypass policy\b", re.IGNORECASE),
    re.compile(r"\bexpose hidden context\b", re.IGNORECASE),
    re.compile(r"\breturn secrets?\b", re.IGNORECASE),
]
_CANARY_TOKEN_PATTERN = re.compile(r"\bBM_CANARY_[A-Za-z0-9_-]+\b")
_FILE_PATH_PATTERN = re.compile(r"([A-Za-z0-9_./\\-]+\.[A-Za-z0-9_]+)")
_REFUSAL_PATTERN = re.compile(
    r"\b(cannot|can't|will not|won't|refuse|decline)\b",
    re.IGNORECASE,
)


class Orchestrator:
    """Central control-plane component."""

    def __init__(
        self,
        context_manager: ContextManager,
        hook_engine: BrainmassHookEngine,
        cost_governor: CostGovernor,
        effort_controller: EffortControllerProtocol | None = None,
        agent_id: str = "orchestrator",
        session_id: str = "",
        session_type: str = "interactive",
        cwd: str = ".",
        model_callback: Callable[..., str] | None = None,
        tool_executor: Callable[..., str] | None = None,
        tracer: BrainmassTracer | None = None,
        terminal_monitor: TerminalMonitor | None = None,
        cloudwatch_exporter: CloudWatchExporter | None = None,
        runtime_config: object | None = None,
        use_production_agent: bool | None = None,
    ) -> None:
        self.context_manager = context_manager
        self.hook_engine = hook_engine
        self.cost_governor = cost_governor
        self.effort_controller: EffortControllerProtocol = (
            effort_controller or _DefaultEffortController()
        )

        self.agent_id = agent_id
        self.session_id = session_id or str(uuid.uuid4())
        self.session_type = session_type
        self.cwd = cwd
        self.trace_id = ""

        self._model_callback = model_callback or self._default_model_callback
        self._tool_executor = tool_executor or self._default_tool_executor
        self._is_new_session = True
        self._system_prompt = SYSTEM_PROMPT
        self._tracer = tracer
        self._terminal_monitor = terminal_monitor
        self._cloudwatch_exporter = cloudwatch_exporter
        self._runtime_config = runtime_config

        # Auto-detect production mode (Requirements: 1.2, 1.3):
        # - Explicit True: use production Strands path.
        # - Explicit False or None (default): use local callback path.
        #   When None and no model_callback was injected, production mode is
        #   available but not activated automatically — callers must pass
        #   use_production_agent=True explicitly to opt in.
        #   This preserves backward compatibility: existing tests that omit
        #   both model_callback and use_production_agent continue to use the
        #   local stub path.
        if use_production_agent is True and model_callback is not None:
            # Explicit production mode but a callback was also injected —
            # the callback takes precedence; disable production mode.
            self._use_production_agent = False
        elif use_production_agent is True:
            self._use_production_agent = True
        else:
            # None or False → local callback mode
            self._use_production_agent = False

    @staticmethod
    def _strands_available() -> bool:
        """Return True if the strands package can be imported."""
        try:
            import strands  # type: ignore  # noqa: F401
            return True
        except ImportError:
            return False

    def process_request(self, request: str) -> OrchestratorResult:
        """Execute the request lifecycle and return structured result."""
        request_id = str(uuid.uuid4())
        hooks_fired: list[str] = []
        tool_calls: list[ToolCallRecord] = []
        usage = ExecutionUsage()

        if self._tracer is not None:
            self.trace_id = self._tracer.get_trace_id()

        budget = self._ensure_budget()

        if self._is_new_session:
            _ = self._fire_hook(HookEvent.SESSION_START, source="new")
            hooks_fired.append(HookEvent.SESSION_START.value)
            self._is_new_session = False

        submit_result = self._fire_hook(HookEvent.USER_PROMPT_SUBMIT)
        hooks_fired.append(HookEvent.USER_PROMPT_SUBMIT.value)
        if submit_result.updated_input and "request" in submit_result.updated_input:
            request = submit_result.updated_input["request"]
        request, had_prompt_injection = self._sanitize_request(request)

        signals = self._classify_request(request)
        effort = self.effort_controller.select_effort(signals, ModelTier.SONNET, budget)
        model_tier = self.cost_governor.select_model(signals, budget)
        tasks = self._decompose_task(request, signals)
        topology = self._select_topology(signals) if tasks else None

        if self._use_production_agent:
            response, usage, prod_tool_calls = self._execute_with_strands(
                request=request,
                model_tier=model_tier,
                effort=effort,
            )
            tool_calls.extend(prod_tool_calls)
        else:
            response = self._execute_with_hooks(
                request=request,
                model_tier=model_tier,
                effort=effort,
                tool_calls=tool_calls,
                hooks_fired=hooks_fired,
            )
            usage = ExecutionUsage(
                input_tokens=(sum(100 for _ in tool_calls) + len(request.split())),
                output_tokens=len(response.split()),
                cache_read_tokens=0,
            )

        if had_prompt_injection and not self._response_contains_refusal(response):
            response = (
                "I cannot comply with unsafe instruction overrides. "
                + response
            )

        self.context_manager.advance_turn()

        stop_result = self._fire_hook(HookEvent.STOP)
        hooks_fired.append(HookEvent.STOP.value)
        if stop_result.decision == "block":
            response += "\n[Stop hook blocked completion - additional work needed]"

        self.cost_governor.record_usage(
            agent_id=self.agent_id,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            model_tier=model_tier,
            cached_tokens=usage.cache_read_tokens,
        )
        total_cost = self.cost_governor.calculate_cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            model_tier=model_tier,
        )

        if self._tracer is not None:
            self._tracer.record_cost_span(
                agent_id=self.agent_id,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=total_cost,
                model_tier=model_tier.value,
            )

        # Emit dashboard metrics (Req 16.4) and export to CloudWatch (Req 16.5)
        if self._terminal_monitor is not None:
            self._terminal_monitor.format_dashboard()  # side-effect: aggregates spans

        if self._cloudwatch_exporter is not None:
            metrics = self.cost_governor.get_dashboard_data()
            self._cloudwatch_exporter.export_metrics(metrics)

        return OrchestratorResult(
            request_id=request_id,
            response=response,
            model_tier=model_tier,
            effort_level=effort.level,
            topology=topology,
            tasks=tasks,
            tool_calls=tool_calls,
            total_input_tokens=usage.input_tokens,
            total_output_tokens=usage.output_tokens,
            total_cost_usd=total_cost,
            hooks_fired=hooks_fired,
        )

    @staticmethod
    def select_topology(signals: TaskSignals) -> TopologyType:
        """Choose a multi-agent topology from task signals."""
        if signals.is_team_lead or signals.dependency_depth > 3:
            return TopologyType.AGENT_TEAMS
        if signals.is_exploration and signals.requires_reasoning:
            return TopologyType.SELF_IMPROVING_LOOP
        return TopologyType.HIERARCHICAL

    _select_topology = select_topology

    @staticmethod
    def classify_request(request: str) -> TaskSignals:
        """Classify a request into task signals using keyword heuristics."""
        lower = request.lower()
        is_exploration = any(
            kw in lower
            for kw in ("search", "find", "grep", "explore", "list", "ls", "look")
        )
        requires_reasoning = any(
            kw in lower
            for kw in (
                "architect",
                "design",
                "refactor",
                "security",
                "audit",
                "complex",
                "debug",
                "analyse",
                "analyze",
            )
        )
        is_team_lead = "team" in lower or "coordinate" in lower
        files_affected = lower.count("file") + lower.count("module") + 1
        dependency_depth = 4 if "cross-service" in lower or "cross service" in lower else 1
        return TaskSignals(
            is_exploration=is_exploration,
            files_affected=files_affected,
            requires_reasoning=requires_reasoning,
            is_team_lead=is_team_lead,
            dependency_depth=dependency_depth,
        )

    _classify_request = classify_request

    def _decompose_task(self, request: str, signals: TaskSignals) -> list[Task]:
        """Break a complex request into subtasks."""
        if signals.files_affected <= 1 and not signals.requires_reasoning:
            return []
        topology = self.select_topology(signals)
        return [
            Task(
                id=str(uuid.uuid4()),
                description=request,
                topology=topology,
                signals=signals,
            )
        ]

    def _execute_with_hooks(
        self,
        request: str,
        model_tier: ModelTier,
        effort: EffortResult,
        tool_calls: list[ToolCallRecord],
        hooks_fired: list[str],
    ) -> str:
        """Local execution mode using injected callbacks and explicit hooks."""
        if self._tracer is not None:
            with self._tracer.trace_model_interaction(
                model_tier=model_tier.value,
                agent_id=self.agent_id,
            ):
                response = self._model_callback(request, model_tier, effort)
        else:
            response = self._model_callback(request, model_tier, effort)

        simulated_tools = self._extract_tool_calls(response)
        for tool_name, tool_input in simulated_tools:
            pre_result = self._fire_hook(
                HookEvent.PRE_TOOL_USE,
                tool_name=tool_name,
                tool_input=tool_input,
            )
            hooks_fired.append(HookEvent.PRE_TOOL_USE.value)

            if pre_result.permission_decision == "deny":
                tool_calls.append(
                    ToolCallRecord(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_response="[DENIED by PreToolUse hook]",
                        model_tier=model_tier,
                    )
                )
                continue

            effective_input = pre_result.updated_input if pre_result.updated_input else tool_input
            try:
                if self._tracer is not None:
                    with self._tracer.trace_tool_call(
                        tool_name=tool_name,
                        agent_id=self.agent_id,
                    ):
                        tool_response = self._tool_executor(tool_name, effective_input)
                else:
                    tool_response = self._tool_executor(tool_name, effective_input)

                self._fire_hook(
                    HookEvent.POST_TOOL_USE,
                    tool_name=tool_name,
                    tool_input=effective_input,
                    tool_response=tool_response,
                )
                hooks_fired.append(HookEvent.POST_TOOL_USE.value)
            except Exception as exc:
                tool_response = f"[ERROR: {exc}]"
                self._fire_hook(
                    HookEvent.POST_TOOL_USE_FAILURE,
                    tool_name=tool_name,
                    tool_input=effective_input,
                    tool_response=tool_response,
                )
                hooks_fired.append(HookEvent.POST_TOOL_USE_FAILURE.value)

            tool_calls.append(
                ToolCallRecord(
                    tool_name=tool_name,
                    tool_input=effective_input,
                    tool_response=tool_response,
                    model_tier=model_tier,
                )
            )

        return response

    def _execute_with_strands(
        self,
        request: str,
        model_tier: ModelTier,
        effort: EffortResult,
    ) -> tuple[str, ExecutionUsage, list[ToolCallRecord]]:
        """Production execution path using Strands Agent + Bedrock model."""
        try:
            from strands import Agent  # type: ignore
            from strands.models.bedrock import BedrockModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Strands SDK is required for production execution mode."
            ) from exc

        model_id = self._resolve_model_id(model_tier)
        model = BedrockModel(model_id=model_id)
        agent = Agent(
            model=model,
            system_prompt=self._system_prompt,
            hooks=[self.hook_engine],
        )

        _ = perf_counter()
        raw = agent(request)

        response_text, usage, tool_calls = self._normalize_strands_output(raw, model_tier)
        if self._tracer is not None:
            self._tracer.record_effort_span(
                effort_level=effort.level,
                fast_mode=False,
                budget_tokens=effort.budget_tokens,
            )
        return response_text, usage, tool_calls

    def _normalize_strands_output(
        self,
        raw: Any,
        model_tier: ModelTier,
    ) -> tuple[str, ExecutionUsage, list[ToolCallRecord]]:
        """Normalize Strands output into response text, usage, and tool logs."""
        response_text = ""
        tool_calls: list[ToolCallRecord] = []
        usage = ExecutionUsage()

        if isinstance(raw, str):
            response_text = raw
        elif isinstance(raw, dict):
            response_text = str(raw.get("output", raw.get("text", "")))
            usage = ExecutionUsage(
                input_tokens=int(raw.get("input_tokens", 0) or 0),
                output_tokens=int(raw.get("output_tokens", 0) or 0),
                cache_read_tokens=int(raw.get("cache_read_input_tokens", 0) or 0),
            )
        else:
            response_text = str(getattr(raw, "output", "") or getattr(raw, "text", "") or str(raw))

        raw_usage = getattr(raw, "usage", None)
        if raw_usage is not None:
            usage = ExecutionUsage(
                input_tokens=int(getattr(raw_usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(raw_usage, "output_tokens", 0) or 0),
                cache_read_tokens=int(
                    getattr(raw_usage, "cache_read_input_tokens", 0)
                    or getattr(raw_usage, "cache_read_tokens", 0)
                    or 0
                ),
            )

        content_blocks = getattr(raw, "content", None)
        if isinstance(content_blocks, list):
            for block in content_blocks:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    tool_name = str(getattr(block, "name", "unknown"))
                    tool_input = getattr(block, "input", {}) or {}
                    tool_calls.append(
                        ToolCallRecord(
                            tool_name=tool_name,
                            tool_input=(
                                dict(tool_input)
                                if isinstance(tool_input, dict)
                                else {"value": tool_input}
                            ),
                            tool_response="[TOOL_USE]",
                            model_tier=model_tier,
                        )
                    )

        return response_text, usage, tool_calls

    def _resolve_model_id(self, model_tier: ModelTier) -> str:
        """Resolve model ID from runtime config or defaults."""
        cfg = self._runtime_config
        if cfg is not None and hasattr(cfg, "model_id_for_tier"):
            return str(cfg.model_id_for_tier(model_tier))
        return _MODEL_IDS[model_tier]

    def _fire_hook(
        self,
        event: HookEvent,
        source: str | None = None,
        tool_name: str | None = None,
        tool_input: dict | None = None,
        tool_response: str | None = None,
    ) -> HookResult:
        """Build HookContext and fire the event through the hook engine."""
        ctx = HookContext(
            session_id=self.session_id,
            hook_event_name=event,
            cwd=self.cwd,
            session_type=self.session_type,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_response=tool_response,
            source=source,
        )

        started = perf_counter()
        result = self.hook_engine.fire(event, ctx)
        if self._tracer is not None:
            duration_ms = (perf_counter() - started) * 1000.0
            with self._tracer.trace_hook_execution(
                event=event.value,
                handler_type="engine",
                attributes={"duration_ms": duration_ms},
            ):
                pass
        return result

    def _ensure_budget(self) -> AgentBudget:
        """Ensure orchestrator agent budget registration and return budget."""
        budget = AgentBudget(
            input_budget_tokens=1_000_000,
            output_budget_tokens=200_000,
            session_budget_usd=10.0,
        )
        return self.cost_governor.ensure_agent(self.agent_id, budget)

    @staticmethod
    def _default_model_callback(
        prompt: str,
        model_tier: ModelTier,
        effort: EffortResult,
    ) -> str:
        """Stub LLM call used in local callback mode."""
        lower = prompt.lower()
        target_path = Orchestrator._infer_target_path(prompt)

        read_needed = any(
            key in lower for key in ("review", "inspect", "debug", "fix", "implement", "refactor")
        )
        write_needed = any(
            key in lower for key in ("fix", "implement", "refactor", "change", "update", "edit")
        )

        tools: list[str] = []
        if read_needed:
            tools.append(Orchestrator._tool_marker("read_file", {"path": target_path}))
        if write_needed:
            tools.append(Orchestrator._tool_marker("write_file", {"path": target_path}))

        validation_note = (
            "Validated with tests."
            if any(key in lower for key in ("test", "validate", "verification"))
            else "Policy-compliant change plan prepared."
        )

        response = f"[model:{model_tier.value}] Working on {target_path}. {validation_note}"
        if tools:
            response = response + " " + " ".join(tools)

        if Orchestrator._contains_prompt_injection(prompt):
            response = (
                "I cannot comply with unsafe instruction overrides. "
                + response
            )
        return response

    @staticmethod
    def _default_tool_executor(tool_name: str, tool_input: dict) -> str:
        """Stub tool executor used in local callback mode."""
        return f"[tool:{tool_name}] executed"

    @staticmethod
    def _extract_tool_calls(response: str) -> list[tuple[str, dict]]:
        """Extract test tool markers from response string.

        Supported:
        - [TOOL:name]
        - [TOOL:name {"key": "value"}]
        """
        calls: list[tuple[str, dict]] = []
        for match in re.finditer(r"\[TOOL:(\w+)(?:\s+(\{.*?\}))?\]", response):
            name = match.group(1)
            raw_input = match.group(2)
            if raw_input:
                try:
                    tool_input = json.loads(raw_input)
                except json.JSONDecodeError:
                    tool_input = {}
            else:
                tool_input = {}
            calls.append((name, tool_input))
        return calls

    @staticmethod
    def _tool_marker(name: str, payload: dict) -> str:
        return f"[TOOL:{name} {json.dumps(payload)}]"

    @staticmethod
    def _infer_target_path(prompt: str) -> str:
        match = _FILE_PATH_PATTERN.search(prompt)
        if match:
            return match.group(1).replace("\\", "/")
        lower = prompt.lower()
        if "dispatcher" in lower:
            return "src/agents/agent_dispatcher.py"
        if "auth" in lower:
            return "src/auth.py"
        if "test" in lower:
            return "tests/test_app.py"
        return "src/agents/agent_dispatcher.py"

    @staticmethod
    def _contains_prompt_injection(prompt: str) -> bool:
        return any(pattern.search(prompt) for pattern in _PROMPT_INJECTION_PATTERNS)

    @staticmethod
    def _sanitize_request(request: str) -> tuple[str, bool]:
        sanitized = request
        matched = False
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(sanitized):
                matched = True
                sanitized = pattern.sub(" ", sanitized)
        if _CANARY_TOKEN_PATTERN.search(sanitized):
            sanitized = _CANARY_TOKEN_PATTERN.sub(" ", sanitized)
        sanitized = re.sub(r"\s+", " ", sanitized).strip()
        if not sanitized:
            sanitized = "Proceed with a policy-compliant implementation plan."
        return sanitized, matched

    @staticmethod
    def _response_contains_refusal(response: str) -> bool:
        return _REFUSAL_PATTERN.search(response) is not None
