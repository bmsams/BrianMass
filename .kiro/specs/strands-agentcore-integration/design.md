# Design Document: Strands AgentCore Integration

## Overview

This document describes how to replace every stub callback in Brainmass v3 with a real
Strands Agents SDK or Amazon Bedrock AgentCore call. The guiding principle is:

> **Real SDK calls become the default implementation of each pluggable callback. Tests inject
> lightweight stubs. The SDK is never imported at module level — only inside the default
> callback functions, guarded by a try/except that raises `RuntimeError` when the package is
> absent.**

There are **15 integration points** across 10 source files. They are ordered below from
lowest-level (types, cost, observability) to highest-level (orchestrator, runtime) so that
each task can build on the previous one.

---

## Architecture

### Callback-Default Pattern

Every integration point follows the same pattern:

```python
# --- Production integration point ---
def _production_agent_callback(
    agent_def: AgentDefinition,
    task: str,
    budget: AgentBudget,
) -> dict:
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for production mode.") from exc

    model = BedrockModel(model_id=_BEDROCK_MODEL_IDS[agent_def.model])
    agent = Agent(model=model, system_prompt=agent_def.system_prompt)
    result = agent(task)
    return _normalize_strands_result(result)
```

The constructor then does:

```python
self._agent_callback = agent_callback or _production_agent_callback
```

Tests inject a stub:

```python
manager = SubagentManager(agent_callback=lambda d, t, b: {"summary": "ok", ...})
```

### SDK Availability Guard

All SDK imports are **lazy** (inside functions, not at module top-level). This ensures:
- `import src.agents.subagent_manager` never fails without the SDK.
- Only calling the production path (no injected callback) raises `RuntimeError`.

### Inheritance Guard Pattern

For classes that need to extend an SDK base class (`HookProvider`,
`SummarizingConversationManager`), we use a conditional base:

```python
try:
    from strands.hooks import HookProvider as _StrandsHookProvider
    _STRANDS_HOOKS_AVAILABLE = True
except ImportError:
    _StrandsHookProvider = object  # type: ignore[assignment,misc]
    _STRANDS_HOOKS_AVAILABLE = False

class BrainmassHookEngine(_StrandsHookProvider):  # type: ignore[misc]
    ...
```

### Canonical Bedrock Model IDs

All integration points use cross-region inference profile IDs (the `us.` prefix), consistent
with the existing `orchestrator.py` `_MODEL_IDS` dict. A shared constant is defined in each
module (not in `core.py` since these are SDK-specific, not domain types):

```python
_BEDROCK_MODEL_IDS: dict[str, str] = {
    "sonnet":  "us.anthropic.claude-sonnet-4-5-v1:0",
    "opus":    "us.anthropic.claude-opus-4-6-v1:0",
    "haiku":   "us.anthropic.claude-haiku-4-5-v1:0",
    "inherit": "us.anthropic.claude-sonnet-4-5-v1:0",
    # Resolved canonical IDs (from AgentLoader.MODEL_MAP) also accepted:
    "claude-sonnet-4-5-20250929": "us.anthropic.claude-sonnet-4-5-v1:0",
    "claude-opus-4-6":            "us.anthropic.claude-opus-4-6-v1:0",
    "claude-haiku-4-5-20251001":  "us.anthropic.claude-haiku-4-5-v1:0",
}
```

> **QA note**: `orchestrator.py` already uses `us.anthropic.claude-*-v1:0` IDs.
> `agent_dispatcher.py` uses `claude-*` (without `us.` prefix). The production callbacks
> will normalise all aliases through `_BEDROCK_MODEL_IDS` to the `us.` form.

---

## Components and Interfaces

### 1. Orchestrator (`src/orchestrator/orchestrator.py`)

**Change**: `use_production_agent` defaults to `True` when no `model_callback` is injected.

```python
def __init__(self, ..., model_callback=None, use_production_agent=None):
    if use_production_agent is None:
        use_production_agent = (model_callback is None)
    self._model_callback = model_callback or self._default_model_callback
    self._use_production_agent = use_production_agent
```

`_execute_with_strands` already exists and is correct. The only change is making it the
default path when no `model_callback` is injected.

**Strands API** (already in the file, confirmed correct):
```python
from strands import Agent
from strands.models.bedrock import BedrockModel

model = BedrockModel(model_id=model_id)
agent = Agent(
    model=model,
    system_prompt=self._system_prompt,
    hooks=[self.hook_engine],
)
raw = agent(request)
```

**QA finding**: The existing `_execute_with_strands` is already production-ready. The only
gap is that `use_production_agent` defaults to `False`. Changing the default to `True` when
no `model_callback` is injected is the only required change.

---

### 2. SubagentManager (`src/agents/subagent_manager.py`)

**Change**: Replace `_default_agent_callback` and `_default_as_tool_callback` no-ops with
real Strands implementations.

```python
# --- Production integration point ---
def _production_agent_callback(
    agent_def: AgentDefinition,
    task: str,
    budget: AgentBudget,
) -> dict:
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for production subagent execution.") from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(task)
    return _normalize_strands_result(raw)


# --- Production integration point ---
def _production_as_tool_callback(agent_def: AgentDefinition) -> dict:
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for Agent.as_tool().") from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    # Agent.as_tool() returns a callable tool descriptor
    tool = agent.as_tool(
        name=f"agent:{agent_def.name}",
        description=agent_def.description,
    )
    return {"name": f"agent:{agent_def.name}", "description": agent_def.description, "tool": tool}
```

**`_normalize_strands_result` helper** (shared across SubagentManager, AgentDispatcher,
LoopRunner — defined once per module):

```python
def _normalize_strands_result(raw: object) -> dict:
    """Extract a standard result dict from any Strands Agent response shape."""
    # Text extraction
    if isinstance(raw, str):
        response_text = raw
    else:
        response_text = str(
            getattr(raw, "output", None)
            or getattr(raw, "text", None)
            or raw
        )

    # Usage extraction
    usage = getattr(raw, "usage", None)
    tokens: dict = {"input": 0, "output": 0, "cache_read": 0}
    if usage is not None:
        tokens = {
            "input": int(getattr(usage, "input_tokens", 0) or 0),
            "output": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_read": int(
                getattr(usage, "cache_read_input_tokens", 0)
                or getattr(usage, "cache_read_tokens", 0)
                or 0
            ),
        }

    # Tool use extraction
    tools_used: list[str] = []
    files_modified: list[str] = []
    content_blocks = getattr(raw, "content", None)
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if getattr(block, "type", None) == "tool_use":
                tool_name = str(getattr(block, "name", "unknown"))
                tools_used.append(tool_name)
                tool_input = getattr(block, "input", {}) or {}
                if isinstance(tool_input, dict) and "file_path" in tool_input:
                    files_modified.append(str(tool_input["file_path"]))

    stop_reason = getattr(raw, "stop_reason", "end_turn") or "end_turn"
    exit_reason = "complete" if stop_reason == "end_turn" else stop_reason

    return {
        "summary": response_text,
        "tokens_consumed": tokens,
        "tools_used": tools_used,
        "files_modified": files_modified,
        "exit_reason": exit_reason,
        "turns_used": int(getattr(raw, "turns", 0) or 0),
    }
```

---

### 3. TeamManager (`src/agents/team_manager.py`)

**Change**: Replace `_default_teammate_callback` with a real Strands `Agent` invocation.

> **QA finding**: `strands.multiagent.Swarm` is referenced in the architecture context but
> its exact Python API is not confirmed in the codebase. The safe approach is to use a plain
> `strands.Agent` call (which is confirmed) as the default, with a comment noting that
> `Swarm` can be substituted when its API is verified. This avoids a broken production path.

```python
# --- Production integration point ---
def _production_teammate_callback(
    agent_def: AgentDefinition,
    task: str,
    budget: AgentBudget,
) -> dict:
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for production teammate execution.") from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(task)
    return _normalize_strands_result(raw)

    # --- Production integration point (Swarm upgrade path) ---
    # When strands.multiagent.Swarm API is confirmed, replace the above with:
    # from strands.multiagent import Swarm
    # swarm = Swarm(agents=[agent])
    # raw = swarm(task)
```

---

### 4. LoopRunner (`src/agents/loop_runner.py`)

**Change**: Replace `_default_agent_callback` with a real Strands `Agent` invocation.

> **QA finding**: `strands.multiagent.GraphBuilder` is referenced in the architecture context
> but its exact Python API is not confirmed in the codebase. The safe approach is to use a
> plain `strands.Agent` call as the default, with a comment noting the `GraphBuilder` upgrade
> path. The review→fix conditional logic is preserved in the `LoopRunner.run()` loop itself
> (which already implements the conditional edge semantics via `acceptance_met`).

```python
# --- Production integration point ---
def _production_loop_callback(
    agent_def: AgentDefinition,
    context_str: str,
    budget: AgentBudget,
) -> dict:
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for production loop execution.") from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(context_str)
    result = _normalize_strands_result(raw)
    # LoopRunner interprets acceptance_met from the response text
    result["acceptance_met"] = "ACCEPTANCE_MET" in result["summary"].upper()
    result["learnings"] = []  # Learnings extracted by LoopRunner from summary
    return result

    # --- Production integration point (GraphBuilder upgrade path) ---
    # When strands.multiagent.GraphBuilder API is confirmed, replace with:
    # from strands.multiagent import GraphBuilder
    # graph = GraphBuilder().add_node("implement", agent).add_node("review", reviewer)
    #     .add_edge("implement", "review")
    #     .add_conditional_edges("review", lambda r: "__end__" if r.get("acceptance_met") else "implement")
    #     .set_entry_point("implement").build()
    # raw = graph(context_str)
```

---

### 5. CompoundLoopOrchestrator (`src/agents/compound_loop.py`)

**Change**: Replace `_default_stage_callback` with a real Strands `Agent` invocation.

> **QA finding**: `strands.multiagent.Workflow` is referenced in the architecture context but
> its exact Python API is not confirmed. Use plain `strands.Agent` as the default with a
> Workflow upgrade comment.

```python
# --- Production integration point ---
def _production_stage_callback(
    agent_def: AgentDefinition,
    loop_context: LoopContext,
    budget: AgentBudget,
) -> dict:
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for production stage execution.") from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(loop_context.current_task)
    result = _normalize_strands_result(raw)
    result["output"] = {"result": result["summary"]}
    result["acceptance_met"] = result["exit_reason"] == "complete"
    return result

    # --- Production integration point (Workflow upgrade path) ---
    # from strands.multiagent import Workflow
    # workflow = Workflow(agents=[agent])
    # raw = workflow(loop_context.current_task)
```

---

### 6. BrainmassHookEngine (`src/hooks/hook_engine.py`)

**Change**: Conditionally inherit from `strands.hooks.HookProvider` and implement
`register_hooks`.

> **QA finding**: The architecture context (section 5.4) shows `HookProvider` with
> `pre_tool_use` / `post_tool_use` methods. The Strands SDK 1.27+ uses event-based
> `HookRegistry.add_callback(EventClass, handler)`. We implement both the method-based
> interface (for compatibility) and the registry-based interface.

```python
try:
    from strands.hooks import HookProvider as _StrandsHookProvider
    _STRANDS_HOOKS_AVAILABLE = True
except ImportError:
    _StrandsHookProvider = object  # type: ignore[assignment,misc]
    _STRANDS_HOOKS_AVAILABLE = False


class BrainmassHookEngine(_StrandsHookProvider):  # type: ignore[misc]

    def register_hooks(self, registry: object) -> None:
        """Register with a Strands HookRegistry.

        Requirements: 6.2
        """
        # --- Production integration point ---
        if not _STRANDS_HOOKS_AVAILABLE:
            return
        try:
            from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent
            registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)  # type: ignore[union-attr]
            registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)    # type: ignore[union-attr]
        except (ImportError, AttributeError) as exc:
            logger.warning("Could not register Strands hook callbacks: %s", exc)

    def _on_before_tool_call(self, event: object) -> None:
        """Translate Strands BeforeToolCallEvent → HookEvent.PRE_TOOL_USE."""
        ctx = HookContext(
            session_id=str(getattr(event, "session_id", "")),
            hook_event_name=HookEvent.PRE_TOOL_USE,
            cwd=str(getattr(event, "cwd", ".")),
            session_type="interactive",
            tool_name=str(getattr(event, "tool_name", None) or ""),
            tool_input=getattr(event, "tool_input", None),
        )
        self.fire(HookEvent.PRE_TOOL_USE, ctx)

    def _on_after_tool_call(self, event: object) -> None:
        """Translate Strands AfterToolCallEvent → HookEvent.POST_TOOL_USE."""
        ctx = HookContext(
            session_id=str(getattr(event, "session_id", "")),
            hook_event_name=HookEvent.POST_TOOL_USE,
            cwd=str(getattr(event, "cwd", ".")),
            session_type="interactive",
            tool_name=str(getattr(event, "tool_name", None) or ""),
            tool_response=str(getattr(event, "tool_response", "") or ""),
        )
        self.fire(HookEvent.POST_TOOL_USE, ctx)

    # Strands HookProvider method-based interface (section 5.4 of architecture_context.md)
    def pre_tool_use(self, tool_name: str, tool_input: object) -> None:
        """Called by Strands SDK before tool execution."""
        ctx = HookContext(
            session_id="",
            hook_event_name=HookEvent.PRE_TOOL_USE,
            cwd=".",
            session_type="interactive",
            tool_name=tool_name,
            tool_input=tool_input if isinstance(tool_input, dict) else {},
        )
        self.fire(HookEvent.PRE_TOOL_USE, ctx)

    def post_tool_use(self, tool_name: str, tool_input: object, tool_response: object) -> None:
        """Called by Strands SDK after tool execution."""
        ctx = HookContext(
            session_id="",
            hook_event_name=HookEvent.POST_TOOL_USE,
            cwd=".",
            session_type="interactive",
            tool_name=tool_name,
            tool_response=str(tool_response),
        )
        self.fire(HookEvent.POST_TOOL_USE, ctx)
```

---

### 7. ContextManager (`src/context/context_manager.py`)

**Change**: Conditionally inherit from `SummarizingConversationManager` and use real
`AgentCoreMemoryStore` as default when `bedrock-agentcore` is installed.

```python
try:
    from strands.session import SummarizingConversationManager as _StrandsBase
    _STRANDS_SESSION_AVAILABLE = True
except ImportError:
    _StrandsBase = object  # type: ignore[assignment,misc]
    _STRANDS_SESSION_AVAILABLE = False


class ContextManager(_StrandsBase):  # type: ignore[misc]
    def __init__(self, session_id: str, window_size: int = 200_000,
                 memory_store: Optional[MemoryStore] = None, ...):
        if _STRANDS_SESSION_AVAILABLE:
            # --- Production integration point ---
            super().__init__(
                max_tokens=window_size,
                summarization_model="us.anthropic.claude-haiku-4-5-v1:0",
            )

        # Default to AgentCoreMemoryStore when bedrock-agentcore is available
        if memory_store is None:
            try:
                # --- Production integration point ---
                from bedrock_agentcore.memory import MemoryClient
                memory_store = AgentCoreMemoryStore(MemoryClient())
            except ImportError:
                memory_store = InMemoryMemoryStore()
        self.memory_client: MemoryStore = memory_store
        ...
```

---

### 8. LearningStore (`src/agents/learning_store.py`)

**Change**: Replace `_default_memory_callback` with a real `MemoryClient` call.

```python
# --- Production integration point ---
def _production_memory_callback(learning_id: str, learning_dict: dict) -> bool:
    try:
        from bedrock_agentcore.memory import MemoryClient
    except ImportError as exc:
        raise RuntimeError("bedrock-agentcore is required for production memory persistence.") from exc
    client = MemoryClient()
    # Use the AgentCoreMemoryStore adapter for consistent API usage
    store = AgentCoreMemoryStore(client)
    store.create_memory(
        namespace="learning-store",
        content=learning_dict,
    )
    return True
```

> **QA note**: `AgentCoreMemoryStore` already exists in `context_manager.py` and handles the
> `create_event` / `create_or_get_memory` API differences. `LearningStore` should import and
> reuse it rather than duplicating the client logic.

---

### 9. SessionTeleporter (`src/session/teleporter.py`)

**Change**: Auto-select production adapters based on environment.

```python
def __init__(
    self,
    storage_backend=None,
    load_backend=None,
    storage_adapter: Optional[SessionStorageAdapter] = None,
    memory_adapter: Optional[SessionMemoryAdapter] = None,
) -> None:
    # Auto-select storage adapter
    if storage_adapter is None:
        bucket = os.environ.get("S3_SESSION_BUCKET")
        if bucket:
            # --- Production integration point ---
            storage_adapter = S3SessionStorageAdapter(bucket=bucket)
        else:
            logger.warning(
                "S3_SESSION_BUCKET not set; using in-memory session storage. "
                "Set S3_SESSION_BUCKET for production use."
            )
            storage_adapter = InMemorySessionStorageAdapter()

    # Auto-select memory adapter
    if memory_adapter is None:
        try:
            # --- Production integration point ---
            from bedrock_agentcore.memory import MemoryClient  # noqa: F401
            memory_adapter = AgentCoreSessionMemoryAdapter()
        except ImportError:
            memory_adapter = InMemorySessionMemoryAdapter()
    ...
```

---

### 10. AgentCore Runtime (`src/runtime/app.py`)

**Status**: Already correct. `BedrockAgentCoreApp` is already imported and `@app.entrypoint`
is already applied. The only change needed is ensuring `_build_orchestrator` passes
`use_production_agent=True` explicitly (it already does via the `Orchestrator` default change
in integration point 1).

**QA finding**: `app.py` already has:
```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload, session):
    ...
```
This is production-ready. No changes required beyond the Orchestrator default fix.

---

### 11. Observability Tracer (`src/observability/tracer.py`)

**Change**: Wire `strands.telemetry.get_tracer` as the backend when available. Add
`backend_tracer` parameter to `BrainmassTracer`.

```python
# In tracer.py get_tracer():
def get_tracer(name: str = "brainmass") -> BrainmassTracer:
    global _default_tracer
    with _lock:
        if _default_tracer is None:
            try:
                # --- Production integration point ---
                from strands.telemetry import get_tracer as _strands_get_tracer
                _strands_backend = _strands_get_tracer(name)
                _default_tracer = BrainmassTracer(
                    service_name=name,
                    backend_tracer=_strands_backend,
                )
            except ImportError:
                _default_tracer = BrainmassTracer(service_name=name)
        return _default_tracer
```

`BrainmassTracer` in `instrumentation.py` gains an optional `backend_tracer` parameter:

```python
class BrainmassTracer:
    def __init__(self, service_name: str = "brainmass",
                 backend_tracer: Optional[object] = None) -> None:
        self._service_name = service_name
        self._backend_tracer = backend_tracer  # strands tracer or None
        self._spans: list[dict] = []
        ...

    @contextmanager
    def trace_model_interaction(self, model_tier: str, agent_id: str):
        if self._backend_tracer is not None:
            # --- Production integration point ---
            with self._backend_tracer.start_as_current_span(
                f"model.{model_tier}", attributes={"agent.id": agent_id}
            ):
                yield
        else:
            yield
```

---

### 12. CostGovernor (`src/cost/cost_governor.py`)

**Change**: Lazily construct `BedrockModel` instances per tier and expose
`get_bedrock_model(tier)`.

```python
class CostGovernor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, _AgentUsage] = {}
        self._bedrock_models: dict[ModelTier, object] = {}
        self._models_initialized: bool = False

    def _ensure_models(self) -> None:
        """Lazily initialize BedrockModel instances for each tier."""
        if self._models_initialized:
            return
        try:
            # --- Production integration point ---
            from strands.models.bedrock import BedrockModel
            _ids = {
                ModelTier.OPUS:   "us.anthropic.claude-opus-4-6-v1:0",
                ModelTier.SONNET: "us.anthropic.claude-sonnet-4-5-v1:0",
                ModelTier.HAIKU:  "us.anthropic.claude-haiku-4-5-v1:0",
            }
            for tier, model_id in _ids.items():
                self._bedrock_models[tier] = BedrockModel(model_id=model_id)
        except ImportError:
            pass  # SDK absent — models stay empty
        finally:
            self._models_initialized = True

    def get_bedrock_model(self, tier: ModelTier) -> Optional[object]:
        """Return the BedrockModel for the given tier, or None if SDK absent."""
        self._ensure_models()
        return self._bedrock_models.get(tier)
```

---

### 13. AgentDispatcher (`src/agents/agent_dispatcher.py`)

**Change**: Replace all four no-op default callbacks with real implementations.

> **QA finding**: The `hooks_register_callback` is a no-op because the `AgentDispatcher`
> already has `hook_engine` injected. The production `_production_hooks_register` should
> call `hook_engine.register_scoped()` directly. However, since `hook_engine` is an instance
> attribute, the production callback must be a closure or method, not a module-level function.
> The cleanest approach: when `hooks_register_callback` is `None` AND `hook_engine` is not
> `None`, use `hook_engine.register_scoped` directly in `_step6_hooks`.

```python
# --- Production integration point ---
def _production_agent_loop(
    agent_def: AgentDefinition,
    task: str,
    system_prompt: str,
    budget: AgentBudget,
) -> dict:
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for production agent execution.") from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=system_prompt)
    raw = agent(task)
    return _normalize_strands_result(raw)


# --- Production integration point ---
def _production_mcp_start(mcp_servers: dict) -> Callable[[], None]:
    try:
        from strands_agents_tools import MCPClient  # strands-agents-tools package
    except ImportError as exc:
        raise RuntimeError("strands-agents-tools is required for MCP server support.") from exc
    clients = {name: MCPClient(**cfg) for name, cfg in mcp_servers.items()}
    for client in clients.values():
        client.start()
    def cleanup() -> None:
        for client in clients.values():
            try:
                client.stop()
            except Exception:
                pass
    return cleanup


# --- Production integration point ---
def _production_skills_load(skill_names: list[str]) -> str:
    from src.skills.skill_registry import SkillRegistry
    registry = SkillRegistry()
    parts = []
    for name in skill_names:
        content = registry.get_skill_content(name)
        if content:
            parts.append(content)
    return "\n\n".join(parts)
```

For `_step6_hooks`, the production path wires directly through `hook_engine`:

```python
def _step6_hooks(self, ctx: DispatchContext) -> None:
    if not ctx.agent_def.hooks:
        return
    if self._hooks_register_callback is not None:
        # Injected callback (tests or custom wiring)
        ctx.hooks_cleanup = self._hooks_register_callback(ctx.agent_def.hooks, ctx.agent_id)
    elif self._hook_engine is not None:
        # --- Production integration point ---
        # Wire directly through the injected hook_engine
        from src.types.core import HookEvent, HookDefinition
        hooks_by_event: dict[HookEvent, list[HookDefinition]] = {}
        for key, defs in ctx.agent_def.hooks.items():
            try:
                event = HookEvent(key) if isinstance(key, str) else key
                hooks_by_event[event] = defs
            except ValueError:
                pass
        if hooks_by_event:
            ctx.hooks_cleanup = self._hook_engine.register_scoped(
                hooks=hooks_by_event,
                scope_id=ctx.agent_id,
                scope="subagent_frontmatter",
            )
```

---

### 14. AgentLoader / AgentRegistry (`src/agents/agent_loader.py`, `src/agents/agent_registry.py`)

**Change**: Add `to_strands_agent()` to both classes.

```python
# AgentLoader.to_strands_agent()
def to_strands_agent(self, agent_def: AgentDefinition) -> object:
    """Construct a real strands.Agent from an AgentDefinition.

    Requirements: 14.1
    """
    # --- Production integration point ---
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents is required for to_strands_agent(). "
            "Install with: pip install strands-agents"
        ) from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)

    if agent_def.tools:
        try:
            from strands_agents_tools import get_tools
            tools = get_tools(agent_def.tools)
        except (ImportError, Exception):
            tools = []
    else:
        try:
            from strands_agents_tools import TOOLS as tools
        except ImportError:
            tools = []

    return Agent(
        model=model,
        system_prompt=agent_def.system_prompt or "",
        tools=tools,
    )
```

```python
# AgentRegistry.to_strands_agent()
def to_strands_agent(self, name: str) -> object:
    """Look up an agent by name and return a real strands.Agent.

    Requirements: 14.2
    """
    agent_def = self.get(name)
    if agent_def is None:
        raise KeyError(f"Agent '{name}' not found in registry.")
    return self._loader.to_strands_agent(agent_def)
```

---

### 15. HookHandlers (`src/hooks/handlers/prompt.py`, `src/hooks/handlers/agent.py`)

**Change**: Add production default callbacks to `PromptHandler` and `AgentHandler`.

```python
# prompt.py — production default model callback
def _production_model_callback(prompt: str) -> str:
    """Invoke Haiku via Strands Agent for prompt-type hook evaluation."""
    # --- Production integration point ---
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
    except ImportError as exc:
        raise RuntimeError("strands-agents is required for PromptHandler production mode.") from exc
    model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-v1:0")
    agent = Agent(model=model)
    return str(agent(prompt))


# agent.py — production default agent callback
def _production_agent_callback(context_prompt: str, agent_config: Optional[dict]) -> str:
    """Invoke a Haiku agent with Read/Grep/Glob tools for agent-type hook evaluation."""
    # --- Production integration point ---
    try:
        from strands import Agent
        from strands.models.bedrock import BedrockModel
        from strands_agents_tools import file_read, grep, glob
    except ImportError as exc:
        raise RuntimeError("strands-agents and strands-agents-tools are required for AgentHandler.") from exc
    model = BedrockModel(model_id="us.anthropic.claude-haiku-4-5-v1:0")
    agent = Agent(model=model, tools=[file_read, grep, glob])
    return str(agent(context_prompt))
```

`PromptHandler` and `AgentHandler` constructors are updated to use these as defaults:

```python
class PromptHandler:
    def __init__(self) -> None:
        self._model_callback: Optional[ModelCallback] = _production_model_callback

class AgentHandler:
    def __init__(self) -> None:
        self._agent_callback: Optional[AgentCallback] = _production_agent_callback
```

---

## Data Models

No new types are added to `src/types/core.py`. All integration points use existing types.

The `_normalize_strands_result` helper function is defined once per module that needs it
(SubagentManager, AgentDispatcher, LoopRunner). It is not shared via a common module to
avoid circular imports.

The `_BEDROCK_MODEL_IDS` dict is defined in each module that needs it (not in `core.py`
since it is SDK-specific infrastructure, not a domain type).

`BrainmassTracer` in `src/observability/instrumentation.py` gains one new optional
constructor parameter: `backend_tracer: Optional[object] = None`.

---

## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions
of a system — essentially, a formal statement about what the system should do. Properties
serve as the bridge between human-readable specifications and machine-verifiable correctness
guarantees.

---

### Property 1: Callback injection always overrides production path

*For any* integration point class and *any* injected callback, calling the class's primary
execution method should invoke the injected callback and never attempt to import the Strands
SDK.

**Validates: Requirements 1.3, 2.4, 3.2, 4.2, 5.2, 13.5, 15.3, 16.1**

---

### Property 2: `_normalize_strands_result` handles all raw output shapes

*For any* raw output value (string, dict with `output`/`text` keys, object with `output`
attribute, object with `usage` attribute, or arbitrary object), `_normalize_strands_result`
should return a dict containing all required keys (`summary`, `tokens_consumed`, `tools_used`,
`files_modified`, `exit_reason`, `turns_used`) where `summary` is always a string.

**Validates: Requirements 1.4, 13.6**

---

### Property 3: Tool call event translation is correct for all events

*For any* mock `BeforeToolCallEvent` with a `tool_name` attribute, `_on_before_tool_call`
should produce a `HookContext` with `hook_event_name == HookEvent.PRE_TOOL_USE` and the same
`tool_name`. *For any* mock `AfterToolCallEvent`, `_on_after_tool_call` should produce a
`HookContext` with `hook_event_name == HookEvent.POST_TOOL_USE`.

**Validates: Requirements 6.3, 6.4**

---

### Property 4: BedrockModel coverage — all tiers have a model when SDK is present

*For all* `ModelTier` enum values, when `CostGovernor._ensure_models()` is called with a
mock `BedrockModel` constructor, `get_bedrock_model(tier)` should return a non-None value
for every tier in `ModelTier`.

**Validates: Requirements 12.1, 12.2**

---

### Property 5: `to_strands_agent` succeeds for all valid AgentDefinitions

*For any* `AgentDefinition` with a valid `model` alias (one of `sonnet`, `opus`, `haiku`,
`inherit`) and any `system_prompt`, `AgentLoader.to_strands_agent(agent_def)` should return
a non-None object when called with a mock `Agent` constructor, and raise `RuntimeError` when
the import fails.

**Validates: Requirements 14.1, 14.3**

---

### Property 6: Missing SDK + no callback raises RuntimeError

*For all* integration point classes, when the production callback is used and the SDK import
raises `ImportError`, the primary execution method should raise `RuntimeError` rather than
returning a silent no-op result.

**Validates: Requirements 1.5, 2.5, 3.4, 4.4, 5.4, 8.4, 14.3, 15.4, 16.4**

---

### Property 7: SessionTeleporter adapter selection is environment-driven

*For any* combination of `S3_SESSION_BUCKET` env var (set/unset) and `bedrock-agentcore`
availability (installed/absent), `SessionTeleporter()` constructed with no arguments should
select the correct adapter pair: S3+AgentCore when both available, InMemory otherwise.

**Validates: Requirements 9.1, 9.2, 9.4**

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `strands` not installed, no callback | `RuntimeError("strands-agents is required...")` |
| `bedrock-agentcore` not installed, no callback | `RuntimeError("bedrock-agentcore is required...")` |
| `strands_agents_tools` not installed, MCP needed | `RuntimeError("strands-agents-tools is required...")` |
| Strands `Agent` raises during execution | Caught in dispatcher/manager, `exit_reason="error"` |
| `BedrockModel` construction fails (bad model ID) | Propagated as `RuntimeError` |
| `MemoryClient` unavailable | Falls back to `InMemoryMemoryStore` with `logger.warning` |
| `S3_SESSION_BUCKET` not set | Falls back to `InMemorySessionStorageAdapter` with `logger.warning` |
| `register_hooks` called without SDK | No-op (guarded by `_STRANDS_HOOKS_AVAILABLE`) |
| `BrainmassTracer` backend unavailable | Falls back to standalone span store |

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required. They are complementary:
- **Unit tests** verify specific examples, edge cases, and error conditions.
- **Property tests** verify universal properties across many generated inputs.

### Unit Tests

Each integration point gets additions to its existing test file in `tests/unit/`. Tests
inject stub callbacks so no SDK is required:

```python
# tests/unit/test_subagent_manager.py — example addition
def test_execute_uses_injected_callback():
    called = []
    def stub(agent_def, task, budget):
        called.append((agent_def.name, task))
        return {"summary": "ok", "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
                "tools_used": [], "files_modified": [], "exit_reason": "complete", "turns_used": 1}
    manager = SubagentManager(agent_callback=stub)
    result = manager.execute(_make_agent_def(), "do work")
    assert called == [("test-agent", "do work")]
    assert result.exit_reason == "complete"

def test_missing_sdk_raises_runtime_error(monkeypatch):
    """When no callback is injected and SDK import fails, RuntimeError is raised."""
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "strands":
            raise ImportError("strands not installed")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)
    manager = SubagentManager()  # no callback injected
    with pytest.raises(RuntimeError, match="strands-agents is required"):
        manager.execute(_make_agent_def(), "task")
```

### Property-Based Tests (Hypothesis)

Property tests live in `tests/property/test_strands_integration_properties.py`.
Minimum 100 iterations per test.

```python
@pytest.mark.property
@given(raw=st.one_of(
    st.text(),
    st.fixed_dictionaries({"output": st.text(), "input_tokens": st.integers(0, 10000)}),
    st.just(None),
))
@settings(max_examples=100)
def test_normalize_strands_result_always_returns_valid_dict(raw):
    # Feature: strands-agentcore-integration, Property 2: normalize handles all shapes
    result = _normalize_strands_result(raw)
    assert isinstance(result["summary"], str)
    assert isinstance(result["tokens_consumed"], dict)
    assert isinstance(result["tools_used"], list)
    assert isinstance(result["files_modified"], list)
    assert isinstance(result["exit_reason"], str)
    assert isinstance(result["turns_used"], int)
```

### Property Test Configuration

- Minimum 100 iterations per property test (`@settings(max_examples=100)`)
- Each property test references its design document property in a comment
- Tag format: `# Feature: strands-agentcore-integration, Property N: <property_text>`
- Each correctness property is implemented by a single property-based test

### Test File Mapping

| Property | Test file | Test function |
|---|---|---|
| Property 1 | `tests/property/test_strands_integration_properties.py` | `test_callback_injection_overrides_production_path` |
| Property 2 | `tests/property/test_strands_integration_properties.py` | `test_normalize_strands_result_handles_all_shapes` |
| Property 3 | `tests/property/test_strands_integration_properties.py` | `test_tool_event_translation_correct` |
| Property 4 | `tests/property/test_strands_integration_properties.py` | `test_bedrock_model_coverage_all_tiers` |
| Property 5 | `tests/property/test_strands_integration_properties.py` | `test_to_strands_agent_valid_definitions` |
| Property 6 | `tests/property/test_strands_integration_properties.py` | `test_missing_sdk_raises_runtime_error` |
| Property 7 | `tests/property/test_strands_integration_properties.py` | `test_session_teleporter_adapter_selection` |
