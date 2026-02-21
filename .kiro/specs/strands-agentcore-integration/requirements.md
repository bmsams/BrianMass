# Requirements Document

## Introduction

Brainmass v3 currently stubs every Strands Agents SDK and Amazon Bedrock AgentCore call behind
pluggable callbacks. All production integration points are marked with
`# --- Production integration point ---` comments. The system runs and all tests pass without
the SDK installed, but it does not actually invoke AWS Bedrock or use the Strands SDK in
production.

This feature replaces every stub with a real Strands / AgentCore call while preserving the
pluggable-callback architecture so that unit tests continue to work without the SDK installed.
The real SDK calls become the **default** implementation of each callback; tests may still
inject lightweight stubs.

## Glossary

- **Strands_SDK**: The `strands-agents` Python package (`strands`, `strands.models.bedrock`,
  `strands.multiagent`, `strands.hooks`, `strands.session`, `strands.telemetry`).
- **AgentCore**: The `bedrock-agentcore` Python package (`bedrock_agentcore.runtime`,
  `bedrock_agentcore.memory`).
- **BedrockModel**: `strands.models.bedrock.BedrockModel` — wraps a Bedrock model ID and
  exposes it to a Strands `Agent`.
- **Strands_Agent**: `strands.Agent` — the core agent class that accepts a `BedrockModel`,
  system prompt, tools list, and hooks list.
- **Swarm**: `strands.multiagent.Swarm` — peer-topology multi-agent coordinator.
- **GraphBuilder**: `strands.multiagent.GraphBuilder` — builds conditional-edge agent graphs
  for self-improving loops.
- **Workflow**: `strands.multiagent.Workflow` — sequential pipeline of agent stages.
- **HookProvider**: `strands.hooks.HookProvider` — base class for Strands hook integration.
- **HookRegistry**: `strands.hooks.HookRegistry` — registry passed to `register_hooks()`.
- **SummarizingConversationManager**: `strands.session.SummarizingConversationManager` —
  LLM-driven context summarisation base class.
- **FileSessionManager**: `strands.session.FileSessionManager` — file-backed session
  persistence for Strands agents.
- **MemoryClient**: `bedrock_agentcore.memory.MemoryClient` — AgentCore long-term memory
  client with semantic retrieval.
- **BedrockAgentCoreApp**: `bedrock_agentcore.runtime.BedrockAgentCoreApp` — AgentCore
  runtime application class with `@app.entrypoint` decorator.
- **Pluggable_Callback**: A constructor parameter of type `Optional[Callable[...]]` that
  defaults to a real SDK call in production but can be overridden in tests.
- **Production_Integration_Point**: A code comment `# --- Production integration point ---`
  marking where the real SDK call lives.

---

## Requirements

### Requirement 1: Orchestrator — Real Strands Agent Execution

**User Story:** As a platform operator, I want the Orchestrator to use a real Strands `Agent`
backed by a `BedrockModel` as its default execution path, so that requests are actually
processed by AWS Bedrock in production.

#### Acceptance Criteria

1. WHEN `use_production_agent=True` is set on the `Orchestrator`, THE `Orchestrator` SHALL
   invoke `strands.Agent` with a `BedrockModel` constructed from the resolved model ID.
2. WHEN the `Orchestrator` is constructed without a `model_callback`, THE `Orchestrator` SHALL
   default `use_production_agent` to `True` and use the real Strands `Agent` path.
3. WHEN the `Orchestrator` is constructed with an explicit `model_callback`, THE `Orchestrator`
   SHALL use that callback instead of the Strands `Agent` (preserving testability).
4. WHEN the Strands `Agent` returns a response, THE `Orchestrator` SHALL normalise the output
   into `(response_text, ExecutionUsage, list[ToolCallRecord])` via `_normalize_strands_output`.
5. IF the `strands` package is not installed and `use_production_agent=True`, THEN THE
   `Orchestrator` SHALL raise a `RuntimeError` with a descriptive message.
6. THE `Orchestrator` SHALL pass `self.hook_engine` in the `hooks` list when constructing the
   Strands `Agent`, so lifecycle hooks fire during real execution.

---

### Requirement 2: SubagentManager — Real Agent.as_tool() Calls

**User Story:** As a platform operator, I want the `SubagentManager` to wrap subagents as real
Strands tools using `Agent.as_tool()`, so that hierarchical agent execution actually runs on
Bedrock.

#### Acceptance Criteria

1. WHEN `SubagentManager` is constructed without an `agent_callback`, THE `SubagentManager`
   SHALL default to a real implementation that constructs a `strands.Agent` and calls it with
   the task string.
2. WHEN `SubagentManager` is constructed without an `as_tool_callback`, THE `SubagentManager`
   SHALL default to a real implementation that calls `Agent.as_tool()` on a constructed
   `strands.Agent` to produce a tool descriptor.
3. WHEN a subagent executes, THE `SubagentManager` SHALL extract `tokens_consumed` from the
   Strands `Agent` response's `usage` attribute.
4. WHEN an `agent_callback` or `as_tool_callback` is explicitly injected, THE
   `SubagentManager` SHALL use the injected callback instead of the real SDK (preserving
   testability).
5. IF the `strands` package is not installed and no callback is injected, THEN THE
   `SubagentManager` SHALL raise a `RuntimeError` when `execute()` is called.

---

### Requirement 3: TeamManager — Real Strands Swarm

**User Story:** As a platform operator, I want the `TeamManager` to use `strands.multiagent.Swarm`
for peer-topology coordination, so that Agent Teams actually run on Bedrock.

#### Acceptance Criteria

1. WHEN `TeamManager` is constructed without a `teammate_callback`, THE `TeamManager` SHALL
   default to a real implementation that uses `strands.multiagent.Swarm` to coordinate
   teammates.
2. WHEN a `teammate_callback` is explicitly injected, THE `TeamManager` SHALL use the injected
   callback instead of the real Swarm (preserving testability).
3. WHEN the Swarm executes, THE `TeamManager` SHALL extract per-agent token usage from the
   Swarm result and record it with the `CostGovernor`.
4. IF the `strands` package is not installed and no callback is injected, THEN THE
   `TeamManager` SHALL raise a `RuntimeError` when `execute_team()` is called.

---

### Requirement 4: LoopRunner — Real GraphBuilder with Conditional Edges

**User Story:** As a platform operator, I want the `LoopRunner` to use
`strands.multiagent.GraphBuilder` with conditional edges for the review→fix cycle, so that
self-improving loops actually run on Bedrock.

#### Acceptance Criteria

1. WHEN `LoopRunner` is constructed without an `agent_callback`, THE `LoopRunner` SHALL
   default to a real implementation that builds a `GraphBuilder` graph with
   `implement → review → (FAIL → implement | PASS → END)` conditional edges.
2. WHEN an `agent_callback` is explicitly injected, THE `LoopRunner` SHALL use the injected
   callback instead of the real graph (preserving testability).
3. WHEN the graph executes an iteration, THE `LoopRunner` SHALL extract `learnings` and
   `acceptance_met` from the graph result.
4. IF the `strands` package is not installed and no callback is injected, THEN THE
   `LoopRunner` SHALL raise a `RuntimeError` when `run()` is called.

---

### Requirement 5: CompoundLoopOrchestrator — Real Strands Workflow

**User Story:** As a platform operator, I want the `CompoundLoopOrchestrator` to use
`strands.multiagent.Workflow` for sequential pipeline stages, so that compound loops actually
run on Bedrock.

#### Acceptance Criteria

1. WHEN `CompoundLoopOrchestrator` is constructed without a `stage_callback`, THE
   `CompoundLoopOrchestrator` SHALL default to a real implementation that uses
   `strands.multiagent.Workflow` to execute each stage.
2. WHEN a `stage_callback` is explicitly injected, THE `CompoundLoopOrchestrator` SHALL use
   the injected callback instead of the real Workflow (preserving testability).
3. WHEN the Workflow executes a stage, THE `CompoundLoopOrchestrator` SHALL extract
   `output`, `acceptance_met`, and `tokens_consumed` from the Workflow result.
4. IF the `strands` package is not installed and no callback is injected, THEN THE
   `CompoundLoopOrchestrator` SHALL raise a `RuntimeError` when `run()` is called.

---

### Requirement 6: HookEngine — Real Strands HookProvider

**User Story:** As a platform operator, I want `BrainmassHookEngine` to extend
`strands.hooks.HookProvider` and implement `register_hooks(registry)`, so that Strands-native
`BeforeToolCallEvent` and `AfterToolCallEvent` are bridged to our `PRE_TOOL_USE` and
`POST_TOOL_USE` events.

#### Acceptance Criteria

1. WHEN the `strands` package is installed, THE `BrainmassHookEngine` SHALL inherit from
   `strands.hooks.HookProvider`.
2. WHEN `register_hooks(registry)` is called, THE `BrainmassHookEngine` SHALL call
   `registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)` and
   `registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)`.
3. WHEN a `BeforeToolCallEvent` fires, THE `BrainmassHookEngine` SHALL translate it into a
   `HookEvent.PRE_TOOL_USE` and delegate to `self.fire()`.
4. WHEN an `AfterToolCallEvent` fires, THE `BrainmassHookEngine` SHALL translate it into a
   `HookEvent.POST_TOOL_USE` and delegate to `self.fire()`.
5. WHEN the `strands` package is not installed, THE `BrainmassHookEngine` SHALL fall back to
   a standalone base class so the system continues to function without the SDK.

---

### Requirement 7: ContextManager — Real SummarizingConversationManager

**User Story:** As a platform operator, I want `ContextManager` to extend
`strands.session.SummarizingConversationManager` and use `bedrock_agentcore.memory.MemoryClient`
for long-term learning persistence, so that context compaction uses real LLM summarisation.

#### Acceptance Criteria

1. WHEN the `strands` package is installed, THE `ContextManager` SHALL inherit from
   `strands.session.SummarizingConversationManager` and call `super().__init__()` with
   `max_tokens` and `summarization_model` parameters.
2. WHEN `compact()` is called in production mode, THE `ContextManager` SHALL delegate
   summarisation to the `SummarizingConversationManager` base class instead of the
   character-truncation stub.
3. WHEN `memory_store` is not injected and the `bedrock-agentcore` package is installed, THE
   `ContextManager` SHALL default to `AgentCoreMemoryStore` backed by a real `MemoryClient`.
4. WHEN `memory_store` is explicitly injected, THE `ContextManager` SHALL use the injected
   store (preserving testability).
5. WHEN the `strands` package is not installed, THE `ContextManager` SHALL fall back to the
   standalone implementation so tests continue to pass.

---

### Requirement 8: LearningStore — Real AgentCore MemoryClient

**User Story:** As a platform operator, I want `LearningStore` to use a real
`bedrock_agentcore.memory.MemoryClient` as its default `memory_callback`, so that learnings
are persisted to AgentCore long-term memory in production.

#### Acceptance Criteria

1. WHEN `LearningStore` is constructed without a `memory_callback`, THE `LearningStore` SHALL
   default to a real implementation that calls `MemoryClient.create_event()` to persist each
   learning.
2. WHEN a `memory_callback` is explicitly injected, THE `LearningStore` SHALL use the injected
   callback instead of the real `MemoryClient` (preserving testability).
3. WHEN `query()` is called in production mode, THE `LearningStore` SHALL use
   `MemoryClient.retrieve_memories()` with the semantic strategy for top-K retrieval.
4. IF the `bedrock-agentcore` package is not installed and no callback is injected, THEN THE
   `LearningStore` SHALL raise a `RuntimeError` when `add()` is called.

---

### Requirement 9: SessionTeleporter — Real Strands SessionManager with S3

**User Story:** As a platform operator, I want `SessionTeleporter` to use a real
`strands.session.FileSessionManager` (or S3-backed equivalent) as its default storage
backend, so that session state is actually persisted across surfaces.

#### Acceptance Criteria

1. WHEN `SessionTeleporter` is constructed without a `storage_adapter`, THE
   `SessionTeleporter` SHALL default to `S3SessionStorageAdapter` when the `S3_SESSION_BUCKET`
   environment variable is set.
2. WHEN `SessionTeleporter` is constructed without a `memory_adapter`, THE
   `SessionTeleporter` SHALL default to `AgentCoreSessionMemoryAdapter` backed by a real
   `MemoryClient` when the `bedrock-agentcore` package is installed.
3. WHEN `storage_adapter` or `memory_adapter` are explicitly injected, THE
   `SessionTeleporter` SHALL use the injected adapters (preserving testability).
4. IF neither `S3_SESSION_BUCKET` nor an injected adapter is available, THEN THE
   `SessionTeleporter` SHALL fall back to `InMemorySessionStorageAdapter` with a warning log.

---

### Requirement 10: AgentCore Runtime — Real BedrockAgentCoreApp

**User Story:** As a platform operator, I want `src/runtime/app.py` to wire a real
`BedrockAgentCoreApp` with `@app.entrypoint`, so that the system can be deployed and invoked
as an AgentCore agent.

#### Acceptance Criteria

1. THE `Runtime` SHALL import `bedrock_agentcore.runtime.BedrockAgentCoreApp` and instantiate
   `app = BedrockAgentCoreApp()`.
2. THE `Runtime` SHALL decorate the `invoke` function with `@app.entrypoint`.
3. WHEN the `bedrock-agentcore` package is not installed, THE `Runtime` SHALL raise a
   `RuntimeError` at import time with a descriptive message.
4. WHEN `invoke` is called by AgentCore, THE `Runtime` SHALL build an `Orchestrator` with
   `use_production_agent=True` and process the request.

---

### Requirement 11: Observability — Real Strands Telemetry Tracer

**User Story:** As a platform operator, I want `src/observability/tracer.py` to use
`strands.telemetry.get_tracer('brainmass-v3')` as the backend for the `BrainmassTracer`, so
that spans are emitted to the Strands telemetry pipeline.

#### Acceptance Criteria

1. WHEN the `strands` package is installed, THE `get_tracer()` function SHALL call
   `strands.telemetry.get_tracer('brainmass-v3')` and use the returned tracer as the backend.
2. WHEN the `strands` package is not installed, THE `get_tracer()` function SHALL fall back to
   the standalone `BrainmassTracer` implementation.
3. THE `BrainmassTracer` SHALL delegate `start_as_current_span` calls to the Strands tracer
   backend when available.

---

### Requirement 12: CostGovernor — Real BedrockModel Instances per Tier

**User Story:** As a platform operator, I want `CostGovernor` to hold real
`strands.models.bedrock.BedrockModel` instances for each model tier, so that model routing
produces real Bedrock API calls.

#### Acceptance Criteria

1. WHEN the `strands` package is installed, THE `CostGovernor` SHALL construct a
   `BedrockModel` instance for each `ModelTier` using the model IDs from `_MODEL_IDS`.
2. WHEN `select_model()` returns a tier, THE `CostGovernor` SHALL expose a
   `get_bedrock_model(tier)` method that returns the corresponding `BedrockModel` instance.
3. WHEN the `strands` package is not installed, THE `CostGovernor` SHALL operate without
   `BedrockModel` instances and `get_bedrock_model()` SHALL return `None`.

---

### Requirement 13: AgentDispatcher — Real Strands Agent Loop + MCP + Skills

**User Story:** As a platform operator, I want `AgentDispatcher` to use a real Strands `Agent`
for its 13-step lifecycle (agent loop, MCP server start, skills load, hooks register), so that
dispatched agents actually run on Bedrock with full tool and skill support.

#### Acceptance Criteria

1. WHEN `AgentDispatcher` is constructed without an `agent_loop_callback`, THE
   `AgentDispatcher` SHALL default to a real implementation that constructs a `strands.Agent`
   with the resolved `BedrockModel`, system prompt, effective tools, and hook engine, then
   calls `agent(task)`.
2. WHEN `AgentDispatcher` is constructed without an `mcp_start_callback`, THE
   `AgentDispatcher` SHALL default to a real implementation that starts MCP server processes
   using `strands_agents_tools.MCPClient` for each server in `agent_def.mcp_servers`.
3. WHEN `AgentDispatcher` is constructed without a `skills_load_callback`, THE
   `AgentDispatcher` SHALL default to a real implementation that loads skill content from
   `SkillRegistry` for each skill in `agent_def.skills`.
4. WHEN `AgentDispatcher` is constructed without a `hooks_register_callback`, THE
   `AgentDispatcher` SHALL default to a real implementation that calls
   `hook_engine.register_scoped()` with the agent's hooks and scope ID.
5. WHEN any callback is explicitly injected, THE `AgentDispatcher` SHALL use the injected
   callback instead of the real SDK (preserving testability).
6. WHEN the Strands `Agent` returns a result, THE `AgentDispatcher` SHALL extract
   `tokens_consumed`, `tools_used`, `files_modified`, and `exit_reason` from the result.
7. IF the `strands` package is not installed and no `agent_loop_callback` is injected, THEN
   THE `AgentDispatcher` SHALL raise a `RuntimeError` when `dispatch()` is called.

---

### Requirement 14: AgentLoader / AgentRegistry — Real strands.Agent from Definitions

**User Story:** As a platform operator, I want `AgentLoader` and `AgentRegistry` to produce
real `strands.Agent` objects from `AgentDefinition` instances, so that loaded agents can be
directly invoked via the Strands SDK.

#### Acceptance Criteria

1. THE `AgentLoader` SHALL expose a `to_strands_agent(agent_def)` method that constructs a
   real `strands.Agent` with the resolved `BedrockModel`, system prompt, and tool list.
2. THE `AgentRegistry` SHALL expose a `to_strands_agent(name)` method that looks up an
   `AgentDefinition` by name and delegates to `AgentLoader.to_strands_agent()`.
3. WHEN the `strands` package is not installed, THEN `to_strands_agent()` SHALL raise a
   `RuntimeError` with a descriptive message.
4. WHEN `agent_def.tools` is `None`, THE `to_strands_agent()` method SHALL use the default
   Strands tool set from `strands_agents_tools`.

---

### Requirement 15: HookHandlers — Real Strands Agent Invocations

**User Story:** As a platform operator, I want `PromptHandler` and `AgentHandler` to use real
`strands.Agent` invocations as their default callbacks, so that hook evaluation actually runs
on Bedrock.

#### Acceptance Criteria

1. WHEN `PromptHandler` has no `model_callback` set, THE `PromptHandler` SHALL default to a
   real implementation that constructs a `strands.Agent` with the Haiku `BedrockModel` and
   invokes it with the resolved prompt.
2. WHEN `AgentHandler` has no `agent_callback` set, THE `AgentHandler` SHALL default to a
   real implementation that constructs a `strands.Agent` with Read, Grep, and Glob tools and
   invokes it with the context prompt.
3. WHEN a `model_callback` or `agent_callback` is explicitly set, THE handlers SHALL use the
   injected callback instead of the real SDK (preserving testability).
4. IF the `strands` package is not installed and no callback is set, THEN THE handlers SHALL
   raise a `RuntimeError` when `execute()` is called.

---

### Requirement 16: Testability — Pluggable Callbacks Remain Intact

**User Story:** As a developer, I want all real SDK calls to remain behind pluggable callbacks,
so that unit tests continue to pass without the Strands SDK or AWS credentials installed.

#### Acceptance Criteria

1. THE `System` SHALL preserve every existing `Optional[Callable[...]]` constructor parameter
   across all 15 integration points.
2. WHEN a callback is explicitly injected, THE `System` SHALL use the injected callback
   regardless of whether the SDK is installed.
3. THE `System` SHALL mark every real SDK call with a `# --- Production integration point ---`
   comment immediately above the SDK import or call site.
4. WHEN the SDK is not installed and no callback is injected, THE `System` SHALL raise a
   `RuntimeError` (not silently return a no-op result).
5. THE existing test suite SHALL pass without modification after the integration is complete.

---

### Requirement 17: Backward Compatibility — Existing Tests Pass

**User Story:** As a developer, I want all existing unit and property tests to continue passing
after the integration, so that the SDK wiring does not break the test suite.

#### Acceptance Criteria

1. THE `System` SHALL ensure that all tests in `tests/unit/` pass when the Strands SDK is not
   installed.
2. THE `System` SHALL ensure that all tests in `tests/property/` pass when the Strands SDK is
   not installed.
3. WHEN a test injects a stub callback, THE `System` SHALL not attempt to import or call the
   real SDK.
4. THE `System` SHALL not change any existing public API signatures (constructor parameters,
   method names, return types).
