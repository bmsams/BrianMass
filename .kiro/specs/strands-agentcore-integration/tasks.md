# Implementation Plan: Strands AgentCore Integration

## Overview

Replace every stub callback in Brainmass v3 with real Strands Agents SDK and Amazon Bedrock
AgentCore calls. Tasks are ordered from lowest-level (shared helpers, cost, observability) to
highest-level (orchestrator, runtime) so each step builds on the previous. All real SDK calls
remain behind pluggable callbacks so existing tests continue to pass without the SDK installed.

---

## Tasks

- [x] 1. Add shared `_normalize_strands_result` helper and `_BEDROCK_MODEL_IDS` constant
  - Add `_BEDROCK_MODEL_IDS: dict[str, str]` mapping all model aliases and canonical IDs to
    `us.anthropic.claude-*-v1:0` cross-region inference profile IDs
  - Add `_normalize_strands_result(raw: object) -> dict` that extracts `summary`,
    `tokens_consumed`, `tools_used`, `files_modified`, `exit_reason`, `turns_used` from any
    Strands `Agent` response shape (str, dict, object with `.output`/`.usage`/`.content`)
  - Place both in a new `src/agents/_strands_utils.py` module so all agent modules can import
    without circular dependencies
  - _Requirements: 1.4, 2.3, 13.6_

  - [ ]* 1.1 Write property test for `_normalize_strands_result`
    - **Property 2: `_normalize_strands_result` handles all raw output shapes**
    - Generate raw values as `st.one_of(st.text(), st.none(), st.fixed_dictionaries(...),
      st.builds(SimpleNamespace, ...))`; assert all six keys present and `summary` is `str`
    - **Validates: Requirements 1.4, 13.6**

- [x] 2. Wire `CostGovernor.get_bedrock_model()` with lazy `BedrockModel` construction
  - Add `_bedrock_models: dict[ModelTier, object]` and `_models_initialized: bool` to
    `CostGovernor.__init__`
  - Add `_ensure_models()` private method: lazy-imports `strands.models.bedrock.BedrockModel`,
    constructs one instance per `ModelTier` using `_BEDROCK_MODEL_IDS`; silently skips if
    `ImportError`
  - Add `get_bedrock_model(tier: ModelTier) -> Optional[object]` public method
  - Mark the `BedrockModel` construction block with `# --- Production integration point ---`
  - _Requirements: 12.1, 12.2, 12.3_

  - [ ]* 2.1 Write property test for `get_bedrock_model` tier coverage
    - **Property 4: BedrockModel coverage — all tiers have a model when SDK is present**
    - Patch `strands.models.bedrock.BedrockModel` with a spy; assert `get_bedrock_model(tier)`
      returns non-None for every `ModelTier` value
    - **Validates: Requirements 12.1, 12.2**

- [x] 3. Wire `BrainmassTracer` backend and `get_tracer()` Strands telemetry
  - Add `backend_tracer: Optional[object] = None` parameter to `BrainmassTracer.__init__` in
    `src/observability/instrumentation.py`
  - Update `trace_model_interaction`, `trace_tool_call`, `trace_hook_execution`,
    `trace_agent_action` context managers to delegate `start_as_current_span` to
    `backend_tracer` when set, falling back to the existing span store
  - In `src/observability/tracer.py` `get_tracer()`: try-import
    `strands.telemetry.get_tracer`, pass result as `backend_tracer`; fall back to standalone
    on `ImportError`
  - Mark both integration points with `# --- Production integration point ---`
  - _Requirements: 11.1, 11.2, 11.3_

- [x] 4. Checkpoint — run existing tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Wire `BrainmassHookEngine` to extend `strands.hooks.HookProvider`
  - Add module-level try/except to conditionally import `HookProvider` from `strands.hooks`;
    fall back to `object` base when `ImportError`
  - Change `class BrainmassHookEngine` declaration to inherit from the conditional base
  - Implement `register_hooks(registry: object) -> None`: try-import
    `BeforeToolCallEvent`, `AfterToolCallEvent`; call `registry.add_callback(...)` for each
  - Add `_on_before_tool_call(event)` → fires `HookEvent.PRE_TOOL_USE` via `self.fire()`
  - Add `_on_after_tool_call(event)` → fires `HookEvent.POST_TOOL_USE` via `self.fire()`
  - Add `pre_tool_use(tool_name, tool_input)` and `post_tool_use(tool_name, tool_input,
    tool_response)` method-based interface (architecture_context.md §5.4)
  - Mark all SDK calls with `# --- Production integration point ---`
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 5.1 Write property test for tool event translation
    - **Property 3: Tool call event translation is correct for all events**
    - Generate mock event objects with random `tool_name` strings; assert `_on_before_tool_call`
      produces `HookEvent.PRE_TOOL_USE` and `_on_after_tool_call` produces `HookEvent.POST_TOOL_USE`
    - **Validates: Requirements 6.3, 6.4**

- [x] 6. Wire `ContextManager` to extend `SummarizingConversationManager`
  - Add module-level try/except to conditionally import
    `strands.session.SummarizingConversationManager`; fall back to `object` base
  - Change `class ContextManager` to inherit from the conditional base
  - In `__init__`: when Strands session is available, call `super().__init__(max_tokens=...,
    summarization_model="us.anthropic.claude-haiku-4-5-v1:0")`
  - In `__init__`: when `memory_store` is `None`, try-import
    `bedrock_agentcore.memory.MemoryClient` and default to `AgentCoreMemoryStore(MemoryClient())`
  - Mark both blocks with `# --- Production integration point ---`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 7. Wire `LearningStore` default `memory_callback` to `AgentCoreMemoryStore`
  - Add `_production_memory_callback(learning_id, learning_dict) -> bool` function that
    imports `bedrock_agentcore.memory.MemoryClient`, constructs `AgentCoreMemoryStore`, and
    calls `create_memory(namespace="learning-store", content=learning_dict)`
  - Change `LearningStore.__init__` to default `memory_callback` to
    `_production_memory_callback` instead of `_default_memory_callback`
  - Raise `RuntimeError` (not silent no-op) when `bedrock-agentcore` is absent and no
    callback is injected
  - Mark with `# --- Production integration point ---`
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 8. Checkpoint — run existing tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. Wire `SubagentManager` production agent and as-tool callbacks
  - Add `_production_agent_callback(agent_def, task, budget) -> dict` using
    `strands.Agent` + `strands.models.bedrock.BedrockModel` + `_normalize_strands_result`
    from `src/agents/_strands_utils.py`
  - Add `_production_as_tool_callback(agent_def) -> dict` using `Agent.as_tool()`
  - Change `SubagentManager.__init__` to default both callbacks to the production versions
  - Raise `RuntimeError` when `strands` absent and no callback injected
  - Mark with `# --- Production integration point ---`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 9.1 Write property test for callback injection override
    - **Property 1: Callback injection always overrides production path**
    - For `SubagentManager`, `TeamManager`, `LoopRunner`, `CompoundLoopOrchestrator`:
      inject a spy callback; assert it is called and no `strands` import is attempted
    - **Validates: Requirements 2.4, 3.2, 4.2, 5.2**

- [x] 10. Wire `TeamManager` production teammate callback
  - Add `_production_teammate_callback(agent_def, task, budget) -> dict` using
    `strands.Agent` + `BedrockModel` + `_normalize_strands_result`
  - Include commented-out `Swarm` upgrade path
  - Change `TeamManager.__init__` to default `teammate_callback` to the production version
  - Raise `RuntimeError` when `strands` absent and no callback injected
  - Mark with `# --- Production integration point ---`
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 11. Wire `LoopRunner` production agent callback
  - Add `_production_loop_callback(agent_def, context_str, budget) -> dict` using
    `strands.Agent` + `BedrockModel` + `_normalize_strands_result`; set `acceptance_met`
    from response text; include commented-out `GraphBuilder` upgrade path
  - Change `LoopRunner.__init__` to default `agent_callback` to the production version
  - Raise `RuntimeError` when `strands` absent and no callback injected
  - Mark with `# --- Production integration point ---`
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 12. Wire `CompoundLoopOrchestrator` production stage callback
  - Add `_production_stage_callback(agent_def, loop_context, budget) -> dict` using
    `strands.Agent` + `BedrockModel` + `_normalize_strands_result`; include commented-out
    `Workflow` upgrade path
  - Change `CompoundLoopOrchestrator.__init__` to default `stage_callback` to the production
    version
  - Raise `RuntimeError` when `strands` absent and no callback injected
  - Mark with `# --- Production integration point ---`
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 13. Checkpoint — run existing tests
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Wire `AgentDispatcher` production callbacks (agent loop, MCP, skills, hooks)
  - Add `_production_agent_loop(agent_def, task, system_prompt, budget) -> dict` using
    `strands.Agent` + `BedrockModel` + `_normalize_strands_result`
  - Add `_production_mcp_start(mcp_servers) -> Callable[[], None]` using
    `strands_agents_tools.MCPClient`; raise `RuntimeError` if package absent
  - Add `_production_skills_load(skill_names) -> str` using `src.skills.skill_registry.SkillRegistry`
  - Update `_step6_hooks` to wire directly through `self._hook_engine.register_scoped()` when
    no `hooks_register_callback` is injected and `hook_engine` is available
  - Change `AgentDispatcher.__init__` to default `agent_loop_callback` to
    `_production_agent_loop`
  - Raise `RuntimeError` when `strands` absent and no `agent_loop_callback` injected
  - Mark all four integration points with `# --- Production integration point ---`
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_

  - [ ]* 14.1 Write property test for `AgentDispatcher` result normalisation
    - **Property 7: AgentDispatcher result normalisation is complete**
    - Generate random Strands result shapes; assert `_normalize_strands_result` always
      produces a dict with all six required keys
    - **Validates: Requirements 13.6**

- [x] 15. Add `to_strands_agent()` to `AgentLoader` and `AgentRegistry`
  - Add `to_strands_agent(agent_def: AgentDefinition) -> object` to `AgentLoader`:
    lazy-import `strands.Agent` + `BedrockModel`; resolve tools via `strands_agents_tools`;
    raise `RuntimeError` when `strands` absent
  - Add `to_strands_agent(name: str) -> object` to `AgentRegistry`: look up by name,
    delegate to `self._loader.to_strands_agent(agent_def)`
  - Mark with `# --- Production integration point ---`
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [ ]* 15.1 Write property test for `to_strands_agent`
    - **Property 5: `to_strands_agent` succeeds for all valid AgentDefinitions**
    - Generate `AgentDefinition` instances with valid model aliases; patch `strands.Agent`
      with a spy; assert non-None return for all valid inputs and `RuntimeError` on import
      failure
    - **Validates: Requirements 14.1, 14.3**

- [x] 16. Wire `PromptHandler` and `AgentHandler` production callbacks
  - In `src/hooks/handlers/prompt.py`: add `_production_model_callback(prompt) -> str` using
    Haiku `BedrockModel` + `strands.Agent`; update `PromptHandler.__init__` to default
    `_model_callback` to `_production_model_callback`
  - In `src/hooks/handlers/agent.py`: add `_production_agent_callback(context_prompt,
    agent_config) -> str` using Haiku `BedrockModel` + `strands.Agent` with
    `file_read`, `grep`, `glob` tools from `strands_agents_tools`; update
    `AgentHandler.__init__` to default `_agent_callback` to `_production_agent_callback`
  - Raise `RuntimeError` when `strands` absent and no callback set
  - Mark with `# --- Production integration point ---`
  - _Requirements: 15.1, 15.2, 15.3, 15.4_

- [x] 17. Wire `Orchestrator` to default `use_production_agent=True`
  - Update `Orchestrator.__init__`: when `use_production_agent` is `None`, set it to
    `True` if no `model_callback` was injected, `False` otherwise
  - Verify `_execute_with_strands` already passes `hooks=[self.hook_engine]` to `Agent`
    (it does — no change needed there)
  - Update module docstring to reference Requirements 1.1–1.6
  - _Requirements: 1.1, 1.2, 1.3, 1.6_

  - [ ]* 17.1 Write unit test for Orchestrator production default
    - Construct `Orchestrator` with no `model_callback`; assert `_use_production_agent` is
      `True`; construct with a stub `model_callback`; assert `_use_production_agent` is `False`
    - _Requirements: 1.2, 1.3_

- [x] 18. Wire `SessionTeleporter` production adapter auto-selection
  - Update `SessionTeleporter.__init__`: when `storage_adapter` is `None`, check
    `os.environ.get("S3_SESSION_BUCKET")`; if set, use `S3SessionStorageAdapter(bucket=...)`
    with `logger.warning` fallback to `InMemorySessionStorageAdapter`
  - When `memory_adapter` is `None`, try-import `bedrock_agentcore.memory.MemoryClient`;
    if available, use `AgentCoreSessionMemoryAdapter()`; else use
    `InMemorySessionMemoryAdapter()`
  - Mark both blocks with `# --- Production integration point ---`
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [ ]* 18.1 Write property test for `SessionTeleporter` adapter selection
    - **Property 7: SessionTeleporter adapter selection is environment-driven**
    - Use `hypothesis` + `monkeypatch` to vary `S3_SESSION_BUCKET` env var and mock
      `bedrock_agentcore` import; assert correct adapter type is selected in each combination
    - **Validates: Requirements 9.1, 9.2, 9.4**

- [x] 19. Write property test for missing SDK raises `RuntimeError`
  - **Property 6: Missing SDK + no callback raises RuntimeError**
  - For each of: `SubagentManager`, `TeamManager`, `LoopRunner`, `CompoundLoopOrchestrator`,
    `AgentDispatcher`, `LearningStore`, `AgentLoader.to_strands_agent`, `PromptHandler`,
    `AgentHandler`: patch the `strands` / `bedrock_agentcore` import to raise `ImportError`;
    assert `RuntimeError` is raised when the primary execution method is called with no
    injected callback
  - **Validates: Requirements 1.5, 2.5, 3.4, 4.4, 5.4, 8.4, 14.3, 15.4, 16.4**

- [x] 20. Final checkpoint — full test suite
  - Run `pytest tests/unit tests/property` and ensure all tests pass.
  - Run `ruff check src tests` and fix any lint errors.
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Implementation order (1 → 20) respects dependency graph: shared utils first, leaf
  components next, orchestrator and runtime last
- Every real SDK call must be marked `# --- Production integration point ---`
- Existing tests must not be modified — they inject stubs and must continue to pass
- Property tests use `@pytest.mark.property` and `@settings(max_examples=100)`
