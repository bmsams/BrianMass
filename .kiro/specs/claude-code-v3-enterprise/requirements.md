# Requirements Document

## Introduction

This document specifies the requirements for the Brainmass v3 Enterprise System — an enterprise-grade agentic coding system built on the AWS Strands Agents SDK and Amazon Bedrock AgentCore. The system achieves behavioral parity with Brainmass's confirmed architecture while adding enterprise governance, cost management, multi-agent coordination, plugin extensibility, and production observability. The system is organized into three tiers (Client, Control, Data) with seven core components, supports three multi-agent topologies, 12 lifecycle hook events, a full plugin ecosystem, semantic context management, cost-aware model routing, and deployment across local, hybrid, and fully cloud-hosted topologies. Implementation targets Python (Strands SDK) with AgentCore for production hosting.

## Glossary

- **Orchestrator**: The core control-plane component responsible for model-driven routing, task decomposition, agent team coordination, and learning persistence. Implemented as a Strands `Agent` with `BedrockModel`.
- **Context_Manager**: The component responsible for semantic context triage, tiered persistence (working memory, session file, learning store), compaction strategy, and cache orchestration. Extends Strands `SummarizingConversationManager` with custom classification logic.
- **Policy_Engine**: The component managing 12 lifecycle hooks, 3 handler types (command, prompt, agent), and enterprise managed settings. Combines Strands `HookProvider` with AgentCore Policy (Cedar language).
- **MCP_Tool_Gateway**: The data-plane component for tool discovery via Tool Search, tool execution, and Brainmass-as-MCP-Server dual mode. Maps to AgentCore Gateway for MCP proxy and API-to-MCP conversion.
- **Worker_Pool**: The data-plane component managing subagents (Agents-as-Tools), Agent Teams (Swarm peer topology), and self-improving loop orchestration (GraphBuilder with conditional edges).
- **Cost_Governor**: The subsystem enforcing per-agent token budgets, model tier routing, and real-time cost monitoring. Custom implementation wrapping Strands `BedrockModel` instances.
- **Plugin_Registry**: The client-tier component for plugin discovery, installation, marketplace management, and capability registration. Plugins bundle commands, agents, skills, hooks, MCP servers, and LSP servers.
- **Agent_Registry**: The subsystem for discovering, loading, validating, and dispatching custom agent definitions from .md files with YAML frontmatter. Custom loader using `python-frontmatter` producing Strands `Agent` objects.
- **Hook_Engine**: The runtime engine that fires 12 lifecycle events and executes 3 handler types (command, prompt, agent) with scoped registration. Implements Strands `HookProvider` with `BeforeToolCallEvent` and `AfterToolCallEvent`.
- **Cache_Manager**: The subsystem managing prompt caching strategy with 5-minute and 1-hour cache durations via Bedrock API `cache_control` blocks.
- **Effort_Controller**: The subsystem selecting Quick/Standard/Deep effort levels based on task complexity, model capability, and budget constraints. Maps to `budget_tokens` parameter in extended thinking.
- **Skill_Registry**: The subsystem for skill discovery, SKILL.md frontmatter parsing, auto-invocation matching, hot-reload, and invocation logging.
- **Session_Teleporter**: The subsystem for serializing session state and transferring it across surfaces (web, mobile, desktop, CLI). Uses Strands `SessionManager` with S3 backend and AgentCore Memory.
- **Learning_Store**: The cross-session persistent store with vector-indexed memory of past mistakes, failed approaches, and discovered conventions. Maps to AgentCore Memory long-term with semantic strategy.
- **Loop_Runner**: The subsystem orchestrating self-improving agent loops ("Ralph Wiggum" pattern) with iteration isolation, structured context files, and git-based checkpointing. Implemented via Strands `GraphBuilder` with conditional edges.
- **Team_Manager**: The subsystem managing Agent Teams with team lead, teammates, shared task lists, mailbox IPC, and file-lock coordination. Maps to Strands `Swarm` with custom file-lock protocol.
- **Quota_Manager**: The subsystem tracking consumption across all Brainmass surfaces (web, mobile, desktop, CLI) in real-time with quota exhaustion prediction.
- **Semantic_Triage**: The classification system categorizing context items into PRESERVE_VERBATIM, PRESERVE_STRUCTURED, COMPRESS_AGGRESSIVE, and EPHEMERAL categories.
- **Model_Tier**: One of three model levels — Opus 4.6 (`claude-opus-4-6`, 1M tokens, $5/$25 per M), Sonnet 4.5 (`claude-sonnet-4-5-20250929`, 200K tokens, $3/$15 per M), Haiku 4.5 (`claude-haiku-4-5-20251001`, 200K tokens, $0.80/$4 per M).
- **Effort_Level**: One of three reasoning depth levels — Quick (budget_tokens: 2000), Standard (budget_tokens: 10000), Deep (budget_tokens: 50000+).
- **Hook_Event**: One of 12 lifecycle events: SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PostToolUseFailure, PermissionRequest, PreCompact, Notification, Stop, SubagentStop, SessionEnd, Setup.
- **Handler_Type**: One of 3 hook handler types: Command (shell scripts, exit 0/2), Prompt (Haiku evaluation with $ARGUMENTS), Agent (sub-agent with Read/Grep/Glob tools).
- **MCP**: Model Context Protocol — the standard protocol for tool integration between AI agents and external services.
- **Cedar**: The policy language used by Amazon Bedrock AgentCore for fine-grained tool-level governance at the Gateway level.
- **Strands_SDK**: The AWS Strands Agents SDK — an open-source Python framework for building AI agents with multi-agent patterns (Agent, Swarm, Graph, Workflow).
- **AgentCore**: Amazon Bedrock AgentCore — the managed platform with Runtime (microVM per session), Memory (short-term + long-term), Gateway (MCP proxy), Identity (OAuth/Okta/Entra/Cognito), Policy (Cedar), Observability (CloudWatch/OTel), and Evaluations (13 built-in).
- **BRAINMASS.md**: Project-level configuration file providing guidance and conventions to the agent system.
- **AgentCore_Runtime**: Serverless hosting with microVM isolation per session, supporting HTTP/MCP/A2A/WebSocket protocols, 8-hour workloads, 100MB payloads, immutable versioning, and consumption-based pricing.
- **Batch_API**: Anthropic's asynchronous processing API offering 50% discount on all model tiers for non-interactive tasks.
- **Context_Category**: One of four semantic triage classifications: PRESERVE_VERBATIM (never summarize), PRESERVE_STRUCTURED (compress format, keep data), COMPRESS_AGGRESSIVE (summarize to key takeaways), EPHEMERAL (drop after use).

## Requirements

### Requirement 1: System Architecture and Core Orchestrator

**User Story:** As a developer, I want a three-tier agentic coding system with a central orchestrator, so that I can delegate complex coding tasks to AI agents with proper routing, decomposition, and lifecycle management.

#### Acceptance Criteria

1. THE Orchestrator SHALL organize the system into three tiers: Client (CLI/IDE Extension, Plugin_Registry), Control (Orchestrator, Context_Manager, Policy_Engine), and Data (MCP_Tool_Gateway, Worker_Pool)
2. WHEN a user submits a request, THE Orchestrator SHALL execute an 8-step processing flow: fire SessionStart hooks, fire UserPromptSubmit hooks, classify request complexity and select effort level, select model tier via Cost_Governor, decompose into tasks if complex, execute tool calls with PreToolUse/PostToolUse hooks, update context via Context_Manager, and fire Stop hooks
3. WHEN the Orchestrator decomposes a complex request, THE Orchestrator SHALL select the appropriate agent topology (hierarchical subagent, Agent Team, or self-improving loop) based on task characteristics
4. THE Orchestrator SHALL support three model tiers: Opus 4.6 (claude-opus-4-6, 1M context, $5/$25 per M tokens), Sonnet 4.5 (claude-sonnet-4-5-20250929, 200K context, $3/$15 per M tokens, premium pricing >200K: $6/$22.50 per M tokens), and Haiku 4.5 (claude-haiku-4-5-20251001, 200K context, $0.80/$4 per M tokens)
5. THE Orchestrator SHALL support the Batch API with 50% discount on all model tiers for non-interactive asynchronous tasks
6. WHEN the Orchestrator processes a request, THE Orchestrator SHALL track cost across the entire request lifecycle and record usage with the Cost_Governor after each API call
7. THE Orchestrator SHALL be implemented as a Strands `Agent` with `BedrockModel` and deployed on AgentCore Runtime with microVM isolation per session

### Requirement 2: Context Management and Semantic Triage

**User Story:** As a developer, I want intelligent context management that preserves critical information across compaction events, so that the agent does not lose important error messages, file paths, or decisions when the context window fills up.

#### Acceptance Criteria

1. THE Context_Manager SHALL classify every new context item into one of four categories: PRESERVE_VERBATIM (never summarize — error messages, file paths, variable names, test output, git diff hunks, API responses), PRESERVE_STRUCTURED (compress format, keep data — decision records, task lists, acceptance criteria, schemas), COMPRESS_AGGRESSIVE (summarize to key takeaways — discussion, reasoning chains, alternatives considered), EPHEMERAL (drop after use — tool call metadata, raw search results, intermediate compilation output)
2. THE Context_Manager SHALL correctly classify 95% or more of context items using pattern matching with defined VERBATIM_PATTERNS (stack traces, exit codes, file paths, declarations, test output, version numbers, environment variables) and STRUCTURED_PATTERNS (decision records, acceptance criteria, schemas)
3. THE Context_Manager SHALL maintain three storage tiers: Working Memory (in-context, subject to compaction), Session File (.brainmass/session-state.json persisting PRESERVE_VERBATIM and PRESERVE_STRUCTURED items), and Learning Store (.brainmass/learnings/ for cross-session vector-indexed memory)
4. WHEN compaction occurs on a 200K token context window, THE Context_Manager SHALL trigger at 83.5% capacity (~167K tokens), preserve verbatim items to the session file, compress discussion, and drop ephemeral items
5. WHEN compaction occurs on a 1M token context window (beta), THE Context_Manager SHALL defer compaction until 85% capacity (~850K tokens) and only compact when hitting the premium pricing threshold unless budget allows
6. WHEN a user initiates manual compaction via /compact, THE Context_Manager SHALL back up critical context via the PreCompact hook before compaction occurs
7. THE Context_Manager SHALL implement context editing by tagging each context block with a staleness score (turns since last reference × 1/reference count), automatically clearing tool call results not referenced in the last N turns, and preserving conversation flow by keeping user messages and key assistant decisions
8. THE Context_Manager SHALL expose context health metrics: freePercent, totalTokens, preservedTokens, compressibleTokens, ephemeralTokens, stalenessDistribution, and cacheHitRate
9. WHEN a session starts or resumes, THE Context_Manager SHALL load the session file into context via the SessionStart hook handler
10. WHEN compaction is about to occur, THE Context_Manager SHALL fire the PreCompact hook to back up critical state to the session file before compaction destroys it
11. THE Context_Manager SHALL ensure all PRESERVE_VERBATIM items survive compaction with 100% fidelity
12. THE Context_Manager SHALL support configurable compaction threshold via the BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE environment variable (1-100)

### Requirement 3: Hook Lifecycle Engine

**User Story:** As a developer, I want a comprehensive hook system with 12 lifecycle events and 3 handler types, so that I can enforce deterministic guardrails, automate workflows, and customize agent behavior at every stage of execution.

#### Acceptance Criteria

1. THE Hook_Engine SHALL support all 12 lifecycle events: SessionStart (session starts/resumes, non-blocking), UserPromptSubmit (user submits prompt, can modify), PreToolUse (before tool executes, can block via exit code 2 or deny decision), PostToolUse (after tool succeeds, non-blocking), PostToolUseFailure (after tool fails, non-blocking), PermissionRequest (tool permission requested, can auto-allow/deny), PreCompact (before compaction, non-blocking, use for backup), Notification (alert sent, non-blocking), Stop (agent finishes, can block completion), SubagentStop (subagent finishes, non-blocking), SessionEnd (session terminates, non-blocking), Setup (triggered via --init/--init-only/--maintenance flags, non-blocking)
2. THE Hook_Engine SHALL support three handler types: Command (shell scripts receiving JSON on stdin, exit 0=allow/2=deny, 10-minute timeout), Prompt (text sent to Haiku model with $ARGUMENTS placeholder for single-turn semantic evaluation), and Agent (sub-agent with Read/Grep/Glob tools for multi-turn codebase verification)
3. THE Hook_Engine SHALL support 7 configuration scopes with precedence: user-wide (~/.brainmass/settings.json), project-shared (.brainmass/settings.json), project-local (.brainmass/settings.local.json), skill frontmatter (scoped to skill lifecycle), subagent frontmatter (Stop auto-converted to SubagentStop), plugin hooks (active when plugin enabled), and enterprise managed (allowManagedHooksOnly blocks all other hooks)
4. WHEN a PreToolUse hook fires, THE Hook_Engine SHALL support three output modes: allow (exit 0), deny (exit code 2 or structured JSON with permissionDecision: deny and reason), and modify input (structured JSON with updatedInput and optional additionalContext, transparent to the model)
5. WHEN a Stop hook fires, THE Hook_Engine SHALL support blocking completion (decision: block with reason) to force the agent to continue working
6. WHEN a PermissionRequest hook fires, THE Hook_Engine SHALL support auto-allow and auto-deny decisions with optional updatedInput
7. THE Hook_Engine SHALL support async hooks (async: true) that fire and forget without blocking main execution, with results logged to OpenTelemetry
8. THE Hook_Engine SHALL support matcher regex patterns against tool_name for tool events and source for SessionStart, with '*' or omitted matching all events
9. THE Hook_Engine SHALL provide the following environment variables to hooks: $BRAINMASS_TOOL_INPUT_FILE_PATH, $BRAINMASS_SESSION_ID, $BRAINMASS_MODEL, $BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE, $ENABLE_TOOL_SEARCH, $MCP_TIMEOUT
10. THE Hook_Engine SHALL pass common input fields to all hook handlers: session_id, hook_event_name, cwd, and session_type (interactive or headless)

### Requirement 4: Cost Governance and Token Economics

**User Story:** As a developer and enterprise administrator, I want cost-aware model routing and per-agent token budgets, so that I can control spending, prevent runaway costs, and route tasks to the cheapest capable model.

#### Acceptance Criteria

1. THE Cost_Governor SHALL enforce per-agent token budgets with four parameters: input_budget_tokens (max input per turn), output_budget_tokens (max output per turn), session_budget_usd (max total cost per session), and team_budget_usd (max total cost across all teammates in an Agent Team)
2. WHEN an agent approaches 80% of its budget, THE Cost_Governor SHALL trigger a model downgrade (Opus → Sonnet → Haiku)
3. WHEN an agent reaches 95% of its budget, THE Cost_Governor SHALL pause the agent and request human approval to continue
4. THE Cost_Governor SHALL route tasks to the cheapest capable model: Haiku for file search, grep, exploration, linting, simple transforms, and prompt hook evaluation; Sonnet for standard code generation, refactoring, implementation, and test writing; Opus for complex reasoning, architecture decisions, cross-service refactoring, and Agent Team leads; Batch API (50% discount) for async non-interactive tasks
5. THE Cost_Governor SHALL select models based on TaskSignals: isExploration=true OR filesAffected<=1 AND !requiresReasoning → Haiku; isTeamLead=true OR requiresReasoning=true OR dependencyDepth>3 → Opus; default → Sonnet; budget override downgrades one tier at warning, forces Haiku at critical
6. THE Cost_Governor SHALL calculate costs using the pricing formula: (regularInput/1M × inputRate) + (output/1M × outputRate) + (cacheRead/1M × cachedRate) + (cacheWrite/1M × cacheWriteRate) with correct rates per model tier
7. THE Cost_Governor SHALL provide a real-time dashboard with token consumption, cost per agent, model tier distribution, and cache hit rates
8. THE Cost_Governor SHALL support per-task cost attribution with total input/output tokens and USD cost tagged to every task
9. THE Cost_Governor SHALL integrate with ccusage CLI (npx ccusage@latest blocks --live) for real-time terminal monitoring
10. THE Cost_Governor SHALL consume SDKRateLimitInfo events for rate limit status including utilization, reset times, and overage information

### Requirement 5: Multi-Agent Orchestration — Hierarchical Subagents

**User Story:** As a developer, I want to delegate well-defined subtasks to specialized worker agents, so that complex tasks can be decomposed and executed in parallel with clear input/output contracts.

#### Acceptance Criteria

1. THE Worker_Pool SHALL implement the Agents-as-Tools pattern where the Orchestrator wraps worker agents as tools callable via natural language using Strands `Agent.as_tool()`
2. WHEN the Orchestrator delegates a task to a subagent, THE Worker_Pool SHALL create an independent context window for the subagent with its own budget and hook scope
3. WHEN a subagent completes, THE Worker_Pool SHALL fire SubagentStop hooks and return a structured AgentResult (summary, turnsUsed, tokensConsumed, toolsUsed, filesModified, exitReason) to the parent
4. THE Worker_Pool SHALL support subagent frontmatter that defines scoped hooks, with Stop hooks automatically converted to SubagentStop
5. THE Orchestrator SHALL synthesize results from multiple subagents and coordinate sequencing for tasks with sequential dependencies

### Requirement 6: Multi-Agent Orchestration — Agent Teams

**User Story:** As a team lead, I want to coordinate a team of peer AI agents that can communicate directly, share task lists, and claim work autonomously, so that complex exploration tasks like security audits can be parallelized across multiple full Brainmass instances.

#### Acceptance Criteria

1. THE Team_Manager SHALL implement the Agent Teams topology with a Team Lead (coordinator that spawns teammates and maintains shared task list) and N Teammates (each a full Brainmass instance with its own context window)
2. THE Team_Manager SHALL enable peer-to-peer communication where teammates message each other directly via a mailbox system, not just through the orchestrator
3. THE Team_Manager SHALL maintain a shared task list with status tracking (pending, claimed, blocked, complete), dependency declarations (task B blocks on task A), and file ownership mapping
4. THE Team_Manager SHALL implement file-lock-based coordination: before editing a file, a teammate checks for .brainmass/locks/{filepath}.lock, creates the lock with teammate ID and timestamp if absent, breaks stale locks (>5 minutes), and releases all locks on task completion
5. THE Team_Manager SHALL implement an inbox/mailbox system using file-system-based IPC where each teammate has an inbox directory, messages are JSON files with sender, recipient, type (task_assignment, finding, question, status_update), and payload
6. WHEN Agent Teams is enabled via settings.json (BRAINMASS_EXPERIMENTAL_AGENT_TEAMS: "1"), THE Team_Manager SHALL store team configuration at ~/.brainmass/teams/{team-name}/config.json and task state at ~/.brainmass/tasks/{team-name}/
7. THE Team_Manager SHALL provide tmux split pane visibility showing each teammate's activity in real-time
8. THE Team_Manager SHALL enforce per-team budgets via the Cost_Governor and provide real-time cost tracking per teammate
9. THE Team_Manager SHALL support task decomposition by file ownership (e.g., Agent 1: auth module, Agent 2: database layer) and maintain running docs of failed approaches so teammates do not repeat mistakes
10. THE Team_Manager SHALL coordinate via git synchronization and handle merge conflicts autonomously
11. THE Team_Manager SHALL map to Strands `Swarm` for the collaboration model with custom file-lock protocol and shared state via `invocation_state`

### Requirement 7: Multi-Agent Orchestration — Self-Improving Loops

**User Story:** As a developer, I want to run autonomous iterative agent loops overnight that accumulate learnings across iterations, so that repetitive tasks with learnable patterns can be completed without human intervention.

#### Acceptance Criteria

1. THE Loop_Runner SHALL implement the "Ralph Wiggum" pattern: each iteration spawns a fresh agent with a clean context window, feeds it a structured context file, has it do work, commit to git, update the context file with learnings, terminate, and loop
2. THE Loop_Runner SHALL maintain a structured context file at .brainmass/loop-context.json containing: currentTask, acceptanceCriteria, constraints, learnings[] (pattern, resolution, confidence, source_iteration), failedApproaches[] (iteration, approach, why_failed), iterationCount, and maxIterations
3. THE Loop_Runner SHALL implement git-based checkpointing where every iteration commits to git, providing an audit trail, enabling rollback, and allowing diff-based review
4. THE Loop_Runner SHALL accumulate learnings across iterations: discovered patterns, error signatures and fixes, codebase conventions, and performance insights
5. THE Loop_Runner SHALL implement four safety controls: live log monitoring (auto-pause if agent loops on same error 3+ times), diff size limits (abort if diff exceeds bounds or touches files outside task scope), stop file (.auto/stop sentinel to gracefully halt at next iteration boundary), and acceptance criteria gate (check all criteria pass before marking complete)
6. THE Loop_Runner SHALL support compound loop orchestration chaining Analysis → Planning → Execution loops where one agent's output becomes the next agent's input
7. THE Loop_Runner SHALL be implemented via Strands `GraphBuilder` with conditional edges for the review → fix cycle
8. THE Learning_Store SHALL persist learnings at .brainmass/learnings/ as a cross-session store with vector-indexed memory loaded selectively based on semantic similarity to the current task

### Requirement 8: Custom Agent Definition Schema

**User Story:** As a developer, I want to define custom specialized agents as Markdown files with YAML frontmatter, so that I can create reusable, isolated AI assistants with specific tools, models, hooks, and system prompts for different coding tasks.

#### Acceptance Criteria

1. THE Agent_Registry SHALL load custom agent definitions from .md files with YAML frontmatter, where the Markdown body becomes the agent's system prompt
2. THE Agent_Registry SHALL require two fields: name (lowercase-with-hyphens, regex ^[a-z][a-z0-9-]*$) and description (detailed, action-oriented with examples, used for auto-dispatch matching)
3. THE Agent_Registry SHALL support optional fields: model (sonnet/opus/haiku/inherit, default inherit), tools (comma-separated allowlist), disallowedTools (comma-separated denylist, mutually exclusive with tools), permissionMode (default/bypassPermissions/plan), color (purple/cyan/green/orange/blue/red), maxTurns (integer limit on agentic loop), hooks (all 12 events scoped to agent lifecycle), skills (comma-separated skill names), mcpServers (agent-scoped with ${ENV_VAR} substitution), and memory (persistent context injected every invocation)
4. THE Agent_Registry SHALL resolve model aliases: sonnet → claude-sonnet-4-5-20250929, opus → claude-opus-4-6, haiku → claude-haiku-4-5-20251001, inherit → parent's model
5. THE Agent_Registry SHALL load agents from four storage locations with precedence: project (.brainmass/agents/*.md) > user (~/.brainmass/agents/*.md) > plugin ({plugin}/agents/*.md, namespaced as plugin-name:agent-name) > CLI inline (--agents JSON flag, ephemeral)
6. THE Agent_Registry SHALL support four invocation modes: auto-dispatch (model sees agents as tools and selects based on description), explicit ("Use the X agent" or /agents menu), CLI (brainmass --agent code-reviewer), and pipeline (SubagentStop hooks chain stages via .brainmass/pipeline-state.json)
7. WHEN an agent is dispatched, THE Agent_Registry SHALL execute a 13-step lifecycle: dispatch, resolve model alias and check budget, build effective tool list, start agent-scoped MCP servers, load referenced skills into context, register agent-scoped hooks (additive), create isolated context window, inject memory + skills + systemPrompt, execute agentic loop, fire SubagentStop hooks on stop, deregister scoped hooks and stop MCP servers, release context, and return AgentResult to parent
8. THE Agent_Registry SHALL validate all YAML frontmatter fields (required check, type check, enum check) and reject invalid definitions with descriptive errors
9. THE Agent_Registry SHALL support hot-reload: new or modified .md files detected without restart via filesystem watcher or /agents command
10. THE Agent_Registry SHALL generate tool definitions for all registered agents so the model sees them as invocable tools with name='agent:{agent.name}' and description=agent.description
11. THE Agent_Registry SHALL support Agent Teams integration where the team lead can spawn custom agents as teammates with mailbox/task coordination
12. THE Agent_Registry SHALL provide 5 built-in agent templates: Code Reviewer (Sonnet, Read/Glob/Grep/Bash, purple), Security Auditor (Opus, read-only, OWASP skill, red), Implementer-Tester (Sonnet, all tools, PostToolUse auto-format, green), Researcher (Haiku, read-only, plan mode, cyan), and Architect (Opus, GitHub MCP, ADR skill, blue)

### Requirement 9: Plugin Ecosystem

**User Story:** As a developer, I want to install, manage, and create shareable plugin packages that bundle commands, agents, skills, hooks, MCP servers, and LSP servers, so that I can extend the system's capabilities and share tools with the community.

#### Acceptance Criteria

1. THE Plugin_Registry SHALL support the standard plugin directory layout: .brainmass-plugin/plugin.json (manifest with name, description, version, author, homepage, license), commands/ (slash commands), agents/ (specialized agents with YAML frontmatter), skills/ (SKILL.md files), hooks/ (event handlers), .mcp.json (MCP server configuration), and .lsp.json (LSP configuration)
2. THE Plugin_Registry SHALL implement marketplace architecture where marketplaces are Git repositories with .brainmass-plugin/marketplace.json catalogs, the official marketplace ships pre-configured, and community marketplaces are added via /plugin marketplace add {owner}/{repo}
3. WHEN a user installs a plugin via /plugin install {name}@{marketplace}, THE Plugin_Registry SHALL register all plugin capabilities (commands, agents, skills, hooks, MCP servers, LSP servers) into their respective registries
4. THE Plugin_Registry SHALL support enable/disable toggling per project context and full uninstallation
5. THE Plugin_Registry SHALL namespace plugin agents as plugin-name:agent-name to avoid collisions with user/project agents
6. THE Plugin_Registry SHALL support the MCP Tool Search feature: when tool definitions exceed the context threshold, switch to lazy loading with tool_reference blocks (requires Sonnet 4+ or Opus 4+, Haiku falls back to eager loading), controlled via ENABLE_TOOL_SEARCH environment variable (auto/auto:N/false), reducing context usage by up to 95%

### Requirement 10: Skills Registry

**User Story:** As a developer, I want reusable skill workflows that the agent can invoke automatically or via slash commands, with transparency about which skills are active and how much context they consume.

#### Acceptance Criteria

1. THE Skill_Registry SHALL discover skills by scanning ~/.brainmass/skills and .brainmass/skills directories and parsing SKILL.md files with YAML frontmatter (name, description, disable-model-invocation, hooks, allowed_tools)
2. THE Skill_Registry SHALL support three invocation modes: auto-invocation (match task description against skill descriptions, trigger when confidence exceeds threshold, disable per-skill with disable-model-invocation: true), slash command (/skill-name with $ARGUMENTS placeholder), and programmatic (plugins invoke via capability registry)
3. THE Skill_Registry SHALL support skill-scoped hooks that only run when the skill is active, covering all 12 hook events
4. THE Skill_Registry SHALL implement transparency requirements: invocation logging (which skills considered, confidence scores, selection/rejection reasons), context consumption tracking (each skill's token contribution visible in observability dashboards), and hot-reload (skills immediately available without restart)
5. THE Skill_Registry SHALL namespace plugin skills as /plugin-name:skill-name

### Requirement 11: Prompt Caching Architecture

**User Story:** As a developer, I want intelligent prompt caching that reduces token costs by 70-80% for stable content, so that repeated system prompts, BRAINMASS.md contents, and tool schemas are not re-processed on every turn.

#### Acceptance Criteria

1. THE Cache_Manager SHALL implement a 7-layer caching strategy: system prompts (1-hour cache, stable), BRAINMASS.md contents (1-hour cache, stable), tool schemas/MCP definitions (1-hour cache, stable), policy/hook definitions (5-minute cache, may change during session), skill instructions (5-minute cache, hot-reload may invalidate), conversation history (never cache, volatile), and current diff state (never cache, volatile)
2. THE Cache_Manager SHALL inject cache_control blocks into API requests with appropriate TTL: { type: 'ephemeral', ttl: 3600 } for 1-hour and { type: 'ephemeral' } for 5-minute default
3. THE Cache_Manager SHALL track cache hit rates and alert via the Cost_Governor if rates fall below 70%
4. THE Cache_Manager SHALL account for cache economics: cache writes cost 1.25x (5-min) or 2x (1-hour) the base input rate, with break-even at 2 reads for 5-minute and 8 reads for 1-hour cache
5. THE Cache_Manager SHALL parse API response usage metrics: cache_creation_input_tokens and cache_read_input_tokens (90% discount on cached reads)

### Requirement 12: Adaptive Thinking and Effort Controls

**User Story:** As a developer, I want the system to automatically adjust reasoning depth based on task complexity, so that simple tasks execute quickly and cheaply while complex tasks get deep analysis.

#### Acceptance Criteria

1. THE Effort_Controller SHALL support three effort levels: Quick (budget_tokens: 2000, minimal thinking, low latency/cost — linting, formatting, simple renames, exploration), Standard (budget_tokens: 10000, moderate thinking, balanced — code generation, refactoring, test writing), and Deep (budget_tokens: 50000+, extended thinking, highest quality — architecture decisions, complex debugging, security analysis)
2. THE Effort_Controller SHALL select effort level based on: task complexity signals (files affected, dependency depth, ambiguity), model capability (Haiku always Quick, Sonnet Quick or Standard, Opus any level), budget constraints (downgrade effort when approaching limits), and explicit override (developer can force effort level)
3. THE Effort_Controller SHALL map effort levels to the extended thinking budget_tokens parameter in the Anthropic Messages API
4. THE Effort_Controller SHALL support Fast Mode for Opus 4.6 via configuration, adding a speed attribute to OTel events and trace spans for visibility

### Requirement 13: Session Teleportation and Cross-Surface Continuity

**User Story:** As a developer, I want to move my coding session seamlessly between web, mobile, desktop, and CLI surfaces, so that I can continue work from any device without losing state, permissions, or active workers.

#### Acceptance Criteria

1. THE Session_Teleporter SHALL serialize all session state into a portable blob: conversation history, tool permissions, active workers/teammates, pending approvals, compaction state, context manager state, cost tracking, and hook registrations
2. WHEN a user invokes /teleport, THE Session_Teleporter SHALL transfer the session from the current surface (web/mobile) to the terminal with full state preservation
3. WHEN a user invokes /desktop, THE Session_Teleporter SHALL hand off the session to the desktop app for visual diff review
4. THE Session_Teleporter SHALL maintain permission inheritance so tool approvals persist across surface transitions
5. THE Session_Teleporter SHALL manage worker lifecycle so active subagents and teammates survive teleport
6. THE Session_Teleporter SHALL maintain audit continuity with a single trace ID across all surfaces for a given session
7. THE Session_Teleporter SHALL maintain unified cost attribution regardless of which surface is active
8. WHEN simultaneous edits occur from multiple surfaces, THE Session_Teleporter SHALL resolve conflicts via last-write-wins with conflict notification
9. THE Session_Teleporter SHALL use Strands `SessionManager` with S3 backend and AgentCore Memory for persistence

### Requirement 14: Security and Governance

**User Story:** As an enterprise administrator, I want comprehensive security controls including data classification, guardrails, HIPAA compliance, and unified quota management, so that the system meets enterprise governance requirements.

#### Acceptance Criteria

1. THE Security subsystem SHALL implement data classification for context items: PII, PHI, financial data, and credentials
2. THE Security subsystem SHALL integrate Bedrock Guardrails via the guardrail_latest_message parameter in Strands for content filtering
3. THE Security subsystem SHALL implement Constitutional AI filter integration for blocking dangerous or unethical suggestions
4. THE Security subsystem SHALL provide a HIPAA compliance path for Enterprise plans processing PHI, with health data handling requiring explicit consent on iOS/Android
5. THE Security subsystem SHALL implement a nest guard to prevent launching Brainmass inside another Brainmass session
6. THE Security subsystem SHALL support unified quota management across all Brainmass surfaces (web, mobile, desktop, CLI) with weekly rate limit architecture for 24/7 agent users
7. THE Security subsystem SHALL support self-serve Enterprise deployment without requiring Sales involvement
8. THE Security subsystem SHALL implement data classification for health and financial data in cross-surface identity propagation
9. THE Security subsystem SHALL integrate with AgentCore Identity (Okta, Entra, Cognito) for authentication and AgentCore Policy (Cedar) for fine-grained tool-level governance at the Gateway level
10. THE Security subsystem SHALL support VPC-only mode across all AgentCore services for network isolation

### Requirement 15: Unified Quota Management

**User Story:** As a developer and administrator, I want real-time tracking of consumption across all Brainmass surfaces with quota exhaustion prediction, so that I can manage usage limits and avoid unexpected service interruptions.

#### Acceptance Criteria

1. THE Quota_Manager SHALL track consumption across all Brainmass surfaces (web, mobile, desktop, CLI) in real-time
2. THE Quota_Manager SHALL predict quota exhaustion based on current run rate
3. WHEN consumption approaches 80% of the quota, THE Quota_Manager SHALL alert the user
4. THE Quota_Manager SHALL provide usage breakdowns by surface and by agent
5. THE Quota_Manager SHALL support weekly rate limits for 24/7 agent users (affecting <5% of users)
6. THE Quota_Manager SHALL consume SDKRateLimitInfo events for rate limit status including utilization, reset times, and overage information
7. THE Quota_Manager SHALL support overage purchase at API rates for Max subscribers

### Requirement 16: Observability and Audit

**User Story:** As a developer and operations engineer, I want comprehensive observability with OpenTelemetry traces, cost attribution, and context health metrics, so that I can monitor, debug, and optimize agent behavior in production.

#### Acceptance Criteria

1. THE Observability subsystem SHALL instrument all components with OpenTelemetry via Strands' built-in instrumentation, emitting structured traces for all agent actions, tool calls, hook executions, and model interactions
2. THE Observability subsystem SHALL emit custom spans for: cost attribution (token counts + USD per trace), context health (free %, staleness scores, cache hit rates), agent coordination (inter-teammate messages, task claims, file locks), effort level (Fast Mode speed attribute), and skill invocation (which skills, confidence, selection reason)
3. THE Observability subsystem SHALL emit structured logs for every hook execution
4. THE Observability subsystem SHALL integrate with ccusage for terminal-based monitoring
5. THE Observability subsystem SHALL integrate with AgentCore Observability for CloudWatch dashboards, X-Ray tracing, and automatic dashboards for token usage, latency, session duration, and error rates
6. THE Observability subsystem SHALL maintain a single trace ID across all surfaces for session teleportation audit continuity

### Requirement 17: Evaluation Framework

**User Story:** As a developer, I want a comprehensive evaluation suite that measures task completion, code quality, safety, context preservation, cost efficiency, multi-agent coordination, learning retention, and latency, so that I can validate and improve system quality.

#### Acceptance Criteria

1. THE Evaluation subsystem SHALL measure 8 dimensions: task completion (≥90% via automated acceptance criteria verification), code quality (≥4.0/5.0 via LLM-as-a-Judge with rubric), safety/security (zero critical violations via guardrail + hook enforcement validation), context preservation (≥95% verbatim items survive compaction via pre/post compaction diff analysis), cost efficiency (≤20% variance from budget estimate via actual vs. predicted comparison), multi-agent coordination (≤85% coordination overhead via useful work tokens / total tokens ratio), learning retention (no repeated mistakes across sessions via cross-session error pattern matching), and latency (p95 ≤30s for standard tasks via end-to-end trace timing)
2. THE Evaluation subsystem SHALL use Strands Evals SDK 0.1.2 with LLM-as-a-Judge, trace-based evaluation, and ActorSimulator for multi-turn scenario testing
3. THE Evaluation subsystem SHALL integrate with AgentCore Evaluations (13 built-in evaluations including correctness, goal_success_rate, context_relevance, harmfulness, stereotyping, helpfulness, coherence, conciseness, tool_selection_accuracy, tool_parameter_accuracy) plus custom evaluators
4. THE Evaluation subsystem SHALL support both on-demand evaluation during development and continuous online evaluation in production

### Requirement 18: Deployment Topology

**User Story:** As a developer and enterprise architect, I want flexible deployment options from local-only to fully cloud-hosted, so that I can choose the topology that fits my security, performance, and cost requirements.

#### Acceptance Criteria

1. THE System SHALL support three deployment topologies: local-only (developer laptop), hybrid (local client + cloud orchestration), and fully cloud-hosted (enterprise SaaS)
2. ALL deployment topologies SHALL support the full feature set including Agent Teams, self-improving loops, plugin ecosystem, and cost governance
3. THE System SHALL support AgentCore Runtime deployment with microVM isolation per session, HTTP/MCP/A2A/WebSocket protocols, workloads up to 8 hours, 100MB payloads, immutable versioning with rollback, and consumption-based pricing
4. THE System SHALL support local development with Strands SDK and Ollama for free, fast iteration, Fargate or Lambda for staging, and AgentCore Runtime for production
5. THE System SHALL deploy via AgentCore CLI or MCP server using `BedrockAgentCoreApp` with the `@app.entrypoint` decorator (3 lines of code to wrap)

### Requirement 19: Configuration System

**User Story:** As a developer, I want a multi-scope configuration system that merges settings from user, project, and local levels, so that I can customize behavior at the right granularity with proper precedence.

#### Acceptance Criteria

1. THE Configuration subsystem SHALL load settings from three scopes with precedence: user (~/.brainmass/settings.json), project (.brainmass/settings.json, version-controlled), and project-local (.brainmass/settings.local.json, gitignored)
2. THE Configuration subsystem SHALL support the settings.json format with hooks (per-event arrays of matcher + handler definitions), feature flags (e.g., BRAINMASS_EXPERIMENTAL_AGENT_TEAMS), and env overrides (e.g., BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE, ENABLE_TOOL_SEARCH)
3. THE Configuration subsystem SHALL support .mcp.json for MCP server configuration with command, args, env (with ${ENV_VAR} substitution), and scope (project/user)
4. THE Configuration subsystem SHALL support enterprise managed settings with organization-level policy enforcement and allowManagedHooksOnly to block user/project/plugin hooks
5. THE Configuration subsystem SHALL support policy distribution via Git-based config repository for enterprise deployments

### Requirement 20: Brainmass-as-MCP-Server Mode

**User Story:** As a developer, I want the system to operate as both an MCP client and an MCP server, so that other tools and clients can consume the system's capabilities via the standard MCP protocol.

#### Acceptance Criteria

1. THE System SHALL support dual-mode operation: MCP client (consuming external MCP servers) and MCP server (exposing tools to other clients via stdio transport)
2. WHEN operating as an MCP server, THE System SHALL expose: Bash, Read, Write, Edit, LS, GrepTool, GlobTool, and Replace tools
3. WHEN operating as an MCP server, THE System SHALL NOT expose MCP servers configured within the system to connecting clients (no passthrough)
4. WHEN a client connects to the MCP server, THE System SHALL spawn a fresh instance with no shared state

### Requirement 21: Learning Store with Vector Index

**User Story:** As a developer, I want cross-session learning persistence with semantic search, so that the agent can recall relevant past mistakes, failed approaches, and discovered conventions when starting new tasks.

#### Acceptance Criteria

1. THE Learning_Store SHALL persist learnings at .brainmass/learnings/ with each entry containing: pattern, resolution, confidence (0-1), and source_iteration
2. THE Learning_Store SHALL add vector embeddings to learning entries using a lightweight embedding model for semantic similarity search
3. WHEN starting a new session or loop iteration, THE Learning_Store SHALL query for learnings relevant to the current task using top-K retrieval with configurable K, not loading all learnings indiscriminately
4. THE Learning_Store SHALL integrate with AgentCore Memory long-term storage with semantic strategy
5. THE Learning_Store SHALL reduce error repetition by 50% or more over 10 iterations of a self-improving loop

### Requirement 22: Compound Loop Orchestration

**User Story:** As a developer, I want to chain multiple agent loop types into pipelines where one agent's output feeds the next, so that I can build continuous delivery workflows (Analysis → Planning → Execution).

#### Acceptance Criteria

1. THE System SHALL support compound loop orchestration chaining Analysis Loop (reads reports/trackers/monitoring → prioritized task list), Planning Loop (takes task list → PRD, subtasks, acceptance criteria), and Execution Loop (implements tasks, commits, runs tests, updates status)
2. WHEN one loop stage completes, THE System SHALL pass its output as input to the next stage in the pipeline
3. THE System SHALL implement pipeline stages as configurable YAML-defined stages
4. THE System SHALL support pipeline chaining via SubagentStop hooks writing status to .brainmass/pipeline-state.json

### Requirement 23: Enterprise Managed Settings

**User Story:** As an enterprise administrator, I want organization-level policy enforcement that overrides user and project settings, so that I can ensure compliance across all developer workstations.

#### Acceptance Criteria

1. THE System SHALL support enterprise managed settings with organization-level hooks and policies
2. WHEN allowManagedHooksOnly is enabled, THE System SHALL block all user, project, and plugin hooks, enforcing only managed hooks
3. THE System SHALL support self-serve Enterprise deployment without requiring Sales involvement
4. THE System SHALL distribute enterprise policies via Git-based config repository

### Requirement 24: Agent Design Best Practices and Anti-Patterns

**User Story:** As a developer, I want clear guidelines on agent design limits and anti-patterns, so that I can create effective agent configurations without dispatch confusion or runaway behavior.

#### Acceptance Criteria

1. THE System SHALL enforce a recommended limit of 3-4 custom agents per project, warning when more than 5 agents are registered (dispatch accuracy drops with too many similar agents)
2. THE Agent_Registry SHALL validate that agent descriptions are specific and action-oriented with concrete examples, rejecting vague descriptions that match too broadly
3. THE Agent_Registry SHALL enforce mutual exclusivity between tools and disallowedTools fields in agent definitions
4. THE System SHALL recommend the color-role mapping convention: red (security/destructive), purple (code review/QA), blue (architecture/planning), green (implementation/building), cyan (research/read-only), orange (DevOps/infrastructure)
5. THE System SHALL warn against anti-patterns: too many agents (>5), vague descriptions, unrestricted write access on reviewers, missing maxTurns, hardcoded secrets in agent files, and loading more than 3 skills per agent

### Requirement 25: Strands SDK Multi-Agent Workflow Patterns

**User Story:** As a developer, I want access to all four Strands SDK multi-agent primitives, so that I can compose agents using the pattern best suited to each task type.

#### Acceptance Criteria

1. THE System SHALL support the Strands Agents-as-Tools pattern for hierarchical delegation where any Agent is wrapped as a tool callable by another Agent
2. THE System SHALL support the Strands Swarm pattern for dynamic agent handoff with shared context, where agents transfer control to specialized agents based on task requirements
3. THE System SHALL support the Strands Graph Orchestration pattern for DAG-based workflows with conditional edges, where nodes are agents or tools and edges define data flow and branching conditions
4. THE System SHALL support the Strands Workflow (Parallel) pattern for fan-out tasks to multiple agents with fan-in results, supporting parallel execution with configurable concurrency limits

### Requirement 26: Implementation Roadmap

**User Story:** As a project manager, I want a phased implementation roadmap with clear deliverables per phase, so that I can plan resources and track progress across the 24-week build.

#### Acceptance Criteria

1. THE System SHALL be implemented in 5 phases over 24 weeks: Phase 1 Foundation (Weeks 1-4, 30-39h: core orchestrator, context manager with semantic triage, hook engine with 12 events and 3 handler types, basic cost tracking), Phase 2 Multi-Agent (Weeks 5-10, 26-34h: hierarchical subagents, Agent Teams topology, self-improving loop framework, learning persistence store), Phase 3 Intelligence (Weeks 11-16, 18-24h: adaptive effort controls, model tier routing, prompt caching, MCP Tool Search, skill registry), Phase 4 Ecosystem (Weeks 17-20, 20-25h: plugin system, marketplace integration, session teleportation, cross-surface continuity), Phase 5 Enterprise (Weeks 21-24, 28-36h: unified quota management, HIPAA compliance, security hardening, evaluation suite, documentation)
2. THE System SHALL include integration tasks (26-35h): compound loop orchestration, enterprise managed settings, learning store vector index, end-to-end integration tests, documentation and BRAINMASS.md templates, and agent registry and dispatch system
3. THE total implementation effort SHALL be 148-193 hours for pure implementation, reduced to approximately 100-140 hours when leveraging AgentCore managed services (eliminating ~30-40 hours of infrastructure work)

### Requirement 27: Community-Proven Hook Patterns

**User Story:** As a developer, I want pre-built hook patterns for common use cases, so that I can quickly set up security guards, auto-formatting, context backup, and other proven workflows.

#### Acceptance Criteria

1. THE System SHALL provide a context backup on compaction pattern: a PreCompact hook that backs up critical state (PRESERVE_VERBATIM items) to .brainmass/backups/ before compaction destroys context
2. THE System SHALL provide a security guard pattern: a PreToolUse hook that blocks dangerous bash commands (rm -rf /, DROP TABLE, curl | bash, chmod 777, git push --force) with structured deny responses
3. THE System SHALL provide an auto-format on file write pattern: a PostToolUse hook on Write/Edit/MultiEdit events that runs a code formatter (e.g., prettier) on the modified file
4. THE System SHALL provide a self-improving loop context file pattern: a structured JSON file at .brainmass/loop-context.json with currentTask, acceptanceCriteria, constraints, learnings, failedApproaches, iterationCount, and maxIterations
5. THE System SHALL provide an agent team task list pattern: a structured JSON file at .brainmass/tasks/{team-name}/tasks.json with task entries containing id, title, assignee, status, dependencies, and files

### Requirement 28: Anthropic Messages API Integration

**User Story:** As a developer, I want the system to correctly integrate with the Anthropic Messages API including all request/response schemas, so that model interactions are properly formatted and all features (caching, thinking, streaming) are utilized.

#### Acceptance Criteria

1. THE System SHALL format API requests conforming to the Anthropic Messages API schema: model, max_tokens (1-128000), messages (role + content), system (string or SystemBlock[]), tools (ToolDefinition[]), tool_choice (auto/any/tool), temperature (0.0-1.0), thinking (type + budget_tokens), and stream
2. THE System SHALL support ContentBlock types: text, image, tool_use, tool_result, thinking, and document
3. THE System SHALL parse API response Usage metrics: input_tokens, output_tokens, cache_creation_input_tokens, and cache_read_input_tokens
4. THE System SHALL support streaming responses for real-time output display

## User Journeys

### Journey 1: Developer Setting Up the System for the First Time

1. Developer installs the Strands SDK (`pip install strands-agents strands-agents-tools`) and AgentCore SDK (`pip install bedrock-agentcore`)
2. Developer creates a project directory with the standard structure (src/, tests/, evals/)
3. Developer creates core type definitions (ModelTier, EffortLevel, ContextCategory, AgentBudget, HookEvent, etc.)
4. Developer creates a configuration loader that reads from 3 scopes (user, project, project-local) with correct precedence
5. Developer creates a BRAINMASS.md file with project conventions and guidance
6. Developer creates .brainmass/settings.json with initial hook configurations (security guard, auto-format)
7. Developer creates .mcp.json with MCP server configurations (e.g., GitHub, database)
8. Developer optionally creates custom agents in .brainmass/agents/ (e.g., code-reviewer.md, security-auditor.md)
9. Developer runs the system locally with Ollama for testing, then deploys to AgentCore Runtime for production via `agentcore deploy --name brainmass-v3 --framework strands`
10. System loads all configurations, registers hooks, discovers agents and skills, starts MCP servers, and presents the CLI interface

### Journey 2: Developer Making a Simple Code Change (Single Agent, Haiku)

1. Developer submits a simple request: "Fix the typo in the README"
2. Orchestrator fires SessionStart hooks (if new session)
3. Orchestrator fires UserPromptSubmit hooks
4. Effort_Controller classifies as Quick (1 file, no reasoning needed)
5. Cost_Governor selects Haiku ($0.80/$4 per M tokens) — cheapest capable model
6. Orchestrator executes: Read file → Edit file → verify change
7. PreToolUse hooks fire before each tool call (security guard checks)
8. PostToolUse hooks fire after each tool call (auto-format on Write/Edit)
9. Context_Manager classifies new items (file content → EPHEMERAL, edit result → EPHEMERAL)
10. Cost_Governor records usage (~$0.04 total for simple fix)
11. Orchestrator fires Stop hooks, returns result
12. Total cost: ~$0.04, latency: <10 seconds

### Journey 3: Developer Implementing a Complex Feature (Sonnet, with Subagents)

1. Developer submits: "Implement JWT authentication with refresh tokens for the API"
2. Orchestrator fires SessionStart and UserPromptSubmit hooks
3. Effort_Controller classifies as Standard (multiple files, moderate complexity)
4. Cost_Governor selects Sonnet ($3/$15 per M tokens) — default workhorse
5. Orchestrator decomposes into subtasks: design auth flow, implement token generation, implement middleware, write tests
6. Orchestrator delegates "research existing auth patterns" to a Researcher subagent (Haiku, read-only, plan mode)
7. Researcher returns findings; Orchestrator delegates implementation to Implementer-Tester subagent (Sonnet)
8. Each subagent operates in isolated context windows with own budgets
9. SubagentStop hooks fire on each completion, returning AgentResult with summary, tokens, files modified
10. Context_Manager preserves error messages and test output as PRESERVE_VERBATIM, compresses discussion as COMPRESS_AGGRESSIVE
11. Cache_Manager caches system prompts and tool schemas (1-hour), saving 30-50% on subsequent turns
12. After 15-25 turns, total cost: ~$0.75
13. Learning_Store records discovered patterns (e.g., "bcrypt.compare is async — always await")

### Journey 4: Team Lead Coordinating an Agent Team for a Security Audit

1. Team lead enables Agent Teams: sets BRAINMASS_EXPERIMENTAL_AGENT_TEAMS: "1" in settings.json
2. Team lead submits: "Perform a comprehensive security audit of the entire codebase"
3. Orchestrator selects Agent Teams topology (complex exploration, cross-agent communication needed)
4. Team_Manager spawns Team Lead (Opus) + 4 Teammates (Sonnet): auth-reviewer, input-validator, db-security-analyst, cross-referencer
5. Each teammate is a custom agent from .brainmass/agents/ with appropriate tools and skills (e.g., security-auditor.md with OWASP skill)
6. Team Lead creates shared task list: auth-review (teammate-1), input-validation (teammate-2), db-security (teammate-3), cross-reference (blocked on first three)
7. Teammates claim tasks via file-lock mechanism, begin working in parallel
8. Teammates communicate findings via mailbox system (JSON messages in inbox directories)
9. tmux split panes show each teammate's real-time activity
10. Cost_Governor tracks per-teammate costs; Team Lead monitors team_budget_usd
11. When first three tasks complete, cross-referencer unblocks and synthesizes findings
12. Git synchronization handles any merge conflicts autonomously
13. Total cost: ~$3.00 for 4-agent team (60-100 turns across all agents)

### Journey 5: Developer Running a Self-Improving Loop Overnight

1. Developer creates .brainmass/loop-context.json with: currentTask, acceptanceCriteria, constraints, maxIterations: 10
2. Developer starts the Loop_Runner and goes to sleep
3. Iteration 1: Fresh agent reads context file, implements initial approach, commits to git, discovers "bcrypt.compare is async", records learning, terminates
4. Iteration 2: Fresh agent reads updated context file (includes iteration 1 learnings), avoids sync bcrypt, implements correctly, discovers JWT_SECRET env issue, records learning, terminates
5. Iteration 3: Fresh agent reads context file with both learnings, implements complete solution, all tests pass, acceptance criteria met
6. Safety controls active throughout: live log monitoring, diff size limits, stop file check at each boundary
7. Learning_Store accumulates: { pattern: "bcrypt.compare is async", resolution: "Always await", confidence: 1.0 }, { pattern: "JWT_SECRET missing", resolution: "Load from .env with dotenv", confidence: 1.0 }
8. Developer reviews git log in the morning — clean audit trail of 3 iterations with progressive improvement

### Journey 6: Developer Creating and Deploying a Custom Agent

1. Developer creates .brainmass/agents/api-implementer.md with YAML frontmatter: name: api-implementer, description (action-oriented with examples), model: sonnet, tools: Read/Write/Edit/Bash, color: green, maxTurns: 100, hooks (PostToolUse auto-format), skills: api-patterns
2. Agent_Registry detects new file via filesystem watcher (hot-reload)
3. Agent_Loader parses YAML frontmatter, validates required fields, normalizes tool lists, resolves model alias
4. Agent_Registry registers agent and generates tool definition (name: 'agent:api-implementer', description from frontmatter)
5. Developer says "Implement the /users CRUD endpoints" — model auto-dispatches to api-implementer based on description match
6. Agent_Runner executes 13-step lifecycle: creates isolated context, resolves model, builds tool list, loads skills, registers scoped hooks, executes agentic loop
7. Agent completes, SubagentStop hooks fire, scoped hooks deregistered, MCP servers stopped, AgentResult returned to parent
8. Developer can also invoke explicitly: "Use the api-implementer agent" or via /agents menu

### Journey 7: Developer Installing and Using a Plugin

1. Developer adds a community marketplace: /plugin marketplace add community-org/brainmass-plugins
2. Developer browses available plugins and installs: /plugin install security-toolkit@community-org/brainmass-plugins
3. Plugin_Registry downloads plugin, validates plugin.json manifest
4. Plugin_Registry registers all capabilities: commands (security-scan), agents (vuln-scanner with namespace security-toolkit:vuln-scanner), skills (owasp-checklist), hooks (PreToolUse security guard), MCP servers (dependency-checker)
5. Developer uses plugin: "Scan the codebase for vulnerabilities" — auto-dispatches to security-toolkit:vuln-scanner agent
6. Plugin hooks fire alongside user/project hooks (additive)
7. Plugin MCP servers start on demand, provide additional tools
8. Developer can toggle plugin off for specific projects or uninstall entirely

### Journey 8: Enterprise Admin Configuring Organization-Wide Policies

1. Admin creates enterprise managed settings in Git-based config repository
2. Admin enables allowManagedHooksOnly to block all user/project/plugin hooks
3. Admin configures organization-level hooks: security guard (PreToolUse blocking dangerous commands), compliance checker (PostToolUse verifying code standards), audit logger (all events logged to central system)
4. Admin configures Cedar policies via AgentCore Policy for fine-grained tool-level governance
5. Admin configures AgentCore Identity with Okta/Entra/Cognito for authentication
6. Admin sets organization-wide token budgets and model tier restrictions
7. Admin enables HIPAA compliance path with data classification for PHI
8. Admin deploys via self-serve Enterprise deployment (no Sales gate)
9. All developer workstations receive managed settings on next session start
10. Developers see managed hooks active; cannot override with local settings

### Journey 9: Developer Teleporting a Session from Web to Terminal

1. Developer is working on a feature in the web interface with 15 turns of conversation
2. Developer needs to switch to terminal for deeper debugging
3. Developer invokes /teleport
4. Session_Teleporter serializes all state: conversation history, tool permissions (already approved Bash, Write), active subagent (researcher still running), compaction state, cost tracking ($0.45 spent so far)
5. State blob stored via Strands SessionManager with S3 backend
6. Terminal session starts, SessionStart hook fires with source: "teleport"
7. Context_Manager loads serialized state, researcher subagent resumes
8. All tool permissions inherited — no re-approval needed
9. Single trace ID maintained across web → terminal for audit continuity
10. Cost_Governor continues tracking from $0.45 — unified quota regardless of surface
11. Developer continues debugging in terminal seamlessly

### Journey 10: System Handling Context Compaction with Semantic Triage

1. Developer is 30 turns into a complex refactoring session on a 200K context window
2. Context_Manager detects 83.5% capacity reached (~167K tokens)
3. PreCompact hook fires — backup script saves critical state to .brainmass/backups/
4. Context_Manager classifies all items: error messages and file paths → PRESERVE_VERBATIM (persisted to .brainmass/session-state.json), decision records and task lists → PRESERVE_STRUCTURED (persisted to session file), discussion and reasoning → COMPRESS_AGGRESSIVE (summarized to key takeaways), tool metadata and search results → EPHEMERAL (dropped)
5. Context editing clears stale tool call results (staleness score above threshold)
6. Compaction executes: 84% token reduction while preserving all critical information
7. SessionStart hook restores PRESERVE_VERBATIM and PRESERVE_STRUCTURED items from session file
8. Context health metrics update: freePercent jumps to ~60%, preservedTokens shows retained items
9. Developer continues working — agent remembers exact error messages, file paths, and decisions
10. Observability dashboard shows compaction event, token savings, and preservation queue

### Journey 11: Cost Governor Downgrading Models When Budget Is Tight

1. Developer starts a session with session_budget_usd: $2.00, using Opus for architecture work
2. After 20 turns, Cost_Governor reports $1.60 spent (80% of budget)
3. Cost_Governor triggers model downgrade: Opus → Sonnet for remaining work
4. Developer is notified of the downgrade with reason: "Approaching budget limit (80%)"
5. After 5 more turns on Sonnet, Cost_Governor reports $1.90 spent (95% of budget)
6. Cost_Governor pauses the agent and requests human approval: "Budget 95% consumed. Continue?"
7. Developer approves continuation with additional $1.00 budget
8. Cost_Governor resumes on Sonnet, tracks against new $3.00 total budget
9. Real-time dashboard shows: cost per agent, model tier distribution (Opus for first 20 turns, Sonnet for remaining), cache hit rates
10. Per-task cost attribution tags each task with input/output tokens and USD

### Journey 12: Developer Using the Evaluation Framework

1. Developer creates evaluation suite in evals/ directory using Strands Evals SDK
2. Developer defines test scenarios for each of the 8 dimensions: task completion (automated acceptance criteria), code quality (LLM-as-a-Judge rubric), safety (guardrail validation), context preservation (compaction diff analysis), cost efficiency (budget variance), multi-agent coordination (overhead ratio), learning retention (error pattern matching), latency (trace timing)
3. Developer runs on-demand evaluation during development
4. ActorSimulator generates multi-turn test scenarios simulating real developer interactions
5. Evaluation results show: task completion 92% (pass), code quality 4.2/5.0 (pass), safety zero violations (pass), context preservation 97% (pass), cost efficiency 15% variance (pass), coordination overhead 78% (pass), learning retention confirmed (pass), p95 latency 25s (pass)
6. Developer integrates with AgentCore Evaluations for continuous online evaluation in production
7. AgentCore provides 13 built-in evaluations (correctness, goal_success_rate, context_relevance, etc.) alongside custom evaluators
8. Dashboard shows evaluation trends over time, highlighting regressions
