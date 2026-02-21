**Claude Code v3 Enterprise System Specification**

Version 1.1 — Gap Analysis Integration

February 2026

*Strands Agents SDK + Claude Code Behavioral Parity*

**Change Log**

**v1.0 (Feb 19, 2026):** Initial 11-section specification covering architecture, multi-agent orchestration, hooks, evaluation, security, and deployment.

**v1.1 (Feb 20, 2026):** Integrated 17 critical gaps identified through research on Anthropic's February 2026 releases (Opus 4.6, Agent Teams, 1M context), community pain points, real-world usage patterns, and the plugin ecosystem. Added 6 new sections, expanded 5 existing sections.

**Table of Contents**

- 1. Executive Summary (updated)

- 2. System Architecture (updated)

- 3. Context Management Architecture (NEW)

- 4. Multi-Agent Orchestration (updated)

- 5. Agent Teams Topology (NEW)

- 6. Self-Improving Agent Loops (NEW)

- 7. Hooks and Policy Engine (updated)

- 8. Plugin Ecosystem (NEW)

- 9. Skills Registry (NEW)

- 10. Token Economics and Cost Governance (NEW)

- 11. Prompt Caching Architecture (NEW)

- 12. Adaptive Thinking and Effort Controls (NEW)

- 13. Evaluation Framework (updated)

- 14. Observability and Audit

- 15. Security and Governance (updated)

- 16. Session Teleportation and Cross-Surface Continuity (NEW)

- 17. Deployment Topology

- 18. Acceptance Criteria (updated)

- 19. Implementation Roadmap (updated)

- A. Appendix: Strands SDK Feature Mapping

- B. Appendix: Hook Events Reference

- C. Appendix: Pricing and Token Economics Reference

**1. Executive Summary**

This specification defines an enterprise-grade agentic coding system built on the AWS Strands Agents SDK, achieving behavioral parity with Claude Code's confirmed architecture while adding enterprise governance, cost management, and multi-agent coordination capabilities.

**1.1 Core Design Principles**

**Deterministic guardrails > prompt promises:** Hooks and policy enforce behavior; CLAUDE.md is guidance, hooks are guarantees.

**Decoupled control/data planes:** Routing, orchestration, and policy separated from tool execution.

**MCP-first tool integration:** All external tools via Model Context Protocol servers and gateways.

**Production observability first-class:** OpenTelemetry traces, cost attribution, and audit logging from day one.

**Context-window-adaptive architecture:** Orchestration strategy changes based on available context (200K vs 1M).

**Cost-aware model routing:** Automatic model tier selection based on task complexity and budget constraints.

**Cross-session learning persistence:** Agents accumulate and apply learnings across iterations.

**Plugin-extensible by design:** Skills, commands, agents, hooks, and MCP servers as shareable packages.

**1.2 What Changed in v1.1**

Research conducted February 19-20, 2026 identified 17 critical gaps in the v1.0 specification. These fall into four categories:

- Anthropic releases (Feb 2026): Opus 4.6 with 1M context, Agent Teams, adaptive thinking, effort controls, Fast Mode, plugin ecosystem, 12 hook events (up from 7), MCP Tool Search.

- Community pain points: Context management crisis (#1 complaint), token cost explosion (#2), skill auto-invocation confusion, cross-surface session loss.

- Real-world patterns: Self-improving loops ("Ralph Wiggum" pattern), compound product pipelines, cross-session learning, cost-aware model routing.

- Enterprise gaps: Unified quota management across all Claude surfaces, HIPAA compliance path, weekly rate limits for 24/7 agents, self-serve Enterprise deployment.

**2. System Architecture**

The system is organized into three tiers with seven components (expanded from six in v1.0 to add the Plugin Registry).

**2.1 Component Overview**

|  |  |  |
|----|----|----|
| **Tier** | **Component** | **Responsibility** |
| Client | CLI / IDE Extension | UI rendering, session store, local state collection, secret redaction, plugin management |
| Client | Plugin Registry | Plugin discovery, installation, marketplace management, capability registration (NEW) |
| Control | Orchestrator | Model-driven routing, task decomposition, agent team coordination, learning persistence |
| Control | Context Manager | Semantic context triage, tiered persistence, compaction strategy, cache orchestration (NEW) |
| Control | Policy Engine | 12 lifecycle hooks (up from 7), 3 handler types, enterprise managed settings |
| Data | MCP Tool Gateway | Tool discovery via Tool Search, execution, Claude-as-MCP-Server dual mode |
| Data | Worker Pool | Subagents, Agent Teams (peer topology), self-improving loop orchestration |

**2.2 Model Tier Architecture**

v1.1 introduces explicit model tier routing as a first-class architectural concern. The system dynamically selects models based on task complexity, budget constraints, and latency requirements.

|  |  |  |
|----|----|----|
| **Model** | **Use Cases** | **Pricing (per M tokens)** |
| Opus 4.6 | Complex reasoning, architecture decisions, hard debugging, Agent Team leads | $5 input / $25 output |
| Sonnet 4.5 | Standard implementation, code generation, refactoring, most coding tasks | $3 input / $15 output (>200K: $6/$22.50) |
| Haiku 4.5 | Search, exploration, linting, simple transforms, prompt hook evaluation | $0.80 input / $4 output |
| Batch API | Async processing, bulk analysis, non-interactive tasks | 50% discount on all tiers |

**3. Context Management Architecture**

Context management is the #1 user pain point with Claude Code today. Auto-compact triggers at approximately 83.5% of the context window and produces lossy summaries that destroy critical information (exact error messages, variable names, nuanced decisions). This section specifies a Semantic Context Triage system that categorizes information by compressibility and applies different retention strategies.

**3.1 Context Categories**

|  |  |  |
|----|----|----|
| **Category** | **Retention** | **Examples** |
| PRESERVE_VERBATIM | Never summarize; persist to session file | Error messages, file paths, variable names, test output, git diff hunks, API responses |
| PRESERVE_STRUCTURED | Compress format, keep data | Decision records, task lists, acceptance criteria, schemas |
| COMPRESS_AGGRESSIVE | Summarize to key takeaways | Discussion, reasoning chains, alternatives considered, exploration attempts |
| EPHEMERAL | Drop after use | Tool call metadata, raw search results, intermediate compilation output |

**3.2 Tiered Persistence Architecture**

The Context Manager maintains three storage tiers that work together to preserve critical information across compaction events:

- Working Memory (in-context): Current conversation turns, active tool calls, recent decisions. Subject to compaction.

- Session File (.claude/session-state.json): Persisted between compactions. Stores PRESERVE_VERBATIM and PRESERVE_STRUCTURED items. Automatically loaded into context after compaction via PreCompact hook backup + SessionStart hook restore.

- Learning Store (.claude/learnings/): Cross-session persistent store. Vector-indexed memory of past mistakes, failed approaches, discovered conventions. Loaded selectively based on semantic similarity to current task.

**3.3 Compaction Strategy**

Compaction behavior adapts to the available context window:

|  |  |  |
|----|----|----|
| **Window Size** | **Trigger Threshold** | **Strategy** |
| 200K tokens | 83.5% (~167K) | Semantic triage: preserve verbatim items to session file, compress discussion, drop ephemeral. Configurable via CLAUDE_AUTOCOMPACT_PCT_OVERRIDE (1-100). |
| 1M tokens (beta) | 85% (~850K) | Deferred compaction. Load entire medium codebases into context. Only compact when hitting premium pricing threshold (>200K input) unless budget allows. |
| Manual (/compact) | User-initiated | Best practice: compact at logical task boundaries. Backup via PreCompact hook before compaction occurs. |

**3.4 Context Editing Integration**

Anthropic's September 2025 context editing feature demonstrated that selectively clearing stale tool calls while preserving conversation flow reduced token consumption by 84% and improved task performance by 29-39%. The Context Manager implements this by: tagging each context block with a staleness score based on recency and reference count, automatically clearing tool call results that haven't been referenced in the last N turns, preserving conversation flow by keeping user messages and key assistant decisions, and surfacing context health metrics (free %, staleness distribution, preservation queue size) in the observability dashboard.

**4. Multi-Agent Orchestration**

v1.1 recognizes three distinct multi-agent topologies, each suited to different task types. The v1.0 spec only covered hierarchical delegation. Agent Teams (Section 5) and Self-Improving Loops (Section 6) are now first-class patterns.

**4.1 Topology Selection Matrix**

|  |  |  |
|----|----|----|
| **Topology** | **When to Use** | **Cost Profile** |
| Hierarchical Subagents | Well-defined subtasks, clear input/output contracts, sequential dependencies | 1 orchestrator + N workers. Each worker has own context. Moderate cost. |
| Agent Teams (Peer) | Complex exploration requiring cross-agent communication, code review, security audit | 1 lead + N teammates, each full Claude instance. High cost (N× context windows). Enable: CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1 |
| Self-Improving Loops | Repetitive tasks with learnable patterns, overnight batch work, iterative refinement | 1 agent per iteration (serial). Low per-iteration cost. Accumulated learning reduces total iterations. |

**4.2 Hierarchical Subagent Pattern (v1.0)**

The Orchestrator decomposes tasks and delegates to Worker Subagents via the Task tool. Workers operate in independent context windows and return structured results. The Orchestrator synthesizes results and coordinates sequencing. Subagent frontmatter can define scoped hooks (e.g., SubagentStop for cleanup). This pattern maps directly to Strands' Agents-as-Tools and Swarm patterns.

**4.3 Strands SDK Multi-Agent Primitives**

- Agents-as-Tools: Wrap any Agent as a tool callable by another Agent. The calling agent provides a natural language request; the callee returns results.

- Swarm: Dynamic agent handoff with shared context. Agents can transfer control to specialized agents based on task requirements.

- Graph Orchestration: DAG-based workflow with conditional edges. Nodes are agents or tools; edges define data flow and branching conditions.

- Workflow (Parallel): Fan-out tasks to multiple agents, fan-in results. Supports parallel execution with configurable concurrency limits.

**5. Agent Teams Topology**

Released February 5, 2026 with Opus 4.6, Agent Teams are fundamentally different from the hierarchical subagent pattern. Instead of a single orchestrator delegating to workers who report back, Agent Teams enable peer-to-peer collaboration where teammates message each other directly, share task lists with dependency tracking, and claim work via file-lock-based coordination.

**5.1 Architecture**

- Team Lead: One session designated as coordinator. Spawns teammates via TeammateTool. Maintains shared task list.

- Teammates: Each is a full Claude Code instance with its own context window. Can message each other directly (not just orchestrator). Claim tasks via file-lock mechanism to prevent race conditions.

- Shared Task List: Central task registry with status tracking (pending, claimed, blocked, complete), dependency declarations (task B blocks on task A), and file ownership mapping.

- Inbox System: Each teammate has a mailbox. Messages are structured (task assignments, findings, questions, status updates). Team lead can broadcast to all.

- Visibility: tmux split panes show each teammate's activity in real-time.

**5.2 Configuration**

Enable via settings.json: { "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1" }. Team configuration stored at ~/.claude/teams/{team-name}/config.json. Task state stored at ~/.claude/tasks/{team-name}/. Each teammate spawned as a separate process with its own context window, connected via file-system-based IPC.

**5.3 Cost Implications**

Agent Teams multiply cost by the number of teammates (each has a full context window). A 16-agent team building a C compiler consumed approximately $20,000 in API costs across 2,000 sessions. The Cost Governance layer (Section 10) must enforce per-team budgets and provide real-time cost tracking per teammate.

**5.4 Task Decomposition Guidelines**

- Good: Assign by file ownership (Agent 1: auth module, Agent 2: database layer, Agent 3: API endpoints, Agent 4: cross-reference findings).

- Bad: Assign by vague scope (Agent 1: fix all bugs) or create unnecessary dependencies (Agent 2: wait for Agent 1).

- Teams maintain running docs of failed approaches so teammates don't repeat mistakes.

- Coordination via git synchronization handles merge conflicts autonomously.

**6. Self-Improving Agent Loops**

The "Ralph Wiggum" pattern (named by community developer Geoffrey Huntley) is a paradigm for autonomous, iterative agent work. Each iteration spawns a fresh agent with a clean context window, feeds it a structured context file (not conversation history), has it do work, commit to git, update the context file with learnings, terminate, and loop.

**6.1 Loop Architecture**

- Iteration Isolation: Each iteration starts with a clean context window. No conversation history carried over. This prevents context window pollution and ensures consistent behavior.

- Structured Context File: A JSON/YAML file containing current task description, acceptance criteria, known constraints, previous iteration learnings, and failed approaches to avoid.

- Git-Based Checkpointing: Every iteration commits to git. This provides an audit trail, enables rollback, and allows diff-based review of agent work.

- Learning Accumulation: The context file grows across iterations with discovered patterns, error signatures and their fixes, codebase conventions learned, and performance insights.

**6.2 Compound Loop Orchestration**

Advanced workflows chain multiple loop types in a pipeline:

- Analysis Loop: AI reads daily reports, issue trackers, and monitoring to identify what to build. Output: prioritized task list.

- Planning Loop: Takes task list, generates PRD, breaks into subtasks, defines acceptance criteria. Output: structured task specifications.

- Execution Loop: Coding agent implements tasks from the plan. Each iteration tackles one task, commits, runs tests, updates status.

One agent's output becomes the next agent's input in a continuous delivery pipeline. The Compound Product system is the reference implementation for this pattern.

**6.3 Safety Controls**

- Live log monitoring: Tail agent output in real-time. If agent loops on same error 3+ times, auto-pause.

- Diff size limits: Abort if a single iteration's diff exceeds expected bounds or touches files outside task scope.

- Stop file: Place a sentinel file (.auto/stop) to gracefully halt the loop at the next iteration boundary.

- Acceptance criteria gate: Each iteration checks if all acceptance criteria pass before marking task complete.

**7. Hooks and Policy Engine**

v1.1 updates this section to reflect the current state of Claude Code hooks: 12 lifecycle events (up from 7 in v1.0), 3 handler types (command, prompt, agent), and integration with skills and subagent frontmatter.

**7.1 Hook Events (Complete)**

|  |  |  |
|----|----|----|
| **Event** | **Fires When** | **Can Block?** |
| SessionStart | New session starts or existing session resumes | No |
| UserPromptSubmit | User submits a prompt | No (can modify) |
| PreToolUse | Before any tool executes | Yes (exit code 2 or deny decision) |
| PostToolUse | After tool completes successfully | No (can provide feedback) |
| PostToolUseFailure | After tool execution fails | No |
| PermissionRequest | When Claude requests tool permission from user | Yes (auto-allow/deny) |
| PreCompact | Before context compaction occurs | No (use for backup) |
| Notification | When Claude sends an alert | No |
| Stop | When agent finishes responding | Yes (can block completion) |
| SubagentStop | When a subagent finishes | No |
| SessionEnd | When session terminates | No |
| Setup | Triggered via --init, --init-only, or --maintenance CLI flags | No |

**7.2 Handler Types**

**Command Handlers:** Shell scripts that receive JSON context on stdin. Best for deterministic tasks (formatting, linting, logging, security checks). Return exit code 0 (allow), 2 (deny), or structured JSON for nuanced control. Timeout: 10 minutes.

**Prompt Handlers:** Send a text prompt to a fast Claude model (Haiku by default, configurable) for single-turn semantic evaluation. Use $ARGUMENTS placeholder for input injection. Best for context-aware decisions (TDD enforcement, style compliance, commit message quality).

**Agent Handlers:** Spawn a sub-agent with access to Read, Grep, and Glob tools for multi-turn codebase verification. Heaviest handler type. Best for deep validation (test coverage verification, security audit, dependency impact analysis).

**7.3 Hook Configuration Scopes**

- User-wide: ~/.claude/settings.json (applies to all projects)

- Project-shared: .claude/settings.json (version-controlled, team-shared)

- Project-local: .claude/settings.local.json (gitignored, personal overrides)

- Skill frontmatter: Hooks scoped to skill lifecycle (active only when skill is invoked)

- Subagent frontmatter: Hooks scoped to subagent lifecycle (Stop auto-converted to SubagentStop)

- Plugin hooks: Bundled with plugin, active when plugin is enabled

- Enterprise managed: Organization-level hooks. allowManagedHooksOnly blocks user/project/plugin hooks.

**7.4 PreToolUse Input Modification**

Since v2.0.10, PreToolUse hooks can modify tool inputs before execution. The hook receives JSON on stdin, outputs modified JSON to stdout. Claude Code uses the modified input transparently. This enables: automatic sandboxing (add --dry-run flags), secret redaction, path correction, team convention enforcement (commit message formatting), and dependency auto-installation. The modification is invisible to the model.

**8. Plugin Ecosystem**

Claude Code's plugin system (public beta since October 2025) enables shareable packages that bundle slash commands, specialized agents, MCP servers, hooks, skills, and LSP servers. Over 9,000 plugins are available across community marketplaces as of February 2026.

**8.1 Plugin Structure**

Every plugin follows a standard directory layout:

- plugin-name/.claude-plugin/plugin.json — Manifest (name, description, version, author)

- plugin-name/commands/ — Slash commands (optional)

- plugin-name/agents/ — Specialized agents with YAML frontmatter (optional)

- plugin-name/skills/ — Agent skills with SKILL.md files (optional)

- plugin-name/hooks/ — Event handlers (optional)

- plugin-name/.mcp.json — MCP server configuration (optional)

- plugin-name/.lsp.json — Language Server Protocol configuration (optional)

**8.2 Marketplace Architecture**

Marketplaces are Git repositories with a .claude-plugin/marketplace.json catalog. The official Anthropic marketplace (claude-plugins-official) ships pre-configured. Community marketplaces are added via: /plugin marketplace add {owner}/{repo}. Plugins installed via: /plugin install {plugin-name}@{marketplace}. Plugins can be toggled on/off per project context.

**8.3 MCP Tool Search**

When MCP tool definitions exceed the context threshold, Tool Search activates automatically (lazy loading). This reduces context usage by up to 95% for systems with many MCP servers. Control via ENABLE_TOOL_SEARCH environment variable: auto (default, activates when needed), auto:N (custom threshold percentage), false (disabled). Requires Sonnet 4+ or Opus 4+ models. Haiku does not support tool_reference blocks.

**9. Skills Registry**

Skills are reusable workflows that Claude Code invokes based on task context matching. v1.1 addresses the community-reported pain point of uncontrolled skill auto-invocation by specifying explicit invocation, conditional activation, and transparency requirements.

**9.1 Skill Invocation Modes**

- Auto-invocation (default): Claude matches task description against skill descriptions. Triggers automatically when confidence exceeds threshold. Disable per-skill with disable-model-invocation: true in frontmatter.

- Slash command: Skills appear in the slash command menu. User explicitly invokes via /skill-name. $ARGUMENTS placeholder captures user input after the command name.

- Programmatic: Plugins can invoke skills via the capability registry during agent execution.

**9.2 Skill Frontmatter Schema**

Skills are defined in SKILL.md files with YAML frontmatter:

- name: Skill identifier (namespaced when part of a plugin: /plugin-name:skill-name)

- description: Natural language description used for auto-invocation matching

- disable-model-invocation: true|false — Opt out of auto-invocation

- hooks: Scoped hooks that only run when this skill is active (all 12 events supported)

- allowed_tools: Restrict which tools this skill can access

**9.3 Transparency Requirements**

- Invocation logging: Record which skills were considered for each task, their confidence scores, and why each was selected or rejected.

- Context consumption tracking: Each skill's contribution to context window usage must be visible in observability dashboards.

- Hot-reload: Skills in ~/.claude/skills or .claude/skills are immediately available without restart.

**10. Token Economics and Cost Governance**

Cost management is the #2 user complaint after context loss. This section specifies the cost governance layer that the v1.0 spec lacked entirely.

**10.1 Per-Agent Token Budgets**

Every agent (orchestrator, worker, teammate) operates within a declared token budget:

- input_budget_tokens: Maximum input tokens per turn (controls context loading)

- output_budget_tokens: Maximum output tokens per turn (controls generation length)

- session_budget_usd: Maximum total cost for the agent's session lifecycle

- team_budget_usd: Maximum total cost across all teammates in an Agent Team

When an agent approaches 80% of its budget, the Cost Governor triggers a model downgrade (Opus → Sonnet → Haiku). At 95%, the agent pauses and requests human approval to continue.

**10.2 Model Tier Routing Logic**

The Cost Governor routes tasks to the cheapest model capable of handling them:

- Haiku ($0.80/$4): File search, grep, exploration, linting, simple transforms, prompt hook evaluation. 1/10th the cost of Sonnet.

- Sonnet ($3/$15): Standard code generation, refactoring, implementation, test writing. Default tier.

- Opus ($5/$25): Complex reasoning, architecture decisions, cross-service refactoring, Agent Team leads. Only used when Sonnet-tier models would likely fail.

- Batch API (50% discount): Async processing for non-interactive tasks (bulk linting, large-scale analysis).

**10.3 Unified Quota Management**

Critical: Usage limits are shared across ALL Claude surfaces (web, mobile, desktop, CLI). The Cost Governor must: track consumption across surfaces in real-time, predict quota exhaustion based on current run rate, alert when approaching limits (80% threshold), provide usage breakdowns by surface and agent, and support overage purchase at API rates for Max subscribers.

**10.4 Cost Monitoring Tools**

- Real-time dashboard: Token consumption, cost per agent, model tier distribution, cache hit rates.

- ccusage CLI integration: npx ccusage@latest blocks --live for real-time terminal monitoring.

- SDKRateLimitInfo events: Consume rate limit status updates including utilization, reset times, and overage information.

- Per-task cost attribution: Every task tagged with total input/output tokens and USD cost.

**11. Prompt Caching Architecture**

Anthropic offers 5-minute and 1-hour cache durations. Cached input tokens cost only 10% of the base rate. For agentic coding systems that reload system prompts, CLAUDE.md contents, and tool definitions on every turn, caching is a massive cost lever (70-80% savings in production).

**11.1 Cache Layer Design**

|  |  |  |
|----|----|----|
| **Layer** | **Cacheable?** | **Strategy** |
| System prompts | Yes (stable) | 1-hour cache. Changes only on deployment. |
| CLAUDE.md contents | Yes (stable) | 1-hour cache. Changes only on git pull. |
| Tool schemas / MCP definitions | Yes (stable) | 1-hour cache. Changes only on server restart. |
| Policy/hook definitions | Yes (stable) | 5-minute cache. May change during session. |
| Skill instructions | Partially | 5-minute cache. Hot-reload may invalidate. |
| Conversation history | No (volatile) | Changes every turn. Never cache. |
| Current diff state | No (volatile) | Changes every tool call. Never cache. |

**11.2 Cache Economics**

Cache writes cost 1.25x (5-min) or 2x (1-hour) the base input rate. Break-even: 2 reads for 5-minute cache, 8 reads for 1-hour cache. In a typical coding session with 20-50 turns, 1-hour caching of stable layers pays for itself after the first 8 turns. The Cost Governor tracks cache hit rates and alerts if rates fall below 70%.

**12. Adaptive Thinking and Effort Controls**

Opus 4.6 introduced adaptive thinking (model picks up contextual clues about thinking depth) and explicit effort controls for developers. This enables per-task reasoning effort allocation — a simple linting fix doesn't need the same thinking budget as a cross-service architectural refactor.

**12.1 Effort Taxonomy**

|  |  |  |
|----|----|----|
| **Level** | **Use Cases** | **Characteristics** |
| Quick (Fast Mode) | Linting, formatting, simple renames, exploration | Minimal extended thinking. Low latency. Low cost. |
| Standard | Code generation, refactoring, test writing, most implementation | Moderate thinking. Balanced latency/quality. |
| Deep | Architecture decisions, complex debugging, security analysis, cross-service refactoring | Extended thinking. Higher latency. Best quality. |

**12.2 Effort Selection Logic**

The Orchestrator selects effort level based on: task complexity signal (number of files touched, dependency depth, ambiguity in requirements), model capability (Haiku always Quick, Sonnet Quick or Standard, Opus any level), budget constraints (downgrade effort when approaching budget limits), and explicit override (developer can force effort level in tool calls or hook configuration). Fast Mode for Opus 4.6 is available via configuration and adds a speed attribute to OTel events and trace spans for visibility.

**13. Evaluation Framework**

The evaluation framework from v1.0 is retained with the addition of cost efficiency and context preservation metrics. The system uses Strands Evals SDK 0.1.2 with LLM-as-a-Judge, trace-based evaluation, and ActorSimulator for multi-turn testing.

**13.1 Evaluation Dimensions (Updated)**

|  |  |  |
|----|----|----|
| **Dimension** | **Target** | **Method** |
| Task completion | ≥90% | Automated acceptance criteria verification |
| Code quality | ≥4.0/5.0 | LLM-as-a-Judge with rubric |
| Safety / security | Zero critical violations | Guardrail + hook enforcement validation |
| Context preservation | ≥95% verbatim items survive compaction (NEW) | Pre/post compaction diff analysis |
| Cost efficiency | ≤20% variance from budget estimate (NEW) | Actual vs. predicted token/USD comparison |
| Multi-agent coordination | ≤85% coordination overhead (NEW) | Useful work tokens / total tokens ratio |
| Learning retention | No repeated mistakes across sessions (NEW) | Cross-session error pattern matching |
| Latency (p95) | ≤30s for standard tasks | End-to-end trace timing |

**14. Observability and Audit**

Retained from v1.0. OpenTelemetry integration via Strands' built-in instrumentation. All agent actions, tool calls, hook executions, and model interactions emit structured traces. New in v1.1: cost attribution spans (token counts + USD per trace), context health metrics (free %, staleness scores, cache hit rates), Agent Team coordination traces (inter-teammate messages, task claims, file locks), and Fast Mode speed attribute on OTel events.

**15. Security and Governance**

Updated from v1.0 to include:

- Unified quota management across all Claude surfaces (web, mobile, desktop, CLI)

- Weekly rate limit architecture for 24/7 agent users (affects \<5% of users)

- HIPAA compliance path for Enterprise plans processing PHI

- Health data handling with explicit consent requirements (iOS/Android)

- Constitutional AI filter integration for dangerous/unethical suggestion blocking

- Bedrock Guardrails via guardrail_latest_message parameter in Strands

- Self-serve Enterprise deployment (no Sales required as of Feb 2026)

- Data classification for health/financial data in cross-surface identity propagation

- Guard against launching Claude Code inside another Claude Code session

**16. Session Teleportation and Cross-Surface Continuity**

Claude Code supports /teleport (move session from web/mobile to terminal) and /desktop (hand off to desktop app for visual diff review). The specification requires:

- Session serialization format: All state (conversation, tool permissions, active workers, pending approvals, compaction state, cost tracking) serialized to a portable blob.

- Permission inheritance: Tool approvals persist across surface transitions.

- Worker lifecycle management: Active subagents and teammates survive teleport.

- Audit continuity: Single trace ID across all surfaces for a given session.

- Cost attribution: Unified quota regardless of which surface is active.

- Conflict resolution: Simultaneous edits from multiple surfaces handled via last-write-wins with conflict notification.

**17. Deployment Topology**

Retained from v1.0. Three deployment options: local-only (developer laptop), hybrid (local client + cloud orchestration), and fully cloud-hosted (enterprise SaaS). All options must support the full feature set specified in this document, including Agent Teams, self-improving loops, plugin ecosystem, and cost governance.

**18. Acceptance Criteria**

Updated from v1.0 to include v1.1 requirements:

- Context triage system correctly categorizes 95%+ of context items

- Compaction preserves all PRESERVE_VERBATIM items across compaction events

- Agent Teams coordinate without edit collisions via file-lock mechanism

- Self-improving loops accumulate learnings that reduce error repetition by 50%+ over 10 iterations

- All 12 hook events fire at correct lifecycle points with all 3 handler types

- Cost Governor enforces budgets and triggers model downgrades at thresholds

- Prompt caching achieves 70%+ cache hit rate for stable layers after warm-up

- Plugin install/uninstall lifecycle works across all marketplace types

- Session teleportation preserves full state across surface transitions

- Effort controls correctly route to Quick/Standard/Deep based on task signals

**19. Implementation Roadmap**

Updated from v1.0. Five phases over 24 weeks (expanded from 20):

|  |  |  |
|----|----|----|
| **Phase** | **Duration** | **Deliverables** |
| Phase 1: Foundation | Weeks 1-4 | Core orchestrator, context manager with semantic triage, hook engine (12 events, 3 handler types), basic cost tracking |
| Phase 2: Multi-Agent | Weeks 5-10 | Hierarchical subagents, Agent Teams topology, self-improving loop framework, learning persistence store |
| Phase 3: Intelligence | Weeks 11-16 | Adaptive effort controls, model tier routing, prompt caching, MCP Tool Search, skill registry |
| Phase 4: Ecosystem | Weeks 17-20 | Plugin system, marketplace integration, session teleportation, cross-surface continuity |
| Phase 5: Enterprise | Weeks 21-24 | Unified quota management, HIPAA compliance, security hardening, evaluation suite, documentation |

**Appendix A: Strands SDK Feature Mapping**

Maps each specification requirement to a specific Strands SDK capability or identifies where custom implementation is needed:

|  |  |  |
|----|----|----|
| **Spec Requirement** | **Strands Capability** | **Custom Work** |
| Model-driven orchestration | Agent class with tool_use loop | None |
| Multi-agent (hierarchical) | Agents-as-Tools, Swarm patterns | None |
| Agent Teams (peer) | N/A | Full implementation: mailbox, task list, file locks |
| Self-improving loops | N/A | Full implementation: loop runner, context file, learning store |
| Hook lifecycle (12 events) | HookProvider interface | Custom handlers for prompt/agent types |
| Context management | SummarizingConversationManager | Extend with semantic triage + tiered persistence |
| Cost governance | N/A | Full implementation: budgets, routing, monitoring |
| Prompt caching | API support (cache_control blocks) | Caching strategy logic |
| Adaptive thinking | N/A (model-side) | Effort selection + routing logic |
| Plugin system | N/A | Full implementation: registry, marketplace, lifecycle |
| Session portability | SessionManager | Extend with cross-surface state sync |
| Observability | OpenTelemetry built-in | Custom spans for cost, context health |
| Security/guardrails | Bedrock Guardrails param | Custom data classification + HIPAA paths |

**Appendix B: Hook Events Quick Reference**

JSON input schema, matcher fields, and output options for all 12 hook events. See Anthropic's official hooks reference at code.claude.com/docs/en/hooks for complete schemas.

**Appendix C: Pricing Reference**

|                      |                   |                    |
|----------------------|-------------------|--------------------|
| **Model**            | **Input (per M)** | **Output (per M)** |
| Opus 4.6             | $5.00            | $25.00            |
| Sonnet 4.5 (≤200K)   | $3.00            | $15.00            |
| Sonnet 4.5 (>200K)  | $6.00            | $22.50            |
| Haiku 4.5            | $0.80            | $4.00             |
| Cached input (5-min) | 10% of base       | N/A                |
| Cache write (5-min)  | 125% of base      | N/A                |
| Cache write (1-hour) | 200% of base      | N/A                |
| Batch API            | 50% of base       | 50% of base        |

**Section 4A: Custom Agent Definition Schema**

This section specifies the complete agent definition format, invocation patterns, and composition model. For the full schema reference, see the companion document: custom_agent_schema.docx.

**4A.1 Agent Definition Format**

Custom agents are Markdown files (.md) with YAML frontmatter stored in .claude/agents/ (project-level) or ~/.claude/agents/ (user-level). Project agents override user agents on name collision. Plugin agents use namespace prefixing (plugin-name:agent-name) to avoid collisions.

**4A.2 Required Fields**

```
• name (string): Unique identifier, lowercase-with-hyphens. Used for invocation and file naming.

• description (string): Detailed, action-oriented with examples. Claude uses this for auto-dispatch matching — this is the primary signal for automatic task delegation.
```

**4A.3 Optional Fields**

```
• model: sonnet | opus | haiku | inherit. Controls which model tier the agent runs on. Default: inherit (uses parent's model).

• tools / disallowedTools: Allowlist or denylist of tool access. Mutually exclusive. Omit both to inherit all parent tools including MCP.

• permissionMode: default | bypassPermissions | plan. Controls mutation rights.

• color: purple | cyan | green | orange | blue | red. Visual identifier in terminal/UI.

• maxTurns: Integer limit on agentic loop iterations. Prevents runaway agents.

• hooks: Scoped hook definitions for all 12 hook events. Active only during agent execution. Stop hooks auto-convert to SubagentStop.

• skills: Comma-separated skill names. Skills' SKILL.md contents injected into agent context on activation.

• mcpServers: Agent-scoped MCP servers. Started on activation, stopped on completion. Support ${ENV_VAR} substitution.

• memory: Persistent context injected on every invocation. Project conventions, stack info, shared knowledge.
```

**4A.4 Invocation Modes**

Four invocation modes are supported:

```
• Auto-dispatch: Claude evaluates requests against agent descriptions and automatically delegates via the Task tool. Agents appear as tools from the model's perspective.

• Explicit: User says "Use the X agent" or selects from /agents interactive menu.

• CLI: claude --agent code-reviewer runs agent as main thread with Task tool for spawning sub-subagents.

• Pipeline: SubagentStop hooks chain stages (PM-Spec → Architect → Implementer → QA) by writing status to .claude/pipeline-state.json.
```

**4A.5 Composition Model**

Agents compose with the skill, hook, and MCP systems through three principles:

```
• Isolation: Each agent runs in its own context window. Agent work does not pollute the parent conversation.

• Additive hooks: Agent-scoped hooks are added to (not replace) existing user/project/managed hooks. Cleaned up when agent completes.

• Lifecycle-scoped MCP: Agent MCP servers start on activation and stop on completion. Tools available only to that agent.
```

**4A.6 Integration with Other Sections**

```
• Section 3 (Context Management): Each agent has its own context manager instance with independent compaction strategy.

• Section 5 (Agent Teams): Custom agents can serve as teammates. Team lead dispatches them with inbox/task coordination.

• Section 6 (Self-Improving Loops): Loop iterations can use custom agents as the fresh-agent-per-iteration executor.

• Section 7 (Hooks): All 12 hook events available for agent-scoped hooks. SubagentStop is the primary lifecycle event.

• Section 8 (Plugins): Plugin agents are custom agents distributed via the plugin ecosystem with namespace prefixing.

• Section 9 (Skills): Agents can reference skills by name. Skills are loaded into agent context on activation.

• Section 10 (Cost Governance): Agent model selection interacts with per-agent budgets. Cost Governor can override model tier.
```

**4A.7 Implementation Reference**

Implementation Task 25 in the Implementation Tasks document specifies the complete agent registry, loader, dispatcher, and runner modules with TypeScript interfaces and acceptance criteria. Estimated effort: 10-14 hours. Dependencies: Task 5 (Orchestrator), Task 3 (Hook Engine), Task 12 (Skills Registry).
