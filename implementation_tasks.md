**Claude Code v3 Implementation Tasks**

LLM-Executable Task Breakdown with Code Contracts

February 2026 — 24 Tasks Across 5 Phases

**How to Use This Document**

This document is designed to be consumed by an LLM (Claude Code, Cursor, Copilot, or similar) as a series of self-contained implementation tasks. Each task includes everything the LLM needs to produce working code: the directory structure, interface contracts (TypeScript types), key implementation logic, test criteria, and dependencies on other tasks.

**For each task:** Copy the task section into your LLM's context. The LLM should be able to produce a working implementation from the task description alone, without needing to reference other sections (though dependency ordering matters).

**Language:** TypeScript (Node.js). The Strands Agents SDK has a Python-first API, but this system targets TypeScript for consistency with Claude Code's own codebase and the MCP ecosystem.

**Testing:** Each task specifies acceptance tests. Use Vitest for unit tests and the Strands Evals SDK for evaluation tests.

**Dependency order:** Tasks are numbered and declare dependencies. Complete dependencies before starting a task.

**Phase 1: Foundation (Weeks 1-4)**

**Task 1: Project Scaffold and Core Types**

**Dependencies:** None

**Estimated effort:** 2-3 hours

**Output:** Project structure, shared type definitions, configuration loader

**Directory Structure**

```
claude-code-v3/

├── src/

│ ├── types/ # Shared type definitions

│ ├── config/ # Configuration loading & validation

│ ├── context/ # Context management system

│ ├── orchestrator/ # Core orchestration engine

│ ├── hooks/ # Hook lifecycle engine

│ ├── agents/ # Agent topologies (subagent, team, loop)

│ ├── cost/ # Token economics & cost governance

│ ├── cache/ # Prompt caching strategy

│ ├── plugins/ # Plugin registry & marketplace

│ ├── skills/ # Skills registry

│ ├── session/ # Session portability & teleportation

│ ├── observability/ # OTel instrumentation

│ └── security/ # Guardrails, data classification

├── tests/

├── evals/ # Strands Evals test suites

├── package.json

└── tsconfig.json
```

**Core Type Definitions (src/types/index.ts)**

Define these interfaces. The LLM should create the full file:

```
// Model tiers

type ModelTier = 'opus' | 'sonnet' | 'haiku';

type EffortLevel = 'quick' | 'standard' | 'deep';

// Context categories for semantic triage

type ContextCategory = 'PRESERVE_VERBATIM' | 'PRESERVE_STRUCTURED' | 'COMPRESS_AGGRESSIVE' | 'EPHEMERAL';

interface ContextItem {

id: string;

category: ContextCategory;

content: string;

tokenCount: number;

createdAt: Date;

lastReferencedAt: Date;

referenceCount: number;

source: 'user' | 'assistant' | 'tool_call' | 'tool_result' | 'system';

}

interface AgentBudget {

inputBudgetTokens: number;

outputBudgetTokens: number;

sessionBudgetUsd: number;

teamBudgetUsd?: number;

currentInputTokens: number;

currentOutputTokens: number;

currentCostUsd: number;

}

// Hook system types

type HookEvent = 'SessionStart' | 'UserPromptSubmit' | 'PreToolUse' | 'PostToolUse'

| 'PostToolUseFailure' | 'PermissionRequest' | 'PreCompact' | 'Notification'

| 'Stop' | 'SubagentStop' | 'SessionEnd' | 'Setup';

type HookHandlerType = 'command' | 'prompt' | 'agent';

interface HookHandler {

type: HookHandlerType;

command?: string; // for 'command' type

prompt?: string; // for 'prompt' type

agentConfig?: object; // for 'agent' type

async?: boolean;

timeout?: number; // ms, default 600000 (10 min)

}

interface HookDefinition {

matcher?: string; // regex, '*' = match all

hooks: HookHandler[];

}

interface AgentConfig {

id: string;

modelTier: ModelTier;

effortLevel: EffortLevel;

budget: AgentBudget;

tools: string[];

hooks: Record\<HookEvent, HookDefinition[]>;

}
```

**Acceptance Criteria**

- All types compile with strict TypeScript (no any types)

- Configuration loader reads from 3 scopes: user (~/.claude/settings.json), project (.claude/settings.json), project-local (.claude/settings.local.json) with correct precedence

- Unit tests for config merging (local overrides project overrides user)

**Task 2: Context Manager with Semantic Triage**

**Dependencies:** Task 1

**Estimated effort:** 6-8 hours

**Output:** src/context/context-manager.ts, src/context/triage.ts, src/context/session-store.ts

**Key Implementation Logic**

The Context Manager is the single most important component. It must:

- Classify every new context item into a ContextCategory using pattern matching: items containing file paths, error messages (stack traces, exit codes), variable names, or test output are PRESERVE_VERBATIM. Items containing decision records, task lists, or schemas are PRESERVE_STRUCTURED. Discussion and reasoning are COMPRESS_AGGRESSIVE. Tool metadata and raw search results are EPHEMERAL.

- Maintain a session file at .claude/session-state.json that persists PRESERVE_VERBATIM and PRESERVE_STRUCTURED items across compaction events.

- Implement a PreCompact hook handler that backs up critical context to the session file before compaction occurs.

- Implement a SessionStart hook handler that loads the session file into context when a session starts or resumes.

- Track staleness scores for all context items: score = (turns since last reference) * (1 / reference count). Items above a staleness threshold are candidates for removal.

- Expose context health metrics: freePercent, totalTokens, preservedTokens, compressibleTokens, ephemeralTokens, stalenessDistribution, cacheHitRate.

**Classification Patterns (src/context/triage.ts)**

```
// Pattern matchers for automatic classification

const VERBATIM_PATTERNS = [

/Error:.*at\s+/, // Stack traces

/exit code \d+/i, // Exit codes

/\[\w\\]+(\[\w\\]+)+/, // File paths

/^\s*(const|let|var|function|class|interface|type)\s+\w+/, // Declarations

/FAIL|PASS|ERROR|assert/i, // Test output

/\d+\\d+\\d+/, // Version numbers

/[A-Z\_]{3,}=/, // Environment variables

];

const STRUCTURED_PATTERNS = [

/DECISION:|DECIDED:|TODO:|TASK:/i,

/acceptance criteria|requirements/i,

/schema|interface|type.*\/,

];
```

**Acceptance Criteria**

- Correctly classifies 95%+ of context items (test with 100-item corpus)

- Session file persists across simulated compaction events

- PRESERVE_VERBATIM items survive compaction with 100% fidelity

- Context health metrics update in real-time

- Staleness scoring correctly identifies unused context items

**Task 3: Hook Lifecycle Engine**

**Dependencies:** Task 1

**Estimated effort:** 8-10 hours

**Output:** src/hooks/hook-engine.ts, src/hooks/handlers/command.ts, src/hooks/handlers/prompt.ts, src/hooks/handlers/agent.ts

**Key Implementation Logic**

Implement the full 12-event hook lifecycle with 3 handler types:

```
class HookEngine {

// Register hooks from all scopes (user, project, local, plugin, skill, subagent)

registerHooks(scope: HookScope, hooks: Record\<HookEvent, HookDefinition[]>): void;

// Fire a hook event. Returns aggregated results from all matching handlers.

async fire(event: HookEvent, context: HookContext): Promise\<HookResult>;

// For PreToolUse: can return 'allow', 'deny', or modified input

// For Stop: can block completion

// For PermissionRequest: can auto-allow or auto-deny

}

interface HookContext {

session_id: string;

hook_event_name: HookEvent;

tool_name?: string; // for tool events

tool_input?: object; // for PreToolUse

tool_response?: string; // for PostToolUse

user_prompt?: string; // for UserPromptSubmit

source?: string; // for SessionStart

model?: string; // for SessionStart

}

interface HookResult {

permissionDecision?: 'allow' | 'deny';

permissionDecisionReason?: string;

updatedInput?: object; // modified tool input

additionalContext?: string; // context injected to model

decision?: 'block' | 'continue';

reason?: string;

}
```

**Handler Implementation Notes**

- Command handlers: Spawn child process, pipe HookContext as JSON to stdin, read stdout as JSON HookResult. Exit code 0 = allow, 2 = deny. Timeout default 10 minutes.

- Prompt handlers: Send prompt to Haiku model (configurable) with $ARGUMENTS replaced by JSON context. Parse response as HookResult.

- Agent handlers: Spawn a Strands Agent with Read, Grep, Glob tools. Pass context as initial message. Collect structured response as HookResult.

- Async hooks (async: true): Fire and forget. Do not block main execution. Log results to OTel.

- Matcher: Regex matching against tool_name for tool events, source for SessionStart. '*' or omitted = match all.

**Acceptance Criteria**

- All 12 events fire at correct lifecycle points (unit test per event)

- Command handler correctly spawns process, pipes stdin, reads stdout

- PreToolUse can deny (exit code 2) and modify input (updatedInput)

- Prompt handler sends to Haiku and parses structured response

- Agent handler spawns subagent with tools and collects result

- Async hooks do not block execution

- Hooks from multiple scopes execute in correct precedence order

**Task 4: Cost Governor**

**Dependencies:** Task 1

**Estimated effort:** 4-6 hours

**Output:** src/cost/cost-governor.ts, src/cost/model-router.ts, src/cost/budget-tracker.ts

**Key Implementation Logic**

```
interface ModelPricing {

inputPerMillion: number;

outputPerMillion: number;

cachedInputPerMillion: number;

cacheWritePerMillion: number;

}

const PRICING: Record\<ModelTier, ModelPricing> = {

opus: { inputPerMillion: 5.00, outputPerMillion: 25.00, cachedInputPerMillion: 0.50, cacheWritePerMillion: 6.25 },

sonnet: { inputPerMillion: 3.00, outputPerMillion: 15.00, cachedInputPerMillion: 0.30, cacheWritePerMillion: 3.75 },

haiku: { inputPerMillion: 0.80, outputPerMillion: 4.00, cachedInputPerMillion: 0.08, cacheWritePerMillion: 1.00 },

};

class CostGovernor {

// Select cheapest model capable of handling the task

selectModel(task: TaskSignals, budget: AgentBudget): ModelTier;

// Track token usage and cost after each API call

recordUsage(agentId: string, inputTokens: number, outputTokens: number, model: ModelTier, cached: boolean): void;

// Check if agent is approaching budget limits

checkBudget(agentId: string): BudgetStatus; // 'ok' | 'warning' | 'critical' | 'exceeded'

// Get real-time cost dashboard data

getDashboard(): CostDashboard;

}

interface TaskSignals {

filesAffected: number;

dependencyDepth: number;

requiresReasoning: boolean;

isExploration: boolean;

isTeamLead: boolean;

}
```

**Model Selection Rules**

- isExploration = true OR filesAffected \<= 1 AND !requiresReasoning: Haiku

- isTeamLead = true OR requiresReasoning = true OR dependencyDepth > 3: Opus

- Default: Sonnet

- Budget override: If budget warning, downgrade one tier. If critical, force Haiku.

**Acceptance Criteria**

- Model selection matches rules for all TaskSignal combinations

- Budget tracking accurately calculates USD from token counts

- Warning at 80%, critical at 95%, exceeded at 100%

- Model downgrade triggers at warning threshold

- Dashboard returns correct aggregated data

**Task 5: Core Orchestrator**

**Dependencies:** Tasks 1-4

**Estimated effort:** 10-12 hours

**Output:** src/orchestrator/orchestrator.ts, src/orchestrator/task-decomposer.ts

**Key Implementation Logic**

The Orchestrator is the main loop that ties everything together. It receives a user request, decomposes it into tasks, selects the appropriate agent topology (subagent, team, loop), routes each task to the correct model tier, manages the hook lifecycle, and tracks costs.

```
class Orchestrator {

constructor(

private contextManager: ContextManager,

private hookEngine: HookEngine,

private costGovernor: CostGovernor,

private config: OrchestratorConfig

) {}

async processRequest(request: string): Promise\<OrchestratorResult> {

// 1. Fire SessionStart hooks (if new session)

// 2. Fire UserPromptSubmit hooks

// 3. Classify request complexity -> select effort level

// 4. Select model tier via CostGovernor

// 5. Decompose into tasks if complex

// 6. For each task:

// a. Fire PreToolUse hooks before any tool call

// b. Execute tool call

// c. Fire PostToolUse hooks

// d. Update context (classify and store new items)

// e. Record usage with CostGovernor

// f. Check budget status

// 7. Fire Stop hooks

// 8. Return result with cost summary

}

}
```

**Acceptance Criteria**

- End-to-end flow: request -> decomposition -> tool calls -> response

- All hook events fire in correct order

- Cost tracking accurate across entire request lifecycle

- Context Manager correctly classifies all new items

- Model tier selection respects budget constraints

**Phase 2: Multi-Agent (Weeks 5-10)**

**Task 6: Hierarchical Subagent Manager**

**Dependencies:** Task 5

**Estimated effort:** 6-8 hours

**Output:** src/agents/subagent-manager.ts

Implement the Agents-as-Tools pattern from Strands. The Orchestrator wraps worker agents as tools callable via natural language. Each worker has its own context window, budget, and hook scope. Workers return structured results. SubagentStop hooks fire on completion.

**Task 7: Agent Teams Manager**

**Dependencies:** Task 6

**Estimated effort:** 12-16 hours

**Output:** src/agents/team-manager.ts, src/agents/teammate.ts, src/agents/mailbox.ts, src/agents/task-list.ts

**Key Components**

- TeamManager: Spawns team lead + N teammates. Manages shared task list. Monitors team cost.

- Teammate: Full agent instance with own context window. Connects to mailbox for inter-agent messages.

- Mailbox: File-system-based message queue. Each teammate has inbox directory. Messages are JSON files with sender, recipient, type (task_assignment, finding, question, status_update), and payload.

- TaskList: Central registry with task status (pending, claimed, blocked, complete), dependencies, file ownership mapping, and file-lock-based claiming to prevent race conditions.

**File Lock Protocol**

```
// Before editing a file, teammate claims it:

// 1. Check if .claude/locks/{filepath}.lock exists

// 2. If not, create lock file with teammate ID and timestamp

// 3. If exists, check if lock is stale (>5 min) and break if so

// 4. On task completion, release all locks
```

**Task 8: Self-Improving Loop Runner**

**Dependencies:** Task 5

**Estimated effort:** 8-10 hours

**Output:** src/agents/loop-runner.ts, src/agents/learning-store.ts, src/agents/context-file.ts

**Key Components**

- LoopRunner: Orchestrates iterations. Each iteration: spawn fresh agent, feed context file, execute task, commit to git, update context file with learnings, terminate, check acceptance criteria, loop or stop.

- LearningStore: Persistent store at .claude/learnings/. Each entry: { pattern: string, resolution: string, confidence: number, source_iteration: number }. Loaded into context at start of each iteration.

- ContextFile: JSON file at .claude/loop-context.json with: currentTask, acceptanceCriteria, constraints, learnings[], failedApproaches[], iterationCount, maxIterations.

- Safety: Stop file (.auto/stop), diff size limits, error repetition detection (abort if same error 3x), acceptance criteria gate.

**Phase 3: Intelligence (Weeks 11-16)**

**Task 9: Prompt Cache Manager**

**Dependencies:** Task 4

**Estimated effort:** 4-6 hours

**Output:** src/cache/cache-manager.ts

Implement the two-tier caching strategy. Identify cache-stable layers (system prompts, CLAUDE.md, tool schemas, policy definitions) and inject cache_control blocks into API requests. Track cache hit rates. Alert if rates fall below 70%. Support both 5-minute and 1-hour durations with automatic selection based on content volatility.

**Task 10: Adaptive Effort Controller**

**Dependencies:** Task 5

**Estimated effort:** 4-5 hours

**Output:** src/orchestrator/effort-controller.ts

Analyze task signals (files affected, dependency depth, ambiguity level) to select Quick/Standard/Deep effort level. Map effort levels to model parameters (thinking budget, temperature, max tokens). Support explicit override via configuration and tool call parameters.

**Task 11: MCP Tool Search Integration**

**Dependencies:** Task 5

**Estimated effort:** 4-5 hours

**Output:** src/orchestrator/tool-search.ts

When total tool definitions exceed a configurable context threshold (default: auto), switch to lazy loading mode. Instead of loading all tool schemas into context, load tool_reference blocks that the model can search. Requires Sonnet 4+ or Opus 4+ models. Haiku fallback: load tools eagerly (no search support). Track context savings from tool search.

**Task 12: Skills Registry**

**Dependencies:** Tasks 3, 5

**Estimated effort:** 6-8 hours

**Output:** src/skills/skill-registry.ts, src/skills/skill-loader.ts, src/skills/skill-matcher.ts

Implement skill discovery (scan ~/.claude/skills and .claude/skills directories), frontmatter parsing (YAML with name, description, hooks, disable-model-invocation, allowed_tools), auto-invocation matching (compare task description against skill descriptions, invoke when confidence > threshold), slash command registration, hot-reload (watch filesystem for changes), and invocation logging (which skills considered, confidence scores, why selected/rejected). Track context consumption per skill.

**Phase 4: Ecosystem (Weeks 17-20)**

**Task 13: Plugin Registry**

**Dependencies:** Tasks 3, 12

**Estimated effort:** 10-12 hours

**Output:** src/plugins/plugin-registry.ts, src/plugins/marketplace.ts, src/plugins/plugin-loader.ts

Implement the full plugin lifecycle: marketplace registration (Git repository with marketplace.json catalog), plugin discovery (/plugin marketplace add {owner}/{repo}), installation (/plugin install {name}@{marketplace}), capability registration (commands, agents, skills, hooks, MCP servers, LSP servers all registered into appropriate registries), enable/disable toggling per project, and uninstallation. Plugin directory structure: .claude-plugin/plugin.json (manifest), commands/, agents/, skills/, hooks/, .mcp.json, .lsp.json.

**Task 14: Session Teleportation**

**Dependencies:** Tasks 2, 5

**Estimated effort:** 6-8 hours

**Output:** src/session/teleporter.ts, src/session/serializer.ts

Implement session serialization (all state → portable JSON blob): conversation history, tool permissions, active workers/teammates, pending approvals, compaction state, context manager state, cost tracking, and hook registrations. Support /teleport (move to another surface) and /desktop (hand off to desktop app). Maintain single trace ID across surface transitions for audit continuity. Handle conflict resolution for simultaneous edits via last-write-wins with notification.

**Task 15: Claude-as-MCP-Server Mode**

**Dependencies:** Task 5

**Estimated effort:** 4-5 hours

**Output:** src/mcp/server-mode.ts

Implement dual-mode operation: the system can act as both an MCP client (consuming external MCP servers) and an MCP server (exposing its tools to other clients via stdio transport). Expose: Bash, Read, Write, Edit, LS, GrepTool, GlobTool, Replace tools. Important: MCP servers configured within the system are NOT exposed to connecting clients (no passthrough). Each client connection spawns a fresh instance with no shared state.

**Phase 5: Enterprise (Weeks 21-24)**

**Task 16: Unified Quota Manager**

**Dependencies:** Task 4

**Estimated effort:** 6-8 hours

**Output:** src/cost/quota-manager.ts

Track consumption across all Claude surfaces (web, mobile, desktop, CLI) in real-time. Predict quota exhaustion based on current run rate. Alert at 80% threshold. Provide usage breakdowns by surface and agent. Support weekly rate limits for 24/7 agent users. Consume SDKRateLimitInfo events for rate limit status including utilization, reset times, and overage information.

**Task 17: Security and Data Classification**

**Dependencies:** Tasks 3, 5

**Estimated effort:** 6-8 hours

**Output:** src/security/data-classifier.ts, src/security/guardrails.ts

Implement data classification for context items (PII, PHI, financial, credentials). Integrate Bedrock Guardrails via guardrail_latest_message parameter. Implement Constitutional AI filter for dangerous/unethical suggestions. Add HIPAA compliance path for Enterprise plans. Prevent Claude Code from launching inside another Claude Code session (nest guard).

**Task 18: Observability Dashboard**

**Dependencies:** All previous tasks

**Estimated effort:** 8-10 hours

**Output:** src/observability/dashboard.ts, src/observability/otel-instrumentation.ts

Instrument all components with OpenTelemetry. Custom spans: cost attribution (tokens + USD per trace), context health (free %, staleness, cache hits), agent coordination (inter-teammate messages, task claims, file locks), effort level (Fast Mode speed attribute), skill invocation (which skills, confidence, selection reason). Emit structured logs for every hook execution. Integrate with ccusage for terminal-based monitoring.

**Task 19: Evaluation Suite**

**Dependencies:** All previous tasks

**Estimated effort:** 8-10 hours

**Output:** evals/ directory with test suites

Build evaluation suite using Strands Evals SDK. Dimensions: task completion (≥90%), code quality (≥4.0/5.0 via LLM-as-a-Judge), safety (zero critical violations), context preservation (95%+ verbatim survival), cost efficiency (≤20% variance), multi-agent coordination (≤85% overhead), learning retention (no repeated mistakes), latency (p95 ≤30s). Use ActorSimulator for multi-turn scenario testing.

**Integration and Polish Tasks**

**Task 20: Compound Loop Orchestration**

**Dependencies:** Tasks 8, 5

**Estimated effort:** 6-8 hours

Chain analysis → planning → execution loops. One agent's output becomes the next agent's input. Implement as a configurable pipeline with YAML-defined stages.

**Task 21: Enterprise Managed Settings**

**Dependencies:** Tasks 3, 16, 17

**Estimated effort:** 4-5 hours

Implement organization-level policy enforcement via managed settings. Support allowManagedHooksOnly (block user/project/plugin hooks). Self-serve Enterprise deployment (no Sales gate). Policy distribution via Git-based config repository.

**Task 22: Learning Store Vector Index**

**Dependencies:** Task 8

**Estimated effort:** 4-6 hours

Add vector embedding to the Learning Store for semantic similarity search. When starting a new session or loop iteration, query the store for learnings relevant to the current task (not just all learnings). Use lightweight embedding model. Store embeddings alongside learning entries. Top-K retrieval with configurable K.

**Task 23: End-to-End Integration Tests**

**Dependencies:** All previous tasks

**Estimated effort:** 8-10 hours

Integration tests covering: full request lifecycle (user prompt → decomposition → tool calls → response), Agent Team coordination (spawn team, assign tasks, verify no edit collisions), self-improving loop (3 iterations with learning accumulation), session teleportation (serialize, deserialize, verify state), cost governance (budget enforcement, model downgrade), hook lifecycle (all 12 events, all 3 handler types).

**Task 24: Documentation and CLAUDE.md Templates**

**Dependencies:** All previous tasks

**Estimated effort:** 4-6 hours

Generate: API documentation for all public interfaces, CLAUDE.md templates for common project types, hook configuration examples for the top 10 use cases (auto-format, security guard, TDD enforcement, commit message quality, deployment gate, test coverage, dependency audit, context backup, cost alerting, skill logging), plugin template for custom enterprise plugins, and runbook for common operational scenarios.

**Total Effort Summary**

|                       |             |                |
|-----------------------|-------------|----------------|
| **Phase**             | **Tasks**   | **Est. Hours** |
| Phase 1: Foundation   | Tasks 1-5   | 30-39 hours    |
| Phase 2: Multi-Agent  | Tasks 6-8   | 26-34 hours    |
| Phase 3: Intelligence | Tasks 9-12  | 18-24 hours    |
| Phase 4: Ecosystem    | Tasks 13-15 | 20-25 hours    |
| Phase 5: Enterprise   | Tasks 16-19 | 28-36 hours    |
| Integration           | Tasks 20-24 | 26-35 hours    |
| TOTAL                 | 24 Tasks    | 148-193 hours  |

**Task 25: Agent Registry and Dispatch System**

**Phase:** 4 (Weeks 17-20)

**Estimated effort:** 10-14 hours

**Dependencies:** Task 5 (Orchestrator), Task 3 (Hook Engine), Task 12 (Skills Registry)

**Output:** src/agents/agent-registry.ts, src/agents/agent-loader.ts, src/agents/agent-dispatcher.ts, src/agents/agent-runner.ts

**25.1 Directory Structure**

```
src/agents/

agent-registry.ts # Discovery, registration, precedence resolution

agent-loader.ts # Parse .md files with YAML frontmatter

agent-dispatcher.ts # Route tasks to agents, generate tool definitions

agent-runner.ts # Isolated execution: context, tools, hooks, MCP, lifecycle

agent-templates.ts # Built-in agent templates (reviewer, auditor, etc.)

\_\_tests\_\_/

agent-registry.test.ts

agent-loader.test.ts

agent-dispatcher.test.ts

agent-runner.test.ts
```

**25.2 Core Interfaces**

```
interface AgentDefinition {

name: string; // Required: lowercase-with-hyphens

description: string; // Required: action-oriented with examples

model: 'sonnet' | 'opus' | 'haiku' | 'inherit';

tools: string[] | null; // null = inherit all from parent

disallowedTools: string[] | null;

permissionMode: 'default' | 'bypassPermissions' | 'plan';

color?: 'purple' | 'cyan' | 'green' | 'orange' | 'blue' | 'red';

maxTurns?: number;

hooks: Record\<HookEvent, HookDefinition[]>;

skills: string[];

mcpServers: Record\<string, McpServerConfig>;

memory?: string;

systemPrompt: string; // Markdown body after frontmatter

source: 'project' | 'user' | 'plugin' | 'cli';

pluginNamespace?: string;

filePath: string;

}

interface AgentResult {

agentName: string;

summary: string; // Agent's final response to delegating agent

turnsUsed: number;

tokensConsumed: { input: number; output: number; cacheRead: number; };

toolsUsed: string[]; // Which tools the agent actually invoked

filesModified: string[]; // Files changed during execution

exitReason: 'complete' | 'maxTurns' | 'budget' | 'error' | 'stopped';

}
```

**25.3 Agent Loader Implementation**

Parse .md files into AgentDefinition objects:

```
import matter from 'gray-matter'; // YAML frontmatter parser

import { readFile, readdir } from 'fs/promises';

import { join } from 'path';

async function loadAgentFile(filePath: string): Promise\<AgentDefinition> {

const raw = await readFile(filePath, 'utf-8');

const { data: fm, content: body } = matter(raw);

// Validate required fields

if (!fm.name || typeof fm.name !== 'string')

throw new Error(\`Agent ${filePath}: missing required 'name' field\`);

if (!fm.description || typeof fm.description !== 'string')

throw new Error(\`Agent ${filePath}: missing required 'description' field\`);

if (!/^[a-z][a-z0-9-]*$/.test(fm.name))

throw new Error(\`Agent ${filePath}: name must be lowercase-with-hyphens\`);

// Parse tools (comma-separated string → array)

const tools = fm.tools

? (typeof fm.tools === 'string'

? fm.tools.split(',').map((t: string) => t.trim())

: fm.tools)

: null;

const disallowedTools = fm.disallowedTools

? (typeof fm.disallowedTools === 'string'

? fm.disallowedTools.split(',').map((t: string) => t.trim())

: fm.disallowedTools)

: null;

// Mutual exclusivity check

if (tools && disallowedTools)

throw new Error(\`Agent ${fm.name}: cannot specify both tools and disallowedTools\`);

// Validate model

const validModels = ['sonnet', 'opus', 'haiku', 'inherit'];

const model = fm.model || 'inherit';

if (!validModels.includes(model))

throw new Error(\`Agent ${fm.name}: invalid model '${model}'\`);

// Parse skills (comma-separated string → array)

const skills = fm.skills

? (typeof fm.skills === 'string'

? fm.skills.split(',').map((s: string) => s.trim())

: fm.skills)

: [];

return {

name: fm.name,

description: fm.description,

model,

tools,

disallowedTools,

permissionMode: fm.permissionMode || 'default',

color: fm.color,

maxTurns: fm.maxTurns,

hooks: fm.hooks || {},

skills,

mcpServers: fm.mcpServers || {},

memory: fm.memory,

systemPrompt: body.trim(),

source: 'project', // caller sets this based on discovery path

filePath

};

}
```

**25.4 Agent Registry Implementation**

```
class AgentRegistry {

private agents = new Map\<string, AgentDefinition>();

private watchers: FSWatcher[] = [];

async loadAll(): Promise\<void> {

// Load in reverse precedence order (later overwrites earlier)

// 1. Plugin agents (namespaced, won't collide)

await this.loadFromDir(getPluginAgentsDir(), 'plugin');

// 2. User agents (~/.claude/agents/)

await this.loadFromDir(getUserAgentsDir(), 'user');

// 3. Project agents (.claude/agents/) - highest priority

await this.loadFromDir(getProjectAgentsDir(), 'project');

}

private async loadFromDir(dir: string, source: AgentSource): Promise\<void> {

if (!existsSync(dir)) return;

const files = (await readdir(dir)).filter(f => f.endsWith('.md'));

for (const file of files) {

try {

const agent = await loadAgentFile(join(dir, file));

agent.source = source;

const key = source === 'plugin'

? \`${agent.pluginNamespace}:${agent.name}\`

: agent.name;

this.agents.set(key, agent);

} catch (err) {

console.warn(\`Failed to load agent ${file}: ${err}\`);

}

}

}

get(name: string): AgentDefinition | undefined {

return this.agents.get(name);

}

list(): AgentDefinition[] {

return Array.from(this.agents.values());

}

// Generate tool definitions so model sees agents as invocable tools

getToolDefinitions(): ToolDefinition[] {

return this.list().map(agent => ({

name: \`agent:${agent.name}\`,

description: agent.description,

input_schema: {

type: 'object',

properties: {

task: { type: 'string', description: 'The task to delegate to this agent' }

},

required: ['task']

}

}));

}

watch(): void {

for (const dir of [getProjectAgentsDir(), getUserAgentsDir()]) {

if (existsSync(dir)) {

const w = watchDir(dir, async () => {

this.agents.clear();

await this.loadAll();

});

this.watchers.push(w);

}

}

}

}
```

**25.5 Agent Runner Implementation**

```
class AgentRunner {

constructor(

private hookEngine: HookEngine,

private costGovernor: CostGovernor,

private skillRegistry: SkillRegistry,

private contextManager: ContextManager

) {}

async run(agent: AgentDefinition, task: string, parentModel: string): Promise\<AgentResult> {

// 1. Resolve model

const model = agent.model === 'inherit'

? parentModel

: resolveModelAlias(agent.model);

// 2. Check budget

const budgetCheck = this.costGovernor.checkBudget(agent.name, model);

const effectiveModel = budgetCheck.override || model;

// 3. Build tool list

const tools = this.resolveTools(agent);

// 4. Start agent-scoped MCP servers

const mcpCleanup = await this.startMcpServers(agent.mcpServers);

// 5. Load skills into context prefix

const skillContext = await this.loadSkills(agent.skills);

// 6. Register agent-scoped hooks

const hookCleanup = this.hookEngine.registerScoped(agent.hooks, agent.name);

// 7. Build system message

const systemMessage = [

agent.memory || '',

skillContext,

agent.systemPrompt

].filter(Boolean).join('\n\n');

// 8. Create isolated context + run agentic loop

const context = this.contextManager.createIsolated();

let turns = 0;

let totalTokens = { input: 0, output: 0, cacheRead: 0 };

let toolsUsed = new Set\<string>();

let filesModified = new Set\<string>();

let exitReason: AgentResult['exitReason'] = 'complete';

const messages = [{ role: 'user', content: task }];

try {

while (true) {

if (agent.maxTurns && turns >= agent.maxTurns) {

exitReason = 'maxTurns';

break;

}

const response = await callModel({

model: effectiveModel,

system: systemMessage,

messages,

tools

});

totalTokens.input += response.usage.input_tokens;

totalTokens.output += response.usage.output_tokens;

totalTokens.cacheRead += response.usage.cache_read_input_tokens || 0;

turns++;

// Check if model is done (no tool_use)

const toolUses = response.content.filter(b => b.type === 'tool_use');

if (toolUses.length === 0) break;

// Execute tools with hook lifecycle

for (const toolUse of toolUses) {

// PreToolUse hook

const preResult = await this.hookEngine.fire('PreToolUse', {

tool_name: toolUse.name,

tool_input: toolUse.input

});

if (preResult?.permissionDecision === 'deny') {

messages.push({ role: 'tool', content: 'Permission denied by hook' });

continue;

}

toolsUsed.add(toolUse.name);

const result = await executeTool(toolUse.name, toolUse.input);

// Track file modifications

if (['Write', 'Edit', 'MultiEdit'].includes(toolUse.name)) {

filesModified.add(toolUse.input.file_path || toolUse.input.path);

}

// PostToolUse hook

await this.hookEngine.fire('PostToolUse', {

tool_name: toolUse.name,

tool_input: toolUse.input,

tool_response: result

});

messages.push({ role: 'assistant', content: [toolUse] });

messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: toolUse.id, content: result }] });

}

}

} catch (err) {

exitReason = 'error';

} finally {

// 9. Fire SubagentStop hook

await this.hookEngine.fire('SubagentStop', {

agent_name: agent.name,

exit_reason: exitReason,

turns_used: turns

});

// 10. Cleanup

hookCleanup();

await mcpCleanup();

context.release();

}

// Extract final summary (last text block from model)

const lastAssistant = messages.filter(m => m.role === 'assistant').pop();

const summary = lastAssistant?.content

?.filter((b: any) => b.type === 'text')

.map((b: any) => b.text).join('\n') || 'Agent completed without summary.';

return {

agentName: agent.name,

summary,

turnsUsed: turns,

tokensConsumed: totalTokens,

toolsUsed: Array.from(toolsUsed),

filesModified: Array.from(filesModified),

exitReason

};

}

}
```

**25.6 Acceptance Criteria**

```
• Agent .md files parsed from .claude/agents/ and ~/.claude/agents/ with all YAML fields validated

• Required fields enforced (name, description). Type and enum validation on all optional fields

• Precedence: project agent overrides user agent with same name

• Auto-dispatch: model sees agents as tools via getToolDefinitions() and delegates based on description

• Isolated context window: agent conversation does not leak to parent

• Model override: agent runs on specified model tier, not parent's model

• Tool restriction: agent only accesses tools in its allowlist (or all minus denylist)

• Scoped hooks: agent hooks fire only during execution, cleaned up after completion

• Skill loading: referenced skills' SKILL.md injected into agent context

• MCP lifecycle: servers started on activation, stopped on completion

• SubagentStop: fires on completion with agent name, exit reason, turns used

• maxTurns: agent stops at limit with graceful summary

• Hot-reload: filesystem watcher detects new/modified .md files without restart

• Pipeline chaining: SubagentStop hooks can trigger next stage via pipeline state file

• Agent Teams: custom agents work as teammates with mailbox/task coordination

• Cost tracking: agent token consumption reported in AgentResult and attributed in cost dashboard
```
