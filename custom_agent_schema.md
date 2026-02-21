**Custom Agent Definition Schema**

Complete YAML Specification, Invocation Patterns, and Composition Model

Supplement to Claude Code v3 Spec v1.1 — Section 4A

February 2026

**1. Overview**

Custom agents in Claude Code are specialized AI assistants defined as Markdown files with YAML frontmatter. Each agent gets its own system prompt, tool access restrictions, model selection, permission mode, scoped hooks, skill references, MCP server configuration, and isolated context window. Claude delegates tasks to agents automatically based on description matching, or users invoke them explicitly.

This document specifies the complete schema so an LLM can: create new agent definitions from natural language descriptions, validate existing agent configurations, compose agents with skills and hooks, build the agent registry and dispatch system in code, and generate agent templates for common patterns (reviewer, implementer, researcher, security auditor).

**1.1 Agent vs. Skill vs. Command**

|  |  |  |  |
|----|----|----|----|
| **Aspect** | **Custom Agent** | **Skill** | **Command** |
| Context | Own context window (isolated) | Runs in caller's context window | Runs in caller's context window |
| Invocation | Auto-dispatch by description OR explicit "Use the X agent" | Auto-invoke by description match OR /slash-command | /slash-command only |
| Result | Returns summary to delegating agent | Modifies caller's context directly | Modifies caller's context directly |
| Model | Can specify own model tier | Inherits caller's model | Inherits caller's model |
| Tools | Can restrict or expand independently | Can restrict via allowed_tools | No tool restrictions |
| Hooks | Scoped hooks (SubagentStop fires on completion) | Scoped hooks (cleaned up when skill exits) | No scoped hooks |
| File format | .md with YAML frontmatter | SKILL.md with YAML frontmatter | .md (plain markdown) |
| Best for | Complex multi-step tasks needing isolation | Context injection and behavior modification | Simple predefined workflows |

**2. Complete YAML Frontmatter Schema**

The agent definition file is a Markdown file (.md) with YAML frontmatter delimited by --- lines. The Markdown body below the frontmatter becomes the agent's system prompt.

**2.1 Full Schema Reference**

```
---

# ===== REQUIRED FIELDS =====

name: code-reviewer # lowercase-with-hyphens, unique within scope

description: | # Multi-line, detailed, action-oriented

Use this agent when you need to review code for quality,

security vulnerabilities, and adherence to best practices.

Examples:

(1) "Review my recent changes" - launches agent to review git diff

(2) "Check this PR for security issues" - focused security review

(3) "Audit the auth module" - deep analysis of specific module

# ===== MODEL SELECTION =====

model: sonnet # sonnet | opus | haiku | inherit

# inherit = same model as main conversation

# omitted = defaults to inherit

# ===== TOOL ACCESS CONTROL =====

tools: Read, Glob, Grep, Bash # Allowlist: comma-separated tool names

# Omitted = inherits ALL tools from parent

# Internal tools: Read, Write, Edit, MultiEdit,

# Bash, Glob, Grep, LS, Task (spawn subagents)

# MCP tools: prefix with server name:

# e.g., github/create_pull_request

disallowedTools: Write, Edit # Denylist: explicitly block these tools

# Use EITHER tools OR disallowedTools, not both

# ===== PERMISSION MODE =====

permissionMode: default # default | bypassPermissions | plan

# default = normal permission prompts

# bypassPermissions = skip all permission checks

# plan = read-only mode, no mutations

# ===== VISUAL IDENTITY =====

color: purple # purple | cyan | green | orange | blue | red

# Background color in tmux/UI for identification

# ===== EXECUTION LIMITS =====

maxTurns: 50 # Maximum agentic loop turns before forced stop

# Prevents runaway agents

# ===== SCOPED HOOKS =====

hooks: # All 12 hook events supported

PreToolUse: # Runs ONLY while this agent is active

\- matcher: "Bash"

hooks:

\- type: command

command: "./scripts/review-bash-safety.sh"

SubagentStop: # Fires when THIS agent completes

\- hooks:

\- type: command

command: "echo 'Agent complete' >> .claude/logs/agent-completions.log"

Stop: # Auto-converted to SubagentStop for subagents

\- hooks:

\- type: command

command: "./scripts/post-review-summary.sh"

# ===== SKILL REFERENCES =====

skills: security-checklist, style-guide # Comma-separated skill names to load

# Skills' instructions injected into agent context

# ===== MCP SERVER CONFIGURATION =====

mcpServers: # Agent-scoped MCP servers

github: # Started when agent activates, stopped when done

command: npx

args: ["-y", "@modelcontextprotocol/server-github"]

env:

GITHUB_TOKEN: "${GITHUB_TOKEN}" # Environment variable substitution

# ===== MEMORY =====

memory: | # Persistent context loaded every invocation

This project uses ESLint with Airbnb config.

Tests use Vitest. CI runs on GitHub Actions.

---

# System Prompt (Markdown body)

You are a senior code reviewer specializing in TypeScript...

When invoked:

1. Run \`git diff\` to see recent changes

2. Focus on modified files

3. Check for security issues, performance, and code quality

Report findings as:

\- **Critical** (must fix before merge)

\- **Warnings** (should fix)

\- **Suggestions** (nice to have)
```

**2.2 Field Reference Table**

|  |  |  |  |
|----|----|----|----|
| **Field** | **Required** | **Type** | **Description** |
| name | Yes | string | Unique identifier. Lowercase-with-hyphens. Used for invocation and file naming. |
| description | Yes | string | Detailed, action-oriented. Include examples. Claude uses this for auto-dispatch matching. |
| model | No | enum | sonnet | opus | haiku | inherit. Default: inherit (parent's model). |
| tools | No | string | Comma-separated allowlist. Omit to inherit all parent tools including MCP. |
| disallowedTools | No | string | Comma-separated denylist. Mutually exclusive with tools field. |
| permissionMode | No | enum | default | bypassPermissions | plan. Controls mutation rights. |
| color | No | enum | purple | cyan | green | orange | blue | red. Visual identifier in UI. |
| maxTurns | No | integer | Maximum agentic turns. Prevents runaway agents. No default (unlimited). |
| hooks | No | object | Scoped hooks. All 12 events. Active only while agent runs. Stop → SubagentStop. |
| skills | No | string | Comma-separated skill names. Skills loaded into agent's context on activation. |
| mcpServers | No | object | Agent-scoped MCP servers. Started on activation, stopped on completion. |
| memory | No | string | Persistent context injected on every invocation. Project conventions, stack info. |

**3. Storage Locations and Precedence**

**3.1 File Locations**

|  |  |  |
|----|----|----|
| **Scope** | **Path** | **Use Case** |
| Project | .claude/agents/*.md | Codebase-specific agents. Version-controlled. Team-shared. |
| User | ~/.claude/agents/*.md | Personal agents. Available in all projects. |
| Plugin | {plugin}/agents/*.md | Installed via plugin system. Namespaced: plugin-name:agent-name. |
| CLI inline | --agents JSON flag | Ephemeral. JSON with same fields as YAML frontmatter. For CI/scripts. |

**3.2 Precedence Rules**

- When agents share the same name, project-level wins over user-level.

- Plugin agents are namespaced (plugin-name:agent-name), so no collision with user/project agents.

- Built-in agents (general-purpose) are always available but overridable.

- Agents loaded at session start. Create or modify files, then restart session or use /agents to reload.

**4. Invocation Patterns**

**4.1 Auto-Dispatch (Default)**

Claude evaluates every user request against registered agent descriptions. When confidence exceeds the dispatch threshold, Claude automatically delegates via the Task tool. The user never explicitly names the agent.

**Dispatch signals:** User request text, agent description field, current conversation context, available tools on the agent.

**Example:** User says "Review my recent changes for security issues." Claude matches against code-reviewer agent (description mentions "review code" + "security") and auto-delegates.

The auto-dispatch is the same mechanism as tool selection — the model sees agents as available tools with descriptions and decides when to invoke them. This is the key insight: agents ARE tools from the model's perspective.

**4.2 Explicit Invocation**

Users can force agent invocation:

- Natural language: "Use the code-reviewer agent to check this PR"

- Slash command: /agents → select agent from interactive menu

- Claude --agent flag: claude --agent code-reviewer (runs agent as main thread, can spawn its own subagents via Task tool)

**4.3 Programmatic Invocation (Pipeline)**

In pipeline architectures (e.g., PM → Architect → Implementer), one agent invokes another via the Task tool:

```
// Agent A's system prompt includes:

// "When specification is complete, delegate to the architect-review agent"

// Claude internally calls: Task(agent='architect-review', task='Review this spec: ...')
```

Pipeline chaining uses SubagentStop hooks to trigger the next stage:

```
hooks:

SubagentStop:

\- hooks:

\- type: command

command: |

# Parse agent output, determine next stage

STATUS=$(jq -r '.tool_response' \< /dev/stdin | grep -o 'READY_FOR\_[A-Z]*')

echo "Next stage: $STATUS" >> .claude/pipeline-state.json
```

**4.4 Agent Teams Integration**

When Agent Teams is enabled, the team lead can spawn teammates that are custom agents:

```
// Team lead's prompt:

// "Spawn 3 teammates using our custom agents:

// 1. code-reviewer for auth module

// 2. security-auditor for API endpoints

// 3. perf-analyzer for database queries"
```

Each teammate inherits the custom agent's configuration (tools, model, hooks) but operates in the Agent Teams coordination model with mailboxes and shared task lists.

**5. Composition Model: Agents + Skills + Hooks**

**5.1 How Agents Compose with Skills**

When an agent references skills in its frontmatter, those skills' SKILL.md instructions are injected into the agent's context window at activation time. This means:

- The agent's context window pays the token cost of loaded skills.

- Skill-scoped hooks also activate (but only for the duration of the agent's execution).

- Skills can reference other skills, creating a chain. But watch context consumption.

```
# Agent that composes with skills:

---

name: secure-implementer

description: Implement features with security-first approach

model: sonnet

skills: security-checklist, owasp-top-10, code-standards

---

You implement features while following security best practices.

Always consult the loaded security checklist and OWASP guidelines.
```

**5.2 How Agents Compose with Hooks**

Agent-scoped hooks have three behaviors:

- Additive: Agent hooks ADD to any user/project/managed hooks already registered. They don't replace them.

- Lifecycle-scoped: Agent hooks only fire during the agent's execution. They're cleaned up when the agent finishes.

- Stop → SubagentStop: When an agent defines a Stop hook, it's automatically converted to SubagentStop since that's the event that fires when a subagent completes.

**5.3 How Agents Compose with MCP Servers**

Agent-scoped MCP servers start when the agent activates and stop when the agent completes. They are available only to that agent's tools list (not leaked to the parent conversation). Environment variables in server config support ${VAR_NAME} substitution from the process environment.

**5.4 Inheritance Diagram**

```
Main Conversation (Orchestrator)

├─ Model: opus (user's choice)

├─ Tools: [all internal + 3 MCP servers]

├─ Hooks: [user + project + managed]

│

└─▶ Custom Agent: code-reviewer

├─ Model: sonnet (overridden in frontmatter)

├─ Tools: [Read, Glob, Grep, Bash] (restricted in frontmatter)

├─ Hooks: [user + project + managed + agent-scoped]

├─ Skills: [security-checklist loaded into context]

├─ MCP: [github server started for this agent only]

├─ Context Window: SEPARATE from main conversation

│

└─▶ Sub-subagent (if agent uses Task tool)

├─ Model: inherits sonnet from parent agent

├─ Tools: inherits from parent agent

└─ Context Window: SEPARATE (third isolation level)
```

**6. Agent Templates for Common Patterns**

These templates are ready to use. Copy to .claude/agents/ and customize the system prompt.

**6.1 Code Reviewer**

```
---

name: code-reviewer

description: |

Reviews code for quality, security vulnerabilities, and adherence to best practices.

Examples:

(1) "Review my recent changes" - analyzes git diff

(2) "Check this PR for issues" - comprehensive PR review

model: sonnet

tools: Read, Glob, Grep, Bash

color: purple

maxTurns: 30

---

You are a senior code reviewer. When invoked:

1. Run \`git diff HEAD~1\` to see recent changes

2. For each modified file, check: security issues, performance, readability, test coverage

3. Report findings as Critical / Warning / Suggestion
```

**6.2 Security Auditor**

```
---

name: security-auditor

description: |

Deep security analysis of codebase. OWASP Top 10 checks, dependency audit,

authentication flow review, input validation analysis.

model: opus

tools: Read, Glob, Grep, Bash

disallowedTools: Write, Edit

color: red

skills: owasp-top-10

maxTurns: 50

---

You are a security specialist. Analyze the codebase for vulnerabilities.

Focus areas: injection, broken auth, sensitive data exposure, XXE,

broken access control, misconfiguration, XSS, deserialization, dependency vulns.
```

**6.3 Implementer-Tester (Pipeline Stage)**

```
---

name: implementer-tester

description: |

Implements features from specs and writes comprehensive tests.

Use after architect-review has approved the design.

model: sonnet

color: green

maxTurns: 100

hooks:

PostToolUse:

\- matcher: "Write|Edit"

hooks:

\- type: command

command: "npx prettier --write \$CLAUDE_TOOL_INPUT_FILE_PATH\"

SubagentStop:

\- hooks:

\- type: command

command: "echo 'READY_FOR_QA' >> .claude/pipeline-state.json"

---

You implement features and write tests. For each task:

1. Read the spec from .claude/specs/

2. Implement the feature

3. Write unit tests (>90% coverage)

4. Run all tests to ensure nothing breaks

5. Commit with conventional commit message
```

**6.4 Research Agent (Read-Only)**

```
---

name: researcher

description: |

Researches codebase to answer questions. Read-only, never modifies files.

Use when you need to understand how something works before making changes.

model: haiku

tools: Read, Glob, Grep

permissionMode: plan

color: cyan

maxTurns: 20

---

You are a codebase researcher. Read and analyze code to answer questions.

Never suggest changes. Just explain what exists and how it works.

Use Haiku model for speed — you’re optimized for fast exploration at 1/10 the cost.
```

**6.5 Architect (with MCP)**

```
---

name: architect-review

description: |

Reviews architectural decisions against platform constraints.

Produces Architecture Decision Records (ADRs).

model: opus

color: blue

skills: architecture-patterns, adr-template

mcpServers:

github:

command: npx

args: ["-y", "@modelcontextprotocol/server-github"]

env:

GITHUB_TOKEN: "${GITHUB_TOKEN}"

maxTurns: 40

---

You are a software architect. Review designs for:

\- Scalability, performance, security implications

\- Alignment with existing patterns in the codebase

\- Cost/complexity tradeoffs

Produce an ADR using the loaded template.
```

**7. Implementation Task: Agent Registry and Dispatch System**

**Task ID:** Task 25 (addition to Implementation Tasks document)

**Dependencies:** Task 5 (Orchestrator), Task 3 (Hook Engine), Task 12 (Skills Registry)

**Estimated effort:** 10-14 hours

**Output:** src/agents/agent-registry.ts, src/agents/agent-loader.ts, src/agents/agent-dispatcher.ts, src/agents/agent-runner.ts

**7.1 Agent Registry (src/agents/agent-registry.ts)**

```
interface AgentDefinition {

name: string;

description: string;

model: 'sonnet' | 'opus' | 'haiku' | 'inherit';

tools: string[] | null; // null = inherit all

disallowedTools: string[] | null;

permissionMode: 'default' | 'bypassPermissions' | 'plan';

color?: string;

maxTurns?: number;

hooks: Record\<HookEvent, HookDefinition[]>;

skills: string[];

mcpServers: Record\<string, McpServerConfig>;

memory?: string;

systemPrompt: string; // Markdown body after frontmatter

source: 'project' | 'user' | 'plugin' | 'cli';

pluginNamespace?: string; // For plugin agents

filePath: string; // Original .md file location

}

class AgentRegistry {

// Scan all agent directories and build registry

async loadAll(): Promise\<void>;

// Register a single agent (for CLI --agents flag)

register(agent: AgentDefinition): void;

// Get agent by name (respects precedence: project > user > plugin)

get(name: string): AgentDefinition | undefined;

// List all registered agents

list(): AgentDefinition[];

// Watch filesystem for changes (hot-reload)

watch(): void;

}
```

**7.2 Agent Loader (src/agents/agent-loader.ts)**

Parses .md files with YAML frontmatter into AgentDefinition objects:

- Parse YAML frontmatter (use gray-matter or js-yaml)

- Validate required fields (name, description)

- Normalize tool lists (comma-separated string → string array)

- Resolve model alias to API model string

- Parse hook definitions from YAML into HookDefinition structures

- Extract Markdown body as systemPrompt

- Handle encoding edge cases (UTF-8 BOM, CRLF line endings)

**7.3 Agent Dispatcher (src/agents/agent-dispatcher.ts)**

```
class AgentDispatcher {

constructor(

private registry: AgentRegistry,

private hookEngine: HookEngine,

private costGovernor: CostGovernor,

private skillRegistry: SkillRegistry

) {}

// Called by the Orchestrator when the model requests Task tool

async dispatch(agentName: string, task: string): Promise\<AgentResult>;

// Generate tool definitions for all agents (used in model's tool list)

getAgentToolDefinitions(): ToolDefinition[];

// Each agent becomes a tool with name='agent:{agent.name}'

// and description=agent.description

}
```

**7.4 Agent Runner (src/agents/agent-runner.ts)**

Executes a single agent invocation with full isolation:

- 1. Create new context window (isolated from parent)

- 2. Resolve model: if 'inherit', use parent's model; otherwise map alias to API string

- 3. Build tool list: if tools specified, restrict; if disallowedTools, filter; else inherit all

- 4. Start agent-scoped MCP servers (if any)

- 5. Load referenced skills into context (inject SKILL.md contents)

- 6. Register agent-scoped hooks with HookEngine

- 7. Inject memory field and systemPrompt as system message

- 8. Execute agentic loop (model → tool_use → tool_result → repeat) up to maxTurns

- 9. Fire SubagentStop hooks on completion

- 10. Clean up: deregister hooks, stop MCP servers, release context window

- 11. Return structured result (summary of agent's work) to parent

**7.5 Acceptance Criteria**

- Agent files parsed correctly from .claude/agents/ and ~/.claude/agents/

- All YAML frontmatter fields validated (required check, type check, enum check)

- Precedence: project agent overrides user agent with same name

- Auto-dispatch: model sees agents as tools and selects based on description

- Isolated context window: agent's conversation does not leak to parent

- Model override: agent runs on specified model, not parent's model

- Tool restriction: agent can only access tools in its allowlist

- Scoped hooks: agent hooks fire only during agent execution, cleaned up after

- Skill loading: referenced skills injected into agent context

- MCP servers: started on activation, stopped on completion

- SubagentStop: fires correctly, Stop hooks auto-converted

- maxTurns: agent stops at turn limit with graceful summary

- Hot-reload: new/modified .md files detected without restart

- Pipeline chaining: SubagentStop hooks can trigger next pipeline stage

- Agent Teams: custom agents work as teammates with mailbox/task coordination

**8. Design Patterns and Best Practices**

**8.1 Agent Design Rules**

- Limit to 3-4 agents maximum per project. More than that and dispatch accuracy drops.

- Description is the dispatch signal. Write it like documentation: specific, action-oriented, with concrete examples.

- Start with Claude generation (/agents → Generate with Claude), then refine the .md file manually.

- Separate concerns: reviewer reads, implementer writes, researcher explores. Don't create "do everything" agents.

- Progressive tool expansion: start restricted, add tools as you validate behavior.

- Use Haiku for read-only agents (researchers, analyzers) — 10x cheaper than Sonnet.

- Use Opus only for agents that need deep reasoning (architecture review, complex debugging).

**8.2 Pipeline Architecture Pattern**

```
Pipeline: PM-Spec → Architect-Review → Implementer-Tester → QA

Each stage:

1. Reads input from .claude/pipeline/ (previous stage's output)

2. Does its work with isolated context

3. Writes output to .claude/pipeline/ (next stage's input)

4. Sets status via SubagentStop hook (READY_FOR_ARCH, READY_FOR_BUILD, etc.)

5. Orchestrator reads status, dispatches next stage
```

**8.3 Color-Role Mapping Convention**

|           |                                      |                            |
|-----------|--------------------------------------|----------------------------|
| **Color** | **Role Pattern**                     | **Model Recommendation**   |
| Red       | Security / destructive operations    | Opus (needs deep analysis) |
| Purple    | Code review / quality assurance      | Sonnet (balanced)          |
| Blue      | Architecture / planning              | Opus (needs reasoning)     |
| Green     | Implementation / building            | Sonnet (default workhorse) |
| Cyan      | Research / read-only exploration     | Haiku (fast + cheap)       |
| Orange    | DevOps / deployment / infrastructure | Sonnet (balanced)          |

**8.4 Anti-Patterns to Avoid**

- Too many agents: More than 4-5 agents causes dispatch confusion. The model can't reliably choose between similar-sounding options.

- Vague descriptions: "Use this agent for development tasks" matches everything. Be specific: "Use this agent when implementing TypeScript features with Prisma ORM in the /src/features directory."

- Unrestricted write access on reviewers: A code reviewer should never have Write/Edit tools. Use disallowedTools to prevent mutations.

- Missing maxTurns: Without a turn limit, a confused agent can loop indefinitely. Set maxTurns based on expected complexity.

- Putting secrets in agent files: Use ${ENV_VAR} substitution in mcpServers config, never hardcode tokens in .md files.

- Skill overloading: Loading 5+ skills into an agent's context eats thousands of tokens before it starts working. Keep skill references to 2-3 max.
