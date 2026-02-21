**Architecture Context Reference**

APIs, Schemas, Patterns, and Code Examples

Companion to Implementation Tasks Document

**Purpose**

This document contains the reference material an LLM needs to implement the Claude Code v3 system. It provides: Anthropic API schemas, Strands SDK interfaces, Claude Code hook event schemas, MCP protocol contracts, community-proven patterns, and configuration file formats. Load this document into context alongside the Implementation Tasks document when coding any task.

**1. Anthropic Messages API**

The core API for model interactions. All agent-to-model communication uses this endpoint.

**1.1 Request Schema**

```
POST https://api.anthropic.com/v1/messages

interface MessagesRequest {

model: string; // 'claude-opus-4-6' | 'claude-sonnet-4-5-20250929' | 'claude-haiku-4-5-20251001'

max_tokens: number; // 1-128000

messages: Message[];

system?: string | SystemBlock[];

tools?: ToolDefinition[];

tool_choice?: { type: 'auto' | 'any' | 'tool'; name?: string };

temperature?: number; // 0.0-1.0, default 1.0

thinking?: { type: 'enabled'; budget_tokens: number }; // Extended thinking

stream?: boolean;

}

interface Message {

role: 'user' | 'assistant';

content: string | ContentBlock[];

}

interface ContentBlock {

type: 'text' | 'image' | 'tool_use' | 'tool_result' | 'thinking' | 'document';

// Fields vary by type

}
```

**1.2 Prompt Caching**

Add cache_control blocks to system messages and tool definitions for caching:

```
// System message with 1-hour cache

system: [{

type: 'text',

text: 'You are a coding assistant...',

cache_control: { type: 'ephemeral', ttl: 3600 } // 1-hour

}]

// Tool definitions with 5-minute cache

tools: [{

name: 'read_file',

description: '...',

input_schema: { ... },

cache_control: { type: 'ephemeral' } // 5-minute default

}]
```

Response includes cache metrics:

```
interface Usage {

input_tokens: number;

output_tokens: number;

cache_creation_input_tokens: number; // Tokens written to cache

cache_read_input_tokens: number; // Tokens read from cache (90% discount)

}
```

**1.3 Extended Thinking (Adaptive)**

Opus 4.6 supports adaptive thinking where the model picks up contextual clues about thinking depth. Developers can also set explicit effort controls:

```
// Enable extended thinking with budget

thinking: {

type: 'enabled',

budget_tokens: 10000 // Quick: 2000, Standard: 10000, Deep: 50000+

}
```

**1.4 Model Identifiers**

|            |                            |                    |
|------------|----------------------------|--------------------|
| **Model**  | **API String**             | **Context Window** |
| Opus 4.6   | claude-opus-4-6            | 1M tokens (beta)   |
| Sonnet 4.5 | claude-sonnet-4-5-20250929 | 200K (1M beta)     |
| Haiku 4.5  | claude-haiku-4-5-20251001  | 200K               |

**2. Hook Event JSON Schemas**

Each hook event sends a JSON context object to the handler on stdin. These are the complete schemas for each event.

**2.1 Common Input Fields (all events)**

```
{

"session_id": "abc123",

"hook_event_name": "PreToolUse",

"cwd": "/home/user/project",

"session_type": "interactive" | "headless"

}
```

**2.2 PreToolUse Input**

```
{

...common,

"tool_name": "Bash" | "Write" | "Edit" | "MultiEdit" | "Read" | "Glob" | "Grep" | ...,

"tool_input": {

"command": "npm test", // for Bash

"file_path": "/src/app.ts", // for Write/Edit

"content": "...", // for Write

"old_string": "...", // for Edit

"new_string": "..." // for Edit

}

}
```

PreToolUse output options:

```
// Allow (default - just exit 0)

// Deny (exit code 2 OR structured JSON)

{ "hookSpecificOutput": {

"hookEventName": "PreToolUse",

"permissionDecision": "deny",

"permissionDecisionReason": "Blocked: rm -rf is not allowed"

}}

// Modify input (transparent to model)

{ "hookSpecificOutput": {

"hookEventName": "PreToolUse",

"permissionDecision": "allow",

"updatedInput": { "command": "npm test -- --dry-run" },

"additionalContext": "Added --dry-run flag for safety"

}}
```

**2.3 PostToolUse Input**

```
{

...common,

"tool_name": "Write",

"tool_input": { "file_path": "/src/app.ts", "content": "..." },

"tool_response": "File written successfully"

}
```

**2.4 Stop Event Input**

```
{

...common,

"stop_reason": "end_turn" | "tool_use" | "max_tokens"

}

// Block completion (force agent to continue)

{ "decision": "block", "reason": "Tests failing. Fix them before completing." }
```

**2.5 PermissionRequest Input**

```
{

...common,

"tool_name": "Bash",

"tool_input": { "command": "npm run lint" }

}

// Auto-allow

{ "hookSpecificOutput": {

"hookEventName": "PermissionRequest",

"decision": { "behavior": "allow", "updatedInput": { "command": "npm run lint" } }

}}
```

**2.6 SessionStart Input**

```
{

...common,

"source": "new" | "resume" | "teleport",

"model": "claude-opus-4-6",

"agent_type": "main" | "subagent" | "teammate" // optional

}
```

**2.7 Environment Variables Available in Hooks**

|  |  |
|----|----|
| **Variable** | **Description** |
| $CLAUDE_TOOL_INPUT_FILE_PATH | File path from current tool input (Write/Edit) |
| $CLAUDE_SESSION_ID | Current session identifier |
| $CLAUDE_MODEL | Current model being used |
| $CLAUDE_AUTOCOMPACT_PCT_OVERRIDE | Custom compaction threshold (1-100) |
| $ENABLE_TOOL_SEARCH | Tool search mode: auto|auto:N|false |
| $MCP_TIMEOUT | MCP server startup timeout (ms) |

**3. Configuration File Formats**

**3.1 settings.json (Hook Configuration)**

```
{

"hooks": {

"PreToolUse": [

{

"matcher": "Bash",

"hooks": [

{ "type": "command", "command": "./scripts/security-check.sh" },

{ "type": "prompt", "prompt": "Evaluate if this bash command is safe: $ARGUMENTS" },

{ "type": "command", "command": "node log.js", "async": true, "timeout": 30000 }

]

},

{

"matcher": "Write|Edit|MultiEdit",

"hooks": [

{ "type": "command", "command": "npx prettier --write \$CLAUDE_TOOL_INPUT_FILE_PATH\" }

]

}

],

"PostToolUse": [ ... ],

"Stop": [ { "matcher": ".*", "hooks": [

{ "type": "command", "command": "./scripts/post-check.sh" }

] } ]

},

"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",

"env": {

"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "90",

"ENABLE_TOOL_SEARCH": "auto:5"

}

}
```

**3.2 plugin.json (Plugin Manifest)**

```
{

"name": "my-plugin",

"description": "Plugin description",

"version": "1.0.0",

"author": { "name": "Developer Name" },

"homepage": "https://github.com/owner/repo",

"license": "MIT"

}
```

**3.3 Skill SKILL.md Frontmatter**

```
---

name: secure-operations

description: Perform operations with security validation

disable-model-invocation: false

hooks:

PreToolUse:

\- matcher: "Bash"

hooks:

\- type: command

command: "./scripts/security-check.sh"

---

# Skill instructions follow here

When performing operations, always validate security constraints first...
```

**3.4 .mcp.json (MCP Server Configuration)**

```
{

"mcpServers": {

"github": {

"command": "npx",

"args": ["-y", "@modelcontextprotocol/server-github"],

"env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "..." },

"scope": "project"

},

"postgres": {

"command": "npx",

"args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://..."],

"scope": "user"

}

}

}
```

**4. Community-Proven Patterns**

**4.1 Context Backup on Compaction**

PreCompact hook that backs up critical state before compaction destroys it:

```
// .claude/hooks/pre-compact-backup.sh

#!/bin/bash

HOOK_INPUT=$(cat)

BACKUP_DIR=".claude/backups"

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "$HOOK_INPUT" > "$BACKUP_DIR/$TIMESTAMP-precompact.json"

# Extract and preserve critical context items

echo "$HOOK_INPUT" | jq '.context_items[] | select(.category == "PRESERVE_VERBATIM")' \

> "$BACKUP_DIR/$TIMESTAMP-preserved.json"

exit 0
```

**4.2 Security Guard (PreToolUse)**

Block dangerous bash commands before execution:

```
#!/bin/bash

HOOK_INPUT=$(cat)

COMMAND=$(echo "$HOOK_INPUT" | jq -r '.tool_input.command // empty')

[[ -z "$COMMAND" ]] && exit 0

# Dangerous patterns

BLOCKED_PATTERNS=(

'rm -rf /'

'DROP TABLE'

'curl.*| bash'

'chmod 777'

'git push.*--force'

)

for pattern in "${BLOCKED_PATTERNS[@]}"; do

if echo "$COMMAND" | grep -qiE "$pattern"; then

jq -n --arg reason "Blocked: matches dangerous pattern '$pattern'" \

'{ "hookSpecificOutput": { "hookEventName": "PreToolUse", "permissionDecision": "deny", "permissionDecisionReason": $reason } }'

exit 2

fi

done

exit 0
```

**4.3 Auto-Format on File Write (PostToolUse)**

```
// settings.json

{

"hooks": {

"PostToolUse": [{

"matcher": "Write|Edit|MultiEdit",

"hooks": [{

"type": "command",

"command": "npx prettier --write \$CLAUDE_TOOL_INPUT_FILE_PATH\"

}]

}]

}

}
```

**4.4 Self-Improving Loop Context File**

```
// .claude/loop-context.json

{

"currentTask": "Implement user authentication with JWT tokens",

"acceptanceCriteria": [

"POST /auth/login returns JWT on valid credentials",

"JWT includes user ID and role claims",

"All tests pass with > 90% coverage"

],

"constraints": [

"Use bcrypt for password hashing",

"Token expiry: 1 hour"

],

"learnings": [

{ "iteration": 1, "pattern": "bcrypt.compare is async", "resolution": "Always await bcrypt.compare()", "confidence": 1.0 },

{ "iteration": 2, "pattern": "JWT_SECRET missing from env", "resolution": "Load from .env with dotenv at app startup", "confidence": 1.0 }

],

"failedApproaches": [

{ "iteration": 1, "approach": "Sync bcrypt.compareSync in middleware", "why_failed": "Blocks event loop under load" }

],

"iterationCount": 3,

"maxIterations": 10

}
```

**4.5 Agent Team Task List**

```
// .claude/tasks/security-audit/tasks.json

{

"tasks": [

{ "id": "auth-review", "title": "Review authentication flows", "assignee": "teammate-1", "status": "complete",

"dependencies": [], "files": ["src/auth/**"] },

{ "id": "input-val", "title": "Check input validation", "assignee": "teammate-2", "status": "in-progress",

"dependencies": [], "files": ["src/api/**"] },

{ "id": "db-security", "title": "Analyze database queries", "assignee": "teammate-3", "status": "pending",

"dependencies": [], "files": ["src/db/**"] },

{ "id": "cross-ref", "title": "Cross-reference findings", "assignee": null, "status": "blocked",

"dependencies": ["auth-review", "input-val", "db-security"], "files": [] }

]

}
```

**5. Strands Agents SDK Quick Reference**

The Strands Agents SDK 1.0 (Python-first, but the patterns translate to TypeScript) provides these key abstractions:

**5.1 Core Agent Pattern**

```
# Python reference (translate to TypeScript)

from strands import Agent, tool

@tool

def read_file(path: str) -> str:

"""Read a file from disk"""

return open(path).read()

agent = Agent(

model='anthropic/claude-sonnet-4-5-20250929',

tools=[read_file],

system_prompt='You are a coding assistant.'

)

result = agent('Read the file at /src/app.ts and summarize it')
```

**5.2 Multi-Agent: Agents-as-Tools**

```
researcher = Agent(tools=[web_search], system_prompt='Research agent')

coder = Agent(tools=[read_file, write_file], system_prompt='Coding agent')

orchestrator = Agent(

tools=[researcher.as_tool(), coder.as_tool()],

system_prompt='Orchestrate research and coding tasks'

)
```

**5.3 Session Management**

```
from strands.session import SessionManager, SummarizingConversationManager

# Persistent sessions

session = SessionManager(storage='dynamodb', table='sessions')

# Conversation summarization (extend this for semantic triage)

convo = SummarizingConversationManager(

max_tokens=100000,

summarization_model='haiku'

)
```

**5.4 Hook Provider Interface**

```
from strands.hooks import HookProvider

class CustomHookProvider(HookProvider):

def pre_tool_use(self, tool_name, tool_input):

# Return 'allow', 'deny', or modified input

pass

def post_tool_use(self, tool_name, tool_input, tool_response):

pass
```

**5.5 Observability**

```
# Built-in OpenTelemetry instrumentation

from strands.telemetry import get_tracer

tracer = get_tracer('claude-code-v3')

with tracer.start_as_current_span('process_request') as span:

span.set_attribute('model_tier', 'sonnet')

span.set_attribute('effort_level', 'standard')

span.set_attribute('cost_usd', 0.045)

span.set_attribute('fast_mode', False)
```

**5.6 Guardrails Integration**

```
agent = Agent(

model='bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0',

guardrail_id='your-guardrail-id',

guardrail_version='DRAFT',

# guardrail_latest_message=True # Apply only to latest message

)
```

**6. Token Budget and Pricing Quick Reference**

**6.1 Pricing Formulas**

```
// Calculate cost for a single API call

function calculateCost(

inputTokens: number,

outputTokens: number,

model: ModelTier,

cacheReadTokens: number = 0,

cacheWriteTokens: number = 0

): number {

const p = PRICING[model];

const regularInput = inputTokens - cacheReadTokens;

return (

(regularInput / 1_000_000) * p.inputPerMillion +

(outputTokens / 1_000_000) * p.outputPerMillion +

(cacheReadTokens / 1_000_000) * p.cachedInputPerMillion +

(cacheWriteTokens / 1_000_000) * p.cacheWritePerMillion

);

}
```

**6.2 Typical Session Profiles**

|                         |           |            |           |               |
|-------------------------|-----------|------------|-----------|---------------|
| **Scenario**            | **Input** | **Output** | **Turns** | **Est. Cost** |
| Simple fix (Haiku)      | 20K       | 5K         | 3-5       | $0.04        |
| Feature impl (Sonnet)   | 100K      | 30K        | 15-25     | $0.75        |
| Complex refactor (Opus) | 200K      | 50K        | 30-50     | $2.25        |
| Agent Team (4x Sonnet)  | 400K      | 120K       | 60-100    | $3.00        |
| 16-agent build (Opus)   | 3.2M      | 800K       | 2000+     | $20,000+     |

**6.3 Cache Savings Estimates**

With proper caching of stable layers (system prompt + CLAUDE.md + tool schemas, typically 5-15K tokens): After the initial cache write, subsequent turns save 90% on those tokens. Over a 25-turn session with 10K cached tokens per turn, savings are approximately: 25 turns x 10K tokens x $3/M x 90% = $0.675 saved (Sonnet). This represents 30-50% of total session cost.

**7. Architectural Decisions and Rationale**

**7.1 Why TypeScript over Python for Implementation**

- Claude Code itself is TypeScript. Plugin ecosystem is JS/TS-first.

- MCP SDK is available in both, but the ecosystem leans TS.

- Strands SDK is Python-first, but the patterns (tool_use loop, agent composition) translate directly.

- For production deployment, consider maintaining a Python adapter layer for Strands-native features.

**7.2 Why File-System IPC for Agent Teams**

- Claude Code uses file-lock-based coordination natively. This is proven at scale (16-agent teams).

- No additional infrastructure needed (no Redis, no message queues).

- Git provides built-in conflict detection and resolution.

- Limitation: Single machine only. For distributed teams, consider upgrading to a shared filesystem (EFS) or message queue.

**7.3 Why Semantic Triage over Simple Summarization**

- Simple summarization (Strands SummarizingConversationManager) is lossy for critical data.

- Error messages, file paths, and variable names cannot be reconstructed from summaries.

- Community reports that agents 'forget everything' after compaction are traced to this exact issue.

- Semantic triage costs slightly more in classification overhead but prevents cascading failures from context loss.

**7.4 Reference URLs**

- Anthropic API: https://docs.anthropic.com/en/api/messages

- Claude Code Hooks: https://code.claude.com/docs/en/hooks

- Claude Code Plugins: https://code.claude.com/docs/en/plugins

- Claude Code MCP: https://code.claude.com/docs/en/mcp

- Strands Agents SDK: https://github.com/strands-agents/sdk-python

- Strands Evals: https://github.com/strands-agents/evals

- MCP Specification: https://modelcontextprotocol.io

- Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching

- Agent Teams Guide: https://code.claude.com/docs/en/agent-teams

**7. Custom Agent YAML Schema Reference**

Complete YAML frontmatter schema for custom agent definitions. Agent files are Markdown (.md) with YAML frontmatter delimited by --- lines.

**7.1 Full Field Specification**

```
---

# REQUIRED

name: string # lowercase-with-hyphens, regex: ^[a-z][a-z0-9-]*$

description: string # Multi-line with | pipe. Include usage examples.

# MODEL

model: enum # 'sonnet' | 'opus' | 'haiku' | 'inherit'

# Alias mapping:

# sonnet → claude-sonnet-4-5-20250929

# opus → claude-opus-4-6

# haiku → claude-haiku-4-5-20251001

# inherit → parent's model

# TOOLS

tools: string # Comma-separated OR YAML array

# Internal: Read, Write, Edit, MultiEdit, Bash, Glob, Grep, LS, Task

# MCP: 'server-name/tool-name' (e.g. 'github/create_pull_request')

disallowedTools: string # Denylist. Mutually exclusive with tools.

# PERMISSIONS

permissionMode: enum # 'default' | 'bypassPermissions' | 'plan'

# VISUAL

color: enum # 'purple' | 'cyan' | 'green' | 'orange' | 'blue' | 'red'

# LIMITS

maxTurns: integer # Max agentic loop iterations. Prevents runaway.

# HOOKS (all 12 events)

hooks: # Scoped to agent lifecycle

PreToolUse: # Matcher + handler array (same schema as settings.json hooks)

\- matcher: 'Bash'

hooks:

\- type: command

command: './scripts/validate.sh'

# SKILLS

skills: string # Comma-separated skill names (loaded into agent context)

# MCP SERVERS (agent-scoped)

mcpServers:

server-name:

command: string # Binary to run

args: string[] # Command arguments

env: # Environment variables (${VAR} substitution supported)

KEY: '${ENV_VAR}'

# MEMORY

memory: string # Persistent context injected every invocation

---

# Markdown body below = system prompt
```

**7.2 Storage Locations**

```
Project scope: .claude/agents/*.md (version-controlled, team-shared)

User scope: ~/.claude/agents/*.md (personal, all projects)

Plugin scope: {plugin}/agents/*.md (namespaced: plugin-name:agent-name)

CLI inline: --agents '{JSON}' (ephemeral, same fields as YAML)

Precedence: project > user > plugin

Loaded at session start. Hot-reload via /agents or filesystem watcher.
```

**7.3 Model Alias Resolution**

```
function resolveModelAlias(alias: string, parentModel?: string): string {

const ALIASES: Record\<string, string> = {

'sonnet': 'claude-sonnet-4-5-20250929',

'opus': 'claude-opus-4-6',

'haiku': 'claude-haiku-4-5-20251001',

};

if (alias === 'inherit') return parentModel || ALIASES['sonnet'];

return ALIASES[alias] || alias;

}
```

**7.4 Agent Execution Lifecycle**

```
1. Dispatch: Orchestrator calls AgentDispatcher.dispatch(agentName, task)

2. Resolve: Map model alias → API model string; check CostGovernor budget

3. Tooling: Build effective tool list (allowlist/denylist/inherit)

4. MCP Start: Start agent-scoped MCP servers (if any)

5. Skills: Load referenced skills into context prefix

6. Hooks: Register agent-scoped hooks with HookEngine (additive)

7. Context: Create isolated context window

8. System: Inject memory + skill context + systemPrompt as system message

9. Loop: Execute agentic loop (model → tool_use → PreToolUse hook →

execute → PostToolUse hook → tool_result → repeat)

10. Stop: Model returns end_turn OR maxTurns reached OR budget exceeded

11. Lifecycle: Fire SubagentStop hooks (status, turns, tokens)

12. Cleanup: Deregister scoped hooks → stop MCP servers → release context

13. Return: AgentResult (summary, tokens, tools used, files modified) to parent
```

**7.5 Integration Points**

```
Context Manager: Each agent gets contextManager.createIsolated()

Hook Engine: hookEngine.registerScoped(agent.hooks, scopeId) → cleanup function

Cost Governor: costGovernor.checkBudget(agentName, model) → may override model tier

Skill Registry: skillRegistry.load(skillNames) → concatenated SKILL.md contents

Agent Teams: Team lead can dispatch custom agents as teammates with mailbox IPC

Pipeline: SubagentStop hooks write to .claude/pipeline-state.json for chaining
```

**7.6 Complete Agent Templates**

Five production-ready agent templates are provided in the companion document custom_agent_schema.docx, Section 6: Code Reviewer (Sonnet, read-only tools, purple), Security Auditor (Opus, read-only, OWASP skill, red), Implementer-Tester (Sonnet, all tools, PostToolUse auto-format hook, green), Researcher (Haiku, read-only, plan mode, cyan), and Architect (Opus, GitHub MCP server, ADR skill, blue).
