# AWS Strands SDK + AgentCore Mapping Guide

**How Every Claude Code v3 Component Maps to AWS Strands Agents SDK + Amazon Bedrock AgentCore**

*Companion to Claude Code v3 Spec v1.1 — February 2026*

---

## 1. Executive Summary

This document maps every component from the Claude Code v3 Enterprise System Specification to its implementation using two AWS services: the **Strands Agents SDK** (open-source Python framework, Apache 2.0) and **Amazon Bedrock AgentCore** (managed platform for deploying and operating agents at scale).

The mapping covers 25 implementation tasks across 5 phases. For each v3 component, we identify which Strands primitive implements it, which AgentCore service hosts it in production, and what code changes are needed versus what comes out of the box.

### 1.1 Why This Stack

- **Strands is production-proven** — used by Amazon Q Developer, AWS Glue, and Kiro. Battle-tested at AWS scale.
- **AgentCore provides infrastructure** — Runtime, Memory, Gateway, Identity, Observability, Policy, Evaluations. So you don't build it yourself.
- **Model-agnostic** — Bedrock (Claude, Nova, Llama), Anthropic API direct, OpenAI, Ollama for local dev.
- **Multi-agent primitives** — Strands 1.0 added Graph, Swarm, and Workflow patterns. Direct analogs to our v3 topologies.
- **Protocol support** — AgentCore supports MCP and A2A natively. Matches our plugin and inter-agent communication requirements.

---

## 2. Architecture Overview

The v3 spec has 3 tiers: Orchestration, Agent Execution, and Infrastructure. Here's how they map:

| v3 Tier | Strands SDK | AgentCore Service |
|---------|-------------|-------------------|
| Orchestration | `Agent` + `GraphBuilder` + `Swarm` | AgentCore Runtime (hosts the orchestrator) |
| Agent Execution | `Agent`, `@tool`, MCP client | AgentCore Runtime (microVM per session) |
| Context Management | `SessionManager`, `SummarizingConversationManager` | AgentCore Memory (short-term + long-term) |
| Tool Integration | MCP servers, `@tool` decorator | AgentCore Gateway (MCP proxy, OAuth, API keys) |
| Security / Identity | Bedrock Guardrails integration | AgentCore Identity + Policy (Cedar) |
| Observability | OpenTelemetry instrumentation | AgentCore Observability (CloudWatch dashboards) |
| Evaluation | `strands-agents-eval` | AgentCore Evaluations (13 built-in + custom) |
| Deployment | Containerized agent code | AgentCore Runtime (serverless, auto-scaling) |

---

## 3. Component-by-Component Mapping

### 3.1 Core Orchestrator → Strands Agent

**v3 Spec:** Section 2 (System Architecture), Task 5
**Strands:** `strands.Agent` class — the core primitive. Model + system_prompt + tools.

```python
from strands import Agent
from strands.models.bedrock import BedrockModel

model = BedrockModel(model_id='us.anthropic.claude-opus-4-6-v1:0')

orchestrator = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[context_manager, cost_governor, agent_registry]
)
```

**AgentCore hosting:** Deploy as AgentCore Runtime with `BedrockAgentCoreApp`. Each user session gets an isolated microVM.

```python
from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload, session):
    return orchestrator(payload['input'])
```

---

### 3.2 Context Management → Strands SessionManager + AgentCore Memory

**v3 Spec:** Section 3 (Context Management Architecture), Task 2

| v3 Feature | Strands Implementation | AgentCore Service |
|------------|----------------------|-------------------|
| Semantic triage (PRESERVE_VERBATIM, COMPRESS, etc.) | Custom `ConversationManager` subclass with classification logic | Custom logic in your agent code (not built-in) |
| Session persistence | `SessionManager` abstraction (file, S3, or custom backend) | AgentCore Memory short-term (automatic per session) |
| Learning store (cross-session) | Custom tool writing to persistent store | AgentCore Memory long-term (semantic, user preference, summary strategies) |
| Compaction strategy | `SummarizingConversationManager` (auto-summarize when threshold hit) | AgentCore Memory summary strategy (built-in) |
| Context backup on compaction | Custom hook in `ConversationManager` | Not built-in — implement as Strands callback |

```python
from strands.session import FileSessionManager
from bedrock_agentcore.memory import MemoryClient

# Short-term: Strands SessionManager for conversation history
session_mgr = FileSessionManager(session_id='user-123')

# Long-term: AgentCore Memory for cross-session learning
memory = MemoryClient()
memory.create_memory(namespace='learning-store',
    content={'pattern': 'auth module requires Opus', 'confidence': 0.95})
```

---

### 3.3 Multi-Agent Orchestration → Strands Multi-Agent Patterns

**v3 Spec:** Sections 4, 5, 6 (Multi-Agent, Agent Teams, Self-Improving Loops), Tasks 6-8

| v3 Topology | Strands Pattern | Code |
|-------------|----------------|------|
| Hierarchical subagents (Agents-as-Tools) | `Agent.as_tool()` / `@tool` wrapping an Agent | `lead = Agent(tools=[reviewer.as_tool()])` |
| Agent Teams (peer collaboration) | `Swarm` (shared context, collaborative handoffs) | `team = Swarm([analyst, writer, reviewer])` |
| Self-improving loops | `Graph` with loop edges + conditional exit | `GraphBuilder` with review → fix cycle |
| Pipeline (PM → Architect → Impl) | `Workflow` or `Graph` (sequential) | `builder.add_edge('pm', 'architect')` |
| Custom agents (YAML frontmatter) | `Agent` with name, system_prompt, tools | Custom loader from `.md` → `Agent()` |

#### 3.3.1 Hierarchical Subagents

```python
from strands import Agent, tool

reviewer = Agent(
    name='code-reviewer',
    model=BedrockModel(model_id='us.anthropic.claude-sonnet-4-5-v1:0'),
    system_prompt='You are a senior code reviewer...',
    tools=[read_file, grep, glob]
)

# Wrap as tool for the orchestrator
@tool
def review_code(task: str) -> str:
    '''Review code for quality and security issues.'''
    return str(reviewer(task))

orchestrator = Agent(tools=[review_code, implement_feature, run_tests])
```

#### 3.3.2 Agent Teams (Swarm)

```python
from strands.multiagent import Swarm

analyst = Agent(name='analyst', tools=[calculator, python_repl])
writer = Agent(name='writer', tools=[file_write])
reviewer = Agent(name='reviewer', tools=[read_file, grep])

# Swarm = peer collaboration with shared context
team = Swarm([analyst, writer, reviewer])
result = team.execute('Analyze auth module and write a security report')
```

#### 3.3.3 Self-Improving Loop (Graph)

```python
from strands.multiagent import GraphBuilder

builder = GraphBuilder()
builder.add_node(implementer, 'implement')
builder.add_node(tester, 'test')
builder.add_node(reviewer, 'review')

builder.add_edge('implement', 'test')
builder.add_edge('test', 'review')
# Conditional loop: if review fails, go back to implement
builder.add_conditional_edge('review', lambda r: 'implement' if 'FAIL' in r else 'END')

graph = builder.build()
result = graph('Implement user authentication with JWT')
```

---

### 3.4 Hook System → Strands Callbacks + AgentCore Policy

**v3 Spec:** Section 7 (Hooks and Policy Engine), Task 3

| v3 Hook Event | Strands Equivalent | AgentCore Equivalent |
|---------------|-------------------|---------------------|
| PreToolUse | `HookProvider.before_tool_call` | AgentCore Policy (Cedar rules via Gateway) |
| PostToolUse | `HookProvider.after_tool_call` | Gateway observability metrics |
| PreCompact | `ConversationManager` callback | Custom (not built-in) |
| SessionStart/End | Agent lifecycle hooks | Runtime session lifecycle events |
| SubagentStop | Graph/Swarm completion events | Runtime async task completion |
| PermissionRequest | Custom tool wrapper | Policy ENFORCE mode (Cedar) |

```python
from strands.hooks import HookProvider, HookRegistry
from strands.hooks import BeforeToolCallEvent, AfterToolCallEvent

class SecurityHook(HookProvider):
    def register_hooks(self, registry: HookRegistry):
        registry.add_callback(BeforeToolCallEvent, self.check_safety)
        registry.add_callback(AfterToolCallEvent, self.log_usage)

    def check_safety(self, event: BeforeToolCallEvent):
        if 'rm -rf' in str(event.tool_args):
            raise ValueError('Blocked: destructive command')

    def log_usage(self, event: AfterToolCallEvent):
        logger.info(f'Tool {event.tool_name} completed')

agent = Agent(tools=[bash], hooks=[SecurityHook()])
```

**AgentCore Policy (production-grade):** Cedar language for fine-grained permissions. Enforced at Gateway level, outside the agent's reasoning loop.

```cedar
# Cedar policy: block destructive S3 operations on production buckets
permit(principal, action == AgentCore::Action::"delete_s3_bucket",
  resource == AgentCore::Gateway::"arn:aws:...")
when { !(context.input.bucket_name like "*prod*") };
```

---

### 3.5 Cost Governance → Strands Model Selection + Bedrock Pricing

**v3 Spec:** Section 10 (Token Economics), Task 4

Strands doesn't have a built-in cost governor. This is custom code that wraps model selection. However, the model-switching capability is native:

```python
from strands.models.bedrock import BedrockModel

# Model tier routing (custom implementation)
MODELS = {
    'haiku':  BedrockModel(model_id='us.anthropic.claude-haiku-4-5-v1:0'),
    'sonnet': BedrockModel(model_id='us.anthropic.claude-sonnet-4-5-v1:0'),
    'opus':   BedrockModel(model_id='us.anthropic.claude-opus-4-6-v1:0'),
}

class CostGovernor:
    def select_model(self, task_complexity, budget_remaining):
        if budget_remaining < 0.10:
            return MODELS['haiku']  # Emergency: cheapest model
        if task_complexity == 'deep_reasoning':
            return MODELS['opus']
        return MODELS['sonnet']  # Default workhorse
```

**AgentCore pricing:** Consumption-based. Runtime charges for active CPU only (not I/O wait). No pre-provisioning. Token costs pass through to Bedrock pricing.

---

### 3.6 Prompt Caching → Bedrock Prompt Caching

**v3 Spec:** Section 11 (Prompt Caching Architecture), Task 9

Prompt caching is a Bedrock feature, not a Strands feature. Strands passes through `cache_control` blocks to the Bedrock API. Your agent code manages cache placement strategy:

```python
# Bedrock handles caching at the API level
# Strands' BedrockModel passes cache_control blocks through
# Configure via Bedrock API: 5-min and 1-hour cache durations
# 90% discount on cached input tokens
```

---

### 3.7 Custom Agents → Strands Agent + Custom Loader

**v3 Spec:** Section 4A (Custom Agent Definition Schema), Task 25

The v3 YAML frontmatter agent format is Claude Code-specific. To use it with Strands, build a loader that parses `.md` files into Strands Agent objects:

```python
import frontmatter  # pip install python-frontmatter
from strands import Agent

def load_agent_from_md(filepath: str) -> Agent:
    post = frontmatter.load(filepath)
    fm = post.metadata
    body = post.content

    model = MODELS.get(fm.get('model', 'sonnet'))
    tools = resolve_tools(fm.get('tools', None))

    return Agent(
        name=fm['name'],
        model=model,
        system_prompt=body,
        tools=tools,
        hooks=parse_hooks(fm.get('hooks', {}))
    )

# Load all agents from directory
agents = {a.name: a for f in Path('.claude/agents').glob('*.md')
          for a in [load_agent_from_md(f)]}
```

---

### 3.8 Plugin Ecosystem → MCP Servers + AgentCore Gateway

**v3 Spec:** Section 8 (Plugin Ecosystem), Task 13

| v3 Plugin Feature | Strands / MCP | AgentCore |
|-------------------|---------------|-----------|
| Plugin tools | MCP servers (native support) | Gateway auto-converts APIs to MCP tools |
| Plugin agents | Load from plugin dir as Agent objects | Deploy as separate Runtime instances |
| Plugin skills | Inject SKILL.md into `system_prompt` | Custom (prepend to prompt) |
| Plugin hooks | `HookProvider` loaded from plugin | Policy (Cedar) for tool-level control |
| Marketplace | Git repos with plugin.json | AWS Marketplace (pre-built agent tools) |
| MCP Tool Search | Built-in `retrieve` tool for semantic search over tool descriptions | Gateway `x_amz_bedrock_agentcore_search` tool |

---

### 3.9 Security → AgentCore Identity + Policy + Guardrails

**v3 Spec:** Section 15 (Security), Task 17

| v3 Security Feature | Strands | AgentCore |
|---------------------|---------|-----------|
| PII/PHI classification | Bedrock Guardrails integration | Guardrails (same, via Bedrock) |
| Constitutional AI | Custom hook/guardrail | Guardrails content filters |
| Agent identity | Not built-in | AgentCore Identity (Okta, Entra, Cognito) |
| Tool permission control | Hook-based allow/deny | Policy (Cedar) at Gateway |
| Session isolation | Separate Agent instances | MicroVM per session (hardware isolation) |
| Credential management | Environment variables | Identity secure token vault (OAuth, API keys) |
| VPC connectivity | Standard networking | VPC-only mode across all services |

---

### 3.10 Observability → Strands OTel + AgentCore Observability

**v3 Spec:** Section 14 (Observability), Task 18

Strands emits OpenTelemetry traces natively. AgentCore Observability provides dashboards, visualization, and alerting on top:

```python
# Strands: Built-in OTel instrumentation
from strands.telemetry import get_tracer
tracer = get_tracer('claude-code-v3')

with tracer.start_as_current_span('agent-execution') as span:
    span.set_attribute('agent.name', agent.name)
    span.set_attribute('model.id', model.model_id)
    result = agent(task)
    span.set_attribute('tokens.input', result.usage.input_tokens)
    span.set_attribute('tokens.output', result.usage.output_tokens)
```

**AgentCore Observability:** Automatic dashboards for token usage, latency, session duration, error rates. Visualizes full agent execution path including tool invocations and model interactions. Integrates with CloudWatch, X-Ray, and any OTel-compatible backend.

---

### 3.11 Evaluation → Strands Evals + AgentCore Evaluations

**v3 Spec:** Section 13 (Evaluation Framework), Task 19

| v3 Eval Dimension | Strands Evals | AgentCore Evaluations |
|-------------------|---------------|----------------------|
| Task completion accuracy | Custom evaluator | `correctness`, `goal_success_rate` |
| Context preservation | Custom evaluator | `context_relevance` |
| Cost efficiency | Custom evaluator (token tracking) | Token usage dashboards |
| Safety compliance | Guardrails integration | `harmfulness`, `stereotyping` |
| Tool selection accuracy | Custom evaluator | `tool_selection_accuracy`, `tool_parameter_accuracy` |
| Response quality | Custom evaluator | `helpfulness`, `coherence`, `conciseness` |

**AgentCore built-in:** 13 prebuilt evaluations. Supports on-demand evaluation during development and continuous online evaluation in production. LLM-as-judge pattern.

---

## 4. Deployment Architecture on AgentCore

### 4.1 Deployment Steps

1. **Build** agent with Strands SDK locally. Test with Ollama or direct Bedrock.
2. **Wrap** with AgentCore SDK (3 lines of code: import, app, `@entrypoint`).
3. **Deploy** via AgentCore CLI or MCP server: `agentcore create` → `agentcore deploy`.
4. **Connect** AgentCore Gateway for tool access (Lambda, OpenAPI, MCP servers).
5. **Enable** AgentCore Memory for persistent context.
6. **Configure** AgentCore Identity for auth (Okta, Cognito, Entra).
7. **Set up** AgentCore Policy (Cedar rules) for tool-level governance.
8. **Enable** AgentCore Observability dashboards and Evaluations.

### 4.2 Production Architecture Diagram

```
                    ┌───────────────────────────────────────┐
                    │     Amazon Bedrock AgentCore           │
                    ├───────────────────────────────────────┤
  Users ──▶        │  Runtime (microVM per session)         │
  (HTTP/           │  ┌─────────────────────────────────┐   │
  WebSocket)       │  │ Strands Orchestrator Agent      │   │
                    │  │  ├─ Cost Governor               │   │
                    │  │  ├─ Context Manager             │   │
                    │  │  ├─ Agent Registry              │   │
                    │  │  └─ Hook Engine                 │   │
                    │  └────┬──────────┬─────────┬──────┘   │
                    │       │          │         │          │
                    │  ┌────┴────┐ ┌───┴───┐ ┌───┴──┐      │
                    │  │ Swarm   │ │ Graph │ │ Agent│      │
                    │  │ Teams   │ │ Loops │ │ Tools│      │
                    │  └─────────┘ └───────┘ └──────┘      │
                    ├───────────────────────────────────────┤
                    │  Gateway   │ Memory   │ Identity      │
                    │  (MCP      │ (short + │ (OAuth,       │
                    │  proxy)    │ long)    │ Okta/Entra)   │
                    ├───────────────────────────────────────┤
                    │  Observability │ Policy   │ Evaluations│
                    │  (OTel +       │ (Cedar)  │ (13 built- │
                    │  CloudWatch)   │          │  in + LLM) │
                    └───────────────────────────────────────┘
```

### 4.3 AgentCore Runtime Configuration

```
Protocols supported:
  HTTP      - REST API endpoints (request/response)
  MCP       - Model Context Protocol (for tool servers)
  A2A       - Agent-to-Agent protocol (multi-agent comm)
  WebSocket - Bidirectional streaming (real-time/voice)

Session features:
  MicroVM isolation per user session
  Long-running workloads up to 8 hours
  100MB payload support (multimodal)
  Immutable versioning with rollback
  Consumption-based pricing (no I/O wait charges)
```

### 4.4 What You Build vs. What AgentCore Provides

| You Build (Strands SDK) | AgentCore Provides (Managed) |
|------------------------|------------------------------|
| Orchestrator agent logic + system prompt | Serverless hosting with auto-scale |
| Custom agents (.md loader) | MicroVM session isolation |
| Cost governor (model selection logic) | Consumption-based pricing (no I/O wait) |
| Context semantic triage logic | Short-term + long-term memory service |
| Hook engine (PreToolUse, PostToolUse) | Policy (Cedar) for production governance |
| Agent registry + dispatcher | Identity (OAuth, Okta, Entra, Cognito) |
| Graph/Swarm topology definitions | Observability dashboards + tracing |
| Self-improving loop logic | Evaluations (13 built-in + custom) |
| Skill loading + frontmatter parsing | Gateway (API → MCP, marketplace tools) |
| OTel instrumentation in code | Browser tool + Code Interpreter |

---

## 5. Implementation Task Mapping

For each of the 25 implementation tasks from the v3 spec, here's the build vs. use-managed breakdown:

| # | Task | Strands Component | Effort | AgentCore Equivalent |
|---|------|-------------------|--------|---------------------|
| 1 | Project scaffold | `pip install strands-agents` | 2h | `agentcore create` |
| 2 | Context Manager | Custom `ConversationManager` | 8-10h | Memory (partial) |
| 3 | Hook lifecycle engine | `HookProvider` + callbacks | 6-8h | Policy (partial) |
| 4 | Cost Governor | Custom model selection | 6-8h | Not managed (build it) |
| 5 | Core Orchestrator | `Agent()` constructor | 4-6h | Runtime hosting |
| 6 | Hierarchical subagents | `Agent.as_tool()` | 4-6h | N/A (SDK pattern) |
| 7 | Agent Teams | `Swarm([agents])` | 6-8h | A2A protocol |
| 8 | Self-improving loops | `GraphBuilder` + conditional edges | 6-8h | N/A (SDK pattern) |
| 9 | Prompt cache manager | Bedrock API pass-through | 4-6h | Bedrock caching |
| 10 | Adaptive effort | Model selection + `budget_tokens` | 3-4h | Not managed (build it) |
| 11 | MCP Tool Search | Built-in `retrieve` tool | 2-3h | Gateway search tool |
| 12 | Skills registry | Custom loader + prompt injection | 4-6h | Not managed (build it) |
| 13 | Plugin registry | MCP server loading | 4-6h | Gateway + Marketplace |
| 14 | Session teleportation | `SessionManager` (S3 backend) | 4-6h | Memory + Runtime |
| 15 | Claude-as-MCP-Server | Strands agent exposed via MCP | 3-4h | Runtime MCP mode |
| 16 | Unified quota manager | Custom tracking | 5-6h | Not managed (build it) |
| 17 | Security + data class. | Bedrock Guardrails | 4-6h | Identity + Policy |
| 18 | Observability dashboard | OTel instrumentation | 4-6h | Observability (managed) |
| 19 | Evaluation suite | `strands-agents-eval` | 4-6h | Evaluations (13 built-in) |
| 20 | Compound loops | Nested Graph/Swarm | 4-6h | N/A (SDK pattern) |
| 21 | Enterprise settings | Config file loading | 3-4h | Runtime versioning |
| 22 | Learning store vector | AgentCore Memory long-term | 4-6h | Memory (managed) |
| 23 | Integration tests | pytest + Strands test utils | 6-8h | Evaluations (continuous) |
| 24 | Documentation | Markdown + docstrings | 4-6h | N/A |
| 25 | Agent registry | Custom `.md` loader | 10-14h | Not managed (build it) |

**Total custom build effort:** ~100-140 hours (vs. 148-193h in pure TypeScript). AgentCore eliminates ~30-40 hours of infrastructure work (hosting, observability dashboards, auth, session isolation).

---

## 6. Key Architecture Decisions

### 6.1 Python (Strands) vs. TypeScript (Claude Code native)

The v3 spec targets TypeScript because Claude Code's ecosystem is TS-first. Strands SDK is Python-first (TypeScript SDK exists but is less mature). **Decision:** Use Python for the Strands implementation. **Rationale:** Strands Python is production-proven at AWS scale (Amazon Q, Glue, Kiro). The TS SDK (`strands-agents/sdk-typescript`) is available for teams that need it, but the multi-agent patterns (Graph, Swarm, Workflow) are more mature in Python.

### 6.2 AgentCore Memory vs. Custom Context Manager

AgentCore Memory provides short-term (session) and long-term (semantic, preference, summary) storage out of the box. However, our v3 semantic triage system (PRESERVE_VERBATIM vs. COMPRESS_AGGRESSIVE classification) is not built into AgentCore. **Decision:** Use AgentCore Memory for persistence, but implement semantic triage as a custom `ConversationManager` that classifies content before storing. This gives us managed persistence with custom intelligence.

### 6.3 AgentCore Policy vs. Custom Hook Engine

Both systems intercept tool calls. AgentCore Policy uses Cedar language and operates at the Gateway level (outside the agent's reasoning loop). Our v3 hook engine operates inside the agent loop with more granular events (12 events including PreCompact, SubagentStop). **Decision:** Use both. AgentCore Policy for production-grade governance (Cedar rules for destructive operations). Custom hook engine for agent-level lifecycle events that Policy can't see (PreCompact, SubagentStop, prompt-type handlers).

### 6.4 Swarm vs. Agent Teams

The v3 spec models Agent Teams after Claude Code's experimental feature (shared task list, mailboxes, file locks). Strands Swarm uses shared context and collaborative handoffs instead. **Decision:** Map Agent Teams to Strands Swarm for the collaboration model, but implement the file-lock protocol and task list as custom shared state via `invocation_state`. The coordination semantics are equivalent; the IPC mechanism differs (file-system vs. in-memory shared context).

### 6.5 Deployment Target

AgentCore Runtime is the primary target for production. For development and testing, use Strands locally with Ollama (free, fast iteration). For staging, deploy to Fargate or Lambda. The same Strands agent code runs unchanged across all environments — only the model provider and infrastructure wrapper change.

---

## 7. Quickstart: From v3 Spec to Running Agent

### 7.1 Local Development

```bash
# Setup
pip install strands-agents strands-agents-tools
pip install bedrock-agentcore  # For AgentCore integration
```

```python
# Minimal orchestrator
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands_tools import calculator, file_read, file_write

@tool
def review_code(task: str) -> str:
    '''Reviews code for quality and security.'''
    reviewer = Agent(
        model=BedrockModel(model_id='us.anthropic.claude-sonnet-4-5-v1:0'),
        system_prompt='You are a code reviewer...',
        tools=[file_read]
    )
    return str(reviewer(task))

orchestrator = Agent(
    model=BedrockModel(model_id='us.anthropic.claude-opus-4-6-v1:0'),
    system_prompt='You are the lead developer...',
    tools=[review_code, file_write, calculator]
)

result = orchestrator('Review the auth module and fix any security issues')
```

### 7.2 Deploy to AgentCore

```python
# Add 3 lines to deploy:
from bedrock_agentcore.runtime import BedrockAgentCoreApp
app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload, session):
    return str(orchestrator(payload['input']))
```

```bash
# Deploy via CLI:
agentcore deploy --name claude-code-v3 --framework strands
```

### 7.3 Add Memory, Gateway, Policy

```python
# Memory (persistent context)
from bedrock_agentcore.memory import MemoryClient
memory = MemoryClient()
```

```bash
# Gateway (connect external tools as MCP)
aws bedrock-agentcore create-gateway --name tool-gateway
aws bedrock-agentcore create-target --gateway-id gw-xxx \
  --type LAMBDA --function-arn arn:aws:lambda:...
```

```cedar
# Policy (Cedar rules for governance)
permit(principal, action == AgentCore::Action::"read_file",
  resource) when { context.input.path like "/src/*" };
```
