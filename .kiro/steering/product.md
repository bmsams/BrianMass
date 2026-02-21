# Product

Brainmass v3 is an enterprise-grade agentic coding platform built on the AWS Strands Agents SDK and Amazon Bedrock AgentCore. It implements behavioral parity with Claude Code's confirmed architecture while adding enterprise governance, cost management, and multi-agent coordination.

## Core Capabilities

- **Multi-agent orchestration** across three topologies: hierarchical subagents, peer Agent Teams, and self-improving loops
- **Semantic context triage** — categorizes context items (PRESERVE_VERBATIM, PRESERVE_STRUCTURED, COMPRESS_AGGRESSIVE, EPHEMERAL) to survive compaction without data loss
- **Cost governance** — per-agent token budgets, automatic model tier routing (Opus → Sonnet → Haiku), and unified quota tracking
- **12-event lifecycle hook engine** — command, prompt, and agent handler types
- **Plugin and skills ecosystem** — shareable packages bundling agents, hooks, MCP servers, and slash commands
- **Session teleportation** — serialize and restore full session state across surfaces

## Key Design Principles

- Deterministic guardrails over prompt promises (hooks enforce behavior)
- Decoupled control/data planes
- MCP-first tool integration
- Cost-aware model routing
- Cross-session learning persistence
