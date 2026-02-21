# Project Structure

```
brainmass-v3/
├── src/
│   ├── agents/          # Multi-agent coordination
│   │   ├── context_file.py      # Self-improving loop context file I/O
│   │   ├── file_lock.py         # File-lock-based IPC for Agent Teams
│   │   ├── loop_runner.py       # Self-improving loop orchestration
│   │   ├── mailbox.py           # Agent Teams inbox/mailbox system
│   │   ├── safety_controls.py   # Loop safety (diff limits, stop file, error detection)
│   │   ├── subagent_manager.py  # Hierarchical subagent execution (Agents-as-Tools)
│   │   └── team_manager.py      # Agent Teams peer topology coordination
│   ├── cache/           # Prompt caching layer
│   ├── config/          # Config loading (.mcp.json, settings.json, plugin.json)
│   ├── context/         # Semantic context triage and compaction
│   │   ├── context_manager.py   # Context window management, session persistence, learning store
│   │   └── triage.py            # classify() — assigns ContextCategory to items
│   ├── cost/            # Token budget and cost governance
│   │   └── cost_governor.py     # Model tier routing, budget enforcement, usage tracking
│   ├── hooks/           # 12-event lifecycle hook engine
│   │   ├── hook_engine.py       # BrainmassHookEngine — fire(), register_scoped()
│   │   ├── pre_tool_use.py      # PreToolUse-specific logic
│   │   └── handlers/            # command.py, prompt.py, agent.py handler types
│   ├── orchestrator/    # Central control plane
│   │   ├── orchestrator.py      # 8-step request processing, topology selection
│   │   ├── effort_controller.py # Effort level selection (Quick/Standard/Deep)
│   │   └── tool_search.py       # MCP Tool Search (lazy loading)
│   ├── plugins/         # Plugin registry and marketplace
│   ├── skills/          # Skills registry (SKILL.md parsing, invocation)
│   ├── types/
│   │   └── core.py      # All enums, dataclasses, and constants (single source of truth)
│   ├── observability/   # OpenTelemetry integration stubs
│   ├── security/        # Guardrails and security controls
│   └── session/         # Session teleportation and serialization
├── tests/
│   ├── unit/            # Unit tests — one file per src module (test_<module>.py)
│   ├── property/        # Hypothesis property-based tests
│   └── integration/     # Integration tests
├── evals/               # Strands Evals SDK evaluation harness
├── pyproject.toml
└── architecture_context.md  # API schemas, SDK patterns, config formats reference
```

## Conventions

- **Types first**: All shared types live in `src/types/core.py`. Never define domain types elsewhere.
- **Pluggable callbacks**: External SDK calls (Strands, AgentCore) are injected as constructor callbacks so components are testable without SDK installation.
- **Requirement annotations**: Module docstrings reference spec requirements (e.g., `Requirements: 5.1, 5.2`).
- **Test co-location**: Unit tests mirror `src/` structure under `tests/unit/`. Property tests go in `tests/property/`.
- **Pytest markers**: `@pytest.mark.property` for Hypothesis tests, `@pytest.mark.integration` for integration tests, `@pytest.mark.slow` for slow tests.
- **Async**: `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio` decorator.
- **Line length**: 100 characters (ruff enforced).
- **Imports**: `from __future__ import annotations` at top of every module.
