---
inclusion: always
---

# Coding Standards

## Core Rules

- Production-grade code only — no stubs, placeholders, or TODO-driven implementations unless explicitly marked as a production integration point
- All external SDK calls (Strands, AgentCore) must be injected as pluggable callbacks so components remain testable without SDK installation
- Mark real integration points with `# --- Production integration point ---` comments and include commented-out production code showing the intended SDK usage
- All shared types (enums, dataclasses, constants) live exclusively in `src/types/core.py` — never define domain types in other modules

## Python Style

- `from __future__ import annotations` at the top of every module
- Full type annotations on all function signatures and class attributes
- `Optional[X]` for nullable fields (not `X | None` union syntax)
- `dataclass` for data-holding types; `Protocol` for structural interfaces
- Line length: 100 characters (ruff enforced)
- Module docstrings must reference spec requirements: `Requirements: 1.2, 3.4`

## Architecture Patterns

- Dependency injection via constructor parameters — never import and instantiate dependencies inside a class
- Pluggable callbacks follow the pattern: `callback: Optional[Callable[..., ReturnType]] = None`, defaulting to a stub that returns a safe no-op result
- Enums for all finite value sets; never use raw strings for states, tiers, or event names
- `asyncio_mode = "auto"` is active — no `@pytest.mark.asyncio` needed; async tests work natively

## Testing

- Unit tests mirror `src/` under `tests/unit/` with `test_<module>.py` naming
- Property-based tests go in `tests/property/` and use `@pytest.mark.property`
- Use `hypothesis` for property tests; write smart generators that constrain to valid input domains
- Do not use mocks to make tests pass — tests must validate real logic
- Both unit tests and property tests are required for new functionality; they complement each other
- Limit test fix attempts to 2 before surfacing the issue to the user

## Hook Engine

- All 12 `HookEvent` values must be handled; never add new events without updating `src/types/core.py`
- Scope precedence order is fixed: `enterprise_managed` → `plugin` → `subagent_frontmatter` → `skill_frontmatter` → `project_local` → `project_shared` → `user_wide`
- `deny` always beats `allow`; `block` always beats `continue` in result merging
- Async handlers are fire-and-forget — they must not affect the merged `HookResult`

## Cost & Model Routing

- Model tier selection is always delegated to `CostGovernor.select_model()` — never hardcode a tier
- Budget status thresholds: OK < 80%, WARNING ≥ 80%, CRITICAL ≥ 95%, EXCEEDED ≥ 100%
- Always call `cost_governor.record_usage()` after every model invocation

## Context Triage

- Every context item must be assigned a `ContextCategory` via `triage.classify()`
- `PRESERVE_VERBATIM`: error messages, file paths, decisions — never compress
- `PRESERVE_STRUCTURED`: structured data that can be reformatted but not lost
- `COMPRESS_AGGRESSIVE`: summaries, logs, verbose output
- `EPHEMERAL`: transient data safe to discard after the current turn
####CRITCAL_RULES###

USEe the strand power and agentcore power to helpbuild the app

DO NOT STUB OUT FILES