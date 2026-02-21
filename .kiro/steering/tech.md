# Tech Stack

## Language & Runtime
- Python 3.11+ (required)
- All type annotations use `from __future__ import annotations`

## Core Dependencies
- `strands-agents` — Strands Agents SDK (Bedrock/Anthropic model integration, Agents-as-Tools, hooks)
- `strands-agents-tools` — built-in tool set for Strands agents
- `bedrock-agentcore` — AWS Bedrock AgentCore runtime deployment
- `python-frontmatter` — YAML frontmatter parsing for agent/skill `.md` files

## Dev Dependencies
- `pytest` + `pytest-asyncio` — test runner (`asyncio_mode = "auto"`)
- `hypothesis` — property-based testing (PBT)
- `pytest-cov` — coverage
- `ruff` — linting and formatting (`line-length = 100`, target `py311`)
- `mypy` — strict type checking

## Build System
- `hatchling` (via `pyproject.toml`)
- Package: `brainmass-v3`, source root: `src/`

## Common Commands

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run all tests (single pass, no watch)
pytest

# Run only unit tests
pytest tests/unit

# Run only property-based tests
pytest -m property

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Lint
ruff check src tests

# Type check
mypy src
```

## Model Tiers & Pricing (per million tokens)
| Tier   | Input  | Output  | Cached Input |
|--------|--------|---------|--------------|
| Opus   | $5.00  | $25.00  | $0.50        |
| Sonnet | $3.00  | $15.00  | $0.30        |
| Haiku  | $0.80  | $4.00   | $0.08        |

## External SDK Integration Notes
- Strands SDK calls are **stubbed** behind pluggable callbacks (`model_callback`, `tool_executor`, `agent_callback`) so the system runs and tests without the SDK installed
- Production integration points are marked with `# --- Production integration point ---` comments
- AgentCore deployment is similarly stubbed
