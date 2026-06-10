# Brainmass

An enterprise-grade agentic coding platform built on the AWS Strands Agents
SDK and Amazon Bedrock AgentCore. Brainmass runs a full request lifecycle —
lifecycle hooks, effort selection, model-tier routing, cost governance,
multi-agent teams, and evaluation suites — around Claude models on Bedrock.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/bmsams/BrianMass.git
cd BrianMass
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Quickstart (no AWS required)

Local mode runs the entire orchestration pipeline — hooks, effort
selection, model-tier routing, cost tracking — with a stub model callback,
so you can exercise the platform without credentials:

```bash
# One-shot request
brainmass "Review src/agents/file_lock.py for race conditions"

# Structured output
brainmass --json "fix the failing tests"

# Interactive session
brainmass
```

The footer line (on stderr) shows the routing decisions:

```
[tier=sonnet effort=standard tokens=108/11 cost=$0.000489]
```

## Production mode (Amazon Bedrock)

With AWS credentials configured and Bedrock model access enabled for
Anthropic Claude, the same CLI executes through a real Strands Agent:

```bash
brainmass --production "Summarize the architecture of this repository"
```

Model IDs default to `us.anthropic.claude-*` inference profiles and can be
overridden with `MODEL_ID_HAIKU`, `MODEL_ID_SONNET`, and `MODEL_ID_OPUS`.

### AgentCore deployment

`src/runtime/app.py` is the Bedrock AgentCore entrypoint for hosted
deployment. Strict mode requires these environment variables:

| Variable | Purpose |
|---|---|
| `AWS_REGION` | Deployment region |
| `AGENTCORE_APP_NAME` | AgentCore application name |
| `BRAINMASS_SESSION_BUCKET` | S3 bucket for session state |
| `MODEL_ID_HAIKU` / `MODEL_ID_SONNET` / `MODEL_ID_OPUS` | Canonical Bedrock model IDs |

## Python API

```python
from src.cli import build_orchestrator

orch = build_orchestrator(session_id="demo")          # local stub mode
# orch = build_orchestrator(session_id="demo", production=True)  # Bedrock

result = orch.process_request("Review the auth module")
print(result.response)
print(result.model_tier, result.effort_level, result.total_cost_usd)
```

## Architecture

| Module | Responsibility |
|---|---|
| `src/orchestrator/` | Central control plane: request lifecycle, task decomposition, topology and effort selection |
| `src/agents/` | Agent teams: dispatcher (13-step lifecycle), loaders, mailbox IPC, file locks, learning store, compound loops |
| `src/hooks/` | 12-event lifecycle hook engine with scope precedence and command/prompt/agent handlers |
| `src/context/` | Context window management, compaction, triage |
| `src/cost/` | Cost governor (per-tier pricing, budgets) and quota manager |
| `src/security/` | Data classifier and enterprise security controls |
| `src/config/` | Configuration loading and enterprise policy management |
| `src/workflow/` | SDLC workflow engine (vibe / design / full-SDLC modes) |
| `src/session/` | Session teleportation across surfaces |
| `src/cache/`, `src/mcp/`, `src/plugins/`, `src/skills/` | Prompt caching, MCP server mode, plugin and skill registries |
| `src/observability/` | OTel-style tracing and instrumentation |
| `src/evals/`, `evals/` | Evaluation suites, including adversarial "hard" evals |

## Evals

```bash
# Adversarial AI-resistance evals through the orchestrator (local mode)
python scripts/run_hard_evals.py --write-sample   # bootstrap a sample input
python scripts/run_hard_evals_live.py             # live variant execution
```

## Development

```bash
.venv/bin/python -m pytest -q        # full suite (unit + property + integration)
.venv/bin/python -m ruff check src/  # lint
```

Tests run entirely offline — no AWS credentials are needed.
