"""AgentCore runtime entrypoint for Brainmass v3.

This module is strict production mode: it validates runtime env and wires
runtime-critical components through concrete execution paths.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.config.enterprise import EnterprisePolicyManager
from src.context.context_manager import ContextManager
from src.cost.cost_governor import CostGovernor
from src.hooks.hook_engine import BrainmassHookEngine
from src.observability.instrumentation import BrainmassTracer
from src.orchestrator.orchestrator import Orchestrator
from src.runtime.config import RuntimeConfig, load_runtime_config

logger = logging.getLogger(__name__)


def _build_orchestrator(config: RuntimeConfig) -> Orchestrator:
    """Build an orchestrator instance for runtime invocation."""
    tracer = BrainmassTracer(service_name=config.agentcore_app_name)
    context_manager = ContextManager(
        session_id="runtime-session",
        window_size=200_000,
    )
    hook_engine = BrainmassHookEngine()
    cost_governor = CostGovernor()

    # Wire enterprise policy into the hook engine (Req 23.1, 23.2)
    enterprise_policy = EnterprisePolicyManager()
    if enterprise_policy.settings.allow_managed_hooks_only:
        hook_engine.set_managed_only(True)
        logger.info("Enterprise managed-only hooks mode active.")

    return Orchestrator(
        context_manager=context_manager,
        hook_engine=hook_engine,
        cost_governor=cost_governor,
        session_type="headless",
        cwd=".",
        tracer=tracer,
        runtime_config=config,
        use_production_agent=True,
    )


_RUNTIME_CONFIG = load_runtime_config(strict=True)

try:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp  # type: ignore
except Exception as exc:  # pragma: no cover - exercised in production env
    raise RuntimeError(
        "bedrock-agentcore runtime package is required for AgentCore entrypoint."
    ) from exc


app = BedrockAgentCoreApp()


@app.entrypoint
async def invoke(payload: dict[str, Any], session: Any) -> dict[str, Any]:
    """AgentCore entrypoint.

    Request payload:
    {
      "input": str,
      "session_id": str | optional,
      "metadata": dict | optional
    }
    """
    request_text = payload.get("input")
    if not isinstance(request_text, str) or not request_text.strip():
        return {
            "status": "error",
            "error": "payload.input must be a non-empty string",
            "request_id": str(uuid.uuid4()),
        }

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        session_id = f"runtime-{uuid.uuid4()}"

    try:
        orchestrator = _build_orchestrator(_RUNTIME_CONFIG)
        orchestrator.session_id = session_id
        result = orchestrator.process_request(request_text)
        return {
            "status": "ok",
            "output": result.response,
            "request_id": result.request_id,
            "usage": {
                "input_tokens": result.total_input_tokens,
                "output_tokens": result.total_output_tokens,
                "cost_usd": result.total_cost_usd,
            },
            "trace_id": getattr(orchestrator, "trace_id", ""),
            "metadata": payload.get("metadata", {}),
        }
    except Exception as exc:
        logger.exception("Runtime invocation failed")
        return {
            "status": "error",
            "error": str(exc),
            "request_id": str(uuid.uuid4()),
        }
