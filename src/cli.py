"""Brainmass command-line interface.

Provides the ``brainmass`` console command (also runnable as
``python -m src.cli``) for invoking the orchestrator from a terminal.

Two execution modes:

1. **Local mode** (default) — runs the full request lifecycle (hooks,
   effort selection, model-tier routing, cost governance) with a local
   stub model callback. No AWS credentials required.
2. **Production mode** (``--production``) — executes through a Strands
   Agent backed by Amazon Bedrock. Requires AWS credentials with Bedrock
   model access for Anthropic Claude.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from importlib.metadata import PackageNotFoundError, version

from src.context.context_manager import ContextManager
from src.cost.cost_governor import CostGovernor
from src.hooks.hook_engine import BrainmassHookEngine
from src.orchestrator.orchestrator import Orchestrator, OrchestratorResult

logger = logging.getLogger(__name__)


def _package_version() -> str:
    try:
        return version("brainmass-v3")
    except PackageNotFoundError:
        return "unknown"


def build_orchestrator(
    session_id: str,
    cwd: str = ".",
    window_size: int = 200_000,
    production: bool = False,
) -> Orchestrator:
    """Construct an Orchestrator wired for CLI use."""
    return Orchestrator(
        context_manager=ContextManager(session_id=session_id, window_size=window_size),
        hook_engine=BrainmassHookEngine(),
        cost_governor=CostGovernor(),
        session_id=session_id,
        session_type="headless",
        cwd=cwd,
        use_production_agent=production or None,
    )


def _result_to_dict(result: OrchestratorResult) -> dict:
    return {
        "request_id": result.request_id,
        "response": result.response,
        "model_tier": result.model_tier.value,
        "effort_level": result.effort_level,
        "topology": result.topology.value if result.topology else None,
        "tasks": [t.description for t in result.tasks],
        "tool_calls": [
            {"tool": c.tool_name, "input": c.tool_input, "cost_usd": c.cost_usd}
            for c in result.tool_calls
        ],
        "usage": {
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "cost_usd": result.total_cost_usd,
        },
        "hooks_fired": result.hooks_fired,
    }


def _print_result(result: OrchestratorResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_result_to_dict(result), indent=2))
        return

    print(result.response)
    print(
        f"\n[tier={result.model_tier.value} effort={result.effort_level} "
        f"tokens={result.total_input_tokens}/{result.total_output_tokens} "
        f"cost=${result.total_cost_usd:.6f}]",
        file=sys.stderr,
    )


def _run_once(orchestrator: Orchestrator, request: str, as_json: bool) -> int:
    try:
        result = orchestrator.process_request(request)
    except Exception as exc:
        logger.exception("Request failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_result(result, as_json)
    return 0


def _run_repl(orchestrator: Orchestrator, as_json: bool) -> int:
    print(
        "Brainmass interactive session — type a request, or 'exit' to quit.",
        file=sys.stderr,
    )
    while True:
        try:
            line = input("brainmass> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            return 0
        _run_once(orchestrator, line, as_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brainmass",
        description=(
            "Brainmass — enterprise agentic coding platform. "
            "Runs locally by default; pass --production for Bedrock execution."
        ),
    )
    parser.add_argument(
        "request",
        nargs="?",
        help="The request to process. Omit to start an interactive session.",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Execute through Strands + Amazon Bedrock (requires AWS credentials).",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session identifier (auto-generated when omitted).",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        help="Working directory for the orchestrator (default: current directory).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200_000,
        help="Context window size in tokens (default: 200000).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full structured result as JSON.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_package_version()}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    session_id = args.session_id or f"cli-{uuid.uuid4().hex[:8]}"
    try:
        orchestrator = build_orchestrator(
            session_id=session_id,
            cwd=args.cwd,
            window_size=args.window_size,
            production=args.production,
        )
    except Exception as exc:
        logger.exception("Failed to build orchestrator")
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.request:
        return _run_once(orchestrator, args.request, args.json)
    return _run_repl(orchestrator, args.json)


if __name__ == "__main__":
    sys.exit(main())
