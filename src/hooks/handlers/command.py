"""Command handler for hook execution.

Spawns a child process, pipes HookContext as JSON to stdin, reads stdout
as JSON HookResult, and interprets exit codes (0=allow, 2=deny).

Requirements: 3.2, 3.7
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import asdict

from src.types.core import (
    HookContext,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
)

logger = logging.getLogger(__name__)

# Events where a handler failure should deny (fail-closed)
_BLOCKING_EVENTS = frozenset({
    HookEvent.PRE_TOOL_USE,
    HookEvent.STOP,
    HookEvent.PERMISSION_REQUEST,
})

# Default timeout: 10 minutes in seconds
_DEFAULT_TIMEOUT_SECS = 600


def _timeout_seconds(handler: HookHandler) -> float:
    """Return the timeout in seconds from the handler config.

    ``handler.timeout`` is stored in milliseconds (default 600_000).
    """
    return handler.timeout / 1000.0


def _build_env(context: HookContext) -> dict[str, str]:
    """Build the environment variables dict for the child process.

    Merges the current process environment with Brainmass-specific vars.
    """
    env = os.environ.copy()
    env["BRAINMASS_SESSION_ID"] = context.session_id
    env["BRAINMASS_HOOK_EVENT"] = (
        context.hook_event_name.value
        if isinstance(context.hook_event_name, HookEvent)
        else str(context.hook_event_name)
    )
    if context.model:
        env["BRAINMASS_MODEL"] = context.model
    if context.cwd:
        env["BRAINMASS_CWD"] = context.cwd
    # Tool input file path — write tool_input to a temp location if present
    if context.tool_input is not None:
        env["BRAINMASS_TOOL_INPUT_FILE_PATH"] = ""  # placeholder; caller may set
    return env


def _serialize_context(context: HookContext) -> str:
    """Serialize a HookContext to JSON for piping to stdin."""
    data = asdict(context)
    # Convert HookEvent enum to its string value
    if isinstance(data.get("hook_event_name"), HookEvent):
        data["hook_event_name"] = data["hook_event_name"].value
    elif hasattr(data.get("hook_event_name"), "value"):
        data["hook_event_name"] = data["hook_event_name"].value
    return json.dumps(data, default=str)


def _parse_stdout(stdout: str) -> dict | None:
    """Try to parse stdout as JSON. Return None on failure."""
    stdout = stdout.strip()
    if not stdout:
        return None
    try:
        return json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None


def _result_from_output(
    exit_code: int,
    stdout: str,
    event: HookEvent,
) -> HookResult:
    """Build a HookResult from the child process exit code and stdout.

    Exit code semantics:
      0 → allow / continue
      2 → deny (with optional reason from stdout)
    Any other exit code is treated as an error (fail-open for non-blocking
    events, fail-closed for blocking events).
    """
    parsed = _parse_stdout(stdout)

    if exit_code == 0:
        # Allow — optionally with updatedInput / additionalContext
        if parsed:
            return HookResult(
                permission_decision=parsed.get("permissionDecision", "allow"),
                permission_decision_reason=parsed.get("permissionDecisionReason"),
                updated_input=parsed.get("updatedInput"),
                additional_context=parsed.get("additionalContext"),
                decision=parsed.get("decision", "continue"),
                reason=parsed.get("reason"),
            )
        return HookResult(permission_decision="allow", decision="continue")

    if exit_code == 2:
        # Deny
        reason = None
        if parsed:
            reason = parsed.get("reason") or parsed.get("permissionDecisionReason")
        elif stdout.strip():
            reason = stdout.strip()
        return HookResult(
            permission_decision="deny",
            permission_decision_reason=reason,
            decision="block",
            reason=reason,
        )

    # Unexpected exit code — treat as error
    logger.warning(
        "Command handler exited with unexpected code %d: %s",
        exit_code,
        stdout[:200] if stdout else "(no output)",
    )
    if event in _BLOCKING_EVENTS:
        return HookResult(
            permission_decision="deny",
            permission_decision_reason=f"Command exited with code {exit_code}",
            decision="block",
            reason=f"Command exited with code {exit_code}",
        )
    return HookResult(permission_decision="allow", decision="continue")


class CommandHandler:
    """Executes command-type hook handlers as child processes.

    The handler's ``command`` field is run as a shell command. The
    ``HookContext`` is serialized to JSON and piped to the process's
    stdin. The process's stdout is read as JSON and parsed into a
    ``HookResult``. Exit code 0 means allow, exit code 2 means deny.

    A configurable timeout (default 10 minutes) kills the process if
    it runs too long.
    """

    def execute(
        self,
        handler: HookHandler,
        context: HookContext,
    ) -> HookResult:
        """Run the command handler synchronously.

        Returns a ``HookResult`` based on exit code and stdout.
        For non-blocking events, errors are treated as allow (fail-open).
        For blocking events (PreToolUse, Stop, PermissionRequest), errors
        are treated as deny (fail-closed).
        """
        if handler.type != HookHandlerType.COMMAND:
            raise ValueError(
                f"CommandHandler received non-command handler type: {handler.type}"
            )
        if not handler.command:
            logger.warning("Command handler has no command configured")
            return HookResult(permission_decision="allow", decision="continue")

        event = context.hook_event_name
        timeout = _timeout_seconds(handler)
        env = _build_env(context)
        stdin_data = _serialize_context(context)

        try:
            result = subprocess.run(
                handler.command,
                shell=True,
                input=stdin_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=context.cwd or None,
            )
            return _result_from_output(result.returncode, result.stdout, event)

        except subprocess.TimeoutExpired:
            logger.error(
                "Command handler timed out after %.0f seconds: %s",
                timeout,
                handler.command,
            )
            if event in _BLOCKING_EVENTS:
                return HookResult(
                    permission_decision="deny",
                    permission_decision_reason="Command timed out",
                    decision="block",
                    reason="Command timed out",
                )
            return HookResult(permission_decision="allow", decision="continue")

        except OSError as exc:
            logger.error(
                "Command handler failed to start: %s — %s",
                handler.command,
                exc,
            )
            if event in _BLOCKING_EVENTS:
                return HookResult(
                    permission_decision="deny",
                    permission_decision_reason=f"Command failed: {exc}",
                    decision="block",
                    reason=f"Command failed: {exc}",
                )
            return HookResult(permission_decision="allow", decision="continue")

    async def execute_async(
        self,
        handler: HookHandler,
        context: HookContext,
    ) -> HookResult:
        """Run the command handler asynchronously (fire-and-forget support).

        Uses ``asyncio.create_subprocess_shell`` for non-blocking execution.
        """
        if handler.type != HookHandlerType.COMMAND:
            raise ValueError(
                f"CommandHandler received non-command handler type: {handler.type}"
            )
        if not handler.command:
            logger.warning("Command handler has no command configured")
            return HookResult(permission_decision="allow", decision="continue")

        event = context.hook_event_name
        timeout = _timeout_seconds(handler)
        env = _build_env(context)
        stdin_data = _serialize_context(context)

        try:
            proc = await asyncio.create_subprocess_shell(
                handler.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=context.cwd or None,
            )
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(input=stdin_data.encode()),
                timeout=timeout,
            )
            stdout = stdout_bytes.decode() if stdout_bytes else ""
            return _result_from_output(proc.returncode or 0, stdout, event)

        except TimeoutError:
            logger.error(
                "Async command handler timed out after %.0f seconds: %s",
                timeout,
                handler.command,
            )
            # Try to kill the process
            try:
                proc.kill()  # type: ignore[possibly-undefined]
            except (ProcessLookupError, OSError):
                pass
            if event in _BLOCKING_EVENTS:
                return HookResult(
                    permission_decision="deny",
                    permission_decision_reason="Command timed out",
                    decision="block",
                    reason="Command timed out",
                )
            return HookResult(permission_decision="allow", decision="continue")

        except OSError as exc:
            logger.error(
                "Async command handler failed to start: %s — %s",
                handler.command,
                exc,
            )
            if event in _BLOCKING_EVENTS:
                return HookResult(
                    permission_decision="deny",
                    permission_decision_reason=f"Command failed: {exc}",
                    decision="block",
                    reason=f"Command failed: {exc}",
                )
            return HookResult(permission_decision="allow", decision="continue")
