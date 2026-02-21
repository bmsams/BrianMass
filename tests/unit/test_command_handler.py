"""Unit tests for the command hook handler.

Tests cover:
- Successful execution with exit code 0 (allow)
- Deny via exit code 2
- JSON stdout parsing into HookResult
- Timeout handling
- Process error handling
- Blocking vs non-blocking event behavior
- Environment variable injection
- Async execution
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.hooks.handlers.command import (
    CommandHandler,
    _build_env,
    _parse_stdout,
    _result_from_output,
    _serialize_context,
    _timeout_seconds,
)
from src.types.core import (
    HookContext,
    HookEvent,
    HookHandler,
    HookHandlerType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def handler():
    return CommandHandler()


@pytest.fixture
def allow_hook():
    return HookHandler(
        type=HookHandlerType.COMMAND,
        command="echo ok",
        timeout=600000,
    )


@pytest.fixture
def context():
    return HookContext(
        session_id="test-session-123",
        hook_event_name=HookEvent.PRE_TOOL_USE,
        cwd="/tmp/project",
        session_type="interactive",
        tool_name="bash",
        tool_input={"command": "ls"},
        model="claude-sonnet-4-5-20250929",
    )


def _non_blocking_context():
    """Return a context for a non-blocking event (SessionStart)."""
    return HookContext(
        session_id="sess-1",
        hook_event_name=HookEvent.SESSION_START,
        cwd="/tmp",
        session_type="interactive",
    )


# ---------------------------------------------------------------------------
# _timeout_seconds
# ---------------------------------------------------------------------------

class TestTimeoutSeconds:
    def test_default_timeout(self, allow_hook):
        assert _timeout_seconds(allow_hook) == 600.0

    def test_custom_timeout(self):
        h = HookHandler(type=HookHandlerType.COMMAND, command="x", timeout=30000)
        assert _timeout_seconds(h) == 30.0

    def test_zero_timeout(self):
        h = HookHandler(type=HookHandlerType.COMMAND, command="x", timeout=0)
        assert _timeout_seconds(h) == 0.0


# ---------------------------------------------------------------------------
# _build_env
# ---------------------------------------------------------------------------

class TestBuildEnv:
    def test_session_id_set(self, context):
        env = _build_env(context)
        assert env["BRAINMASS_SESSION_ID"] == "test-session-123"

    def test_hook_event_set(self, context):
        env = _build_env(context)
        assert env["BRAINMASS_HOOK_EVENT"] == "PreToolUse"

    def test_model_set(self, context):
        env = _build_env(context)
        assert env["BRAINMASS_MODEL"] == "claude-sonnet-4-5-20250929"

    def test_model_not_set_when_none(self):
        ctx = HookContext(
            session_id="s1",
            hook_event_name=HookEvent.SESSION_START,
            cwd="/tmp",
            session_type="interactive",
            model=None,
        )
        env = _build_env(ctx)
        assert "BRAINMASS_MODEL" not in env

    def test_cwd_set(self, context):
        env = _build_env(context)
        assert env["BRAINMASS_CWD"] == "/tmp/project"


# ---------------------------------------------------------------------------
# _serialize_context
# ---------------------------------------------------------------------------

class TestSerializeContext:
    def test_produces_valid_json(self, context):
        result = _serialize_context(context)
        data = json.loads(result)
        assert data["session_id"] == "test-session-123"
        assert data["hook_event_name"] == "PreToolUse"
        assert data["cwd"] == "/tmp/project"

    def test_tool_input_included(self, context):
        data = json.loads(_serialize_context(context))
        assert data["tool_input"] == {"command": "ls"}

    def test_none_fields_serialized(self):
        ctx = HookContext(
            session_id="s1",
            hook_event_name=HookEvent.STOP,
            cwd="/tmp",
            session_type="headless",
        )
        data = json.loads(_serialize_context(ctx))
        assert data["tool_name"] is None


# ---------------------------------------------------------------------------
# _parse_stdout
# ---------------------------------------------------------------------------

class TestParseStdout:
    def test_valid_json(self):
        assert _parse_stdout('{"key": "value"}') == {"key": "value"}

    def test_empty_string(self):
        assert _parse_stdout("") is None

    def test_whitespace_only(self):
        assert _parse_stdout("   \n  ") is None

    def test_invalid_json(self):
        assert _parse_stdout("not json") is None

    def test_json_with_whitespace(self):
        assert _parse_stdout('  {"a": 1}  ') == {"a": 1}


# ---------------------------------------------------------------------------
# _result_from_output
# ---------------------------------------------------------------------------

class TestResultFromOutput:
    def test_exit_0_no_output(self):
        r = _result_from_output(0, "", HookEvent.PRE_TOOL_USE)
        assert r.permission_decision == "allow"
        assert r.decision == "continue"

    def test_exit_0_with_json(self):
        stdout = json.dumps({
            "permissionDecision": "allow",
            "updatedInput": {"cmd": "safe-ls"},
            "additionalContext": "modified for safety",
        })
        r = _result_from_output(0, stdout, HookEvent.PRE_TOOL_USE)
        assert r.permission_decision == "allow"
        assert r.updated_input == {"cmd": "safe-ls"}
        assert r.additional_context == "modified for safety"

    def test_exit_2_deny(self):
        r = _result_from_output(2, "", HookEvent.PRE_TOOL_USE)
        assert r.permission_decision == "deny"
        assert r.decision == "block"

    def test_exit_2_with_reason_json(self):
        stdout = json.dumps({"reason": "dangerous command"})
        r = _result_from_output(2, stdout, HookEvent.PRE_TOOL_USE)
        assert r.permission_decision == "deny"
        assert r.reason == "dangerous command"

    def test_exit_2_with_plain_text_reason(self):
        r = _result_from_output(2, "not allowed", HookEvent.PRE_TOOL_USE)
        assert r.permission_decision == "deny"
        assert r.reason == "not allowed"

    def test_unexpected_exit_blocking_event(self):
        r = _result_from_output(1, "error", HookEvent.PRE_TOOL_USE)
        assert r.permission_decision == "deny"

    def test_unexpected_exit_non_blocking_event(self):
        r = _result_from_output(1, "error", HookEvent.SESSION_START)
        assert r.permission_decision == "allow"


# ---------------------------------------------------------------------------
# CommandHandler.execute (synchronous)
# ---------------------------------------------------------------------------

class TestCommandHandlerExecute:
    def test_exit_0_allow(self, handler, allow_hook, context):
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="echo ok", returncode=0, stdout="", stderr=""
            )
            result = handler.execute(allow_hook, context)
        assert result.permission_decision == "allow"

    def test_exit_2_deny(self, handler, context):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="check.sh")
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="check.sh", returncode=2,
                stdout='{"reason": "blocked"}', stderr=""
            )
            result = handler.execute(hook, context)
        assert result.permission_decision == "deny"
        assert result.reason == "blocked"

    def test_json_result_with_updated_input(self, handler, context):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="modify.sh")
        stdout = json.dumps({
            "permissionDecision": "allow",
            "updatedInput": {"command": "safe-cmd"},
        })
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="modify.sh", returncode=0, stdout=stdout, stderr=""
            )
            result = handler.execute(hook, context)
        assert result.permission_decision == "allow"
        assert result.updated_input == {"command": "safe-cmd"}

    def test_timeout_blocking_event_denies(self, handler, context):
        hook = HookHandler(
            type=HookHandlerType.COMMAND, command="slow.sh", timeout=1000
        )
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="slow.sh", timeout=1)
            result = handler.execute(hook, context)
        assert result.permission_decision == "deny"
        assert "timed out" in result.reason.lower()

    def test_timeout_non_blocking_event_allows(self, handler):
        hook = HookHandler(
            type=HookHandlerType.COMMAND, command="slow.sh", timeout=1000
        )
        ctx = _non_blocking_context()
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="slow.sh", timeout=1)
            result = handler.execute(hook, ctx)
        assert result.permission_decision == "allow"

    def test_os_error_blocking_event_denies(self, handler, context):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="missing.sh")
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("No such file")
            result = handler.execute(hook, context)
        assert result.permission_decision == "deny"

    def test_os_error_non_blocking_event_allows(self, handler):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="missing.sh")
        ctx = _non_blocking_context()
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("No such file")
            result = handler.execute(hook, ctx)
        assert result.permission_decision == "allow"

    def test_no_command_returns_allow(self, handler, context):
        hook = HookHandler(type=HookHandlerType.COMMAND, command=None)
        result = handler.execute(hook, context)
        assert result.permission_decision == "allow"

    def test_wrong_handler_type_raises(self, handler, context):
        hook = HookHandler(type=HookHandlerType.PROMPT, prompt="test")
        with pytest.raises(ValueError, match="non-command"):
            handler.execute(hook, context)

    def test_stdin_receives_context_json(self, handler, allow_hook, context):
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="echo ok", returncode=0, stdout="", stderr=""
            )
            handler.execute(allow_hook, context)
        call_kwargs = mock_run.call_args
        stdin_data = call_kwargs.kwargs.get("input") or call_kwargs[1].get("input")
        parsed = json.loads(stdin_data)
        assert parsed["session_id"] == "test-session-123"
        assert parsed["hook_event_name"] == "PreToolUse"

    def test_env_vars_passed_to_subprocess(self, handler, allow_hook, context):
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="echo ok", returncode=0, stdout="", stderr=""
            )
            handler.execute(allow_hook, context)
        call_kwargs = mock_run.call_args
        env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
        assert env["BRAINMASS_SESSION_ID"] == "test-session-123"
        assert env["BRAINMASS_MODEL"] == "claude-sonnet-4-5-20250929"

    def test_cwd_passed_to_subprocess(self, handler, allow_hook, context):
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="echo ok", returncode=0, stdout="", stderr=""
            )
            handler.execute(allow_hook, context)
        call_kwargs = mock_run.call_args
        cwd = call_kwargs.kwargs.get("cwd") or call_kwargs[1].get("cwd")
        assert cwd == "/tmp/project"

    def test_stop_event_blocking(self, handler):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="check.sh")
        ctx = HookContext(
            session_id="s1",
            hook_event_name=HookEvent.STOP,
            cwd="/tmp",
            session_type="interactive",
        )
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="check.sh", returncode=0,
                stdout=json.dumps({"decision": "block", "reason": "not done"}),
                stderr=""
            )
            result = handler.execute(hook, ctx)
        assert result.decision == "block"
        assert result.reason == "not done"

    def test_permission_request_event(self, handler):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="perm.sh")
        ctx = HookContext(
            session_id="s1",
            hook_event_name=HookEvent.PERMISSION_REQUEST,
            cwd="/tmp",
            session_type="interactive",
            tool_name="write_file",
        )
        with patch("src.hooks.handlers.command.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args="perm.sh", returncode=2,
                stdout='{"reason": "no write access"}', stderr=""
            )
            result = handler.execute(hook, ctx)
        assert result.permission_decision == "deny"
        assert result.reason == "no write access"


# ---------------------------------------------------------------------------
# CommandHandler.execute_async
# ---------------------------------------------------------------------------

class TestCommandHandlerExecuteAsync:
    def test_async_exit_0_allow(self, handler, context):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="echo ok")

        async def _mock_create_subprocess(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            return proc

        with patch(
            "src.hooks.handlers.command.asyncio.create_subprocess_shell",
            side_effect=_mock_create_subprocess,
        ):
            result = asyncio.get_event_loop().run_until_complete(
                handler.execute_async(hook, context)
            )
        assert result.permission_decision == "allow"

    def test_async_exit_2_deny(self, handler, context):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="deny.sh")

        async def _mock_create_subprocess(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(
                return_value=(b'{"reason": "denied async"}', b"")
            )
            proc.returncode = 2
            return proc

        with patch(
            "src.hooks.handlers.command.asyncio.create_subprocess_shell",
            side_effect=_mock_create_subprocess,
        ):
            result = asyncio.get_event_loop().run_until_complete(
                handler.execute_async(hook, context)
            )
        assert result.permission_decision == "deny"
        assert result.reason == "denied async"

    def test_async_timeout_blocking(self, handler, context):
        hook = HookHandler(
            type=HookHandlerType.COMMAND, command="slow.sh", timeout=100
        )

        async def _mock_create_subprocess(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(side_effect=TimeoutError())
            proc.kill = MagicMock()
            return proc

        with patch(
            "src.hooks.handlers.command.asyncio.create_subprocess_shell",
            side_effect=_mock_create_subprocess,
        ):
            result = asyncio.get_event_loop().run_until_complete(
                handler.execute_async(hook, context)
            )
        assert result.permission_decision == "deny"
        assert "timed out" in result.reason.lower()

    def test_async_timeout_non_blocking(self, handler):
        hook = HookHandler(
            type=HookHandlerType.COMMAND, command="slow.sh", timeout=100
        )
        ctx = _non_blocking_context()

        async def _mock_create_subprocess(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(side_effect=TimeoutError())
            proc.kill = MagicMock()
            return proc

        with patch(
            "src.hooks.handlers.command.asyncio.create_subprocess_shell",
            side_effect=_mock_create_subprocess,
        ):
            result = asyncio.get_event_loop().run_until_complete(
                handler.execute_async(hook, ctx)
            )
        assert result.permission_decision == "allow"

    def test_async_wrong_type_raises(self, handler, context):
        hook = HookHandler(type=HookHandlerType.AGENT, agent_config={})
        with pytest.raises(ValueError, match="non-command"):
            asyncio.get_event_loop().run_until_complete(
                handler.execute_async(hook, context)
            )

    def test_async_no_command_returns_allow(self, handler, context):
        hook = HookHandler(type=HookHandlerType.COMMAND, command=None)
        result = asyncio.get_event_loop().run_until_complete(
            handler.execute_async(hook, context)
        )
        assert result.permission_decision == "allow"

    def test_async_os_error_non_blocking(self, handler):
        hook = HookHandler(type=HookHandlerType.COMMAND, command="missing.sh")
        ctx = _non_blocking_context()

        with patch(
            "src.hooks.handlers.command.asyncio.create_subprocess_shell",
            side_effect=OSError("not found"),
        ):
            result = asyncio.get_event_loop().run_until_complete(
                handler.execute_async(hook, ctx)
            )
        assert result.permission_decision == "allow"
