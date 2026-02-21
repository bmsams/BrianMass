"""Unit tests for PreToolUse processor.

Covers the three modes (allow, deny, modify) and edge cases around
result interpretation and hook engine integration.
"""

from __future__ import annotations

from src.hooks.hook_engine import BrainmassHookEngine
from src.hooks.pre_tool_use import (
    PreToolUseResult,
    _interpret_result,
    process_pre_tool_use,
)
from src.types.core import (
    HookDefinition,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_with_callback(callback):
    """Create a BrainmassHookEngine with a custom handler callback."""
    engine = BrainmassHookEngine()
    engine.set_handler_callback(callback)
    return engine


def _register_hook(engine, matcher=None, scope="project_shared"):
    """Register a minimal PreToolUse hook definition."""
    defn = HookDefinition(
        matcher=matcher,
        hooks=[HookHandler(type=HookHandlerType.COMMAND, command="echo ok")],
    )
    engine.register(HookEvent.PRE_TOOL_USE, defn, scope=scope)


# ===================================================================
# _interpret_result — unit tests for the pure decision logic
# ===================================================================


class TestInterpretResult:
    """Tests for _interpret_result (pure function, no engine needed)."""

    def test_empty_result_means_allow(self):
        result = _interpret_result(HookResult())
        assert result.allowed is True
        assert result.modified_input is None
        assert result.additional_context is None
        assert result.deny_reason is None

    def test_explicit_allow(self):
        result = _interpret_result(HookResult(permission_decision="allow"))
        assert result.allowed is True
        assert result.modified_input is None

    def test_deny_with_reason(self):
        result = _interpret_result(
            HookResult(
                permission_decision="deny",
                permission_decision_reason="Blocked by policy",
            )
        )
        assert result.allowed is False
        assert result.deny_reason == "Blocked by policy"

    def test_deny_falls_back_to_reason_field(self):
        result = _interpret_result(
            HookResult(permission_decision="deny", reason="fallback reason")
        )
        assert result.allowed is False
        assert result.deny_reason == "fallback reason"

    def test_deny_default_reason(self):
        result = _interpret_result(HookResult(permission_decision="deny"))
        assert result.allowed is False
        assert result.deny_reason == "Denied by PreToolUse hook"

    def test_modify_input(self):
        new_input = {"command": "ls -la"}
        result = _interpret_result(
            HookResult(
                permission_decision="allow",
                updated_input=new_input,
            )
        )
        assert result.allowed is True
        assert result.modified_input == {"command": "ls -la"}

    def test_modify_with_additional_context(self):
        result = _interpret_result(
            HookResult(
                updated_input={"file": "safe.txt"},
                additional_context="Hook injected safety note",
            )
        )
        assert result.allowed is True
        assert result.modified_input == {"file": "safe.txt"}
        assert result.additional_context == "Hook injected safety note"

    def test_deny_overrides_updated_input(self):
        """Even if updated_input is present, deny wins."""
        result = _interpret_result(
            HookResult(
                permission_decision="deny",
                permission_decision_reason="nope",
                updated_input={"should": "be ignored"},
            )
        )
        assert result.allowed is False
        assert result.modified_input is None

    def test_additional_context_without_modified_input(self):
        result = _interpret_result(
            HookResult(additional_context="extra info")
        )
        assert result.allowed is True
        assert result.modified_input is None
        assert result.additional_context == "extra info"


# ===================================================================
# process_pre_tool_use — integration with BrainmassHookEngine
# ===================================================================


class TestProcessPreToolUse:
    """Tests for process_pre_tool_use with a real engine."""

    def test_no_hooks_registered_allows(self):
        engine = BrainmassHookEngine()
        result = process_pre_tool_use(engine, "bash", {"command": "echo hi"})
        assert result.allowed is True
        assert result.modified_input is None

    def test_allow_hook(self):
        def _allow_callback(handler, event, context):
            return HookResult(permission_decision="allow")

        engine = _make_engine_with_callback(_allow_callback)
        _register_hook(engine, matcher="bash")

        result = process_pre_tool_use(engine, "bash", {"command": "echo hi"})
        assert result.allowed is True

    def test_deny_hook(self):
        def _deny_callback(handler, event, context):
            return HookResult(
                permission_decision="deny",
                permission_decision_reason="bash is forbidden",
            )

        engine = _make_engine_with_callback(_deny_callback)
        _register_hook(engine, matcher="bash")

        result = process_pre_tool_use(engine, "bash", {"command": "rm -rf /"})
        assert result.allowed is False
        assert result.deny_reason == "bash is forbidden"

    def test_modify_hook(self):
        safe_input = {"command": "echo safe"}

        def _modify_callback(handler, event, context):
            return HookResult(
                permission_decision="allow",
                updated_input=safe_input,
                additional_context="Input was sanitized",
            )

        engine = _make_engine_with_callback(_modify_callback)
        _register_hook(engine, matcher="bash")

        result = process_pre_tool_use(engine, "bash", {"command": "rm -rf /"})
        assert result.allowed is True
        assert result.modified_input == safe_input
        assert result.additional_context == "Input was sanitized"

    def test_non_matching_hook_allows(self):
        """Hook registered for 'write' should not fire for 'bash'."""
        def _deny_callback(handler, event, context):
            return HookResult(permission_decision="deny")

        engine = _make_engine_with_callback(_deny_callback)
        _register_hook(engine, matcher="write")

        result = process_pre_tool_use(engine, "bash", {"command": "echo hi"})
        assert result.allowed is True

    def test_context_fields_forwarded(self):
        """Verify session_id, cwd, session_type reach the hook context."""
        captured = {}

        def _capture_callback(handler, event, context):
            captured["session_id"] = context.session_id
            captured["cwd"] = context.cwd
            captured["session_type"] = context.session_type
            captured["tool_name"] = context.tool_name
            captured["tool_input"] = context.tool_input
            return HookResult(permission_decision="allow")

        engine = _make_engine_with_callback(_capture_callback)
        _register_hook(engine)

        process_pre_tool_use(
            engine,
            "bash",
            {"command": "ls"},
            session_id="sess-42",
            cwd="/home/dev",
            session_type="headless",
        )

        assert captured["session_id"] == "sess-42"
        assert captured["cwd"] == "/home/dev"
        assert captured["session_type"] == "headless"
        assert captured["tool_name"] == "bash"
        assert captured["tool_input"] == {"command": "ls"}


# ===================================================================
# PreToolUseResult dataclass
# ===================================================================


class TestPreToolUseResult:
    def test_defaults(self):
        r = PreToolUseResult(allowed=True)
        assert r.allowed is True
        assert r.modified_input is None
        assert r.additional_context is None
        assert r.deny_reason is None

    def test_denied_result(self):
        r = PreToolUseResult(allowed=False, deny_reason="blocked")
        assert r.allowed is False
        assert r.deny_reason == "blocked"
