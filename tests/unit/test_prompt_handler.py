"""Unit tests for the prompt hook handler.

Tests cover:
- $ARGUMENTS substitution in prompt templates
- Model response parsing (plain JSON, markdown fences, invalid)
- HookResult construction from parsed responses
- Error handling (no callback, callback raises, empty prompt)
- Blocking vs non-blocking event behaviour on failures
"""

from __future__ import annotations

import json

import pytest

from src.hooks.handlers.prompt import (
    PromptHandler,
    _default_fail_result,
    _parse_model_response,
    _result_from_response,
    _serialize_context,
    _substitute_arguments,
)
from src.types.core import (
    HookContext,
    HookEvent,
    HookHandler,
    HookHandlerType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**overrides) -> HookContext:
    defaults = dict(
        session_id="sess-001",
        hook_event_name=HookEvent.PRE_TOOL_USE,
        cwd="/tmp/project",
        session_type="interactive",
    )
    defaults.update(overrides)
    return HookContext(**defaults)


def _make_handler(**overrides) -> HookHandler:
    defaults = dict(
        type=HookHandlerType.PROMPT,
        prompt="Evaluate this: $ARGUMENTS",
    )
    defaults.update(overrides)
    return HookHandler(**defaults)


# ---------------------------------------------------------------------------
# _serialize_context
# ---------------------------------------------------------------------------

class TestSerializeContext:
    def test_produces_valid_json(self):
        ctx = _make_context()
        result = _serialize_context(ctx)
        data = json.loads(result)
        assert data["session_id"] == "sess-001"
        assert data["hook_event_name"] == "PreToolUse"

    def test_includes_optional_fields(self):
        ctx = _make_context(tool_name="bash", tool_input={"cmd": "ls"})
        data = json.loads(_serialize_context(ctx))
        assert data["tool_name"] == "bash"
        assert data["tool_input"] == {"cmd": "ls"}


# ---------------------------------------------------------------------------
# _substitute_arguments
# ---------------------------------------------------------------------------

class TestSubstituteArguments:
    def test_replaces_arguments_placeholder(self):
        ctx = _make_context()
        prompt = "Check this: $ARGUMENTS"
        result = _substitute_arguments(prompt, ctx)
        assert "$ARGUMENTS" not in result
        # The serialized context should be embedded
        assert "sess-001" in result

    def test_no_placeholder_returns_unchanged(self):
        ctx = _make_context()
        prompt = "No placeholder here"
        result = _substitute_arguments(prompt, ctx)
        assert result == "No placeholder here"

    def test_multiple_placeholders_all_replaced(self):
        ctx = _make_context()
        prompt = "First: $ARGUMENTS, Second: $ARGUMENTS"
        result = _substitute_arguments(prompt, ctx)
        assert "$ARGUMENTS" not in result


# ---------------------------------------------------------------------------
# _parse_model_response
# ---------------------------------------------------------------------------

class TestParseModelResponse:
    def test_plain_json(self):
        raw = json.dumps({"permissionDecision": "deny", "reason": "unsafe"})
        parsed = _parse_model_response(raw)
        assert parsed is not None
        assert parsed["permissionDecision"] == "deny"

    def test_json_in_code_fence(self):
        raw = '```json\n{"permissionDecision": "allow"}\n```'
        parsed = _parse_model_response(raw)
        assert parsed is not None
        assert parsed["permissionDecision"] == "allow"

    def test_json_in_plain_code_fence(self):
        raw = '```\n{"decision": "block"}\n```'
        parsed = _parse_model_response(raw)
        assert parsed is not None
        assert parsed["decision"] == "block"

    def test_empty_string_returns_none(self):
        assert _parse_model_response("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_model_response("   \n  ") is None

    def test_invalid_json_returns_none(self):
        assert _parse_model_response("not json at all") is None

    def test_json_with_surrounding_whitespace(self):
        raw = '  \n  {"permissionDecision": "allow"}  \n  '
        parsed = _parse_model_response(raw)
        assert parsed is not None
        assert parsed["permissionDecision"] == "allow"


# ---------------------------------------------------------------------------
# _result_from_response
# ---------------------------------------------------------------------------

class TestResultFromResponse:
    def test_parsed_allow(self):
        parsed = {"permissionDecision": "allow", "reason": "looks good"}
        result = _result_from_response(parsed, HookEvent.PRE_TOOL_USE)
        assert result.permission_decision == "allow"
        assert result.reason == "looks good"

    def test_parsed_deny(self):
        parsed = {"permissionDecision": "deny", "permissionDecisionReason": "bad"}
        result = _result_from_response(parsed, HookEvent.PRE_TOOL_USE)
        assert result.permission_decision == "deny"
        assert result.permission_decision_reason == "bad"

    def test_parsed_with_updated_input(self):
        parsed = {"updatedInput": {"cmd": "safe-cmd"}, "additionalContext": "sanitized"}
        result = _result_from_response(parsed, HookEvent.PRE_TOOL_USE)
        assert result.updated_input == {"cmd": "safe-cmd"}
        assert result.additional_context == "sanitized"

    def test_none_parsed_blocking_event_denies(self):
        result = _result_from_response(None, HookEvent.PRE_TOOL_USE)
        assert result.permission_decision == "deny"
        assert result.decision == "block"

    def test_none_parsed_nonblocking_event_allows(self):
        result = _result_from_response(None, HookEvent.POST_TOOL_USE)
        assert result.permission_decision == "allow"
        assert result.decision == "continue"

    def test_defaults_for_missing_fields(self):
        parsed = {}
        result = _result_from_response(parsed, HookEvent.SESSION_START)
        assert result.permission_decision == "allow"
        assert result.decision == "continue"


# ---------------------------------------------------------------------------
# _default_fail_result
# ---------------------------------------------------------------------------

class TestDefaultFailResult:
    def test_blocking_event_returns_deny(self):
        result = _default_fail_result(HookEvent.PRE_TOOL_USE)
        assert result.permission_decision == "deny"
        assert result.decision == "block"

    def test_nonblocking_event_returns_allow(self):
        result = _default_fail_result(HookEvent.POST_TOOL_USE)
        assert result.permission_decision == "allow"
        assert result.decision == "continue"

    def test_stop_event_is_blocking(self):
        result = _default_fail_result(HookEvent.STOP)
        assert result.permission_decision == "deny"

    def test_permission_request_is_blocking(self):
        result = _default_fail_result(HookEvent.PERMISSION_REQUEST)
        assert result.permission_decision == "deny"


# ---------------------------------------------------------------------------
# PromptHandler.execute
# ---------------------------------------------------------------------------

class TestPromptHandlerExecute:
    def test_wrong_handler_type_raises(self):
        ph = PromptHandler()
        handler = HookHandler(type=HookHandlerType.COMMAND, command="echo hi")
        ctx = _make_context()
        with pytest.raises(ValueError, match="non-prompt"):
            ph.execute(handler, ctx)

    def test_empty_prompt_returns_allow(self):
        ph = PromptHandler()
        handler = _make_handler(prompt=None)
        ctx = _make_context()
        result = ph.execute(handler, ctx)
        assert result.permission_decision == "allow"

    def test_no_callback_returns_allow(self):
        ph = PromptHandler()
        handler = _make_handler()
        ctx = _make_context()
        result = ph.execute(handler, ctx)
        assert result.permission_decision == "allow"

    def test_callback_receives_resolved_prompt(self):
        received = []

        def stub(prompt: str) -> str:
            received.append(prompt)
            return json.dumps({"permissionDecision": "allow"})

        ph = PromptHandler()
        ph.set_model_callback(stub)
        handler = _make_handler(prompt="Check: $ARGUMENTS")
        ctx = _make_context(session_id="test-123")
        ph.execute(handler, ctx)

        assert len(received) == 1
        assert "test-123" in received[0]
        assert "$ARGUMENTS" not in received[0]

    def test_allow_response(self):
        def stub(prompt: str) -> str:
            return json.dumps({"permissionDecision": "allow", "reason": "safe"})

        ph = PromptHandler()
        ph.set_model_callback(stub)
        result = ph.execute(_make_handler(), _make_context())
        assert result.permission_decision == "allow"
        assert result.reason == "safe"

    def test_deny_response(self):
        def stub(prompt: str) -> str:
            return json.dumps({
                "permissionDecision": "deny",
                "permissionDecisionReason": "dangerous",
            })

        ph = PromptHandler()
        ph.set_model_callback(stub)
        result = ph.execute(_make_handler(), _make_context())
        assert result.permission_decision == "deny"
        assert result.permission_decision_reason == "dangerous"

    def test_block_decision_for_stop_event(self):
        def stub(prompt: str) -> str:
            return json.dumps({"decision": "block", "reason": "not done yet"})

        ph = PromptHandler()
        ph.set_model_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.STOP)
        result = ph.execute(_make_handler(), ctx)
        assert result.decision == "block"
        assert result.reason == "not done yet"

    def test_updated_input_passthrough(self):
        def stub(prompt: str) -> str:
            return json.dumps({
                "permissionDecision": "allow",
                "updatedInput": {"cmd": "safe-ls"},
                "additionalContext": "sanitized input",
            })

        ph = PromptHandler()
        ph.set_model_callback(stub)
        result = ph.execute(_make_handler(), _make_context())
        assert result.updated_input == {"cmd": "safe-ls"}
        assert result.additional_context == "sanitized input"

    def test_callback_exception_nonblocking_allows(self):
        def stub(prompt: str) -> str:
            raise RuntimeError("model down")

        ph = PromptHandler()
        ph.set_model_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.POST_TOOL_USE)
        result = ph.execute(_make_handler(), ctx)
        assert result.permission_decision == "allow"
        assert result.decision == "continue"

    def test_callback_exception_blocking_denies(self):
        def stub(prompt: str) -> str:
            raise RuntimeError("model down")

        ph = PromptHandler()
        ph.set_model_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.PRE_TOOL_USE)
        result = ph.execute(_make_handler(), ctx)
        assert result.permission_decision == "deny"
        assert result.decision == "block"

    def test_unparseable_response_nonblocking_allows(self):
        def stub(prompt: str) -> str:
            return "I think this is fine, go ahead."

        ph = PromptHandler()
        ph.set_model_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.SESSION_START)
        result = ph.execute(_make_handler(), ctx)
        assert result.permission_decision == "allow"

    def test_unparseable_response_blocking_denies(self):
        def stub(prompt: str) -> str:
            return "I think this is fine, go ahead."

        ph = PromptHandler()
        ph.set_model_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.PRE_TOOL_USE)
        result = ph.execute(_make_handler(), ctx)
        assert result.permission_decision == "deny"

    def test_markdown_fenced_json_response(self):
        def stub(prompt: str) -> str:
            return '```json\n{"permissionDecision": "deny", "reason": "blocked"}\n```'

        ph = PromptHandler()
        ph.set_model_callback(stub)
        result = ph.execute(_make_handler(), _make_context())
        assert result.permission_decision == "deny"
        assert result.reason == "blocked"

    def test_set_model_callback_to_none_resets(self):
        def stub(prompt: str) -> str:
            return json.dumps({"permissionDecision": "deny"})

        ph = PromptHandler()
        ph.set_model_callback(stub)
        ph.set_model_callback(None)
        # With no callback, should return default allow
        result = ph.execute(_make_handler(), _make_context())
        assert result.permission_decision == "allow"

    def test_empty_string_prompt_treated_as_no_prompt(self):
        ph = PromptHandler()
        handler = _make_handler(prompt="")
        ctx = _make_context()
        result = ph.execute(handler, ctx)
        assert result.permission_decision == "allow"
