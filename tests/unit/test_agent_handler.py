"""Unit tests for the agent hook handler.

Tests cover:
- Context prompt building from HookContext + agent_config
- Agent response parsing (plain JSON, markdown fences, embedded JSON, invalid)
- HookResult construction from parsed responses
- Error handling (no callback, callback raises)
- Blocking vs non-blocking event behaviour on failures
"""

from __future__ import annotations

import json

import pytest

from src.hooks.handlers.agent import (
    AgentHandler,
    _build_context_prompt,
    _default_fail_result,
    _parse_agent_response,
    _result_from_response,
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
        type=HookHandlerType.AGENT,
        agent_config={"instructions": "Verify code quality", "tools": ["Read", "Grep", "Glob"]},
    )
    defaults.update(overrides)
    return HookHandler(**defaults)


# ---------------------------------------------------------------------------
# _build_context_prompt
# ---------------------------------------------------------------------------

class TestBuildContextPrompt:
    def test_includes_hook_context_json(self):
        ctx = _make_context(session_id="test-42")
        prompt = _build_context_prompt(ctx, None)
        assert "test-42" in prompt
        assert "PreToolUse" in prompt

    def test_includes_agent_instructions(self):
        ctx = _make_context()
        config = {"instructions": "Check for security issues"}
        prompt = _build_context_prompt(ctx, config)
        assert "Check for security issues" in prompt

    def test_includes_tool_list(self):
        ctx = _make_context()
        config = {"tools": ["Read", "Grep", "Glob"]}
        prompt = _build_context_prompt(ctx, config)
        assert "Read" in prompt
        assert "Grep" in prompt
        assert "Glob" in prompt

    def test_no_config_still_includes_context(self):
        ctx = _make_context(session_id="no-config")
        prompt = _build_context_prompt(ctx, None)
        assert "no-config" in prompt

    def test_empty_config_still_includes_context(self):
        ctx = _make_context()
        prompt = _build_context_prompt(ctx, {})
        assert "sess-001" in prompt


# ---------------------------------------------------------------------------
# _parse_agent_response
# ---------------------------------------------------------------------------

class TestParseAgentResponse:
    def test_plain_json(self):
        raw = json.dumps({"permissionDecision": "deny", "reason": "unsafe"})
        parsed = _parse_agent_response(raw)
        assert parsed is not None
        assert parsed["permissionDecision"] == "deny"

    def test_json_in_code_fence(self):
        raw = '```json\n{"permissionDecision": "allow"}\n```'
        parsed = _parse_agent_response(raw)
        assert parsed is not None
        assert parsed["permissionDecision"] == "allow"

    def test_json_in_plain_code_fence(self):
        raw = '```\n{"decision": "block"}\n```'
        parsed = _parse_agent_response(raw)
        assert parsed is not None
        assert parsed["decision"] == "block"

    def test_embedded_json_in_text(self):
        raw = 'After analysis, here is my result: {"permissionDecision": "deny", "reason": "found vulnerability"} end.'
        parsed = _parse_agent_response(raw)
        assert parsed is not None
        assert parsed["permissionDecision"] == "deny"

    def test_empty_string_returns_none(self):
        assert _parse_agent_response("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_agent_response("   \n  ") is None

    def test_invalid_text_returns_none(self):
        assert _parse_agent_response("no json here at all") is None


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

    def test_updated_input_passthrough(self):
        parsed = {"updatedInput": {"cmd": "safe"}, "additionalContext": "sanitized"}
        result = _result_from_response(parsed, HookEvent.PRE_TOOL_USE)
        assert result.updated_input == {"cmd": "safe"}
        assert result.additional_context == "sanitized"


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
# AgentHandler.execute
# ---------------------------------------------------------------------------

class TestAgentHandlerExecute:
    def test_wrong_handler_type_raises(self):
        ah = AgentHandler()
        handler = HookHandler(type=HookHandlerType.COMMAND, command="echo hi")
        ctx = _make_context()
        with pytest.raises(ValueError, match="non-agent"):
            ah.execute(handler, ctx)

    def test_no_callback_returns_allow(self):
        ah = AgentHandler()
        handler = _make_handler()
        ctx = _make_context()
        result = ah.execute(handler, ctx)
        assert result.permission_decision == "allow"

    def test_callback_receives_context_prompt(self):
        received_prompts = []
        received_configs = []

        def stub(prompt: str, config: dict | None) -> str:
            received_prompts.append(prompt)
            received_configs.append(config)
            return json.dumps({"permissionDecision": "allow"})

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        handler = _make_handler()
        ctx = _make_context(session_id="test-123")
        ah.execute(handler, ctx)

        assert len(received_prompts) == 1
        assert "test-123" in received_prompts[0]
        assert "Verify code quality" in received_prompts[0]
        assert received_configs[0] == handler.agent_config

    def test_allow_response(self):
        def stub(prompt: str, config: dict | None) -> str:
            return json.dumps({"permissionDecision": "allow", "reason": "safe"})

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        result = ah.execute(_make_handler(), _make_context())
        assert result.permission_decision == "allow"
        assert result.reason == "safe"

    def test_deny_response(self):
        def stub(prompt: str, config: dict | None) -> str:
            return json.dumps({
                "permissionDecision": "deny",
                "permissionDecisionReason": "found vulnerability",
            })

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        result = ah.execute(_make_handler(), _make_context())
        assert result.permission_decision == "deny"
        assert result.permission_decision_reason == "found vulnerability"

    def test_block_decision_for_stop_event(self):
        def stub(prompt: str, config: dict | None) -> str:
            return json.dumps({"decision": "block", "reason": "not done yet"})

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.STOP)
        result = ah.execute(_make_handler(), ctx)
        assert result.decision == "block"
        assert result.reason == "not done yet"

    def test_updated_input_passthrough(self):
        def stub(prompt: str, config: dict | None) -> str:
            return json.dumps({
                "permissionDecision": "allow",
                "updatedInput": {"cmd": "safe-ls"},
                "additionalContext": "sanitized input",
            })

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        result = ah.execute(_make_handler(), _make_context())
        assert result.updated_input == {"cmd": "safe-ls"}
        assert result.additional_context == "sanitized input"

    def test_callback_exception_nonblocking_allows(self):
        def stub(prompt: str, config: dict | None) -> str:
            raise RuntimeError("agent crashed")

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.POST_TOOL_USE)
        result = ah.execute(_make_handler(), ctx)
        assert result.permission_decision == "allow"
        assert result.decision == "continue"

    def test_callback_exception_blocking_denies(self):
        def stub(prompt: str, config: dict | None) -> str:
            raise RuntimeError("agent crashed")

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.PRE_TOOL_USE)
        result = ah.execute(_make_handler(), ctx)
        assert result.permission_decision == "deny"
        assert result.decision == "block"

    def test_unparseable_response_nonblocking_allows(self):
        def stub(prompt: str, config: dict | None) -> str:
            return "I checked the codebase and everything looks fine."

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.SESSION_START)
        result = ah.execute(_make_handler(), ctx)
        assert result.permission_decision == "allow"

    def test_unparseable_response_blocking_denies(self):
        def stub(prompt: str, config: dict | None) -> str:
            return "I checked the codebase and everything looks fine."

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        ctx = _make_context(hook_event_name=HookEvent.PRE_TOOL_USE)
        result = ah.execute(_make_handler(), ctx)
        assert result.permission_decision == "deny"

    def test_markdown_fenced_json_response(self):
        def stub(prompt: str, config: dict | None) -> str:
            return '```json\n{"permissionDecision": "deny", "reason": "blocked"}\n```'

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        result = ah.execute(_make_handler(), _make_context())
        assert result.permission_decision == "deny"
        assert result.reason == "blocked"

    def test_set_agent_callback_to_none_resets(self):
        def stub(prompt: str, config: dict | None) -> str:
            return json.dumps({"permissionDecision": "deny"})

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        ah.set_agent_callback(None)
        result = ah.execute(_make_handler(), _make_context())
        assert result.permission_decision == "allow"

    def test_no_agent_config_still_works(self):
        def stub(prompt: str, config: dict | None) -> str:
            return json.dumps({"permissionDecision": "allow"})

        ah = AgentHandler()
        ah.set_agent_callback(stub)
        handler = _make_handler(agent_config=None)
        result = ah.execute(handler, _make_context())
        assert result.permission_decision == "allow"
