"""Property-based tests for hook engine behavior.

Properties covered:
- Property 5: Hook matcher regex correctness
- Property 6: PreToolUse allow/deny/modify behavior
- Property 7: Async hooks do not block execution
"""

from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.hooks.hook_engine import BrainmassHookEngine, _matches
from src.types.core import (
    HookContext,
    HookDefinition,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
)

ASCII_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=20,
)


def _pre_tool_context(tool_name: str = "bash") -> HookContext:
    return HookContext(
        session_id="property-session",
        hook_event_name=HookEvent.PRE_TOOL_USE,
        cwd=".",
        session_type="interactive",
        tool_name=tool_name,
    )


@pytest.mark.property
@settings(max_examples=100)
@given(tool_name=ASCII_NAME, alternative=ASCII_NAME, other=ASCII_NAME)
def test_property_5_hook_matcher_regex_correctness(
    tool_name: str,
    alternative: str,
    other: str,
) -> None:
    """Feature: claude-code-v3-enterprise, Property 5."""
    assert _matches(None, tool_name) is True
    assert _matches("*", tool_name) is True
    assert _matches(tool_name, tool_name) is True
    assert _matches(tool_name, alternative) is (tool_name == alternative)

    union_pattern = f"{tool_name}|{alternative}"
    assert _matches(union_pattern, tool_name) is True
    assert _matches(union_pattern, alternative) is True
    if other != tool_name and other != alternative:
        assert _matches(union_pattern, other) is False


@pytest.mark.property
@settings(max_examples=100)
@given(action=st.sampled_from(["allow", "deny", "modify"]))
def test_property_6_pre_tool_use_allow_deny_modify(action: str) -> None:
    """Feature: claude-code-v3-enterprise, Property 6."""
    engine = BrainmassHookEngine()
    definition = HookDefinition(
        matcher="*",
        hooks=[HookHandler(type=HookHandlerType.COMMAND, command="test-hook")],
    )
    engine.register(HookEvent.PRE_TOOL_USE, definition, scope="project_local")

    if action == "allow":
        expected = HookResult(permission_decision="allow")
    elif action == "deny":
        expected = HookResult(permission_decision="deny", permission_decision_reason="blocked")
    else:
        expected = HookResult(
            permission_decision="allow",
            updated_input={"command": "safe-command"},
            additional_context="rewritten by hook",
        )

    engine.set_handler_callback(lambda _h, _e, _c: expected)
    result = engine.fire(HookEvent.PRE_TOOL_USE, _pre_tool_context("bash"))

    assert result.permission_decision == expected.permission_decision
    assert result.updated_input == expected.updated_input
    assert result.additional_context == expected.additional_context


@pytest.mark.property
@settings(max_examples=100)
@given(async_count=st.integers(min_value=1, max_value=10))
def test_property_7_async_hooks_do_not_block_execution(async_count: int) -> None:
    """Feature: claude-code-v3-enterprise, Property 7."""
    engine = BrainmassHookEngine()
    sync_calls = {"count": 0}

    def callback(handler: HookHandler, _event: HookEvent, _context: HookContext) -> HookResult:
        if handler.command == "sync":
            sync_calls["count"] += 1
            return HookResult(permission_decision="allow")
        return HookResult(permission_decision="deny")

    engine.set_handler_callback(callback)

    for idx in range(async_count):
        engine.register(
            HookEvent.PRE_TOOL_USE,
            HookDefinition(
                hooks=[
                    HookHandler(
                        type=HookHandlerType.COMMAND,
                        command=f"async-{idx}",
                        is_async=True,
                    )
                ]
            ),
            scope="project_local",
        )

    engine.register(
        HookEvent.PRE_TOOL_USE,
        HookDefinition(
            hooks=[HookHandler(type=HookHandlerType.COMMAND, command="sync", is_async=False)]
        ),
        scope="project_local",
    )

    started = time.perf_counter()
    result = engine.fire(HookEvent.PRE_TOOL_USE, _pre_tool_context("bash"))
    elapsed = time.perf_counter() - started

    assert result.permission_decision == "allow"
    assert sync_calls["count"] == 1
    assert elapsed < 0.5

