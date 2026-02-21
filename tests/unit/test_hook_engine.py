"""Unit tests for BrainmassHookEngine.

Covers: event firing, scoped registration, 7-scope precedence,
matcher regex logic, async hooks, and managed-only mode.
"""

from __future__ import annotations

import pytest

from src.hooks.hook_engine import (
    SCOPE_PRECEDENCE,
    BrainmassHookEngine,
    _matches,
    _merge_results,
)
from src.types.core import (
    HookContext,
    HookDefinition,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    event: HookEvent = HookEvent.PRE_TOOL_USE,
    tool_name: str | None = "bash",
    source: str | None = None,
) -> HookContext:
    return HookContext(
        session_id="test-session",
        hook_event_name=event,
        cwd="/tmp",
        session_type="interactive",
        tool_name=tool_name,
        source=source,
    )


def _defn(matcher: str | None = None, is_async: bool = False) -> HookDefinition:
    return HookDefinition(
        matcher=matcher,
        hooks=[
            HookHandler(type=HookHandlerType.COMMAND, command="echo ok", is_async=is_async)
        ],
    )


# ===================================================================
# _matches() — matcher regex logic
# ===================================================================

class TestMatches:
    def test_none_matches_everything(self):
        assert _matches(None, "bash") is True
        assert _matches(None, None) is True

    def test_star_matches_everything(self):
        assert _matches("*", "bash") is True
        assert _matches("*", None) is True

    def test_exact_match(self):
        assert _matches("bash", "bash") is True
        assert _matches("bash", "grep") is False

    def test_regex_pattern(self):
        assert _matches("bash|grep", "bash") is True
        assert _matches("bash|grep", "grep") is True
        assert _matches("bash|grep", "ls") is False

    def test_regex_prefix(self):
        assert _matches("file_.*", "file_read") is True
        assert _matches("file_.*", "grep") is False

    def test_none_value_with_specific_matcher(self):
        assert _matches("bash", None) is False

    def test_invalid_regex_returns_false(self):
        assert _matches("[invalid", "bash") is False


# ===================================================================
# _merge_results()
# ===================================================================

class TestMergeResults:
    def test_empty_list(self):
        merged = _merge_results([])
        assert merged.permission_decision is None

    def test_deny_overrides_allow(self):
        r1 = HookResult(permission_decision="allow")
        r2 = HookResult(permission_decision="deny", permission_decision_reason="blocked")
        merged = _merge_results([r1, r2])
        assert merged.permission_decision == "deny"
        assert merged.permission_decision_reason == "blocked"

    def test_block_overrides_continue(self):
        r1 = HookResult(decision="continue")
        r2 = HookResult(decision="block", reason="not done")
        merged = _merge_results([r1, r2])
        assert merged.decision == "block"
        assert merged.reason == "not done"

    def test_first_updated_input_wins(self):
        """First non-None updated_input wins (highest-precedence scope)."""
        r1 = HookResult(updated_input={"a": 1})
        r2 = HookResult(updated_input={"b": 2})
        merged = _merge_results([r1, r2])
        assert merged.updated_input == {"a": 1}

    def test_allow_only_when_no_deny(self):
        r1 = HookResult(permission_decision="allow")
        r2 = HookResult()
        merged = _merge_results([r1, r2])
        assert merged.permission_decision == "allow"


# ===================================================================
# Registration
# ===================================================================

class TestRegistration:
    def test_register_single_hook(self):
        engine = BrainmassHookEngine()
        engine.register(HookEvent.PRE_TOOL_USE, _defn())
        regs = engine.get_registrations(HookEvent.PRE_TOOL_USE)
        assert len(regs) == 1

    def test_register_multiple_events(self):
        engine = BrainmassHookEngine()
        engine.register(HookEvent.PRE_TOOL_USE, _defn())
        engine.register(HookEvent.POST_TOOL_USE, _defn())
        assert len(engine.get_registrations(HookEvent.PRE_TOOL_USE)) == 1
        assert len(engine.get_registrations(HookEvent.POST_TOOL_USE)) == 1
        assert len(engine.get_registrations()) == 2

    def test_invalid_scope_raises(self):
        engine = BrainmassHookEngine()
        with pytest.raises(ValueError, match="Unknown scope"):
            engine.register(HookEvent.STOP, _defn(), scope="invalid_scope")

    def test_all_seven_scopes_accepted(self):
        engine = BrainmassHookEngine()
        for scope in SCOPE_PRECEDENCE:
            engine.register(HookEvent.STOP, _defn(), scope=scope)
        assert len(engine.get_registrations(HookEvent.STOP)) == 7


# ===================================================================
# Scoped registration and cleanup
# ===================================================================

class TestScopedRegistration:
    def test_register_scoped_returns_cleanup(self):
        engine = BrainmassHookEngine()
        hooks = {HookEvent.PRE_TOOL_USE: [_defn()]}
        cleanup = engine.register_scoped(hooks, scope_id="agent-1")
        assert callable(cleanup)
        assert len(engine.get_registrations(HookEvent.PRE_TOOL_USE)) == 1

    def test_cleanup_removes_hooks(self):
        engine = BrainmassHookEngine()
        hooks = {
            HookEvent.PRE_TOOL_USE: [_defn()],
            HookEvent.STOP: [_defn()],
        }
        cleanup = engine.register_scoped(hooks, scope_id="agent-1")
        assert len(engine.get_registrations()) == 2
        cleanup()
        assert len(engine.get_registrations()) == 0

    def test_cleanup_only_removes_own_hooks(self):
        engine = BrainmassHookEngine()
        engine.register(HookEvent.PRE_TOOL_USE, _defn(), scope="user_wide")
        hooks = {HookEvent.PRE_TOOL_USE: [_defn()]}
        cleanup = engine.register_scoped(hooks, scope_id="agent-1")
        assert len(engine.get_registrations(HookEvent.PRE_TOOL_USE)) == 2
        cleanup()
        assert len(engine.get_registrations(HookEvent.PRE_TOOL_USE)) == 1

    def test_auto_generated_scope_id(self):
        engine = BrainmassHookEngine()
        hooks = {HookEvent.STOP: [_defn()]}
        cleanup = engine.register_scoped(hooks)
        assert len(engine.get_registrations(HookEvent.STOP)) == 1
        cleanup()
        assert len(engine.get_registrations(HookEvent.STOP)) == 0

    def test_double_cleanup_is_safe(self):
        engine = BrainmassHookEngine()
        hooks = {HookEvent.STOP: [_defn()]}
        cleanup = engine.register_scoped(hooks, scope_id="x")
        cleanup()
        cleanup()  # should not raise
        assert len(engine.get_registrations()) == 0


# ===================================================================
# Precedence resolution
# ===================================================================

class TestPrecedence:
    def test_enterprise_managed_fires_first(self):
        """Enterprise managed hooks should execute before all others."""
        engine = BrainmassHookEngine()
        call_order: list[str] = []

        def _cb(handler, event, context):
            call_order.append(handler.command or "")
            return HookResult()

        engine.set_handler_callback(_cb)

        engine.register(
            HookEvent.PRE_TOOL_USE,
            HookDefinition(hooks=[HookHandler(type=HookHandlerType.COMMAND, command="user")]),
            scope="user_wide",
        )
        engine.register(
            HookEvent.PRE_TOOL_USE,
            HookDefinition(hooks=[HookHandler(type=HookHandlerType.COMMAND, command="enterprise")]),
            scope="enterprise_managed",
        )
        engine.register(
            HookEvent.PRE_TOOL_USE,
            HookDefinition(hooks=[HookHandler(type=HookHandlerType.COMMAND, command="plugin")]),
            scope="plugin",
        )

        engine.fire(HookEvent.PRE_TOOL_USE, _ctx())
        assert call_order == ["enterprise", "plugin", "user"]

    def test_full_seven_scope_order(self):
        engine = BrainmassHookEngine()
        call_order: list[str] = []

        def _cb(handler, event, context):
            call_order.append(handler.command or "")
            return HookResult()

        engine.set_handler_callback(_cb)

        # Register in reverse precedence order
        for scope in reversed(SCOPE_PRECEDENCE):
            engine.register(
                HookEvent.STOP,
                HookDefinition(
                    hooks=[HookHandler(type=HookHandlerType.COMMAND, command=scope)]
                ),
                scope=scope,
            )

        engine.fire(HookEvent.STOP, _ctx(event=HookEvent.STOP))
        assert call_order == SCOPE_PRECEDENCE

    def test_same_scope_preserves_insertion_order(self):
        engine = BrainmassHookEngine()
        call_order: list[str] = []

        def _cb(handler, event, context):
            call_order.append(handler.command or "")
            return HookResult()

        engine.set_handler_callback(_cb)

        for i in range(3):
            engine.register(
                HookEvent.STOP,
                HookDefinition(
                    hooks=[HookHandler(type=HookHandlerType.COMMAND, command=f"hook-{i}")]
                ),
                scope="project_local",
            )

        engine.fire(HookEvent.STOP, _ctx(event=HookEvent.STOP))
        assert call_order == ["hook-0", "hook-1", "hook-2"]


# ===================================================================
# Event firing
# ===================================================================

class TestFiring:
    def test_fire_returns_empty_result_when_no_hooks(self):
        engine = BrainmassHookEngine()
        result = engine.fire(HookEvent.SESSION_START, _ctx(event=HookEvent.SESSION_START))
        assert result.permission_decision is None
        assert result.decision is None

    def test_fire_pre_tool_use_with_matching_tool(self):
        engine = BrainmassHookEngine()
        engine.set_handler_callback(
            lambda h, e, c: HookResult(permission_decision="deny", permission_decision_reason="blocked")
        )
        engine.register(HookEvent.PRE_TOOL_USE, _defn(matcher="bash"))
        result = engine.fire(HookEvent.PRE_TOOL_USE, _ctx(tool_name="bash"))
        assert result.permission_decision == "deny"

    def test_fire_pre_tool_use_non_matching_tool(self):
        engine = BrainmassHookEngine()
        engine.set_handler_callback(
            lambda h, e, c: HookResult(permission_decision="deny")
        )
        engine.register(HookEvent.PRE_TOOL_USE, _defn(matcher="bash"))
        result = engine.fire(HookEvent.PRE_TOOL_USE, _ctx(tool_name="grep"))
        assert result.permission_decision is None  # no handler matched

    def test_fire_session_start_matches_source(self):
        engine = BrainmassHookEngine()
        engine.set_handler_callback(
            lambda h, e, c: HookResult(decision="continue")
        )
        engine.register(HookEvent.SESSION_START, _defn(matcher="new"))
        result = engine.fire(
            HookEvent.SESSION_START,
            _ctx(event=HookEvent.SESSION_START, source="new"),
        )
        assert result.decision == "continue"

    def test_fire_session_start_non_matching_source(self):
        engine = BrainmassHookEngine()
        engine.set_handler_callback(
            lambda h, e, c: HookResult(decision="continue")
        )
        engine.register(HookEvent.SESSION_START, _defn(matcher="new"))
        result = engine.fire(
            HookEvent.SESSION_START,
            _ctx(event=HookEvent.SESSION_START, source="resume"),
        )
        assert result.decision is None

    def test_fire_all_12_events(self):
        """Every HookEvent value can be fired without error."""
        engine = BrainmassHookEngine()
        engine.register(HookEvent.PRE_TOOL_USE, _defn())  # just one registration
        for event in HookEvent:
            ctx = _ctx(event=event)
            result = engine.fire(event, ctx)
            assert isinstance(result, HookResult)

    def test_fire_stop_with_block(self):
        engine = BrainmassHookEngine()
        engine.set_handler_callback(
            lambda h, e, c: HookResult(decision="block", reason="not done yet")
        )
        engine.register(HookEvent.STOP, _defn())
        result = engine.fire(HookEvent.STOP, _ctx(event=HookEvent.STOP))
        assert result.decision == "block"
        assert result.reason == "not done yet"

    def test_wildcard_matcher_matches_all_tools(self):
        engine = BrainmassHookEngine()
        call_count = 0

        def _cb(h, e, c):
            nonlocal call_count
            call_count += 1
            return HookResult()

        engine.set_handler_callback(_cb)
        engine.register(HookEvent.PRE_TOOL_USE, _defn(matcher="*"))
        engine.fire(HookEvent.PRE_TOOL_USE, _ctx(tool_name="anything"))
        assert call_count == 1


# ===================================================================
# Async hooks
# ===================================================================

class TestAsyncHooks:
    def test_async_hook_does_not_contribute_to_result(self):
        """Async hooks fire-and-forget; their results are not merged."""
        engine = BrainmassHookEngine()
        engine.set_handler_callback(
            lambda h, e, c: HookResult(permission_decision="deny")
        )
        # Register one async hook only
        engine.register(HookEvent.PRE_TOOL_USE, _defn(is_async=True))
        result = engine.fire(HookEvent.PRE_TOOL_USE, _ctx())
        # Async result is not merged → default empty result
        assert result.permission_decision is None

    def test_sync_and_async_mix(self):
        """Sync hooks contribute to result; async hooks do not."""
        engine = BrainmassHookEngine()
        engine.set_handler_callback(
            lambda h, e, c: HookResult(permission_decision="allow")
        )
        # Sync hook
        engine.register(HookEvent.PRE_TOOL_USE, _defn(is_async=False))
        # Async hook
        engine.register(HookEvent.PRE_TOOL_USE, _defn(is_async=True))
        result = engine.fire(HookEvent.PRE_TOOL_USE, _ctx())
        assert result.permission_decision == "allow"


# ===================================================================
# Managed-only mode
# ===================================================================

class TestManagedOnly:
    def test_managed_only_blocks_non_enterprise_hooks(self):
        engine = BrainmassHookEngine()
        call_order: list[str] = []

        def _cb(handler, event, context):
            call_order.append(handler.command or "")
            return HookResult()

        engine.set_handler_callback(_cb)
        engine.register(
            HookEvent.STOP,
            HookDefinition(hooks=[HookHandler(type=HookHandlerType.COMMAND, command="enterprise")]),
            scope="enterprise_managed",
        )
        engine.register(
            HookEvent.STOP,
            HookDefinition(hooks=[HookHandler(type=HookHandlerType.COMMAND, command="user")]),
            scope="user_wide",
        )

        engine.set_managed_only(True)
        engine.fire(HookEvent.STOP, _ctx(event=HookEvent.STOP))
        assert call_order == ["enterprise"]

    def test_managed_only_can_be_toggled_off(self):
        engine = BrainmassHookEngine()
        call_count = 0

        def _cb(h, e, c):
            nonlocal call_count
            call_count += 1
            return HookResult()

        engine.set_handler_callback(_cb)
        engine.register(HookEvent.STOP, _defn(), scope="user_wide")

        engine.set_managed_only(True)
        engine.fire(HookEvent.STOP, _ctx(event=HookEvent.STOP))
        assert call_count == 0

        engine.set_managed_only(False)
        engine.fire(HookEvent.STOP, _ctx(event=HookEvent.STOP))
        assert call_count == 1


# ===================================================================
# register_hooks (Strands integration stub)
# ===================================================================

class TestRegisterHooks:
    def test_register_hooks_accepts_registry_object(self):
        """register_hooks should accept any object (stub for Strands HookRegistry)."""
        engine = BrainmassHookEngine()
        engine.register_hooks(object())  # should not raise


# ===================================================================
# Handler callback
# ===================================================================

class TestHandlerCallback:
    def test_default_handler_returns_empty_result(self):
        engine = BrainmassHookEngine()
        engine.register(HookEvent.PRE_TOOL_USE, _defn())
        result = engine.fire(HookEvent.PRE_TOOL_USE, _ctx())
        assert result.permission_decision is None

    def test_custom_callback_is_invoked(self):
        engine = BrainmassHookEngine()
        invoked = []

        def _cb(handler, event, context):
            invoked.append((handler.command, event))
            return HookResult(permission_decision="allow")

        engine.set_handler_callback(_cb)
        engine.register(HookEvent.PRE_TOOL_USE, _defn())
        result = engine.fire(HookEvent.PRE_TOOL_USE, _ctx())
        assert len(invoked) == 1
        assert invoked[0][1] == HookEvent.PRE_TOOL_USE
        assert result.permission_decision == "allow"

    def test_reset_callback_to_none(self):
        engine = BrainmassHookEngine()
        engine.set_handler_callback(lambda h, e, c: HookResult(decision="block"))
        engine.set_handler_callback(None)
        engine.register(HookEvent.STOP, _defn())
        result = engine.fire(HookEvent.STOP, _ctx(event=HookEvent.STOP))
        assert result.decision is None
