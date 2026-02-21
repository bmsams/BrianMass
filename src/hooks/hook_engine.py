"""Brainmass Hook Engine — fires 12 lifecycle events with scoped registration.

Conditionally inherits from ``strands.hooks.HookProvider`` when the Strands
SDK is installed, enabling direct integration with the Strands agent lifecycle.
Falls back to a plain ``object`` base when the SDK is absent so the engine
remains fully functional without the SDK.

Requirements: 3.1, 3.3, 6.1, 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from time import perf_counter

from src.observability.instrumentation import BrainmassTracer, HookLogger
from src.types.core import (
    HookContext,
    HookDefinition,
    HookEvent,
    HookHandler,
    HookResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional Strands HookProvider base (Requirements: 6.1, 6.2)
# ---------------------------------------------------------------------------

try:
    from strands.hooks import HookProvider as _StrandsHookProvider  # type: ignore[import-untyped]

    _STRANDS_HOOKS_AVAILABLE = True
except ImportError:
    _StrandsHookProvider = object  # type: ignore[assignment,misc]
    _STRANDS_HOOKS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Scope precedence (highest → lowest).  Index 0 = highest priority.
# ---------------------------------------------------------------------------
SCOPE_PRECEDENCE: list[str] = [
    "enterprise_managed",
    "plugin",
    "subagent_frontmatter",
    "skill_frontmatter",
    "project_local",
    "project_shared",
    "user_wide",
]

# Tool-related events where matcher applies against tool_name
_TOOL_EVENTS: set[HookEvent] = {
    HookEvent.PRE_TOOL_USE,
    HookEvent.POST_TOOL_USE,
    HookEvent.POST_TOOL_USE_FAILURE,
    HookEvent.PERMISSION_REQUEST,
}

# Events where matcher applies against source
_SOURCE_EVENTS: set[HookEvent] = {
    HookEvent.SESSION_START,
}


def _matches(matcher: str | None, value: str | None) -> bool:
    """Check if a matcher pattern matches a value.

    Rules (Requirement 3.8):
    - ``None`` or ``'*'`` matches everything.
    - Otherwise the matcher is treated as a regex and tested against *value*.
    - If *value* is ``None`` the match succeeds only for wildcard matchers.
    """
    if matcher is None or matcher == "*":
        return True
    if value is None:
        return False
    try:
        return re.fullmatch(matcher, value) is not None
    except re.error:
        logger.warning("Invalid matcher regex %r — treating as no-match", matcher)
        return False


def _merge_results(results: list[HookResult]) -> HookResult:
    """Merge multiple HookResults into a single aggregate result.

    Handlers run in precedence order (highest-precedence scope first).
    Merge rules:
    - ``permission_decision``: ``'deny'`` always wins over ``'allow'``.
    - ``decision``: ``'block'`` always wins over ``'continue'``.
    - ``updated_input`` / ``additional_context``: first non-None value wins
      (i.e. highest-precedence scope takes priority).
    - ``reason``: first non-None value wins.
    """
    merged = HookResult()
    for r in results:
        if r.permission_decision == "deny":
            merged.permission_decision = "deny"
            merged.permission_decision_reason = r.permission_decision_reason or merged.permission_decision_reason
        elif r.permission_decision == "allow" and merged.permission_decision is None:
            merged.permission_decision = "allow"

        # First-writer-wins: highest-precedence scope's modification sticks
        if r.updated_input is not None and merged.updated_input is None:
            merged.updated_input = r.updated_input
        if r.additional_context is not None and merged.additional_context is None:
            merged.additional_context = r.additional_context

        if r.decision == "block":
            merged.decision = "block"
            merged.reason = r.reason or merged.reason
        elif r.decision == "continue" and merged.decision is None:
            merged.decision = "continue"

        if r.reason is not None and merged.reason is None:
            merged.reason = r.reason
    return merged


# ---------------------------------------------------------------------------
# Registration entry — internal bookkeeping
# ---------------------------------------------------------------------------

class _Registration:
    """Internal record of a registered hook."""

    __slots__ = ("event", "definition", "scope", "scope_id")

    def __init__(
        self,
        event: HookEvent,
        definition: HookDefinition,
        scope: str,
        scope_id: str,
    ) -> None:
        self.event = event
        self.definition = definition
        self.scope = scope
        self.scope_id = scope_id


# ---------------------------------------------------------------------------
# BrainmassHookEngine
# ---------------------------------------------------------------------------

class BrainmassHookEngine(_StrandsHookProvider):  # type: ignore[misc]
    """Hook engine supporting 12 lifecycle events, scoped registration, and
    7-scope precedence resolution.

    Conditionally inherits from ``strands.hooks.HookProvider`` when the Strands
    SDK is installed. Implements ``register_hooks`` to wire Strands
    ``BeforeToolCallEvent`` → ``HookEvent.PRE_TOOL_USE`` and
    ``AfterToolCallEvent`` → ``HookEvent.POST_TOOL_USE``.

    Also exposes the method-based ``pre_tool_use`` / ``post_tool_use`` interface
    from architecture_context.md §5.4 for direct SDK integration.

    Usage::

        engine = BrainmassHookEngine()
        engine.register(HookEvent.PRE_TOOL_USE, definition, scope="project_local")
        result = engine.fire(HookEvent.PRE_TOOL_USE, context)
    """

    def __init__(
        self,
        tracer: BrainmassTracer | None = None,
        hook_logger: HookLogger | None = None,
    ) -> None:
        # event → list[_Registration]  (order of insertion preserved)
        self._registrations: dict[HookEvent, list[_Registration]] = defaultdict(list)
        # scope_id → list[_Registration]  (for cleanup)
        self._scoped: dict[str, list[_Registration]] = defaultdict(list)
        # Flag: when True, only enterprise_managed hooks are allowed
        self._allow_managed_only: bool = False
        self._tracer = tracer
        self._hook_logger = hook_logger

    # ------------------------------------------------------------------
    # Strands HookProvider integration (Requirements: 6.1, 6.2, 6.3, 6.4, 6.5)
    # ------------------------------------------------------------------

    def register_hooks(self, registry: object) -> None:
        """Register with a Strands ``HookRegistry``.

        Wires ``BeforeToolCallEvent`` → ``_on_before_tool_call`` and
        ``AfterToolCallEvent`` → ``_on_after_tool_call`` so that Strands
        tool lifecycle events are translated into Brainmass hook events.

        Requirements: 6.2
        """
        # --- Production integration point ---
        if not _STRANDS_HOOKS_AVAILABLE:
            return
        try:
            from strands.hooks import (  # type: ignore[import-untyped]
                AfterToolCallEvent,
                BeforeToolCallEvent,
            )

            registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)  # type: ignore[union-attr]
            registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)  # type: ignore[union-attr]
        except (ImportError, AttributeError) as exc:
            logger.warning("Could not register Strands hook callbacks: %s", exc)

    def _on_before_tool_call(self, event: object) -> None:
        """Translate Strands ``BeforeToolCallEvent`` → ``HookEvent.PRE_TOOL_USE``.

        Requirements: 6.3
        """
        # event.tool_use is a dict with "name" and "input" keys per Strands API
        tool_use = getattr(event, "tool_use", {}) or {}
        tool_name = str(tool_use.get("name", "") if isinstance(tool_use, dict) else "")
        tool_input = tool_use.get("input", {}) if isinstance(tool_use, dict) else {}

        ctx = HookContext(
            session_id=str(getattr(event, "session_id", "")),
            hook_event_name=HookEvent.PRE_TOOL_USE,
            cwd=str(getattr(event, "cwd", ".")),
            session_type="interactive",
            tool_name=tool_name,
            tool_input=tool_input if isinstance(tool_input, dict) else {},
        )
        self.fire(HookEvent.PRE_TOOL_USE, ctx)

    def _on_after_tool_call(self, event: object) -> None:
        """Translate Strands ``AfterToolCallEvent`` → ``HookEvent.POST_TOOL_USE``.

        Requirements: 6.4
        """
        tool_use = getattr(event, "tool_use", {}) or {}
        tool_name = str(tool_use.get("name", "") if isinstance(tool_use, dict) else "")
        result = getattr(event, "result", {}) or {}
        tool_response = ""
        if isinstance(result, dict):
            content = result.get("content", [])
            if isinstance(content, list) and content:
                first = content[0]
                tool_response = str(first.get("text", "") if isinstance(first, dict) else first)
            else:
                tool_response = str(result)
        else:
            tool_response = str(result)

        ctx = HookContext(
            session_id=str(getattr(event, "session_id", "")),
            hook_event_name=HookEvent.POST_TOOL_USE,
            cwd=str(getattr(event, "cwd", ".")),
            session_type="interactive",
            tool_name=tool_name,
            tool_response=tool_response,
        )
        self.fire(HookEvent.POST_TOOL_USE, ctx)

    def pre_tool_use(self, tool_name: str, tool_input: object) -> None:
        """Method-based interface called by Strands SDK before tool execution.

        Requirements: 6.5 (architecture_context.md §5.4)
        """
        ctx = HookContext(
            session_id="",
            hook_event_name=HookEvent.PRE_TOOL_USE,
            cwd=".",
            session_type="interactive",
            tool_name=tool_name,
            tool_input=tool_input if isinstance(tool_input, dict) else {},
        )
        self.fire(HookEvent.PRE_TOOL_USE, ctx)

    def post_tool_use(self, tool_name: str, tool_input: object, tool_response: object) -> None:
        """Method-based interface called by Strands SDK after tool execution.

        Requirements: 6.5 (architecture_context.md §5.4)
        """
        ctx = HookContext(
            session_id="",
            hook_event_name=HookEvent.POST_TOOL_USE,
            cwd=".",
            session_type="interactive",
            tool_name=tool_name,
            tool_response=str(tool_response),
        )
        self.fire(HookEvent.POST_TOOL_USE, ctx)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        event: HookEvent,
        definition: HookDefinition,
        scope: str = "project_local",
        scope_id: str | None = None,
    ) -> None:
        """Register a hook definition for an event.

        Parameters
        ----------
        event:
            One of the 12 ``HookEvent`` values.
        definition:
            The ``HookDefinition`` (matcher + handlers).
        scope:
            One of the 7 scope strings in ``SCOPE_PRECEDENCE``.
        scope_id:
            Optional identifier used for scoped cleanup (e.g. agent name).
        """
        if scope not in SCOPE_PRECEDENCE:
            raise ValueError(
                f"Unknown scope {scope!r}. Must be one of {SCOPE_PRECEDENCE}"
            )
        reg = _Registration(
            event=event,
            definition=definition,
            scope=scope,
            scope_id=scope_id or "",
        )
        self._registrations[event].append(reg)
        if scope_id:
            self._scoped[scope_id].append(reg)

    def register_scoped(
        self,
        hooks: dict[HookEvent, list[HookDefinition]],
        scope_id: str | None = None,
        scope: str = "skill_frontmatter",
    ) -> Callable[[], None]:
        """Register a batch of hooks and return a cleanup function.

        Parameters
        ----------
        hooks:
            Mapping of ``HookEvent`` → list of ``HookDefinition``.
        scope_id:
            Unique identifier for this scope (auto-generated if omitted).
        scope:
            The scope level for all hooks in this batch.

        Returns
        -------
        A callable that, when invoked, removes all hooks registered in
        this batch.
        """
        sid = scope_id or f"scoped-{uuid.uuid4().hex[:8]}"
        for event, defs in hooks.items():
            for defn in defs:
                self.register(event, defn, scope=scope, scope_id=sid)

        def _cleanup() -> None:
            self._remove_scope(sid)

        return _cleanup

    def _remove_scope(self, scope_id: str) -> None:
        """Remove all registrations associated with *scope_id*."""
        regs = self._scoped.pop(scope_id, [])
        reg_set = set(id(r) for r in regs)
        for event in list(self._registrations):
            self._registrations[event] = [
                r for r in self._registrations[event] if id(r) not in reg_set
            ]
            if not self._registrations[event]:
                del self._registrations[event]

    # ------------------------------------------------------------------
    # Enterprise managed mode
    # ------------------------------------------------------------------

    def set_managed_only(self, enabled: bool) -> None:
        """When *enabled*, only ``enterprise_managed`` hooks will fire."""
        self._allow_managed_only = enabled

    # ------------------------------------------------------------------
    # Firing
    # ------------------------------------------------------------------

    def fire(
        self,
        event: HookEvent,
        context: HookContext,
    ) -> HookResult:
        """Fire all matching handlers for *event* synchronously.

        Handlers are executed in 7-scope precedence order (enterprise_managed
        first, user_wide last).  Within the same scope, handlers run in
        registration order.

        Async handlers (``is_async=True``) are scheduled on the running
        event loop (or a new one) and do **not** block this call.

        Returns a merged ``HookResult``.
        """
        ordered = self._resolve_handlers(event, context)
        sync_results: list[HookResult] = []

        for handler in ordered:
            if handler.is_async:
                self._fire_async(handler, event, context)
            else:
                result = self._execute_handler(handler, event, context)
                sync_results.append(result)

        return _merge_results(sync_results) if sync_results else HookResult()

    async def fire_async(
        self,
        event: HookEvent,
        context: HookContext,
    ) -> HookResult:
        """Async variant of :meth:`fire` — awaits async handlers."""
        ordered = self._resolve_handlers(event, context)
        sync_results: list[HookResult] = []
        async_tasks: list[asyncio.Task] = []

        for handler in ordered:
            if handler.is_async:
                task = asyncio.create_task(
                    self._execute_handler_async(handler, event, context)
                )
                async_tasks.append(task)
            else:
                result = self._execute_handler(handler, event, context)
                sync_results.append(result)

        # Async results are fire-and-forget; we don't merge them.
        # They would be logged to OTel in production.
        return _merge_results(sync_results) if sync_results else HookResult()

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def _resolve_handlers(
        self,
        event: HookEvent,
        context: HookContext,
    ) -> list[HookHandler]:
        """Return matching handlers sorted by scope precedence."""
        regs = self._registrations.get(event, [])

        # Filter by managed-only policy
        if self._allow_managed_only:
            regs = [r for r in regs if r.scope == "enterprise_managed"]

        # Filter by matcher
        matched: list[_Registration] = []
        for reg in regs:
            matcher = reg.definition.matcher
            if event in _TOOL_EVENTS:
                if _matches(matcher, context.tool_name):
                    matched.append(reg)
            elif event in _SOURCE_EVENTS:
                if _matches(matcher, context.source):
                    matched.append(reg)
            else:
                # Non-tool, non-source events: matcher applies to all
                if _matches(matcher, None) or matcher is None or matcher == "*":
                    matched.append(reg)

        # Sort by scope precedence — highest-precedence scopes run FIRST
        # so they can deny/block early and their updated_input wins via
        # first-writer-wins in _merge_results.
        # SCOPE_PRECEDENCE[0] = enterprise_managed (highest) → sorted first.
        scope_order = {s: i for i, s in enumerate(SCOPE_PRECEDENCE)}
        matched.sort(key=lambda r: scope_order.get(r.scope, len(SCOPE_PRECEDENCE)))

        # Flatten handlers
        handlers: list[HookHandler] = []
        for reg in matched:
            handlers.extend(reg.definition.hooks)
        return handlers

    # ------------------------------------------------------------------
    # Handler execution
    # ------------------------------------------------------------------

    def _execute_handler(
        self,
        handler: HookHandler,
        event: HookEvent,
        context: HookContext,
    ) -> HookResult:
        """Execute a single handler synchronously.

        In production, this dispatches to the appropriate handler
        implementation (command, prompt, or agent).  For now it invokes
        a pluggable callback if one has been set, otherwise returns a
        default allow result.
        """
        start = perf_counter()
        if self._tracer is not None:
            with self._tracer.trace_hook_execution(
                event=event.value,
                handler_type=handler.type.value,
            ):
                if self._handler_callback is not None:
                    result = self._handler_callback(handler, event, context)
                else:
                    result = HookResult()
        else:
            if self._handler_callback is not None:
                result = self._handler_callback(handler, event, context)
            else:
                result = HookResult()

        if self._hook_logger is not None:
            duration_ms = (perf_counter() - start) * 1000.0
            self._hook_logger.log_hook_execution(
                event=event.value,
                handler_type=handler.type.value,
                matcher="*",
                duration_ms=duration_ms,
                result=result.__dict__,
            )
        return result

    async def _execute_handler_async(
        self,
        handler: HookHandler,
        event: HookEvent,
        context: HookContext,
    ) -> HookResult:
        """Async wrapper around :meth:`_execute_handler`."""
        return self._execute_handler(handler, event, context)

    def _fire_async(
        self,
        handler: HookHandler,
        event: HookEvent,
        context: HookContext,
    ) -> None:
        """Schedule an async handler without blocking.

        In production the result would be logged to OpenTelemetry.
        """
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._execute_handler_async(handler, event, context))
        except RuntimeError:
            # No running loop — run in a new thread-based loop (best effort)
            logger.debug("No running event loop; skipping async hook for %s", event)

    # ------------------------------------------------------------------
    # Handler callback (pluggable for testing / handler integration)
    # ------------------------------------------------------------------

    _handler_callback: Callable[[HookHandler, HookEvent, HookContext], HookResult] | None = None

    def set_handler_callback(
        self,
        callback: Callable[[HookHandler, HookEvent, HookContext], HookResult] | None,
    ) -> None:
        """Set a callback invoked for every handler execution.

        This is the integration point for the command/prompt/agent handler
        implementations (Task 3.2–3.4).  Pass ``None`` to reset.
        """
        self._handler_callback = callback

    def set_observability(
        self,
        tracer: BrainmassTracer | None = None,
        hook_logger: HookLogger | None = None,
    ) -> None:
        """Attach tracer/logger for hook observability."""
        self._tracer = tracer
        self._hook_logger = hook_logger

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_registrations(
        self,
        event: HookEvent | None = None,
    ) -> list[_Registration]:
        """Return current registrations, optionally filtered by event."""
        if event is not None:
            return list(self._registrations.get(event, []))
        result: list[_Registration] = []
        for regs in self._registrations.values():
            result.extend(regs)
        return result
