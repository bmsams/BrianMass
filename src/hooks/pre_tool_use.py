"""PreToolUse processor — applies allow/deny/modify decisions.

Fires PreToolUse hooks via the :class:`BrainmassHookEngine` and interprets
the merged :class:`HookResult` into one of three modes:

1. **Allow** — tool executes with original input.
2. **Deny** — tool does NOT execute; a reason is returned.
3. **Modify** — tool executes with ``updated_input``; transparent to model.

Requirement 3.4.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.hooks.hook_engine import BrainmassHookEngine
from src.types.core import HookContext, HookEvent, HookResult


@dataclass
class PreToolUseResult:
    """Outcome of processing PreToolUse hooks for a single tool call."""

    allowed: bool
    """Whether the tool is permitted to execute."""

    modified_input: dict | None = None
    """If hooks provided ``updated_input``, the replacement input dict."""

    additional_context: str | None = None
    """Extra context string injected by hooks (transparent to the model)."""

    deny_reason: str | None = None
    """Human-readable reason when the tool call is denied."""


def process_pre_tool_use(
    engine: BrainmassHookEngine,
    tool_name: str,
    tool_input: dict,
    *,
    session_id: str = "",
    cwd: str = ".",
    session_type: str = "interactive",
) -> PreToolUseResult:
    """Fire PreToolUse hooks and return a processed result.

    Parameters
    ----------
    engine:
        The hook engine instance that manages registered hooks.
    tool_name:
        Name of the tool about to be invoked (e.g. ``"bash"``).
    tool_input:
        The original input dict for the tool call.
    session_id:
        Current session identifier (forwarded to hook context).
    cwd:
        Working directory (forwarded to hook context).
    session_type:
        ``"interactive"`` or ``"headless"`` (forwarded to hook context).

    Returns
    -------
    PreToolUseResult
        Contains ``allowed``, ``modified_input``, ``additional_context``,
        and ``deny_reason`` so the caller can decide how to proceed.
    """
    context = HookContext(
        session_id=session_id,
        hook_event_name=HookEvent.PRE_TOOL_USE,
        cwd=cwd,
        session_type=session_type,
        tool_name=tool_name,
        tool_input=tool_input,
    )

    hook_result: HookResult = engine.fire(HookEvent.PRE_TOOL_USE, context)

    return _interpret_result(hook_result)


def _interpret_result(hook_result: HookResult) -> PreToolUseResult:
    """Convert a raw :class:`HookResult` into a :class:`PreToolUseResult`.

    Decision logic:
    - ``permission_decision == "deny"`` → denied with reason.
    - ``updated_input`` is present → allowed with modified input.
    - Otherwise → allowed with original input (no modification).
    """
    # Deny takes absolute precedence.
    if hook_result.permission_decision == "deny":
        return PreToolUseResult(
            allowed=False,
            deny_reason=(
                hook_result.permission_decision_reason
                or hook_result.reason
                or "Denied by PreToolUse hook"
            ),
        )

    # Allow — possibly with modified input.
    return PreToolUseResult(
        allowed=True,
        modified_input=hook_result.updated_input,
        additional_context=hook_result.additional_context,
    )
