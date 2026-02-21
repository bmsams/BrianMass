"""Agent handler for hook execution.

Spawns a Strands Agent with Read, Grep, Glob tools for multi-turn
codebase verification.  The agent's response is parsed as a JSON
``HookResult``.

In production the agent call goes through a Strands ``Agent`` configured
with Read/Grep/Glob tools.  For testability the actual invocation is
behind a pluggable callback (``set_agent_callback``), following the same
pattern as ``prompt.py``.

Requirements: 3.2, 15.1, 15.2, 15.3, 15.4
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
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

# Type alias for the pluggable agent invocation callback.
# Receives a context prompt string (built from HookContext + agent_config),
# returns the raw agent response text.
AgentCallback = Callable[[str, dict | None], str]


def _production_agent_callback(context_prompt: str, agent_config: dict | None) -> str:
    """Production agent callback using Strands Agent with file_read/file_write/editor tools.

    Spawns a multi-turn Haiku agent with codebase inspection tools to
    evaluate the hook context and return a JSON HookResult string.

    --- Production integration point ---
    # from strands import Agent
    # from strands.models.bedrock import BedrockModel
    # from strands_tools import file_read, file_write, editor
    # from src.agents._strands_utils import _BEDROCK_MODEL_IDS
    #
    # model_id = _BEDROCK_MODEL_IDS["haiku"]
    # model = BedrockModel(model_id=model_id)
    # agent = Agent(
    #     model=model,
    #     system_prompt=(
    #         "You are a codebase verification agent. "
    #         "Use the provided tools to inspect the codebase and evaluate "
    #         "the hook context. Respond ONLY with a JSON object containing: "
    #         "permissionDecision (allow|deny), decision (continue|block), "
    #         "and optionally reason, updatedInput, additionalContext."
    #     ),
    #     tools=[file_read, file_write, editor],
    # )
    # raw = agent(context_prompt)
    # return str(getattr(raw, 'output', None) or str(raw))

    Args:
        context_prompt: Built from HookContext + agent_config instructions.
        agent_config: Optional dict with 'instructions' and 'tools' keys.

    Returns:
        Raw agent response text (expected to be JSON).

    Raises:
        RuntimeError: If strands package is not installed.
    """
    try:
        from strands import Agent  # type: ignore
        from strands.models.bedrock import BedrockModel  # type: ignore

        from src.agents._strands_utils import _BEDROCK_MODEL_IDS
    except ImportError as exc:
        raise RuntimeError(
            "strands package is required for production AgentHandler. "
            "Install it with: pip install strands-agents"
        ) from exc

    # Import tools individually to handle platform-specific failures gracefully
    tools: list[object] = []
    for tool_name in ("file_read", "file_write", "editor"):
        try:
            import importlib
            mod = importlib.import_module(f"strands_tools.{tool_name}")
            fn = getattr(mod, tool_name, None)
            if fn is not None:
                tools.append(fn)
        except Exception:
            pass

    model_id = _BEDROCK_MODEL_IDS["haiku"]
    model = BedrockModel(model_id=model_id)

    system_prompt = (
        "You are a codebase verification agent. "
        "Use the provided tools to inspect the codebase and evaluate "
        "the hook context. Respond ONLY with a JSON object containing: "
        "permissionDecision (allow|deny), decision (continue|block), "
        "and optionally reason, updatedInput, additionalContext."
    )
    if agent_config and agent_config.get("instructions"):
        system_prompt = agent_config["instructions"]

    agent = Agent(model=model, system_prompt=system_prompt, tools=tools)
    raw = agent(context_prompt)
    return str(getattr(raw, "output", None) or str(raw))


def _build_context_prompt(context: HookContext, agent_config: dict | None) -> str:
    """Build a context prompt from the HookContext and agent_config.

    The prompt includes the serialized hook context and any instructions
    from the agent_config.  This is what gets sent to the agent for
    multi-turn verification.
    """
    data = asdict(context)
    # Convert HookEvent enum to its string value
    if isinstance(data.get("hook_event_name"), HookEvent):
        data["hook_event_name"] = data["hook_event_name"].value
    elif hasattr(data.get("hook_event_name"), "value"):
        data["hook_event_name"] = data["hook_event_name"].value

    parts = []

    # Add agent instructions if provided
    if agent_config:
        instructions = agent_config.get("instructions", "")
        if instructions:
            parts.append(f"Instructions: {instructions}")

        tools = agent_config.get("tools", [])
        if tools:
            parts.append(f"Available tools: {', '.join(tools)}")

    parts.append(f"Hook context:\n{json.dumps(data, default=str, indent=2)}")

    return "\n\n".join(parts)


def _parse_agent_response(response_text: str) -> dict | None:
    """Try to parse the agent response as JSON.

    The agent may return plain JSON, wrap it in markdown code fences,
    or include explanatory text around the JSON.  We attempt to extract
    JSON from all formats.
    """
    text = response_text.strip()
    if not text:
        return None

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code fences (```json ... ``` or ``` ... ```)
    for prefix in ("```json", "```"):
        if prefix in text:
            start = text.index(prefix) + len(prefix)
            end = text.find("```", start)
            if end != -1:
                inner = text[start:end].strip()
                try:
                    return json.loads(inner)
                except (json.JSONDecodeError, ValueError):
                    pass

    # Try to find JSON object in the response text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidate = text[brace_start : brace_end + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _result_from_response(
    parsed: dict | None,
    event: HookEvent,
) -> HookResult:
    """Build a HookResult from the parsed agent response.

    If the agent returned valid JSON with recognised fields, map them
    into a ``HookResult``.  Otherwise fall back to allow/continue for
    non-blocking events, deny/block for blocking events.
    """
    if parsed is None:
        if event in _BLOCKING_EVENTS:
            return HookResult(
                permission_decision="deny",
                permission_decision_reason="Agent response could not be parsed",
                decision="block",
                reason="Agent response could not be parsed",
            )
        return HookResult(permission_decision="allow", decision="continue")

    return HookResult(
        permission_decision=parsed.get("permissionDecision", "allow"),
        permission_decision_reason=parsed.get("permissionDecisionReason"),
        updated_input=parsed.get("updatedInput"),
        additional_context=parsed.get("additionalContext"),
        decision=parsed.get("decision", "continue"),
        reason=parsed.get("reason"),
    )


def _default_fail_result(event: HookEvent) -> HookResult:
    """Return a safe default when the agent call fails entirely."""
    if event in _BLOCKING_EVENTS:
        return HookResult(
            permission_decision="deny",
            permission_decision_reason="Agent handler invocation failed",
            decision="block",
            reason="Agent handler invocation failed",
        )
    return HookResult(permission_decision="allow", decision="continue")


class AgentHandler:
    """Executes agent-type hook handlers via a multi-turn agent invocation.

    The handler's ``agent_config`` dict configures the agent (instructions,
    tools, etc.).  A context prompt is built from the ``HookContext`` and
    agent config, then sent to the agent for multi-turn codebase
    verification using Read, Grep, and Glob tools.

    The actual agent call is behind a pluggable callback so that:
    - Unit tests can supply a deterministic stub.
    - Production code can wire in a real Strands Agent with tools.

    Use ``set_agent_callback`` to inject the real or stub implementation.
    """

    def __init__(self) -> None:
        self._agent_callback: AgentCallback | None = None

    # ------------------------------------------------------------------
    # Pluggable agent invocation
    # ------------------------------------------------------------------

    def set_agent_callback(self, callback: AgentCallback | None) -> None:
        """Set (or clear) the callback used to invoke the agent.

        Args:
            callback: A callable that takes a context prompt string and
                an optional agent_config dict, and returns the raw agent
                response text.  Pass ``None`` to reset.
        """
        self._agent_callback = callback

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        handler: HookHandler,
        context: HookContext,
    ) -> HookResult:
        """Run the agent handler synchronously.

        1. Validate handler type.
        2. Build a context prompt from HookContext + agent_config.
        3. Invoke the agent callback.
        4. Parse the response as JSON into a HookResult.
        5. On any error, return a safe default (allow for non-blocking,
           deny for blocking events).
        """
        if handler.type != HookHandlerType.AGENT:
            raise ValueError(
                f"AgentHandler received non-agent handler type: {handler.type}"
            )

        event = context.hook_event_name

        # Build the context prompt
        context_prompt = _build_context_prompt(context, handler.agent_config)

        # Invoke the agent
        if self._agent_callback is None:
            logger.warning(
                "No agent callback configured for AgentHandler; "
                "returning default allow"
            )
            return HookResult(permission_decision="allow", decision="continue")

        try:
            response_text = self._agent_callback(context_prompt, handler.agent_config)
        except Exception as exc:
            logger.error("Agent handler invocation failed: %s", exc)
            return _default_fail_result(event)

        # Parse and return
        parsed = _parse_agent_response(response_text)
        return _result_from_response(parsed, event)
