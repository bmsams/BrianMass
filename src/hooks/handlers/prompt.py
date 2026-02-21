"""Prompt handler for hook execution.

Sends a prompt to the Haiku model (via Strands Agent) with ``$ARGUMENTS``
replaced by the serialized HookContext.  The model's response is parsed
as a JSON ``HookResult``.

In production the model call goes through a Strands ``Agent`` configured
with the Haiku model.  For testability the actual invocation is behind a
pluggable callback (``set_model_callback``), similar to how
``hook_engine.py`` uses ``set_handler_callback``.

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

# Type alias for the pluggable model invocation callback.
# Receives the fully-resolved prompt string, returns the raw model
# response text.
ModelCallback = Callable[[str], str]


def _production_model_callback(prompt: str) -> str:
    """Production model callback using Strands Agent with Haiku BedrockModel.

    Invokes a single-turn Haiku agent to evaluate the hook prompt and
    return a JSON HookResult string.

    --- Production integration point ---
    # from strands import Agent
    # from strands.models.bedrock import BedrockModel
    # from src.agents._strands_utils import _BEDROCK_MODEL_IDS
    #
    # model_id = _BEDROCK_MODEL_IDS["haiku"]
    # model = BedrockModel(model_id=model_id)
    # agent = Agent(model=model, system_prompt=(
    #     "You are a security policy evaluator. "
    #     "Respond ONLY with a JSON object containing: "
    #     "permissionDecision (allow|deny), decision (continue|block), "
    #     "and optionally reason, updatedInput, additionalContext."
    # ))
    # raw = agent(prompt)
    # return str(getattr(raw, 'output', None) or str(raw))

    Args:
        prompt: The fully-resolved hook prompt with $ARGUMENTS substituted.

    Returns:
        Raw model response text (expected to be JSON).

    Raises:
        RuntimeError: If strands package is not installed.
    """
    try:
        from strands import Agent  # type: ignore
        from strands.models.bedrock import BedrockModel  # type: ignore

        from src.agents._strands_utils import _BEDROCK_MODEL_IDS
    except ImportError as exc:
        raise RuntimeError(
            "strands package is required for production PromptHandler. "
            "Install it with: pip install strands-agents"
        ) from exc

    model_id = _BEDROCK_MODEL_IDS["haiku"]
    model = BedrockModel(model_id=model_id)
    agent = Agent(
        model=model,
        system_prompt=(
            "You are a security policy evaluator. "
            "Respond ONLY with a JSON object containing: "
            "permissionDecision (allow|deny), decision (continue|block), "
            "and optionally reason, updatedInput, additionalContext."
        ),
    )
    raw = agent(prompt)
    return str(getattr(raw, "output", None) or str(raw))


def _serialize_context(context: HookContext) -> str:
    """Serialize a HookContext to a JSON string for prompt substitution."""
    data = asdict(context)
    # Convert HookEvent enum to its string value
    if isinstance(data.get("hook_event_name"), HookEvent):
        data["hook_event_name"] = data["hook_event_name"].value
    elif hasattr(data.get("hook_event_name"), "value"):
        data["hook_event_name"] = data["hook_event_name"].value
    return json.dumps(data, default=str)


def _substitute_arguments(prompt: str, context: HookContext) -> str:
    """Replace ``$ARGUMENTS`` in the prompt template with the serialized context."""
    serialized = _serialize_context(context)
    return prompt.replace("$ARGUMENTS", serialized)


def _parse_model_response(response_text: str) -> dict | None:
    """Try to parse the model response as JSON.

    The model may return plain JSON or wrap it in markdown code fences.
    We attempt to extract JSON from both formats.
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

    return None


def _result_from_response(
    parsed: dict | None,
    event: HookEvent,
) -> HookResult:
    """Build a HookResult from the parsed model response.

    If the model returned valid JSON with recognised fields, map them
    into a ``HookResult``.  Otherwise fall back to allow/continue.
    """
    if parsed is None:
        # Could not parse — fail-open for non-blocking, fail-closed for blocking
        if event in _BLOCKING_EVENTS:
            return HookResult(
                permission_decision="deny",
                permission_decision_reason="Model response could not be parsed",
                decision="block",
                reason="Model response could not be parsed",
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
    """Return a safe default when the model call fails entirely."""
    if event in _BLOCKING_EVENTS:
        return HookResult(
            permission_decision="deny",
            permission_decision_reason="Prompt handler model call failed",
            decision="block",
            reason="Prompt handler model call failed",
        )
    return HookResult(permission_decision="allow", decision="continue")


class PromptHandler:
    """Executes prompt-type hook handlers via a model invocation.

    The handler's ``prompt`` field is a template string.  ``$ARGUMENTS``
    is replaced with the serialized ``HookContext``.  The resulting
    prompt is sent to the Haiku model for single-turn semantic
    evaluation.  The model's response is parsed as JSON into a
    ``HookResult``.

    The actual model call is behind a pluggable callback so that:
    - Unit tests can supply a deterministic stub.
    - Production code can wire in a real Strands Agent / Bedrock call.

    Use ``set_model_callback`` to inject the real or stub implementation.
    """

    def __init__(self) -> None:
        self._model_callback: ModelCallback | None = None

    # ------------------------------------------------------------------
    # Pluggable model invocation
    # ------------------------------------------------------------------

    def set_model_callback(self, callback: ModelCallback | None) -> None:
        """Set (or clear) the callback used to invoke the model.

        Args:
            callback: A callable that takes a prompt string and returns
                the raw model response text.  Pass ``None`` to reset.
        """
        self._model_callback = callback

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        handler: HookHandler,
        context: HookContext,
    ) -> HookResult:
        """Run the prompt handler synchronously.

        1. Validate handler type and prompt presence.
        2. Replace ``$ARGUMENTS`` in the prompt with the serialized context.
        3. Invoke the model callback (or stub).
        4. Parse the response as JSON into a ``HookResult``.
        5. On any error, return a safe default (allow for non-blocking,
           deny for blocking events).
        """
        if handler.type != HookHandlerType.PROMPT:
            raise ValueError(
                f"PromptHandler received non-prompt handler type: {handler.type}"
            )

        event = context.hook_event_name

        if not handler.prompt:
            logger.warning("Prompt handler has no prompt configured")
            return HookResult(permission_decision="allow", decision="continue")

        # Build the final prompt with $ARGUMENTS replaced
        resolved_prompt = _substitute_arguments(handler.prompt, context)

        # Invoke the model
        if self._model_callback is None:
            logger.warning(
                "No model callback configured for PromptHandler; "
                "returning default allow"
            )
            return HookResult(permission_decision="allow", decision="continue")

        try:
            response_text = self._model_callback(resolved_prompt)
        except Exception as exc:
            logger.error("Prompt handler model call failed: %s", exc)
            return _default_fail_result(event)

        # Parse and return
        parsed = _parse_model_response(response_text)
        return _result_from_response(parsed, event)
