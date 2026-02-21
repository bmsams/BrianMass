"""API type definitions for the Anthropic Messages API.

Defines request/response dataclasses, a fluent MessageRequestBuilder, and
a ResponseParser for handling API interactions including streaming.

Requirements: 28.1, 28.2, 28.3, 28.4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

# ---------------------------------------------------------------------------
# Content Block types (Requirement 28.2)
# ---------------------------------------------------------------------------

@dataclass
class TextBlock:
    """A text content block."""
    type: str = "text"
    text: str = ""


@dataclass
class ImageBlock:
    """An image content block with base64 or URL source."""
    type: str = "image"
    source: dict = field(default_factory=dict)
    # source: {type: "base64"|"url", media_type: str, data: str}


@dataclass
class ToolUseBlock:
    """A tool_use content block representing an assistant tool call."""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class ToolResultBlock:
    """A tool_result content block representing a tool execution result."""
    type: str = "tool_result"
    tool_use_id: str = ""
    content: str | list[dict] = ""
    is_error: bool = False


@dataclass
class ThinkingBlock:
    """An extended thinking content block."""
    type: str = "thinking"
    thinking: str = ""


@dataclass
class DocumentBlock:
    """A document content block for PDF or other document types."""
    type: str = "document"
    source: dict = field(default_factory=dict)
    # source: {type: "base64"|"url", media_type: str, data: str}


# Union of all content block types
ContentBlock = Union[
    TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock, DocumentBlock
]


# ---------------------------------------------------------------------------
# Tool Definition (Requirement 28.1)
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """A tool definition for the tools parameter of the Messages API."""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialise to API-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ---------------------------------------------------------------------------
# System Block (Requirement 28.1)
# ---------------------------------------------------------------------------

@dataclass
class SystemBlock:
    """A system prompt block with optional cache control."""
    type: str = "text"
    text: str = ""
    cache_control: dict | None = None

    def to_dict(self) -> dict:
        """Serialise to API-compatible dict."""
        result: dict[str, Any] = {"type": self.type, "text": self.text}
        if self.cache_control is not None:
            result["cache_control"] = self.cache_control
        return result


# ---------------------------------------------------------------------------
# Usage (Requirement 28.3)
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    """Token usage metrics from an API response."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# ---------------------------------------------------------------------------
# Message Request (Requirement 28.1)
# ---------------------------------------------------------------------------

@dataclass
class MessageRequest:
    """A complete request payload for the Anthropic Messages API.

    Fields:
    - model: Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    - max_tokens: Maximum output tokens (1-128000).
    - messages: Conversation messages with ``role`` and ``content``.
    - system: System prompt -- a plain string or list of SystemBlocks.
    - tools: Tool definitions available to the model.
    - tool_choice: Tool selection strategy (``"auto"`` | ``"any"`` | ``{"type": "tool", "name": ...}``).
    - temperature: Sampling temperature (0.0-1.0).
    - thinking: Extended thinking config (``{"type": "enabled", "budget_tokens": N}``).
    - stream: Whether to stream the response.
    """
    model: str = ""
    max_tokens: int = 4096
    messages: list[dict] = field(default_factory=list)
    system: str | list[SystemBlock] | None = None
    tools: list[ToolDefinition] = field(default_factory=list)
    tool_choice: str | dict | None = None
    temperature: float | None = None
    thinking: dict | None = None
    stream: bool = False

    def to_dict(self) -> dict:
        """Serialise to a dict suitable for JSON encoding and API submission."""
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self.messages,
        }

        if self.system is not None:
            if isinstance(self.system, str):
                payload["system"] = self.system
            else:
                payload["system"] = [block.to_dict() for block in self.system]

        if self.tools:
            payload["tools"] = [tool.to_dict() for tool in self.tools]

        if self.tool_choice is not None:
            payload["tool_choice"] = self.tool_choice

        if self.temperature is not None:
            payload["temperature"] = self.temperature

        if self.thinking is not None:
            payload["thinking"] = self.thinking

        if self.stream:
            payload["stream"] = True

        return payload


# ---------------------------------------------------------------------------
# Message Response (Requirement 28.3)
# ---------------------------------------------------------------------------

@dataclass
class MessageResponse:
    """A parsed response from the Anthropic Messages API."""
    id: str = ""
    type: str = "message"
    role: str = "assistant"
    content: list[ContentBlock] = field(default_factory=list)
    model: str = ""
    stop_reason: str | None = None
    usage: Usage = field(default_factory=Usage)


# ---------------------------------------------------------------------------
# Stream Event (Requirement 28.4)
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """A single event from a streaming response.

    Event types include:
    - ``message_start``: Contains the initial message metadata.
    - ``content_block_start``: Start of a new content block.
    - ``content_block_delta``: Incremental content update.
    - ``content_block_stop``: End of a content block.
    - ``message_delta``: Final message-level updates (stop_reason, usage).
    - ``message_stop``: Signals the end of the stream.
    """
    type: str = ""
    index: int | None = None
    delta: dict | None = None
    message: dict | None = None
    content_block: dict | None = None
    usage: Usage | None = None

    @property
    def delta_text(self) -> str:
        """Extract text from a content_block_delta event.

        Returns the ``text`` field from the delta dict, or an empty string
        if not present or not a text delta.
        """
        if self.delta and isinstance(self.delta, dict):
            return self.delta.get("text", "")
        return ""


# ---------------------------------------------------------------------------
# MessageRequestBuilder -- fluent API (Requirement 28.1)
# ---------------------------------------------------------------------------

class MessageRequestBuilder:
    """Fluent builder for constructing validated API request dicts.

    Supports two API styles:

    1. **set_ prefix** (used by tests)::

        request = (
            MessageRequestBuilder()
            .set_model("claude-sonnet-4-5")
            .set_max_tokens(4096)
            .add_message("user", "Hello")
            .set_system("You are helpful.")
            .set_temperature(0.7)
            .set_thinking(budget_tokens=10000)
            .build()
        )

    2. **Short-hand fluent** (convenience)::

        request = (
            MessageRequestBuilder()
            .model("claude-sonnet-4-5")
            .max_tokens(4096)
            .messages([{"role": "user", "content": "Hello"}])
            .system("You are helpful.")
            .temperature(0.7)
            .thinking(10000)
            .build()
        )

    ``build()`` returns a plain ``dict`` ready for JSON serialisation.

    Validation on ``build()``:
    - ``model`` is required (non-empty).
    - ``max_tokens`` must be in [1, 128000].
    - ``temperature`` (if set) must be in [0.0, 1.0].
    """

    def __init__(self) -> None:
        self._model: str = ""
        self._max_tokens: int = 4096
        self._messages: list[dict] = []
        self._system: str | list[SystemBlock] | None = None
        self._tools: list[ToolDefinition] = []
        self._tool_choice: str | dict | None = None
        self._temperature: float | None = None
        self._thinking: dict | None = None
        self._stream: bool = False
        self._cache_controls: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # set_ prefix API (matches existing tests)
    # ------------------------------------------------------------------

    def set_model(self, name: str) -> MessageRequestBuilder:
        """Set the model identifier."""
        self._model = name
        return self

    def set_max_tokens(self, n: int) -> MessageRequestBuilder:
        """Set the maximum output tokens."""
        self._max_tokens = n
        return self

    def set_temperature(self, t: float) -> MessageRequestBuilder:
        """Set the sampling temperature."""
        self._temperature = t
        return self

    def set_system(self, sys: str | list[SystemBlock]) -> MessageRequestBuilder:
        """Set the system prompt (string or list of SystemBlocks)."""
        self._system = sys
        return self

    def set_thinking(self, budget_tokens: int) -> MessageRequestBuilder:
        """Enable extended thinking with the given token budget."""
        self._thinking = {"type": "enabled", "budget_tokens": budget_tokens}
        return self

    def add_message(
        self,
        role: str,
        content: str | list[dict],
        cache_control: dict | None = None,
    ) -> MessageRequestBuilder:
        """Append a message to the conversation.

        When *cache_control* is provided, the content is wrapped in a
        content block list with the ``cache_control`` annotation attached.
        """
        msg: dict[str, Any] = {"role": role}
        if cache_control is not None:
            # Wrap content into a content block with cache_control metadata
            if isinstance(content, str):
                msg["content"] = [
                    {"type": "text", "text": content, "cache_control": cache_control}
                ]
            else:
                msg["content"] = content
        else:
            msg["content"] = content
        self._messages.append(msg)
        return self

    # ------------------------------------------------------------------
    # Short-hand fluent API (convenience)
    # ------------------------------------------------------------------

    def model(self, name: str) -> MessageRequestBuilder:
        """Set the model identifier (short-hand for ``set_model``)."""
        return self.set_model(name)

    def max_tokens(self, n: int) -> MessageRequestBuilder:
        """Set the maximum output tokens (short-hand for ``set_max_tokens``)."""
        return self.set_max_tokens(n)

    def messages(self, msgs: list[dict]) -> MessageRequestBuilder:
        """Set the conversation messages (replaces any existing)."""
        self._messages = msgs
        return self

    def system(self, sys: str | list[SystemBlock]) -> MessageRequestBuilder:
        """Set the system prompt (short-hand for ``set_system``)."""
        return self.set_system(sys)

    def tools(self, tool_defs: list[ToolDefinition]) -> MessageRequestBuilder:
        """Set the available tool definitions."""
        self._tools = tool_defs
        return self

    def tool_choice(self, choice: str | dict) -> MessageRequestBuilder:
        """Set the tool choice strategy."""
        self._tool_choice = choice
        return self

    def temperature(self, t: float) -> MessageRequestBuilder:
        """Set the sampling temperature (short-hand for ``set_temperature``)."""
        return self.set_temperature(t)

    def thinking(self, budget_tokens: int) -> MessageRequestBuilder:
        """Enable extended thinking (short-hand for ``set_thinking``)."""
        return self.set_thinking(budget_tokens)

    def stream(self, enabled: bool = True) -> MessageRequestBuilder:
        """Enable or disable streaming."""
        self._stream = enabled
        return self

    def with_cache_control(
        self, block_index: int, duration: str = "ephemeral"
    ) -> MessageRequestBuilder:
        """Inject cache control on a system block at *block_index*.

        The ``cache_control`` dict ``{"type": duration}`` will be applied
        when ``build()`` is called, provided the system prompt is a list
        of SystemBlocks and the index is valid.
        """
        self._cache_controls[block_index] = {"type": duration}
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """Validate and construct an API request dict.

        Returns:
            A plain dict suitable for JSON serialisation and API submission.

        Raises:
            ValueError: If validation fails.
        """
        errors: list[str] = []

        if not self._model:
            errors.append("'model' is required and must be non-empty")

        if not (1 <= self._max_tokens <= 128_000):
            errors.append(
                f"'max_tokens' must be between 1 and 128000, got {self._max_tokens}"
            )

        if self._temperature is not None:
            if not (0.0 <= self._temperature <= 1.0):
                errors.append(
                    f"'temperature' must be between 0.0 and 1.0, got {self._temperature}"
                )

        if errors:
            raise ValueError(
                f"MessageRequest validation failed: {'; '.join(errors)}"
            )

        # Apply cache controls to system blocks
        system_value = self._system
        if isinstance(system_value, list) and self._cache_controls:
            system_value = list(system_value)  # shallow copy
            for idx, cc in self._cache_controls.items():
                if 0 <= idx < len(system_value):
                    original = system_value[idx]
                    system_value[idx] = SystemBlock(
                        type=original.type,
                        text=original.text,
                        cache_control=cc,
                    )

        # Build the dict
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": self._messages,
        }

        if system_value is not None:
            if isinstance(system_value, str):
                payload["system"] = system_value
            else:
                payload["system"] = [block.to_dict() for block in system_value]

        if self._tools:
            payload["tools"] = [tool.to_dict() for tool in self._tools]

        if self._tool_choice is not None:
            payload["tool_choice"] = self._tool_choice

        if self._temperature is not None:
            payload["temperature"] = self._temperature

        if self._thinking is not None:
            payload["thinking"] = self._thinking

        if self._stream:
            payload["stream"] = True

        return payload

    def build_request(self) -> MessageRequest:
        """Validate and construct a :class:`MessageRequest` dataclass.

        An alternative to ``build()`` that returns a typed dataclass
        instead of a plain dict.
        """
        # Validate via build() first (raises on error)
        self.build()

        # Apply cache controls to system blocks
        system_value = self._system
        if isinstance(system_value, list) and self._cache_controls:
            system_value = list(system_value)
            for idx, cc in self._cache_controls.items():
                if 0 <= idx < len(system_value):
                    original = system_value[idx]
                    system_value[idx] = SystemBlock(
                        type=original.type,
                        text=original.text,
                        cache_control=cc,
                    )

        return MessageRequest(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=self._messages,
            system=system_value,
            tools=self._tools,
            tool_choice=self._tool_choice,
            temperature=self._temperature,
            thinking=self._thinking,
            stream=self._stream,
        )


# ---------------------------------------------------------------------------
# ResponseParser (Requirement 28.3, 28.4)
# ---------------------------------------------------------------------------

class ResponseParser:
    """Parses raw API response dicts into typed dataclasses."""

    @staticmethod
    def parse_usage(raw: dict) -> Usage:
        """Parse a usage dict into a :class:`Usage` instance.

        Handles both the top-level ``usage`` object and the ``message_delta``
        usage that appears at stream end.
        """
        return Usage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cache_creation_input_tokens=raw.get("cache_creation_input_tokens", 0),
            cache_read_input_tokens=raw.get("cache_read_input_tokens", 0),
        )

    @staticmethod
    def parse_content_block(raw: dict) -> ContentBlock:
        """Parse a single raw content block dict into a typed ContentBlock."""
        block_type = raw.get("type", "text")

        if block_type == "text":
            return TextBlock(text=raw.get("text", ""))
        if block_type == "image":
            return ImageBlock(source=raw.get("source", {}))
        if block_type == "tool_use":
            return ToolUseBlock(
                id=raw.get("id", ""),
                name=raw.get("name", ""),
                input=raw.get("input", {}),
            )
        if block_type == "tool_result":
            return ToolResultBlock(
                tool_use_id=raw.get("tool_use_id", ""),
                content=raw.get("content", ""),
                is_error=raw.get("is_error", False),
            )
        if block_type == "thinking":
            return ThinkingBlock(thinking=raw.get("thinking", ""))
        if block_type == "document":
            return DocumentBlock(source=raw.get("source", {}))

        # Unknown block type -- return as TextBlock with a warning marker
        return TextBlock(text=raw.get("text", f"[unknown block type: {block_type}]"))

    @classmethod
    def parse_content_blocks(cls, raw_list: list[dict]) -> list[ContentBlock]:
        """Parse a list of raw content block dicts."""
        return [cls.parse_content_block(block) for block in raw_list]

    @classmethod
    def parse(cls, raw: dict) -> MessageResponse:
        """Parse a complete API response dict into a :class:`MessageResponse`.

        This is the primary entry point for parsing non-streaming responses.

        Handles the standard response format::

            {
                "id": "msg_...",
                "type": "message",
                "role": "assistant",
                "content": [...],
                "model": "claude-...",
                "stop_reason": "end_turn",
                "usage": {...}
            }
        """
        content_blocks: list[ContentBlock] = []
        raw_content = raw.get("content", [])
        if isinstance(raw_content, list):
            content_blocks = cls.parse_content_blocks(raw_content)

        usage = cls.parse_usage(raw.get("usage", {}))

        return MessageResponse(
            id=raw.get("id", ""),
            type=raw.get("type", "message"),
            role=raw.get("role", "assistant"),
            content=content_blocks,
            model=raw.get("model", ""),
            stop_reason=raw.get("stop_reason"),
            usage=usage,
        )

    # Alias for backward compatibility
    parse_response = parse

    @classmethod
    def parse_stream_event(cls, raw: dict) -> StreamEvent:
        """Parse a single streaming event dict into a :class:`StreamEvent`.

        Handles all event types:
        - ``message_start``: Contains ``message`` with initial metadata.
        - ``content_block_start``: Contains ``content_block`` and ``index``.
        - ``content_block_delta``: Contains ``delta`` and ``index``.
        - ``content_block_stop``: Contains ``index``.
        - ``message_delta``: Contains ``delta`` (stop_reason) and ``usage``.
        - ``message_stop``: Terminal event.
        """
        event_type = raw.get("type", "")

        usage: Usage | None = None
        raw_usage = raw.get("usage")
        if raw_usage is not None:
            usage = cls.parse_usage(raw_usage)

        return StreamEvent(
            type=event_type,
            index=raw.get("index"),
            delta=raw.get("delta"),
            message=raw.get("message"),
            content_block=raw.get("content_block"),
            usage=usage,
        )
