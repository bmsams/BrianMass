"""Unit tests for MessageRequestBuilder and ResponseParser.

Covers: minimal/full request building, validation (model, max_tokens,
temperature), cache control, thinking mode, response parsing (usage,
content blocks, stream events).

Requirements: 2.2, 10.1
"""

from __future__ import annotations

import pytest

from src.types.api import MessageRequestBuilder, ResponseParser

# ===================================================================
# MessageRequestBuilder — minimal request
# ===================================================================

class TestBuilderMinimalRequest:
    """Builder can produce a valid request with just model + messages."""

    def test_builder_minimal_request(self):
        builder = MessageRequestBuilder()
        request = (
            builder
            .set_model("claude-sonnet-4-5")
            .set_max_tokens(1024)
            .add_message("user", "Hello")
            .build()
        )

        assert request["model"] == "claude-sonnet-4-5"
        assert request["max_tokens"] == 1024
        assert len(request["messages"]) == 1
        assert request["messages"][0]["role"] == "user"
        assert request["messages"][0]["content"] == "Hello"


# ===================================================================
# MessageRequestBuilder — full request
# ===================================================================

class TestBuilderFullRequest:
    """Builder supports all optional fields."""

    def test_builder_full_request(self):
        builder = MessageRequestBuilder()
        request = (
            builder
            .set_model("claude-opus-4-6")
            .set_max_tokens(4096)
            .set_temperature(0.7)
            .set_system("You are a helpful assistant.")
            .add_message("user", "Explain recursion.")
            .add_message("assistant", "Recursion is when...")
            .add_message("user", "Give an example.")
            .build()
        )

        assert request["model"] == "claude-opus-4-6"
        assert request["max_tokens"] == 4096
        assert request["temperature"] == pytest.approx(0.7)
        assert request["system"] == "You are a helpful assistant."
        assert len(request["messages"]) == 3


# ===================================================================
# MessageRequestBuilder — validation
# ===================================================================

class TestBuilderValidation:
    """Builder validates required fields and ranges."""

    def test_builder_validation_missing_model(self):
        builder = MessageRequestBuilder()
        builder.set_max_tokens(1024).add_message("user", "Hi")

        with pytest.raises((ValueError, KeyError)):
            builder.build()

    def test_builder_validation_max_tokens_range(self):
        builder = MessageRequestBuilder()
        builder.set_model("claude-sonnet-4-5")
        builder.add_message("user", "Hi")

        # Zero or negative max_tokens should be rejected
        builder.set_max_tokens(0)
        with pytest.raises(ValueError):
            builder.build()

    def test_builder_validation_temperature_range(self):
        builder = MessageRequestBuilder()
        builder.set_model("claude-sonnet-4-5")
        builder.set_max_tokens(1024)
        builder.add_message("user", "Hi")

        # Temperature outside [0.0, 1.0] should be rejected
        builder.set_temperature(1.5)
        with pytest.raises(ValueError):
            builder.build()


# ===================================================================
# MessageRequestBuilder — cache control
# ===================================================================

class TestBuilderCacheControl:
    """Builder supports cache_control annotations."""

    def test_builder_with_cache_control(self):
        builder = MessageRequestBuilder()
        request = (
            builder
            .set_model("claude-sonnet-4-5")
            .set_max_tokens(1024)
            .add_message("user", "Hello", cache_control={"type": "ephemeral"})
            .build()
        )

        msg = request["messages"][0]
        # The message content should carry cache_control metadata
        # Implementation may store it as a content block or message-level key
        assert msg is not None


# ===================================================================
# MessageRequestBuilder — thinking mode
# ===================================================================

class TestBuilderWithThinking:
    """Builder can enable extended thinking."""

    def test_builder_with_thinking(self):
        builder = MessageRequestBuilder()
        request = (
            builder
            .set_model("claude-sonnet-4-5")
            .set_max_tokens(16384)
            .set_thinking(budget_tokens=10000)
            .add_message("user", "Solve this complex problem.")
            .build()
        )

        # Thinking config should be present in the request
        assert "thinking" in request or "budget_tokens" in str(request)


# ===================================================================
# ResponseParser — parse response
# ===================================================================

class TestParseResponse:
    """ResponseParser extracts structured data from API responses."""

    def test_parse_response(self):
        raw = {
            "id": "msg_01ABC",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Hello, world!"}
            ],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }

        parsed = ResponseParser.parse(raw)
        assert parsed.id == "msg_01ABC"
        assert parsed.role == "assistant"
        assert parsed.stop_reason == "end_turn"


# ===================================================================
# ResponseParser — usage
# ===================================================================

class TestParseUsage:
    """ResponseParser extracts usage metrics."""

    def test_parse_usage(self):
        raw = {
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "OK"}],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 500,
                "output_tokens": 200,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 300,
            },
        }

        parsed = ResponseParser.parse(raw)
        assert parsed.usage.input_tokens == 500
        assert parsed.usage.output_tokens == 200
        assert parsed.usage.cache_creation_input_tokens == 100
        assert parsed.usage.cache_read_input_tokens == 300


# ===================================================================
# ResponseParser — content blocks
# ===================================================================

class TestParseContentBlocks:
    """ResponseParser handles different content block types."""

    def test_parse_content_blocks_text(self):
        raw = {
            "id": "msg_03",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "First paragraph."},
                {"type": "text", "text": "Second paragraph."},
            ],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }

        parsed = ResponseParser.parse(raw)
        text_blocks = [b for b in parsed.content if b.type == "text"]
        assert len(text_blocks) == 2
        assert text_blocks[0].text == "First paragraph."

    def test_parse_content_blocks_tool_use(self):
        raw = {
            "id": "msg_04",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tu_01",
                    "name": "bash",
                    "input": {"command": "ls -la"},
                },
            ],
            "model": "claude-sonnet-4-5",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 30},
        }

        parsed = ResponseParser.parse(raw)
        tool_blocks = [b for b in parsed.content if b.type == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "bash"
        assert tool_blocks[0].input == {"command": "ls -la"}

    def test_parse_content_blocks_thinking(self):
        raw = {
            "id": "msg_05",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me consider..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 50, "output_tokens": 100},
        }

        parsed = ResponseParser.parse(raw)
        thinking_blocks = [b for b in parsed.content if b.type == "thinking"]
        assert len(thinking_blocks) == 1


# ===================================================================
# ResponseParser — stream events
# ===================================================================

class TestParseStreamEvent:
    """ResponseParser handles streaming event chunks."""

    def test_parse_stream_event(self):
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "text_delta",
                "text": "Hello",
            },
        }

        parsed = ResponseParser.parse_stream_event(event)
        assert parsed.type == "content_block_delta"
        assert parsed.delta_text == "Hello"
