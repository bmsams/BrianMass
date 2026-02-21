"""Property-based tests for API request schema conformance.

Property 22: API request schema conformance
- Any valid MessageRequest built via builder should have model, max_tokens,
  and messages.
- max_tokens is always 1-128000.
- temperature is always 0.0-1.0.
- Parsed response always contains usage metrics.

Validates: Requirements 2.2, 10.1
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.types.api import MessageRequestBuilder, ResponseParser

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_model = st.sampled_from([
    "claude-opus-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
])

_max_tokens = st.integers(min_value=1, max_value=128000)

_temperature = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_message_role = st.sampled_from(["user", "assistant"])
_message_content = st.text(min_size=1, max_size=200)
_message = st.fixed_dictionaries({
    "role": _message_role,
    "content": _message_content,
})
_messages = st.lists(_message, min_size=1, max_size=5)

_input_tokens = st.integers(min_value=0, max_value=5_000_000)
_output_tokens = st.integers(min_value=0, max_value=5_000_000)
_cache_creation = st.integers(min_value=0, max_value=1_000_000)
_cache_read = st.integers(min_value=0, max_value=1_000_000)


# ---------------------------------------------------------------------------
# Property 22a: Valid request has model, max_tokens, messages
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    model=_model,
    max_tokens=_max_tokens,
    messages=_messages,
)
def test_property_22a_valid_request_has_required_fields(
    model: str,
    max_tokens: int,
    messages: list[dict],
) -> None:
    """Any valid MessageRequest built via builder should have model,
    max_tokens, and messages.

    Feature: claude-code-v3-enterprise, Property 22a.
    Validates: Requirements 2.2
    """
    builder = MessageRequestBuilder()
    builder.set_model(model)
    builder.set_max_tokens(max_tokens)
    for msg in messages:
        builder.add_message(msg["role"], msg["content"])

    request = builder.build()

    assert "model" in request
    assert request["model"] == model
    assert "max_tokens" in request
    assert request["max_tokens"] == max_tokens
    assert "messages" in request
    assert len(request["messages"]) == len(messages)


# ---------------------------------------------------------------------------
# Property 22b: max_tokens is always 1-128000
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=50)
@given(
    model=_model,
    max_tokens=_max_tokens,
)
def test_property_22b_max_tokens_in_valid_range(
    model: str,
    max_tokens: int,
) -> None:
    """max_tokens is always in [1, 128000] for any valid built request.

    Feature: claude-code-v3-enterprise, Property 22b.
    Validates: Requirements 2.2
    """
    builder = MessageRequestBuilder()
    builder.set_model(model)
    builder.set_max_tokens(max_tokens)
    builder.add_message("user", "test")

    request = builder.build()

    assert 1 <= request["max_tokens"] <= 128000


# ---------------------------------------------------------------------------
# Property 22c: temperature is always 0.0-1.0
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=50)
@given(
    model=_model,
    temperature=_temperature,
)
def test_property_22c_temperature_in_valid_range(
    model: str,
    temperature: float,
) -> None:
    """temperature is always in [0.0, 1.0] for any valid built request.

    Feature: claude-code-v3-enterprise, Property 22c.
    Validates: Requirements 2.2
    """
    builder = MessageRequestBuilder()
    builder.set_model(model)
    builder.set_max_tokens(1024)
    builder.set_temperature(temperature)
    builder.add_message("user", "test")

    request = builder.build()

    if "temperature" in request:
        assert 0.0 <= request["temperature"] <= 1.0


# ---------------------------------------------------------------------------
# Property 22d: parsed response always contains usage metrics
# ---------------------------------------------------------------------------

@pytest.mark.property
@settings(max_examples=50)
@given(
    input_tokens=_input_tokens,
    output_tokens=_output_tokens,
    cache_creation=_cache_creation,
    cache_read=_cache_read,
)
def test_property_22d_parsed_response_has_usage(
    input_tokens: int,
    output_tokens: int,
    cache_creation: int,
    cache_read: int,
) -> None:
    """Parsed response always contains usage metrics.

    Feature: claude-code-v3-enterprise, Property 22d.
    Validates: Requirements 10.1
    """
    raw = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "OK"}],
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
    }

    parsed = ResponseParser.parse(raw)

    assert parsed.usage is not None
    assert parsed.usage.input_tokens == input_tokens
    assert parsed.usage.output_tokens == output_tokens
    assert parsed.usage.cache_creation_input_tokens == cache_creation
    assert parsed.usage.cache_read_input_tokens == cache_read
