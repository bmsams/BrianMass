"""Property-based tests for cache layer strategy.

Properties covered:
- Property 20: Cache layer duration assignment
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.cache.cache_manager import (
    LAYER_DURATIONS,
    CacheDuration,
    CacheLayerType,
    CacheManager,
)


@pytest.mark.property
@settings(max_examples=100)
@given(layer=st.sampled_from(list(CacheLayerType)))
def test_property_20_cache_layer_duration_assignment(layer: CacheLayerType) -> None:
    """Feature: claude-code-v3-enterprise, Property 20."""
    manager = CacheManager()
    expected_duration = LAYER_DURATIONS[layer]
    assert manager.get_duration(layer) == expected_duration

    blocks = [{"type": "text", "text": "payload", "cache_layer": layer.value}]
    injected = manager.inject_cache_controls(blocks)
    cache_control = injected[0].get("cache_control")

    if expected_duration == CacheDuration.ONE_HOUR:
        assert cache_control == {"type": "ephemeral", "ttl": 3600}
    elif expected_duration == CacheDuration.FIVE_MINUTE:
        assert cache_control == {"type": "ephemeral"}
    else:
        assert cache_control is None

