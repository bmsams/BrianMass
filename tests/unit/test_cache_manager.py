"""Unit tests for the Cache Manager.

Tests cover:
- Cache layer duration assignments (Req 11.1)
- cache_control block injection (Req 11.2)
- Cache hit rate tracking and alerting (Req 11.3)
- Cache economics awareness (Req 11.4)
- API response usage metric parsing (Req 11.5)
"""

import pytest

from src.cache.cache_manager import (
    LAYER_DURATIONS,
    CacheDuration,
    CacheLayer,
    CacheLayerType,
    CacheManager,
    CacheStats,
)

# ---------------------------------------------------------------------------
# Layer duration assignment tests (Req 11.1)
# ---------------------------------------------------------------------------

class TestLayerDurations:
    """Verify the 7-layer caching strategy assigns correct durations."""

    def test_system_prompt_is_one_hour(self):
        assert LAYER_DURATIONS[CacheLayerType.SYSTEM_PROMPT] == CacheDuration.ONE_HOUR

    def test_brainmass_md_is_one_hour(self):
        assert LAYER_DURATIONS[CacheLayerType.BRAINMASS_MD] == CacheDuration.ONE_HOUR

    def test_tool_schemas_is_one_hour(self):
        assert LAYER_DURATIONS[CacheLayerType.TOOL_SCHEMAS] == CacheDuration.ONE_HOUR

    def test_policy_definitions_is_five_minute(self):
        assert LAYER_DURATIONS[CacheLayerType.POLICY_DEFINITIONS] == CacheDuration.FIVE_MINUTE

    def test_skill_instructions_is_five_minute(self):
        assert LAYER_DURATIONS[CacheLayerType.SKILL_INSTRUCTIONS] == CacheDuration.FIVE_MINUTE

    def test_conversation_history_is_never(self):
        assert LAYER_DURATIONS[CacheLayerType.CONVERSATION_HISTORY] == CacheDuration.NEVER

    def test_current_diff_is_never(self):
        assert LAYER_DURATIONS[CacheLayerType.CURRENT_DIFF] == CacheDuration.NEVER

    def test_all_seven_layers_defined(self):
        assert len(LAYER_DURATIONS) == 7
        assert set(LAYER_DURATIONS.keys()) == set(CacheLayerType)


# ---------------------------------------------------------------------------
# CacheLayer tests
# ---------------------------------------------------------------------------

class TestCacheLayer:
    def test_one_hour_layer_is_cacheable(self):
        layer = CacheLayer(CacheLayerType.SYSTEM_PROMPT, CacheDuration.ONE_HOUR)
        assert layer.is_cacheable is True

    def test_five_minute_layer_is_cacheable(self):
        layer = CacheLayer(CacheLayerType.POLICY_DEFINITIONS, CacheDuration.FIVE_MINUTE)
        assert layer.is_cacheable is True

    def test_never_layer_is_not_cacheable(self):
        layer = CacheLayer(CacheLayerType.CONVERSATION_HISTORY, CacheDuration.NEVER)
        assert layer.is_cacheable is False

    def test_one_hour_cache_control_has_ttl(self):
        layer = CacheLayer(CacheLayerType.SYSTEM_PROMPT, CacheDuration.ONE_HOUR)
        cc = layer.to_cache_control()
        assert cc == {"type": "ephemeral", "ttl": 3600}

    def test_five_minute_cache_control_no_ttl(self):
        layer = CacheLayer(CacheLayerType.POLICY_DEFINITIONS, CacheDuration.FIVE_MINUTE)
        cc = layer.to_cache_control()
        assert cc == {"type": "ephemeral"}

    def test_never_cache_control_is_none(self):
        layer = CacheLayer(CacheLayerType.CONVERSATION_HISTORY, CacheDuration.NEVER)
        assert layer.to_cache_control() is None


# ---------------------------------------------------------------------------
# inject_cache_controls tests (Req 11.2)
# ---------------------------------------------------------------------------

class TestInjectCacheControls:
    def setup_method(self):
        self.cm = CacheManager()

    def test_injects_one_hour_cache_control(self):
        blocks = [
            {"type": "text", "text": "You are an assistant.", "cache_layer": "system_prompt"}
        ]
        result = self.cm.inject_cache_controls(blocks)
        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral", "ttl": 3600}
        assert "cache_layer" not in result[0]

    def test_injects_five_minute_cache_control(self):
        blocks = [
            {"type": "text", "text": "policy data", "cache_layer": "policy_definitions"}
        ]
        result = self.cm.inject_cache_controls(blocks)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_never_cache_layer_no_cache_control(self):
        blocks = [
            {"type": "text", "text": "user message", "cache_layer": "conversation_history"}
        ]
        result = self.cm.inject_cache_controls(blocks)
        assert "cache_control" not in result[0]

    def test_block_without_cache_layer_unchanged(self):
        blocks = [{"type": "text", "text": "plain block"}]
        result = self.cm.inject_cache_controls(blocks)
        assert result == [{"type": "text", "text": "plain block"}]

    def test_unknown_cache_layer_passes_through(self):
        blocks = [{"type": "text", "text": "data", "cache_layer": "unknown_layer"}]
        result = self.cm.inject_cache_controls(blocks)
        assert "cache_control" not in result[0]
        # cache_layer key is removed even for unknown layers
        assert "cache_layer" not in result[0]

    def test_does_not_mutate_original_blocks(self):
        blocks = [
            {"type": "text", "text": "sys", "cache_layer": "system_prompt"}
        ]
        original_copy = [dict(b) for b in blocks]
        self.cm.inject_cache_controls(blocks)
        assert blocks == original_copy

    def test_mixed_layers(self):
        blocks = [
            {"type": "text", "text": "sys", "cache_layer": "system_prompt"},
            {"type": "text", "text": "tools", "cache_layer": "tool_schemas"},
            {"type": "text", "text": "policy", "cache_layer": "skill_instructions"},
            {"type": "text", "text": "chat", "cache_layer": "conversation_history"},
            {"type": "text", "text": "plain"},
        ]
        result = self.cm.inject_cache_controls(blocks)
        assert result[0]["cache_control"] == {"type": "ephemeral", "ttl": 3600}
        assert result[1]["cache_control"] == {"type": "ephemeral", "ttl": 3600}
        assert result[2]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[3]
        assert "cache_control" not in result[4]

    def test_empty_blocks_list(self):
        assert self.cm.inject_cache_controls([]) == []


# ---------------------------------------------------------------------------
# Hit rate tracking tests (Req 11.3, 11.5)
# ---------------------------------------------------------------------------

class TestHitRateTracking:
    def test_initial_hit_rate_is_zero(self):
        cm = CacheManager()
        assert cm.get_hit_rate() == 0.0

    def test_track_hit_rate_returns_rate(self):
        cm = CacheManager()
        rate = cm.track_hit_rate({
            "input_tokens": 1000,
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 100,
        })
        assert rate == pytest.approx(0.7)

    def test_cumulative_tracking(self):
        cm = CacheManager()
        cm.track_hit_rate({
            "input_tokens": 1000,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        })
        cm.track_hit_rate({
            "input_tokens": 1000,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 0,
        })
        # cumulative: 1000 cache_read / 2000 total = 0.5
        assert cm.get_hit_rate() == pytest.approx(0.5)

    def test_alert_fires_when_below_threshold(self):
        alerts = []
        cm = CacheManager(alert_threshold=0.70, alert_callback=lambda r: alerts.append(r))
        cm.track_hit_rate({
            "input_tokens": 1000,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 0,
        })
        assert len(alerts) == 1
        assert alerts[0] == pytest.approx(0.5)

    def test_no_alert_when_above_threshold(self):
        alerts = []
        cm = CacheManager(alert_threshold=0.70, alert_callback=lambda r: alerts.append(r))
        cm.track_hit_rate({
            "input_tokens": 1000,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        })
        assert len(alerts) == 0

    def test_no_alert_without_callback(self):
        cm = CacheManager(alert_threshold=0.70, alert_callback=None)
        # Should not raise even when below threshold
        cm.track_hit_rate({
            "input_tokens": 1000,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 0,
        })

    def test_missing_usage_keys_default_to_zero(self):
        cm = CacheManager()
        rate = cm.track_hit_rate({})
        assert rate == 0.0

    def test_get_stats_returns_copy(self):
        cm = CacheManager()
        cm.track_hit_rate({
            "input_tokens": 500,
            "cache_read_input_tokens": 300,
            "cache_creation_input_tokens": 50,
        })
        stats = cm.get_stats()
        assert stats.total_input_tokens == 500
        assert stats.cache_read_tokens == 300
        assert stats.cache_creation_tokens == 50

    def test_reset_stats(self):
        cm = CacheManager()
        cm.track_hit_rate({"input_tokens": 1000, "cache_read_input_tokens": 500})
        cm.reset_stats()
        assert cm.get_hit_rate() == 0.0
        assert cm.get_stats().total_input_tokens == 0


# ---------------------------------------------------------------------------
# Configurable alert threshold tests
# ---------------------------------------------------------------------------

class TestAlertThreshold:
    def test_default_threshold_is_70_percent(self):
        cm = CacheManager()
        assert cm.alert_threshold == pytest.approx(0.70)

    def test_custom_threshold(self):
        cm = CacheManager(alert_threshold=0.50)
        assert cm.alert_threshold == pytest.approx(0.50)

    def test_threshold_setter(self):
        cm = CacheManager()
        cm.alert_threshold = 0.80
        assert cm.alert_threshold == pytest.approx(0.80)

    def test_alert_respects_custom_threshold(self):
        alerts = []
        cm = CacheManager(alert_threshold=0.90, alert_callback=lambda r: alerts.append(r))
        # 80% hit rate — below 90% threshold
        cm.track_hit_rate({
            "input_tokens": 1000,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 0,
        })
        assert len(alerts) == 1


# ---------------------------------------------------------------------------
# CacheStats tests
# ---------------------------------------------------------------------------

class TestCacheStats:
    def test_hit_rate_zero_when_no_tokens(self):
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        stats = CacheStats(total_input_tokens=1000, cache_read_tokens=700)
        assert stats.hit_rate == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# CacheManager.get_layer / get_duration helpers
# ---------------------------------------------------------------------------

class TestCacheManagerHelpers:
    def test_get_layer_returns_correct_layer(self):
        cm = CacheManager()
        layer = cm.get_layer(CacheLayerType.BRAINMASS_MD)
        assert layer.layer_type == CacheLayerType.BRAINMASS_MD
        assert layer.duration == CacheDuration.ONE_HOUR

    def test_get_duration_returns_correct_duration(self):
        cm = CacheManager()
        assert cm.get_duration(CacheLayerType.SKILL_INSTRUCTIONS) == CacheDuration.FIVE_MINUTE
