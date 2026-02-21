"""Unit tests for the MCP Tool Search module.

Tests cover:
- Lazy vs eager loading decision based on token threshold (Req 9.6)
- Haiku fallback to eager loading (Req 9.6)
- Sonnet/Opus support for tool_reference blocks (Req 9.6)
- ENABLE_TOOL_SEARCH env var parsing (auto / auto:N / false)
- Context savings tracking
"""



from src.orchestrator.tool_search import (
    DEFAULT_CONTEXT_THRESHOLD,
    ENV_VAR_NAME,
    LoadingMode,
    ToolDefinition,
    ToolReference,
    ToolSearchResult,
    _estimate_reference_tokens,
    _parse_env_config,
    _supports_tool_search,
    process_tools,
)
from src.types.core import ModelTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(name: str = "tool_a", desc: str = "A tool", tokens: int = 500) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=desc,
        input_schema={"type": "object"},
        token_count=tokens,
    )


def _make_tools(count: int, tokens_each: int = 500) -> list[ToolDefinition]:
    return [_make_tool(name=f"tool_{i}", tokens=tokens_each) for i in range(count)]


# ---------------------------------------------------------------------------
# _supports_tool_search
# ---------------------------------------------------------------------------

class TestSupportsToolSearch:
    def test_opus_supports(self):
        assert _supports_tool_search(ModelTier.OPUS) is True

    def test_sonnet_supports(self):
        assert _supports_tool_search(ModelTier.SONNET) is True

    def test_haiku_does_not_support(self):
        assert _supports_tool_search(ModelTier.HAIKU) is False


# ---------------------------------------------------------------------------
# _parse_env_config
# ---------------------------------------------------------------------------

class TestParseEnvConfig:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR_NAME, raising=False)
        enabled, threshold = _parse_env_config()
        assert enabled is True
        assert threshold == DEFAULT_CONTEXT_THRESHOLD

    def test_auto(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        enabled, threshold = _parse_env_config()
        assert enabled is True
        assert threshold == DEFAULT_CONTEXT_THRESHOLD

    def test_auto_with_custom_threshold(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto:5000")
        enabled, threshold = _parse_env_config()
        assert enabled is True
        assert threshold == 5000

    def test_false_disables(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "false")
        enabled, threshold = _parse_env_config()
        assert enabled is False

    def test_false_case_insensitive(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "FALSE")
        enabled, threshold = _parse_env_config()
        assert enabled is False

    def test_auto_with_invalid_number_falls_back(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto:abc")
        enabled, threshold = _parse_env_config()
        assert enabled is True
        assert threshold == DEFAULT_CONTEXT_THRESHOLD

    def test_auto_with_zero_falls_back(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto:0")
        enabled, threshold = _parse_env_config()
        assert enabled is True
        assert threshold == DEFAULT_CONTEXT_THRESHOLD

    def test_auto_with_negative_falls_back(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto:-5")
        enabled, threshold = _parse_env_config()
        assert enabled is True
        assert threshold == DEFAULT_CONTEXT_THRESHOLD

    def test_whitespace_trimmed(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "  auto:8000  ")
        enabled, threshold = _parse_env_config()
        assert enabled is True
        assert threshold == 8000


# ---------------------------------------------------------------------------
# _estimate_reference_tokens
# ---------------------------------------------------------------------------

class TestEstimateReferenceTokens:
    def test_returns_positive(self):
        tool = _make_tool(name="x", desc="y")
        assert _estimate_reference_tokens(tool) >= 1

    def test_longer_description_costs_more(self):
        short = _make_tool(name="t", desc="short")
        long = _make_tool(name="t", desc="a much longer description for this tool")
        assert _estimate_reference_tokens(long) > _estimate_reference_tokens(short)


# ---------------------------------------------------------------------------
# ToolSearchResult properties
# ---------------------------------------------------------------------------

class TestToolSearchResult:
    def test_tokens_saved(self):
        r = ToolSearchResult(
            mode=LoadingMode.LAZY,
            tools=[],
            total_tokens_original=10000,
            total_tokens_after=500,
        )
        assert r.tokens_saved == 9500

    def test_savings_percent(self):
        r = ToolSearchResult(
            mode=LoadingMode.LAZY,
            tools=[],
            total_tokens_original=10000,
            total_tokens_after=500,
        )
        assert r.savings_percent == 95.0

    def test_savings_percent_zero_original(self):
        r = ToolSearchResult(
            mode=LoadingMode.EAGER,
            tools=[],
            total_tokens_original=0,
            total_tokens_after=0,
        )
        assert r.savings_percent == 0.0

    def test_eager_no_savings(self):
        r = ToolSearchResult(
            mode=LoadingMode.EAGER,
            tools=[],
            total_tokens_original=5000,
            total_tokens_after=5000,
        )
        assert r.tokens_saved == 0
        assert r.savings_percent == 0.0


# ---------------------------------------------------------------------------
# process_tools — eager loading scenarios
# ---------------------------------------------------------------------------

class TestProcessToolsEager:
    def test_below_threshold_uses_eager(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(5, tokens_each=100)  # 500 total < 10000
        result = process_tools(tools, ModelTier.SONNET)
        assert result.mode == LoadingMode.EAGER
        assert len(result.tools) == 5
        assert all(isinstance(t, ToolDefinition) for t in result.tools)
        assert result.tokens_saved == 0

    def test_at_threshold_uses_eager(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(20, tokens_each=500)  # 10000 == threshold
        result = process_tools(tools, ModelTier.SONNET)
        assert result.mode == LoadingMode.EAGER

    def test_haiku_always_eager_even_above_threshold(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(30, tokens_each=1000)  # 30000 >> threshold
        result = process_tools(tools, ModelTier.HAIKU)
        assert result.mode == LoadingMode.EAGER
        assert all(isinstance(t, ToolDefinition) for t in result.tools)
        assert result.tokens_saved == 0

    def test_disabled_via_env_always_eager(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "false")
        tools = _make_tools(30, tokens_each=1000)
        result = process_tools(tools, ModelTier.OPUS)
        assert result.mode == LoadingMode.EAGER
        assert result.tokens_saved == 0

    def test_empty_tools_list(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        result = process_tools([], ModelTier.SONNET)
        assert result.mode == LoadingMode.EAGER
        assert result.tools == []
        assert result.total_tokens_original == 0


# ---------------------------------------------------------------------------
# process_tools — lazy loading scenarios
# ---------------------------------------------------------------------------

class TestProcessToolsLazy:
    def test_above_threshold_sonnet_uses_lazy(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(30, tokens_each=1000)  # 30000 > 10000
        result = process_tools(tools, ModelTier.SONNET)
        assert result.mode == LoadingMode.LAZY
        assert all(isinstance(t, ToolReference) for t in result.tools)
        assert result.tokens_saved > 0

    def test_above_threshold_opus_uses_lazy(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(30, tokens_each=1000)
        result = process_tools(tools, ModelTier.OPUS)
        assert result.mode == LoadingMode.LAZY

    def test_lazy_preserves_tool_count(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(15, tokens_each=1000)
        result = process_tools(tools, ModelTier.SONNET)
        assert len(result.tools) == 15

    def test_lazy_references_have_correct_names(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = [_make_tool(name="read_file", desc="Read a file", tokens=5000),
                 _make_tool(name="write_file", desc="Write a file", tokens=6000)]
        result = process_tools(tools, ModelTier.SONNET)
        assert result.mode == LoadingMode.LAZY
        names = [t.name for t in result.tools]
        assert names == ["read_file", "write_file"]

    def test_lazy_references_have_descriptions(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = [_make_tool(name="grep", desc="Search files", tokens=6000),
                 _make_tool(name="bash", desc="Run commands", tokens=5000)]
        result = process_tools(tools, ModelTier.OPUS)
        descs = [t.description for t in result.tools]
        assert descs == ["Search files", "Run commands"]

    def test_context_savings_are_significant(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(50, tokens_each=2000)  # 100000 total
        result = process_tools(tools, ModelTier.SONNET)
        assert result.savings_percent > 50.0

    def test_custom_threshold_via_parameter(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(5, tokens_each=100)  # 500 total
        # With threshold=200, 500 > 200 → lazy
        result = process_tools(tools, ModelTier.SONNET, threshold=200)
        assert result.mode == LoadingMode.LAZY

    def test_custom_threshold_via_env(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto:300")
        tools = _make_tools(5, tokens_each=100)  # 500 > 300
        result = process_tools(tools, ModelTier.SONNET)
        assert result.mode == LoadingMode.LAZY

    def test_parameter_threshold_overrides_env(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto:300")
        tools = _make_tools(5, tokens_each=100)  # 500 total
        # param threshold=1000 > 500 → eager
        result = process_tools(tools, ModelTier.SONNET, threshold=1000)
        assert result.mode == LoadingMode.EAGER


# ---------------------------------------------------------------------------
# process_tools — metrics tracking
# ---------------------------------------------------------------------------

class TestProcessToolsMetrics:
    def test_original_tokens_tracked(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(10, tokens_each=300)
        result = process_tools(tools, ModelTier.SONNET)
        assert result.total_tokens_original == 3000

    def test_after_tokens_equal_original_when_eager(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(5, tokens_each=100)
        result = process_tools(tools, ModelTier.SONNET)
        assert result.total_tokens_after == result.total_tokens_original

    def test_after_tokens_less_than_original_when_lazy(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR_NAME, "auto")
        tools = _make_tools(30, tokens_each=1000)
        result = process_tools(tools, ModelTier.SONNET)
        assert result.total_tokens_after < result.total_tokens_original
