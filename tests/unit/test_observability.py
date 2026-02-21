"""Unit tests for observability instrumentation basics."""

from __future__ import annotations

from src.observability.instrumentation import BrainmassTracer, HookLogger, TerminalMonitor


def test_tracer_records_spans() -> None:
    tracer = BrainmassTracer(service_name="brainmass-v3")
    with tracer.start_span("test.span"):
        pass
    spans = tracer.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "test.span"


def test_hook_logger_records_structured_entries() -> None:
    tracer = BrainmassTracer()
    logger = HookLogger(session_id="s1", tracer=tracer)
    entry = logger.log_hook_execution(
        event="PreToolUse",
        handler_type="command",
        matcher="*",
        duration_ms=12.3,
        result={"permission_decision": "allow"},
    )
    assert entry["event"] == "PreToolUse"
    assert logger.get_entries()[0]["status"] == "ok"


def test_terminal_monitor_summary() -> None:
    tracer = BrainmassTracer()
    tracer.record_cost_span(
        agent_id="a1",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.1,
        model_tier="sonnet",
    )
    monitor = TerminalMonitor(tracer=tracer)
    summary = monitor.get_metrics_summary()
    assert summary["total_input_tokens"] == 100
    assert summary["total_output_tokens"] == 50


# ---------------------------------------------------------------------------
# Task 24.4 — comprehensive observability unit tests
# Requirements: 16.1, 16.2, 16.3
# ---------------------------------------------------------------------------

import pytest

from src.observability.instrumentation import (
    CloudWatchExporter,
)
from src.observability.tracer import (
    get_cloudwatch_exporter,
    get_hook_logger,
    get_terminal_monitor,
    get_tracer,
    reset_singletons,
)
from src.types.core import HookEvent

# ---------------------------------------------------------------------------
# Tracer factory / singleton (Req 16.1)
# ---------------------------------------------------------------------------

class TestGetTracer:
    def setup_method(self) -> None:
        reset_singletons()

    def teardown_method(self) -> None:
        reset_singletons()

    def test_returns_brainmass_tracer_instance(self) -> None:
        tracer = get_tracer()
        assert isinstance(tracer, BrainmassTracer)

    def test_singleton_same_object(self) -> None:
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_custom_name_used_on_first_call(self) -> None:
        tracer = get_tracer("my-service")
        assert tracer.service_name == "my-service"

    def test_get_hook_logger_returns_hook_logger(self) -> None:
        assert isinstance(get_hook_logger(), HookLogger)

    def test_get_terminal_monitor_returns_terminal_monitor(self) -> None:
        assert isinstance(get_terminal_monitor(), TerminalMonitor)

    def test_get_cloudwatch_exporter_returns_exporter(self) -> None:
        assert isinstance(get_cloudwatch_exporter(), CloudWatchExporter)

    def test_reset_singletons_creates_fresh_instances(self) -> None:
        t1 = get_tracer()
        reset_singletons()
        t2 = get_tracer()
        assert t1 is not t2


# ---------------------------------------------------------------------------
# Span recording for each span type (Req 16.2)
# ---------------------------------------------------------------------------

class TestSpanTypes:
    def setup_method(self) -> None:
        self.tracer = BrainmassTracer(service_name="test")

    def test_trace_agent_action_records_span(self) -> None:
        with self.tracer.trace_agent_action("agent-1", "dispatch"):
            pass
        spans = self.tracer.get_spans(name_prefix="agent.")
        assert len(spans) == 1
        assert spans[0].attributes["agent.id"] == "agent-1"
        assert spans[0].attributes["agent.action"] == "dispatch"

    def test_trace_tool_call_records_span(self) -> None:
        with self.tracer.trace_tool_call("Bash", agent_id="orch"):
            pass
        spans = self.tracer.get_spans(name_prefix="tool.")
        assert len(spans) == 1
        assert spans[0].attributes["tool.name"] == "Bash"

    def test_trace_hook_execution_records_span(self) -> None:
        with self.tracer.trace_hook_execution("PreToolUse", "command"):
            pass
        spans = self.tracer.get_spans(name_prefix="hook.")
        assert len(spans) == 1
        assert spans[0].attributes["hook.event"] == "PreToolUse"

    def test_trace_model_interaction_records_span(self) -> None:
        with self.tracer.trace_model_interaction("sonnet", agent_id="orch"):
            pass
        spans = self.tracer.get_spans(name_prefix="model.")
        assert len(spans) == 1
        assert spans[0].attributes["model.tier"] == "sonnet"

    def test_record_cost_span(self) -> None:
        span = self.tracer.record_cost_span(
            agent_id="a1", input_tokens=500, output_tokens=200, cost_usd=0.05, model_tier="haiku"
        )
        assert span.attributes["cost.input_tokens"] == 500
        assert span.attributes["cost.usd"] == 0.05

    def test_record_context_health_span(self) -> None:
        span = self.tracer.record_context_health_span(
            free_pct=72.5,
            staleness_scores={"fresh": 10, "warm": 3, "stale": 1},
            cache_hit_rate=0.85,
        )
        assert span.attributes["context.free_pct"] == 72.5
        assert span.attributes["context.cache_hit_rate"] == 0.85

    def test_record_coordination_span(self) -> None:
        span = self.tracer.record_coordination_span(
            event_type="task_claim", agent_id="worker-1", details={"task_id": "t1"}
        )
        assert span.attributes["coordination.event_type"] == "task_claim"
        assert span.attributes["coordination.agent_id"] == "worker-1"

    def test_record_effort_span(self) -> None:
        span = self.tracer.record_effort_span(
            effort_level="deep", fast_mode=False, budget_tokens=50_000
        )
        assert span.attributes["effort.level"] == "deep"
        assert span.attributes["effort.budget_tokens"] == 50_000

    def test_record_skill_span(self) -> None:
        span = self.tracer.record_skill_span(
            skill_name="security-auditor",
            confidence=0.87,
            selection_reason="keyword match",
            matched=True,
        )
        assert span.attributes["skill.name"] == "security-auditor"
        assert span.attributes["skill.matched"] is True

    def test_span_error_status_on_exception(self) -> None:
        with pytest.raises(ValueError):
            with self.tracer.start_span("failing.span"):
                raise ValueError("boom")
        spans = self.tracer.get_spans()
        assert spans[-1].status == "error"

    def test_nested_spans_set_parent_id(self) -> None:
        with self.tracer.start_span("outer") as outer:
            with self.tracer.start_span("inner") as inner:
                pass
        assert inner.parent_span_id == outer.span_id

    def test_trace_id_consistent_across_spans(self) -> None:
        with self.tracer.start_span("s1"):
            pass
        with self.tracer.start_span("s2"):
            pass
        spans = self.tracer.get_spans()
        assert spans[0].trace_id == spans[1].trace_id == self.tracer.get_trace_id()

    def test_set_trace_id_propagates(self) -> None:
        self.tracer.set_trace_id("fixed-trace-id")
        with self.tracer.start_span("s"):
            pass
        assert self.tracer.get_spans()[-1].trace_id == "fixed-trace-id"


# ---------------------------------------------------------------------------
# HookLogger — all 12 HookEvent types (Req 16.3)
# ---------------------------------------------------------------------------

class TestHookLoggerAllEvents:
    def setup_method(self) -> None:
        self.tracer = BrainmassTracer()
        self.logger = HookLogger(session_id="sess-1", tracer=self.tracer)

    def test_all_12_hook_events_logged(self) -> None:
        for event in HookEvent:
            self.logger.log_hook_execution(
                event=event.value,
                handler_type="command",
                matcher="*",
                duration_ms=1.0,
            )
        entries = self.logger.get_entries()
        logged_events = {e["event"] for e in entries}
        assert logged_events == {ev.value for ev in HookEvent}

    def test_error_entry_has_error_status(self) -> None:
        entry = self.logger.log_hook_execution(
            event="PreToolUse",
            handler_type="command",
            matcher="Bash",
            duration_ms=5.0,
            error="timeout",
        )
        assert entry["status"] == "error"
        assert entry["error"] == "timeout"

    def test_ok_entry_has_ok_status(self) -> None:
        entry = self.logger.log_hook_execution(
            event="PostToolUse",
            handler_type="prompt",
            matcher=None,
            duration_ms=2.5,
        )
        assert entry["status"] == "ok"

    def test_filter_by_event(self) -> None:
        self.logger.log_hook_execution("PreToolUse", "command", "*", 1.0)
        self.logger.log_hook_execution("PostToolUse", "command", "*", 1.0)
        pre = self.logger.get_entries(event="PreToolUse")
        assert len(pre) == 1
        assert pre[0]["event"] == "PreToolUse"

    def test_hook_logger_emits_otel_span(self) -> None:
        self.logger.log_hook_execution("Stop", "agent", "*", 3.0)
        hook_log_spans = self.tracer.get_spans(name_prefix="hook.log.")
        assert len(hook_log_spans) == 1

    def test_clear_entries(self) -> None:
        self.logger.log_hook_execution("Stop", "command", "*", 1.0)
        self.logger.clear_entries()
        assert self.logger.get_entries() == []


# ---------------------------------------------------------------------------
# TerminalMonitor — format_dashboard produces non-empty output (Req 16.4)
# ---------------------------------------------------------------------------

class TestTerminalMonitor:
    def test_format_dashboard_non_empty(self) -> None:
        tracer = BrainmassTracer()
        tracer.record_cost_span("orch", 1000, 500, 0.25, "sonnet")
        monitor = TerminalMonitor(tracer=tracer)
        dashboard = monitor.format_dashboard()
        assert len(dashboard) > 0
        assert "BRAINMASS" in dashboard

    def test_format_dashboard_includes_token_counts(self) -> None:
        tracer = BrainmassTracer()
        tracer.record_cost_span("orch", 1234, 567, 0.10, "haiku")
        monitor = TerminalMonitor(tracer=tracer)
        dashboard = monitor.format_dashboard()
        assert "1,234" in dashboard
        assert "567" in dashboard

    def test_metrics_summary_aggregates_multiple_cost_spans(self) -> None:
        tracer = BrainmassTracer()
        tracer.record_cost_span("a1", 100, 50, 0.01, "haiku")
        tracer.record_cost_span("a2", 200, 100, 0.02, "sonnet")
        monitor = TerminalMonitor(tracer=tracer)
        summary = monitor.get_metrics_summary()
        assert summary["total_input_tokens"] == 300
        assert summary["total_output_tokens"] == 150

    def test_metrics_summary_no_tracer(self) -> None:
        monitor = TerminalMonitor(tracer=None)
        summary = monitor.get_metrics_summary()
        assert summary["total_input_tokens"] == 0
        assert summary["cache_hit_rate"] == 0.0

    def test_metrics_summary_span_counts(self) -> None:
        tracer = BrainmassTracer()
        tracer.record_cost_span("a", 10, 5, 0.001, "haiku")
        tracer.record_effort_span("quick", False, 2000)
        monitor = TerminalMonitor(tracer=tracer)
        summary = monitor.get_metrics_summary()
        assert "cost" in summary["span_counts"]
        assert "effort" in summary["span_counts"]

    def test_cache_hit_rate_from_context_health_spans(self) -> None:
        tracer = BrainmassTracer()
        tracer.record_context_health_span(80.0, {}, 0.75)
        tracer.record_context_health_span(70.0, {}, 0.85)
        monitor = TerminalMonitor(tracer=tracer)
        summary = monitor.get_metrics_summary()
        assert abs(summary["cache_hit_rate"] - 0.80) < 0.01


# ---------------------------------------------------------------------------
# CloudWatchExporter (Req 16.5)
# ---------------------------------------------------------------------------

class TestCloudWatchExporter:
    def test_export_calls_callback(self) -> None:
        received: list[dict] = []
        exporter = CloudWatchExporter(export_callback=received.append)
        exporter.export_metrics({"total_cost_usd": 1.23, "input_tokens": 500})
        assert len(received) == 1
        assert received[0]["namespace"] == "Brainmass/V3"

    def test_export_history_recorded(self) -> None:
        exporter = CloudWatchExporter()
        exporter.export_metrics({"x": 1})
        exporter.export_metrics({"y": 2})
        assert len(exporter.get_export_history()) == 2

    def test_metric_data_flattens_numerics(self) -> None:
        received: list[dict] = []
        exporter = CloudWatchExporter(export_callback=received.append)
        exporter.export_metrics({"tokens": 100, "cost": 0.5})
        metric_names = {m["MetricName"] for m in received[0]["metric_data"]}
        assert "tokens" in metric_names
        assert "cost" in metric_names

    def test_dashboard_config_has_required_widgets(self) -> None:
        exporter = CloudWatchExporter()
        config = exporter.get_dashboard_config()
        titles = {w["properties"]["title"] for w in config["widgets"]}
        assert "Token Consumption" in titles
        assert "Cost (USD)" in titles
        assert "Cache Hit Rate" in titles
