"""OTel instrumentation and observability for Brainmass v3.

Provides a pluggable tracing backend, structured hook logging,
terminal-based monitoring, CloudWatch export, and session trace
continuity across surface transitions.

Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SpanRecord — internal storage for completed and in-flight spans
# ---------------------------------------------------------------------------

@dataclass
class SpanRecord:
    """A single recorded span, stored in-memory for query and export."""
    name: str
    attributes: dict
    start_time: datetime
    end_time: datetime | None = None
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str = ""
    status: str = "ok"  # 'ok', 'error'


# ---------------------------------------------------------------------------
# BrainmassTracer — central instrumentation hub (Req 16.1, 16.2, 16.6)
# ---------------------------------------------------------------------------

class BrainmassTracer:
    """Central instrumentation hub for all Brainmass observability.

    Wraps OpenTelemetry tracer functionality behind a pluggable backend.
    All spans are stored internally as ``SpanRecord`` instances so they
    can be queried and tested without an actual OTel SDK.

    Thread-safe: span storage and trace-id access are guarded by a lock.

    Parameters
    ----------
    service_name : str
        Logical service name attached to all spans.
    trace_id : str | None
        Optional persistent trace ID for session trace continuity
        (Requirement 16.6).  Auto-generated if omitted.
    backend : Callable | None
        Optional callback invoked with each completed ``SpanRecord``.
        Enables pluggable export to OTel, stdout, or any sink.
    """

    def __init__(
        self,
        service_name: str = "brainmass",
        trace_id: str | None = None,
        backend: Callable[[SpanRecord], None] | None = None,
        backend_tracer: object | None = None,
    ) -> None:
        self.service_name = service_name
        self._lock = threading.Lock()
        self._trace_id = trace_id or uuid.uuid4().hex
        self._spans: list[SpanRecord] = []
        self._backend = backend
        # Optional Strands telemetry tracer — delegates start_as_current_span when set.
        # Requirements: 11.1, 11.2, 11.3
        self._backend_tracer = backend_tracer
        # Stack of active span IDs for parent-child linkage (per-thread)
        self._active_spans: threading.local = threading.local()

    # ------------------------------------------------------------------
    # Session trace continuity (Requirement 16.6)
    # ------------------------------------------------------------------

    def set_trace_id(self, trace_id: str) -> None:
        """Set or update the persistent trace ID.

        This allows trace continuity across surface transitions
        (e.g. IDE to terminal to web).
        """
        with self._lock:
            self._trace_id = trace_id

    def get_trace_id(self) -> str:
        """Return the current session trace ID."""
        with self._lock:
            return self._trace_id

    # ------------------------------------------------------------------
    # Core span creation (Requirement 16.1)
    # ------------------------------------------------------------------

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict | None = None,
    ) -> Generator[SpanRecord, None, None]:
        """Context manager that creates, yields, and finalises a span.

        Automatically sets ``trace_id``, ``span_id``, ``parent_span_id``,
        and records start/end times.  On exception the span status is set
        to ``'error'`` and the exception is re-raised.

        Parameters
        ----------
        name : str
            Human-readable span name (e.g. ``'tool.Bash'``).
        attributes : dict | None
            Arbitrary key-value pairs attached to the span.
        """
        span = SpanRecord(
            name=name,
            attributes=attributes or {},
            start_time=datetime.now(UTC),
            trace_id=self.get_trace_id(),
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self._current_parent_span_id(),
        )
        self._push_span_id(span.span_id)
        try:
            yield span
        except Exception:
            span.status = "error"
            raise
        finally:
            span.end_time = datetime.now(UTC)
            self._pop_span_id()
            with self._lock:
                self._spans.append(span)
            if self._backend is not None:
                try:
                    self._backend(span)
                except Exception:
                    logger.debug("Tracer backend raised; span %s dropped", span.name)

    # ------------------------------------------------------------------
    # Convenience: agent / tool / hook / model spans (Requirement 16.1)
    # ------------------------------------------------------------------

    @contextmanager
    def trace_agent_action(
        self,
        agent_id: str,
        action: str,
        attributes: dict | None = None,
    ) -> Generator[SpanRecord, None, None]:
        """Trace an agent action (turn, delegation, completion).

        When a Strands backend_tracer is set, the Strands SDK handles its own
        OTel span emission internally during Agent execution.  BrainmassTracer
        records its own span in the internal store for querying and testing.

        --- Production integration point ---
        # The Strands Tracer is passed to Agent(tracer=backend_tracer) at
        # construction time; it emits agent/tool spans automatically via its
        # own start_agent_span / start_tool_call_span API.  We do not call
        # start_as_current_span here because that method does not exist on
        # the Strands Tracer class.
        """
        attrs = {"agent.id": agent_id, "agent.action": action}
        if attributes:
            attrs.update(attributes)
        with self.start_span(f"agent.{action}", attrs) as span:
            yield span

    @contextmanager
    def trace_tool_call(
        self,
        tool_name: str,
        agent_id: str = "",
        attributes: dict | None = None,
    ) -> Generator[SpanRecord, None, None]:
        """Trace a tool invocation.

        --- Production integration point ---
        # Strands Tracer emits tool spans via start_tool_call_span internally.
        # BrainmassTracer records its own span for internal querying.
        """
        attrs = {"tool.name": tool_name, "agent.id": agent_id}
        if attributes:
            attrs.update(attributes)
        with self.start_span(f"tool.{tool_name}", attrs) as span:
            yield span

    @contextmanager
    def trace_hook_execution(
        self,
        event: str,
        handler_type: str,
        attributes: dict | None = None,
    ) -> Generator[SpanRecord, None, None]:
        """Trace a hook handler execution.

        --- Production integration point ---
        # Hook spans are Brainmass-specific; no Strands SDK delegation needed.
        """
        attrs = {"hook.event": event, "hook.handler_type": handler_type}
        if attributes:
            attrs.update(attributes)
        with self.start_span(f"hook.{event}", attrs) as span:
            yield span

    @contextmanager
    def trace_model_interaction(
        self,
        model_tier: str,
        agent_id: str = "",
        attributes: dict | None = None,
    ) -> Generator[SpanRecord, None, None]:
        """Trace a model API call.

        --- Production integration point ---
        # Strands Tracer emits model/LLM spans internally during Agent execution.
        # BrainmassTracer records its own span for cost attribution and querying.
        """
        attrs = {"model.tier": model_tier, "agent.id": agent_id}
        if attributes:
            attrs.update(attributes)
        with self.start_span(f"model.{model_tier}", attrs) as span:
            yield span

    # ------------------------------------------------------------------
    # Custom spans (Requirement 16.2)
    # ------------------------------------------------------------------

    def record_cost_span(
        self,
        agent_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        model_tier: str,
    ) -> SpanRecord:
        """Emit a cost attribution span.

        Captures token consumption and dollar cost for a single API call
        so that cost can be attributed per-agent in dashboards.
        """
        attrs = {
            "cost.agent_id": agent_id,
            "cost.input_tokens": input_tokens,
            "cost.output_tokens": output_tokens,
            "cost.usd": cost_usd,
            "cost.model_tier": model_tier,
        }
        return self._emit_instant_span("cost.attribution", attrs)

    def record_context_health_span(
        self,
        free_pct: float,
        staleness_scores: dict[str, float],
        cache_hit_rate: float,
    ) -> SpanRecord:
        """Emit a context health span.

        Reports the context window's free capacity, per-item staleness,
        and prompt-cache hit rate for monitoring context pressure.
        """
        attrs = {
            "context.free_pct": free_pct,
            "context.staleness_scores": staleness_scores,
            "context.cache_hit_rate": cache_hit_rate,
        }
        return self._emit_instant_span("context.health", attrs)

    def record_coordination_span(
        self,
        event_type: str,
        agent_id: str,
        details: dict,
    ) -> SpanRecord:
        """Emit an agent coordination span.

        Records inter-agent events such as mailbox messages, task claims,
        and file-lock acquisitions.

        Parameters
        ----------
        event_type : str
            One of ``'message'``, ``'task_claim'``, ``'file_lock'``.
        agent_id : str
            The agent originating or receiving the coordination event.
        details : dict
            Arbitrary payload describing the coordination event.
        """
        attrs = {
            "coordination.event_type": event_type,
            "coordination.agent_id": agent_id,
            "coordination.details": details,
        }
        return self._emit_instant_span(f"coordination.{event_type}", attrs)

    def record_effort_span(
        self,
        effort_level: str,
        fast_mode: bool,
        budget_tokens: int,
    ) -> SpanRecord:
        """Emit an effort-level span.

        Captures the selected effort level (quick/standard/deep), whether
        fast mode is active, and the budget_tokens allocation.
        """
        attrs = {
            "effort.level": effort_level,
            "effort.fast_mode": fast_mode,
            "effort.budget_tokens": budget_tokens,
        }
        return self._emit_instant_span("effort.selection", attrs)

    def record_skill_span(
        self,
        skill_name: str,
        confidence: float,
        selection_reason: str,
        matched: bool,
    ) -> SpanRecord:
        """Emit a skill invocation span.

        Records which skill was selected (or rejected), the NL-match
        confidence, and the reason for selection.
        """
        attrs = {
            "skill.name": skill_name,
            "skill.confidence": confidence,
            "skill.selection_reason": selection_reason,
            "skill.matched": matched,
        }
        return self._emit_instant_span("skill.invocation", attrs)

    # ------------------------------------------------------------------
    # Span queries (for testing and dashboard consumers)
    # ------------------------------------------------------------------

    def get_spans(
        self,
        name_prefix: str | None = None,
    ) -> list[SpanRecord]:
        """Return recorded spans, optionally filtered by name prefix."""
        with self._lock:
            if name_prefix is None:
                return list(self._spans)
            return [s for s in self._spans if s.name.startswith(name_prefix)]

    def clear_spans(self) -> None:
        """Remove all recorded spans."""
        with self._lock:
            self._spans.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_instant_span(self, name: str, attributes: dict) -> SpanRecord:
        """Create and store a zero-duration 'instant' span."""
        now = datetime.now(UTC)
        span = SpanRecord(
            name=name,
            attributes=attributes,
            start_time=now,
            end_time=now,
            trace_id=self.get_trace_id(),
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=self._current_parent_span_id(),
        )
        with self._lock:
            self._spans.append(span)
        if self._backend is not None:
            try:
                self._backend(span)
            except Exception:
                logger.debug("Tracer backend raised; span %s dropped", span.name)
        return span

    def _current_parent_span_id(self) -> str:
        """Return the current active span ID, or empty string if none."""
        stack = getattr(self._active_spans, "stack", None)
        if stack:
            return stack[-1]
        return ""

    def _push_span_id(self, span_id: str) -> None:
        """Push a span ID onto the per-thread active stack."""
        if not hasattr(self._active_spans, "stack"):
            self._active_spans.stack = []
        self._active_spans.stack.append(span_id)

    def _pop_span_id(self) -> None:
        """Pop the most recent span ID from the per-thread active stack."""
        stack = getattr(self._active_spans, "stack", None)
        if stack:
            stack.pop()


# ---------------------------------------------------------------------------
# HookLogger — structured hook execution logging (Requirement 16.3)
# ---------------------------------------------------------------------------

class HookLogger:
    """Structured logging for hook executions.

    Emits JSON-structured log entries for every hook handler invocation,
    including timing, matcher details, and error information.

    Parameters
    ----------
    session_id : str
        Session identifier included in every log entry.
    tracer : BrainmassTracer | None
        Optional tracer for emitting OTel spans alongside log entries.
    """

    def __init__(
        self,
        session_id: str = "",
        tracer: BrainmassTracer | None = None,
    ) -> None:
        self.session_id = session_id
        self._tracer = tracer
        self._lock = threading.Lock()
        self._log_entries: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_hook_execution(
        self,
        event: str,
        handler_type: str,
        matcher: str | None,
        duration_ms: float,
        result: dict | None = None,
        error: str | None = None,
        agent_id: str = "",
    ) -> dict:
        """Record a structured log entry for a hook execution.

        Parameters
        ----------
        event : str
            The hook event name (e.g. ``'PreToolUse'``).
        handler_type : str
            Handler type: ``'command'``, ``'prompt'``, or ``'agent'``.
        matcher : str | None
            The matcher pattern that selected this handler.
        duration_ms : float
            Wall-clock execution time in milliseconds.
        result : dict | None
            Serialisable result payload from the handler.
        error : str | None
            Error message if the handler failed.
        agent_id : str
            The agent context in which the hook ran.

        Returns
        -------
        dict
            The structured log entry that was recorded.
        """
        entry: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "agent_id": agent_id,
            "event": event,
            "handler_type": handler_type,
            "matcher": matcher,
            "duration_ms": duration_ms,
            "result": result,
            "error": error,
            "status": "error" if error else "ok",
        }

        with self._lock:
            self._log_entries.append(entry)

        # Emit to Python logger as structured JSON
        if error:
            logger.warning("hook_execution: %s", json.dumps(entry, default=str))
        else:
            logger.info("hook_execution: %s", json.dumps(entry, default=str))

        # Optionally emit an OTel span
        if self._tracer is not None:
            attrs = {
                "hook.event": event,
                "hook.handler_type": handler_type,
                "hook.matcher": matcher or "*",
                "hook.duration_ms": duration_ms,
                "hook.agent_id": agent_id,
            }
            if error:
                attrs["hook.error"] = error
            self._tracer._emit_instant_span(f"hook.log.{event}", attrs)

        return entry

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_entries(
        self,
        event: str | None = None,
    ) -> list[dict]:
        """Return recorded log entries, optionally filtered by event."""
        with self._lock:
            if event is None:
                return list(self._log_entries)
            return [e for e in self._log_entries if e["event"] == event]

    def clear_entries(self) -> None:
        """Remove all recorded log entries."""
        with self._lock:
            self._log_entries.clear()


# ---------------------------------------------------------------------------
# TerminalMonitor — terminal-based monitoring dashboard (Requirement 16.4)
# ---------------------------------------------------------------------------

class TerminalMonitor:
    """Terminal-based monitoring dashboard.

    Aggregates span data from a ``BrainmassTracer`` to produce a
    human-readable dashboard string and a metrics summary dict
    suitable for programmatic consumption.

    Parameters
    ----------
    tracer : BrainmassTracer | None
        The tracer instance to read spans from.  When ``None``, the
        monitor operates in degraded mode and returns empty metrics.
    """

    def __init__(self, tracer: BrainmassTracer | None = None) -> None:
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def format_dashboard(self) -> str:
        """Return a formatted string showing current session metrics.

        The output is designed for terminal display and includes token
        usage, cost breakdown, active agents, and cache performance.
        """
        metrics = self.get_metrics_summary()
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  BRAINMASS v3 SESSION MONITOR")
        lines.append("=" * 60)
        lines.append("")

        # Token usage
        lines.append("  TOKEN USAGE")
        lines.append("  -----------")
        lines.append(f"  Input tokens:  {metrics['total_input_tokens']:>12,}")
        lines.append(f"  Output tokens: {metrics['total_output_tokens']:>12,}")
        lines.append(f"  Total cost:    ${metrics['total_cost_usd']:>11.4f}")
        lines.append("")

        # Active agents
        lines.append("  ACTIVE AGENTS")
        lines.append("  -------------")
        if metrics["active_agents"]:
            for agent_id in metrics["active_agents"]:
                lines.append(f"    - {agent_id}")
        else:
            lines.append("    (none)")
        lines.append("")

        # Cache performance
        lines.append("  CACHE PERFORMANCE")
        lines.append("  -----------------")
        lines.append(f"  Hit rate: {metrics['cache_hit_rate']:.1%}")
        lines.append("")

        # Span counts by category
        lines.append("  SPAN SUMMARY")
        lines.append("  ------------")
        for category, count in sorted(metrics["span_counts"].items()):
            lines.append(f"  {category:<25s} {count:>6}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_metrics_summary(self) -> dict:
        """Return a dict with token usage, cost, active agents, and cache hit rate.

        Keys:
        - ``total_input_tokens``: int
        - ``total_output_tokens``: int
        - ``total_cost_usd``: float
        - ``active_agents``: list[str]
        - ``cache_hit_rate``: float
        - ``span_counts``: dict[str, int]  (category prefix -> count)
        """
        if self._tracer is None:
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "active_agents": [],
                "cache_hit_rate": 0.0,
                "span_counts": {},
            }

        spans = self._tracer.get_spans()

        total_input = 0
        total_output = 0
        total_cost = 0.0
        agents: set[str] = set()
        cache_hit_rates: list[float] = []
        span_counts: dict[str, int] = {}

        for span in spans:
            # Aggregate cost spans
            if span.name == "cost.attribution":
                total_input += span.attributes.get("cost.input_tokens", 0)
                total_output += span.attributes.get("cost.output_tokens", 0)
                total_cost += span.attributes.get("cost.usd", 0.0)
                agent_id = span.attributes.get("cost.agent_id", "")
                if agent_id:
                    agents.add(agent_id)

            # Collect agent IDs from any span that has one
            for key in ("agent.id", "cost.agent_id", "coordination.agent_id",
                        "hook.agent_id"):
                val = span.attributes.get(key, "")
                if val:
                    agents.add(val)

            # Collect cache hit rates from context health spans
            if span.name == "context.health":
                rate = span.attributes.get("context.cache_hit_rate")
                if rate is not None:
                    cache_hit_rates.append(rate)

            # Count spans by category prefix (first dotted segment)
            category = span.name.split(".")[0] if "." in span.name else span.name
            span_counts[category] = span_counts.get(category, 0) + 1

        avg_cache_hit = (
            sum(cache_hit_rates) / len(cache_hit_rates)
            if cache_hit_rates
            else 0.0
        )

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": total_cost,
            "active_agents": sorted(agents),
            "cache_hit_rate": avg_cache_hit,
            "span_counts": span_counts,
        }


# ---------------------------------------------------------------------------
# CloudWatchExporter — pluggable CloudWatch metrics export (Req 16.5)
# ---------------------------------------------------------------------------

class CloudWatchExporter:
    """Pluggable CloudWatch metrics exporter.

    Provides a callback-based interface for exporting metrics to
    Amazon CloudWatch.  No actual AWS SDK calls are made; the
    ``export_callback`` receives a plain dict suitable for
    ``put_metric_data``.

    Parameters
    ----------
    export_callback : Callable | None
        Function called with a metrics dict on every ``export_metrics``
        invocation.  When ``None``, metrics are logged but not exported.
    namespace : str
        CloudWatch metric namespace.
    """

    def __init__(
        self,
        export_callback: Callable[[dict], None] | None = None,
        namespace: str = "Brainmass/V3",
    ) -> None:
        self._export_callback = export_callback
        self._namespace = namespace
        self._lock = threading.Lock()
        self._export_history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_metrics(self, metrics: dict) -> None:
        """Export a metrics dict via the pluggable callback.

        The metrics dict is wrapped in a CloudWatch-compatible envelope
        with namespace and timestamp, then passed to the callback.

        Parameters
        ----------
        metrics : dict
            Arbitrary metrics payload.  Typical keys include
            ``token_consumption``, ``cost_per_agent``, ``cache_hit_rates``.
        """
        envelope: dict = {
            "namespace": self._namespace,
            "timestamp": datetime.now(UTC).isoformat(),
            "metric_data": self._to_metric_data(metrics),
        }

        with self._lock:
            self._export_history.append(envelope)

        if self._export_callback is not None:
            try:
                self._export_callback(envelope)
            except Exception:
                logger.warning(
                    "CloudWatch export callback failed for namespace %s",
                    self._namespace,
                )
        else:
            logger.debug(
                "CloudWatch export (no callback): %s",
                json.dumps(envelope, default=str),
            )

    def get_dashboard_config(self) -> dict:
        """Return a CloudWatch dashboard JSON config template.

        The template defines widgets for token consumption, cost
        tracking, cache performance, and agent activity.
        """
        return {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "title": "Token Consumption",
                        "metrics": [
                            [self._namespace, "InputTokens"],
                            [self._namespace, "OutputTokens"],
                            [self._namespace, "CachedTokens"],
                        ],
                        "period": 60,
                        "stat": "Sum",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Cost (USD)",
                        "metrics": [
                            [self._namespace, "CostUSD"],
                        ],
                        "period": 60,
                        "stat": "Sum",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Cache Hit Rate",
                        "metrics": [
                            [self._namespace, "CacheHitRate"],
                        ],
                        "period": 60,
                        "stat": "Average",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Active Agents",
                        "metrics": [
                            [self._namespace, "ActiveAgentCount"],
                        ],
                        "period": 60,
                        "stat": "Maximum",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Model Tier Distribution",
                        "metrics": [
                            [self._namespace, "OpusCalls"],
                            [self._namespace, "SonnetCalls"],
                            [self._namespace, "HaikuCalls"],
                        ],
                        "period": 60,
                        "stat": "Sum",
                    },
                },
                {
                    "type": "metric",
                    "properties": {
                        "title": "Hook Execution Latency",
                        "metrics": [
                            [self._namespace, "HookDurationMs"],
                        ],
                        "period": 60,
                        "stat": "Average",
                    },
                },
            ],
        }

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_export_history(self) -> list[dict]:
        """Return all previously exported metric envelopes."""
        with self._lock:
            return list(self._export_history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_metric_data(metrics: dict) -> list[dict]:
        """Flatten a metrics dict into CloudWatch MetricData entries.

        Each top-level numeric value becomes a single metric datum.
        Nested dicts are flattened with dot-separated names.
        """
        data: list[dict] = []

        def _flatten(prefix: str, obj: object) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    _flatten(f"{prefix}.{key}" if prefix else key, val)
            elif isinstance(obj, (int, float)):
                data.append({
                    "MetricName": prefix,
                    "Value": obj,
                    "Unit": "Count" if isinstance(obj, int) else "None",
                })

        _flatten("", metrics)
        return data
