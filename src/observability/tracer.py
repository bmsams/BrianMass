"""Observability tracer facade for Brainmass v3.

Public entry points for all instrumentation consumers. Wraps the concrete
BrainmassTracer, HookLogger, and TerminalMonitor from instrumentation.py
behind simple factory/singleton accessors so callers never import the
implementation module directly.

Requirements: 16.1
"""

from __future__ import annotations

import threading

from src.observability.instrumentation import (
    BrainmassTracer,
    CloudWatchExporter,
    HookLogger,
    TerminalMonitor,
)

# ---------------------------------------------------------------------------
# Module-level singletons (lazy-initialised, thread-safe)
# ---------------------------------------------------------------------------

_lock = threading.RLock()

_default_tracer: BrainmassTracer | None = None
_hook_logger: HookLogger | None = None
_terminal_monitor: TerminalMonitor | None = None
_cloudwatch_exporter: CloudWatchExporter | None = None


# ---------------------------------------------------------------------------
# Public factory / accessor functions
# ---------------------------------------------------------------------------


def get_tracer(name: str = "brainmass") -> BrainmassTracer:
    """Return the default BrainmassTracer instance, creating it if needed.

    The tracer is a module-level singleton so all components share the same
    span store and trace ID within a process.

    When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, the Strands SDK automatically
    emits its own spans to that endpoint during Agent execution. BrainmassTracer
    operates as a standalone span store for Brainmass-specific instrumentation.

    Args:
        name: Service name attached to all spans. Defaults to 'brainmass'.

    Returns:
        The shared BrainmassTracer instance.

    Requirements: 11.1, 11.2, 11.3
    """
    global _default_tracer
    with _lock:
        if _default_tracer is None:
            _default_tracer = BrainmassTracer(service_name=name)
        return _default_tracer


def get_hook_logger() -> HookLogger:
    """Return the shared HookLogger singleton.

    The hook logger is wired to the default tracer so hook execution spans
    are emitted alongside structured log entries.

    Returns:
        The shared HookLogger instance.

    Example::

        from src.observability.tracer import get_hook_logger

        hook_logger = get_hook_logger()
        hook_logger.log_hook_execution(
            event='PreToolUse',
            handler_type='command',
            matcher='Bash',
            duration_ms=12.4,
        )
    """
    global _hook_logger
    with _lock:
        if _hook_logger is None:
            _hook_logger = HookLogger(tracer=get_tracer())
        return _hook_logger


def get_terminal_monitor() -> TerminalMonitor:
    """Return the shared TerminalMonitor singleton.

    The monitor reads spans from the default tracer to produce dashboard
    output and metrics summaries.

    Returns:
        The shared TerminalMonitor instance.

    Example::

        from src.observability.tracer import get_terminal_monitor

        monitor = get_terminal_monitor()
        print(monitor.format_dashboard())
    """
    global _terminal_monitor
    with _lock:
        if _terminal_monitor is None:
            _terminal_monitor = TerminalMonitor(tracer=get_tracer())
        return _terminal_monitor


def get_cloudwatch_exporter() -> CloudWatchExporter:
    """Return the shared CloudWatchExporter singleton.

    Returns:
        The shared CloudWatchExporter instance.

    # --- Production integration point ---
    # In production, pass a real boto3 CloudWatch callback:
    #
    #   import boto3
    #   cw = boto3.client('cloudwatch', region_name='us-east-1')
    #   def _export(envelope):
    #       cw.put_metric_data(
    #           Namespace=envelope['namespace'],
    #           MetricData=envelope['metric_data'],
    #       )
    #   exporter = CloudWatchExporter(export_callback=_export)
    """
    global _cloudwatch_exporter
    with _lock:
        if _cloudwatch_exporter is None:
            _cloudwatch_exporter = CloudWatchExporter()
        return _cloudwatch_exporter


def reset_singletons() -> None:
    """Reset all module-level singletons.

    Intended for use in tests only — allows each test to start with a
    fresh tracer/logger/monitor without cross-test span contamination.
    """
    global _default_tracer, _hook_logger, _terminal_monitor, _cloudwatch_exporter
    with _lock:
        _default_tracer = None
        _hook_logger = None
        _terminal_monitor = None
        _cloudwatch_exporter = None
