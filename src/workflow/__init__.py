"""Brainmass v3 — Development Workflow Engine.

Provides a structured SDLC workflow built on Strands SDK GraphBuilder
and the existing AgentDispatcher/AgentRegistry infrastructure.

Three modes:
- ``vibe``  — fast, minimal guardrails, direct coding
- ``design`` — research + high-level design exploration
- ``sdlc``   — 5-phase gated workflow: EARS → Journey → Design → Test → Code

Usage::

    from src.workflow.engine import WorkflowEngine
    from src.types.workflow import WorkflowMode

    engine = WorkflowEngine(mode=WorkflowMode.SDLC)
    result = engine.run("Add user authentication with MFA support")
"""

from src.types.workflow import (
    WorkflowMode,
    WorkflowPhase,
    WorkflowState,
)

__all__ = [
    "WorkflowMode",
    "WorkflowPhase",
    "WorkflowState",
]
