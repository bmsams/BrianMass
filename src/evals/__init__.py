"""Brainmass v3 — Evaluation Suite for SDLC Workflow Agents.

Production-grade evaluation framework built on ``strands-agents-evals`` that
tests every phase of the SDLC workflow pipeline with diverse, adversarial,
and edge-case scenarios.

Evaluators:
- ``EARSSpecEvaluator``  — validates requirements completeness, EARS format, testability
- ``JourneyMapEvaluator`` — validates journey coverage, persona diversity, traceability
- ``DesignDocEvaluator``  — validates correctness properties, component coverage
- ``TDDEvaluator``        — validates test-first discipline, coverage, RED state
- ``CoderEvaluator``      — validates R/Y/G task tracking, implementation quality
- ``TraceabilityEvaluator`` — validates cross-phase traceability matrix completeness
- ``WorkflowE2EEvaluator`` — validates end-to-end workflow execution

Usage::

    from src.evals.runner import run_all_evaluations
    reports = run_all_evaluations()
"""
