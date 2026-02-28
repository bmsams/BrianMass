"""Unit tests for src/workflow/engine.py.

Tests cover:
- WorkflowEngine creation with all three modes
- Vibe mode: single phase, no gates
- Design mode: two phases, no gates
- SDLC mode: full 5-phase pipeline with approval gates
- Approval gate retry on CHANGES_REQUESTED
- Workflow abort at a gate
- Artifact persistence to disk
- State persistence and loading
- Traceability matrix generation
- Validation report generation
- Agent resolution with and without registry
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.types.core import AgentDefinition
from src.types.workflow import (
    GateDecision,
    GateResult,
    TraceabilityRow,
    WorkflowMode,
    WorkflowPhase,
)
from src.workflow.engine import (
    DEFAULT_WORKFLOW_DIR,
    PHASE_AGENT_MAP,
    PHASE_ARTIFACT_NAMES,
    WorkflowEngine,
    WorkflowResult,
    _default_approval_callback,
    _default_phase_callback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_phase_callback(responses: dict[str, str] | None = None):
    """Return a phase callback that returns canned responses by agent name."""
    calls: list[dict] = []

    def callback(agent_def: AgentDefinition, task: str, prior: dict[str, str]) -> str:
        calls.append({"agent": agent_def.name, "task": task, "prior": prior})
        if responses and agent_def.name in responses:
            return responses[agent_def.name]
        return f"[{agent_def.name}] output for: {task[:40]}"

    callback.calls = calls  # type: ignore[attr-defined]
    return callback


def _make_approval_callback(
    decisions: list[GateResult] | None = None,
):
    """Return an approval callback that returns predefined decisions."""
    idx = [0]
    calls: list[dict] = []

    def callback(
        phase: WorkflowPhase,
        summary: str,
        artifact_paths: list[str],
    ) -> GateResult:
        calls.append({"phase": phase, "summary": summary, "paths": artifact_paths})
        if decisions and idx[0] < len(decisions):
            result = decisions[idx[0]]
            idx[0] += 1
            return result
        return GateResult(decision=GateDecision.APPROVED)

    callback.calls = calls  # type: ignore[attr-defined]
    return callback


# ---------------------------------------------------------------------------
# Tests: Mode selection and basic operation
# ---------------------------------------------------------------------------

class TestWorkflowEngineCreation:
    def test_default_mode_is_sdlc(self):
        engine = WorkflowEngine()
        assert engine.mode == WorkflowMode.SDLC

    def test_vibe_mode(self):
        engine = WorkflowEngine(mode=WorkflowMode.VIBE)
        assert engine.mode == WorkflowMode.VIBE

    def test_design_mode(self):
        engine = WorkflowEngine(mode=WorkflowMode.DESIGN)
        assert engine.mode == WorkflowMode.DESIGN


class TestDefaultCallbacks:
    def test_default_phase_callback_returns_string(self):
        agent_def = AgentDefinition(name="test", description="test")
        result = _default_phase_callback(agent_def, "do stuff", {})
        assert isinstance(result, str)
        assert "test" in result

    def test_default_approval_callback_auto_approves(self):
        result = _default_approval_callback(
            WorkflowPhase.EARS_SPEC, "summary", ["/tmp/x"]
        )
        assert result.decision == GateDecision.APPROVED


# ---------------------------------------------------------------------------
# Tests: Vibe mode
# ---------------------------------------------------------------------------

class TestVibeMode:
    def test_vibe_executes_single_coder_phase(self, tmp_path):
        cb = _make_phase_callback({"sdlc-coder": "vibe output"})
        engine = WorkflowEngine(
            mode=WorkflowMode.VIBE,
            phase_callback=cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("quick fix")
        assert isinstance(result, WorkflowResult)
        assert not result.aborted
        assert "vibe output" in result.final_summary
        assert result.state.phase_statuses.get("code") == "complete"

    def test_vibe_writes_artifact(self, tmp_path):
        cb = _make_phase_callback({"sdlc-coder": "# Code\nDone"})
        engine = WorkflowEngine(
            mode=WorkflowMode.VIBE,
            phase_callback=cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("quick fix")

        artifact_path = result.state.artifacts.get("code")
        assert artifact_path is not None
        assert Path(artifact_path).read_text(encoding="utf-8") == "# Code\nDone"

    def test_vibe_persists_state(self, tmp_path):
        engine = WorkflowEngine(
            mode=WorkflowMode.VIBE,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("fix bug")

        # Find state.json in the workflow dir
        state_files = list(tmp_path.rglob("state.json"))
        assert len(state_files) == 1
        data = json.loads(state_files[0].read_text(encoding="utf-8"))
        assert data["mode"] == "vibe"
        assert data["feature_description"] == "fix bug"


# ---------------------------------------------------------------------------
# Tests: Design mode
# ---------------------------------------------------------------------------

class TestDesignMode:
    def test_design_executes_two_phases(self, tmp_path):
        cb = _make_phase_callback({
            "sdlc-ears-spec": "EARS spec output",
            "sdlc-design-doc": "Design doc output",
        })
        engine = WorkflowEngine(
            mode=WorkflowMode.DESIGN,
            phase_callback=cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("new auth feature")

        assert not result.aborted
        assert "EARS spec output" in result.final_summary
        assert "Design doc output" in result.final_summary
        assert len(cb.calls) == 2
        assert cb.calls[0]["agent"] == "sdlc-ears-spec"
        assert cb.calls[1]["agent"] == "sdlc-design-doc"

    def test_design_passes_prior_artifacts(self, tmp_path):
        cb = _make_phase_callback({
            "sdlc-ears-spec": "spec content",
            "sdlc-design-doc": "design content",
        })
        engine = WorkflowEngine(
            mode=WorkflowMode.DESIGN,
            phase_callback=cb,
            workflow_dir=str(tmp_path),
        )
        engine.run("feature")

        # Second call should have ears_spec in prior
        assert "ears_spec" in cb.calls[1]["prior"]
        assert cb.calls[1]["prior"]["ears_spec"] == "spec content"


# ---------------------------------------------------------------------------
# Tests: SDLC mode
# ---------------------------------------------------------------------------

class TestSDLCMode:
    def test_sdlc_executes_all_five_phases(self, tmp_path):
        cb = _make_phase_callback()
        gate_cb = _make_approval_callback()
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            phase_callback=cb,
            approval_callback=gate_cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("add MFA")

        assert not result.aborted
        assert len(cb.calls) == 5
        expected_agents = [
            "sdlc-ears-spec",
            "sdlc-journey-mapper",
            "sdlc-design-doc",
            "sdlc-tdd-enforcer",
            "sdlc-coder",
        ]
        actual_agents = [c["agent"] for c in cb.calls]
        assert actual_agents == expected_agents

    def test_sdlc_fires_gates_after_each_phase(self, tmp_path):
        gate_cb = _make_approval_callback()
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            approval_callback=gate_cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("add MFA")

        # 5 phases = 5 gate calls
        assert len(gate_cb.calls) == 5

    def test_sdlc_chains_artifacts_to_later_phases(self, tmp_path):
        cb = _make_phase_callback({
            "sdlc-ears-spec": "EARS: req1",
            "sdlc-journey-mapper": "Journey: step1",
            "sdlc-design-doc": "Design: cp1",
            "sdlc-tdd-enforcer": "Tests: test1",
            "sdlc-coder": "Code: task1",
        })
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            phase_callback=cb,
            workflow_dir=str(tmp_path),
        )
        engine.run("feature")

        # The coder (5th phase) should see all 4 prior artifacts
        coder_prior = cb.calls[4]["prior"]
        assert "ears_spec" in coder_prior
        assert "customer_journey" in coder_prior
        assert "design_doc" in coder_prior
        assert "test_first" in coder_prior

    def test_sdlc_generates_validation_report(self, tmp_path):
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        assert "Workflow Completion Report" in result.final_summary
        assert result.state.phase_statuses.get("validation") == "complete"

    def test_sdlc_writes_all_artifacts(self, tmp_path):
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        artifact_dir = list(tmp_path.iterdir())[0]
        files = {f.name for f in artifact_dir.iterdir()}
        expected = {
            "01-ears-spec.md",
            "02-customer-journey.md",
            "03-design-doc.md",
            "04-tests.md",
            "05-tasks.md",
            "06-validation-report.md",
            "traceability-matrix.md",
            "state.json",
        }
        assert expected.issubset(files)


# ---------------------------------------------------------------------------
# Tests: Approval gates
# ---------------------------------------------------------------------------

class TestApprovalGates:
    def test_changes_requested_retries_phase(self, tmp_path):
        """When gate returns CHANGES_REQUESTED, phase is re-executed."""
        cb = _make_phase_callback()
        gate_cb = _make_approval_callback([
            GateResult(decision=GateDecision.CHANGES_REQUESTED, feedback="add more detail"),
            GateResult(decision=GateDecision.APPROVED),
            # Remaining phases auto-approve
        ])
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            phase_callback=cb,
            approval_callback=gate_cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        # First phase ran twice (initial + retry), rest ran once each = 6 total
        assert len(cb.calls) == 6
        # The retry should include feedback in the task
        assert "Reviewer Feedback" in cb.calls[1]["task"]
        assert "add more detail" in cb.calls[1]["task"]

    def test_abort_stops_workflow(self, tmp_path):
        gate_cb = _make_approval_callback([
            GateResult(decision=GateDecision.ABORTED),
        ])
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            approval_callback=gate_cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        assert result.aborted
        assert "aborted" in result.final_summary.lower()

    def test_max_retry_attempts(self, tmp_path):
        """After 3 CHANGES_REQUESTED, auto-approves last version."""
        gate_cb = _make_approval_callback([
            GateResult(decision=GateDecision.CHANGES_REQUESTED, feedback="nope"),
            GateResult(decision=GateDecision.CHANGES_REQUESTED, feedback="still nope"),
            GateResult(decision=GateDecision.CHANGES_REQUESTED, feedback="try again"),
            # After 3 attempts, auto-approves; remaining phases auto-approve
        ])
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            phase_callback=_make_phase_callback(),
            approval_callback=gate_cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        # Should not be aborted — auto-approved after max retries
        assert not result.aborted
        # Gate decisions should show 3 changes_requested for first phase
        decisions = result.state.gate_decisions
        ears_decisions = [d for d in decisions if d["phase"] == "ears_spec"]
        assert len(ears_decisions) == 3


# ---------------------------------------------------------------------------
# Tests: State persistence and loading
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_state_round_trip(self, tmp_path):
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("test feature")

        state_files = list(tmp_path.rglob("state.json"))
        assert len(state_files) == 1

        loaded = WorkflowEngine.load_state(str(state_files[0]))
        assert loaded.workflow_id == result.state.workflow_id
        assert loaded.mode == WorkflowMode.SDLC
        assert loaded.feature_description == "test feature"


# ---------------------------------------------------------------------------
# Tests: Traceability matrix
# ---------------------------------------------------------------------------

class TestTraceability:
    def test_traceability_extracts_ears_ids(self, tmp_path):
        cb = _make_phase_callback({
            "sdlc-ears-spec": "### EARS-001\nReq 1\n### EARS-002\nReq 2",
            "sdlc-journey-mapper": "STEP-001 maps to EARS-001",
            "sdlc-design-doc": "CP-001 covers EARS-001",
            "sdlc-tdd-enforcer": "test_auth validates EARS-001",
            "sdlc-coder": "TASK-001 implements EARS-001",
        })
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            phase_callback=cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        assert result.traceability is not None
        assert len(result.traceability.rows) == 2

        # EARS-001 should have full traceability
        row1 = next(r for r in result.traceability.rows if r.ears_id == "EARS-001")
        assert "STEP-001" in row1.journey_steps
        assert "CP-001" in row1.correctness_props
        assert row1.status == "full"

        # EARS-002 should have no traceability
        row2 = next(r for r in result.traceability.rows if r.ears_id == "EARS-002")
        assert row2.status == "missing"

    def test_traceability_matrix_written_to_disk(self, tmp_path):
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        matrix_files = list(tmp_path.rglob("traceability-matrix.md"))
        assert len(matrix_files) == 1
        content = matrix_files[0].read_text(encoding="utf-8")
        assert "Traceability Matrix" in content

    def test_orphan_tasks_detected(self, tmp_path):
        cb = _make_phase_callback({
            "sdlc-ears-spec": "### EARS-001\nReq 1",
            "sdlc-journey-mapper": "",
            "sdlc-design-doc": "",
            "sdlc-tdd-enforcer": "",
            "sdlc-coder": "TASK-001 no ref\nTASK-002 EARS-001",
        })
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            phase_callback=cb,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("feature")

        assert result.traceability is not None
        assert "TASK-001" in result.traceability.orphan_tasks


# ---------------------------------------------------------------------------
# Tests: Agent resolution
# ---------------------------------------------------------------------------

class TestAgentResolution:
    def test_resolves_from_registry(self, tmp_path):
        registry = MagicMock()
        custom_def = AgentDefinition(
            name="sdlc-coder",
            description="Custom coder",
            model="opus",
            system_prompt="Custom prompt",
        )
        registry.get.return_value = custom_def

        engine = WorkflowEngine(
            mode=WorkflowMode.VIBE,
            agent_registry=registry,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("fix")

        registry.get.assert_called_with("sdlc-coder")

    def test_falls_back_to_stub_without_registry(self, tmp_path):
        """Engine works without a registry by creating stub definitions."""
        engine = WorkflowEngine(
            mode=WorkflowMode.VIBE,
            workflow_dir=str(tmp_path),
        )
        result = engine.run("fix")
        assert not result.aborted


# ---------------------------------------------------------------------------
# Tests: Phase agent map constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_all_gated_phases_have_agents(self):
        for phase in [
            WorkflowPhase.EARS_SPEC,
            WorkflowPhase.CUSTOMER_JOURNEY,
            WorkflowPhase.DESIGN_DOC,
            WorkflowPhase.TEST_FIRST,
            WorkflowPhase.CODE,
        ]:
            assert phase in PHASE_AGENT_MAP

    def test_all_phases_have_artifact_names(self):
        for phase in WorkflowPhase:
            assert phase in PHASE_ARTIFACT_NAMES
