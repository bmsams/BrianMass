"""Unit tests for src/types/workflow.py.

Tests cover:
- Enum values for all workflow enums
- EARS requirement sentence rendering
- SDLC phase ordering
- Dataclass defaults and field types
"""

from __future__ import annotations

from src.types.workflow import (
    CorrectnessProperty,
    CorrectnessType,
    CustomerJourney,
    EARSRequirement,
    EARSType,
    GateDecision,
    GateResult,
    JourneyStep,
    MoSCoWPriority,
    SDLC_PHASE_ORDER,
    TaskColor,
    TraceabilityMatrix,
    TraceabilityRow,
    WorkflowMode,
    WorkflowPhase,
    WorkflowState,
    WorkflowTask,
)


class TestEnums:
    def test_workflow_mode_values(self):
        assert WorkflowMode.VIBE.value == "vibe"
        assert WorkflowMode.DESIGN.value == "design"
        assert WorkflowMode.SDLC.value == "sdlc"

    def test_workflow_phase_values(self):
        assert WorkflowPhase.EARS_SPEC.value == "ears_spec"
        assert WorkflowPhase.CUSTOMER_JOURNEY.value == "customer_journey"
        assert WorkflowPhase.DESIGN_DOC.value == "design_doc"
        assert WorkflowPhase.TEST_FIRST.value == "test_first"
        assert WorkflowPhase.CODE.value == "code"
        assert WorkflowPhase.VALIDATION.value == "validation"

    def test_ears_type_values(self):
        assert len(EARSType) == 5

    def test_correctness_type_values(self):
        assert len(CorrectnessType) == 5

    def test_task_color_values(self):
        assert TaskColor.RED.value == "red"
        assert TaskColor.GREEN.value == "green"
        assert TaskColor.DONE.value == "done"

    def test_gate_decision_values(self):
        assert GateDecision.APPROVED.value == "approved"
        assert GateDecision.CHANGES_REQUESTED.value == "changes_requested"
        assert GateDecision.ABORTED.value == "aborted"

    def test_moscow_priority_values(self):
        assert len(MoSCoWPriority) == 4


class TestSDLCPhaseOrder:
    def test_has_six_phases(self):
        assert len(SDLC_PHASE_ORDER) == 6

    def test_starts_with_ears_ends_with_validation(self):
        assert SDLC_PHASE_ORDER[0] == WorkflowPhase.EARS_SPEC
        assert SDLC_PHASE_ORDER[-1] == WorkflowPhase.VALIDATION


class TestEARSRequirement:
    def test_ubiquitous_sentence(self):
        req = EARSRequirement(
            id="EARS-001",
            type=EARSType.UBIQUITOUS,
            system="authentication module",
            action="encrypt all passwords",
        )
        assert req.to_sentence() == "The authentication module shall encrypt all passwords."

    def test_event_driven_sentence(self):
        req = EARSRequirement(
            id="EARS-002",
            type=EARSType.EVENT_DRIVEN,
            system="system",
            action="validate credentials",
            trigger="a user submits the login form",
        )
        assert "When a user submits the login form" in req.to_sentence()

    def test_state_driven_sentence(self):
        req = EARSRequirement(
            id="EARS-003",
            type=EARSType.STATE_DRIVEN,
            system="system",
            action="maintain a session token",
            precondition="the user is authenticated",
        )
        assert "While the user is authenticated" in req.to_sentence()

    def test_optional_sentence(self):
        req = EARSRequirement(
            id="EARS-004",
            type=EARSType.OPTIONAL,
            system="system",
            action="require a second factor",
            feature="MFA is enabled",
        )
        assert "Where MFA is enabled" in req.to_sentence()

    def test_unwanted_sentence(self):
        req = EARSRequirement(
            id="EARS-005",
            type=EARSType.UNWANTED,
            system="system",
            action="lock the account",
            unwanted_condition="5 failed login attempts occur",
        )
        assert "If 5 failed login attempts occur" in req.to_sentence()

    def test_default_priority_is_must(self):
        req = EARSRequirement(
            id="EARS-001",
            type=EARSType.UBIQUITOUS,
            system="sys",
            action="do",
        )
        assert req.priority == MoSCoWPriority.MUST

    def test_acceptance_criteria_default_empty(self):
        req = EARSRequirement(
            id="EARS-001",
            type=EARSType.UBIQUITOUS,
            system="sys",
            action="do",
        )
        assert req.acceptance_criteria == []


class TestJourneyStep:
    def test_default_path_type(self):
        step = JourneyStep(
            id="STEP-001",
            persona="New User",
            action="click signup",
            touchpoint="Web UI",
            expected_outcome="form shown",
        )
        assert step.path_type == "happy"

    def test_requirement_refs(self):
        step = JourneyStep(
            id="STEP-001",
            persona="User",
            action="login",
            touchpoint="API",
            expected_outcome="session",
            requirement_refs=["EARS-001", "EARS-002"],
        )
        assert len(step.requirement_refs) == 2


class TestWorkflowTask:
    def test_default_status_is_red(self):
        task = WorkflowTask(id="TASK-001", title="impl", component="auth")
        assert task.status == TaskColor.RED
        assert task.tests_total == 0
        assert task.tests_passing == 0


class TestGateResult:
    def test_approved_no_feedback(self):
        gr = GateResult(decision=GateDecision.APPROVED)
        assert gr.feedback is None

    def test_changes_requested_with_feedback(self):
        gr = GateResult(
            decision=GateDecision.CHANGES_REQUESTED,
            feedback="needs more detail",
        )
        assert gr.feedback == "needs more detail"


class TestTraceabilityRow:
    def test_default_status(self):
        row = TraceabilityRow(ears_id="EARS-001")
        assert row.status == "pending"
        assert row.journey_steps == []


class TestWorkflowState:
    def test_default_fields(self):
        state = WorkflowState(
            workflow_id="wf-test",
            mode=WorkflowMode.SDLC,
            current_phase=WorkflowPhase.EARS_SPEC,
        )
        assert state.phase_statuses == {}
        assert state.artifacts == {}
        assert state.gate_decisions == []
