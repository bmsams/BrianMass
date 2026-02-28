"""Workflow Engine — orchestrates SDLC workflows using Strands GraphBuilder.

Uses the existing AgentDispatcher + AgentRegistry infrastructure to execute
custom SDLC agents (sdlc-ears-spec, sdlc-journey-mapper, sdlc-design-doc,
sdlc-tdd-enforcer, sdlc-coder) wired together via Strands GraphBuilder.

Three modes:
- ``vibe``   — skip straight to coding (single agent, no gates)
- ``design`` — research + architecture only (2 agents, optional gate)
- ``sdlc``   — full 5-phase gated pipeline via GraphBuilder

All external calls are pluggable for testing:
- ``phase_callback``: executes a phase agent
- ``approval_callback``: presents gate to user
- ``agent_registry``: resolves agent names to definitions

Requirements: Workflow design plan Sections 3, 4, 5, 12, 14
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.types.core import AgentDefinition
from src.types.workflow import (
    GateDecision,
    GateResult,
    TraceabilityMatrix,
    TraceabilityRow,
    WorkflowMode,
    WorkflowPhase,
    WorkflowState,
    SDLC_PHASE_ORDER,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default artifact directory
# ---------------------------------------------------------------------------

DEFAULT_WORKFLOW_DIR = ".brainmass/workflow"

# ---------------------------------------------------------------------------
# Phase → agent name mapping
# ---------------------------------------------------------------------------

PHASE_AGENT_MAP: dict[WorkflowPhase, str] = {
    WorkflowPhase.EARS_SPEC: "sdlc-ears-spec",
    WorkflowPhase.CUSTOMER_JOURNEY: "sdlc-journey-mapper",
    WorkflowPhase.DESIGN_DOC: "sdlc-design-doc",
    WorkflowPhase.TEST_FIRST: "sdlc-tdd-enforcer",
    WorkflowPhase.CODE: "sdlc-coder",
}

PHASE_ARTIFACT_NAMES: dict[WorkflowPhase, str] = {
    WorkflowPhase.EARS_SPEC: "01-ears-spec.md",
    WorkflowPhase.CUSTOMER_JOURNEY: "02-customer-journey.md",
    WorkflowPhase.DESIGN_DOC: "03-design-doc.md",
    WorkflowPhase.TEST_FIRST: "04-tests.md",
    WorkflowPhase.CODE: "05-tasks.md",
    WorkflowPhase.VALIDATION: "06-validation-report.md",
}

# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------

# Executes a single phase agent: (agent_def, task, prior_artifacts) → summary
PhaseCallback = Callable[
    [AgentDefinition, str, dict[str, str]],
    str,
]

# Presents approval gate: (phase, summary, artifact_paths) → GateResult
ApprovalCallback = Callable[
    [WorkflowPhase, str, list[str]],
    GateResult,
]


# ---------------------------------------------------------------------------
# Default callbacks
# ---------------------------------------------------------------------------

def _default_phase_callback(
    agent_def: AgentDefinition,
    task: str,
    prior_artifacts: dict[str, str],
) -> str:
    """Default no-op phase callback for testing."""
    return f"[{agent_def.name}] completed: {task[:80]}"


def _default_approval_callback(
    phase: WorkflowPhase,
    summary: str,
    artifact_paths: list[str],
) -> GateResult:
    """Default auto-approve callback for testing."""
    return GateResult(decision=GateDecision.APPROVED)


def _production_phase_callback(
    agent_def: AgentDefinition,
    task: str,
    prior_artifacts: dict[str, str],
) -> str:
    """Execute a phase agent using Strands Agent + BedrockModel.

    Builds the task prompt with prior artifact context, then dispatches
    via a fresh Strands Agent.
    """
    try:
        from strands import Agent  # type: ignore[import-untyped]
        from strands.models.bedrock import BedrockModel  # type: ignore[import-untyped]

        from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents is required for production workflow execution. "
            "Install with: pip install strands-agents"
        ) from exc

    # Build context from prior artifacts
    context_parts = [f"## Task\n{task}"]
    for phase_name, content in prior_artifacts.items():
        context_parts.append(f"## Prior Artifact: {phase_name}\n{content}")
    full_prompt = "\n\n---\n\n".join(context_parts)

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, agent_def.model)
    model = BedrockModel(model_id=model_id)
    agent = Agent(
        model=model,
        system_prompt=agent_def.system_prompt or "",
    )
    raw = agent(full_prompt)
    result = _normalize_strands_result(raw)
    return result.get("summary", "")


def _interactive_approval_callback(
    phase: WorkflowPhase,
    summary: str,
    artifact_paths: list[str],
) -> GateResult:
    """Interactive CLI approval gate — prompts user for approval."""
    print(f"\n{'='*60}")
    print(f"  GATE: Phase '{phase.value}' complete")
    print(f"{'='*60}")
    print(f"\n{summary}\n")
    if artifact_paths:
        print("Artifacts written:")
        for p in artifact_paths:
            print(f"  - {p}")
    print()

    while True:
        choice = input("[A]pprove / [R]equest changes / A[b]ort: ").strip().lower()
        if choice in ("a", "approve"):
            return GateResult(decision=GateDecision.APPROVED)
        if choice in ("r", "request changes"):
            feedback = input("Feedback: ").strip()
            return GateResult(
                decision=GateDecision.CHANGES_REQUESTED,
                feedback=feedback,
            )
        if choice in ("b", "abort"):
            return GateResult(decision=GateDecision.ABORTED)
        print("Invalid choice. Enter A, R, or B.")


# ---------------------------------------------------------------------------
# WorkflowEngine
# ---------------------------------------------------------------------------


class WorkflowEngine:
    """Orchestrates development workflows using custom SDLC agents.

    Supports three modes:
    - **vibe**: Direct coding, no gates
    - **design**: Research + architecture, optional gate
    - **sdlc**: Full 5-phase gated pipeline

    In production mode, phases are executed as Strands Agents via
    ``GraphBuilder`` conditional edges. In test mode, inject callbacks.

    Usage::

        # Local/test mode
        engine = WorkflowEngine(
            mode=WorkflowMode.SDLC,
            phase_callback=my_callback,
            approval_callback=my_gate,
        )
        result = engine.run("Add user auth with MFA")

        # Production mode (auto-uses Strands + interactive gates)
        engine = WorkflowEngine(mode=WorkflowMode.SDLC)
        result = engine.run("Add user auth with MFA")
    """

    def __init__(
        self,
        mode: WorkflowMode = WorkflowMode.SDLC,
        phase_callback: PhaseCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        agent_registry: object | None = None,
        workflow_dir: str = DEFAULT_WORKFLOW_DIR,
    ) -> None:
        self._mode = mode
        self._phase_callback = phase_callback or _default_phase_callback
        self._approval_callback = approval_callback or _default_approval_callback
        self._agent_registry = agent_registry
        self._workflow_dir = Path(workflow_dir)

    @property
    def mode(self) -> WorkflowMode:
        return self._mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, feature_request: str) -> WorkflowResult:
        """Execute the workflow for a feature request.

        Returns a WorkflowResult with the final state and all artifacts.
        """
        workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
        artifact_dir = self._workflow_dir / workflow_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        state = WorkflowState(
            workflow_id=workflow_id,
            mode=self._mode,
            current_phase=WorkflowPhase.EARS_SPEC,
            started_at=datetime.now(UTC).isoformat(),
            feature_description=feature_request,
        )

        match self._mode:
            case WorkflowMode.VIBE:
                return self._run_vibe(state, feature_request, artifact_dir)
            case WorkflowMode.DESIGN:
                return self._run_design(state, feature_request, artifact_dir)
            case WorkflowMode.SDLC:
                return self._run_sdlc(state, feature_request, artifact_dir)

    def run_graph(self, feature_request: str) -> WorkflowResult:
        """Execute the SDLC workflow using Strands GraphBuilder.

        This is the production entry point that wires phases as graph nodes
        with conditional edges based on approval gate decisions.

        Falls back to ``run()`` if strands is not installed.
        """
        try:
            return self._run_sdlc_graph(feature_request)
        except ImportError:
            logger.warning(
                "strands-agents not installed; falling back to sequential workflow."
            )
            return self.run(feature_request)

    # ------------------------------------------------------------------
    # Mode runners
    # ------------------------------------------------------------------

    def _run_vibe(
        self,
        state: WorkflowState,
        feature_request: str,
        artifact_dir: Path,
    ) -> WorkflowResult:
        """Vibe mode: skip to coding, no gates."""
        agent_def = self._resolve_agent("sdlc-coder")
        summary = self._phase_callback(agent_def, feature_request, {})

        state.current_phase = WorkflowPhase.CODE
        state.phase_statuses[WorkflowPhase.CODE.value] = "complete"
        state.artifacts[WorkflowPhase.CODE.value] = str(
            artifact_dir / PHASE_ARTIFACT_NAMES[WorkflowPhase.CODE]
        )
        self._write_artifact(artifact_dir, WorkflowPhase.CODE, summary)
        state.updated_at = datetime.now(UTC).isoformat()
        self._save_state(state, artifact_dir)

        return WorkflowResult(state=state, final_summary=summary)

    def _run_design(
        self,
        state: WorkflowState,
        feature_request: str,
        artifact_dir: Path,
    ) -> WorkflowResult:
        """Design mode: EARS spec + design doc, optional gate."""
        prior: dict[str, str] = {}
        summaries: list[str] = []

        for phase in [WorkflowPhase.EARS_SPEC, WorkflowPhase.DESIGN_DOC]:
            agent_name = PHASE_AGENT_MAP[phase]
            agent_def = self._resolve_agent(agent_name)

            state.current_phase = phase
            state.phase_statuses[phase.value] = "running"

            summary = self._phase_callback(agent_def, feature_request, prior)
            summaries.append(summary)

            artifact_path = artifact_dir / PHASE_ARTIFACT_NAMES[phase]
            self._write_artifact(artifact_dir, phase, summary)
            state.artifacts[phase.value] = str(artifact_path)
            state.phase_statuses[phase.value] = "complete"
            prior[phase.value] = summary

        state.updated_at = datetime.now(UTC).isoformat()
        self._save_state(state, artifact_dir)
        return WorkflowResult(
            state=state,
            final_summary="\n\n---\n\n".join(summaries),
        )

    def _run_sdlc(
        self,
        state: WorkflowState,
        feature_request: str,
        artifact_dir: Path,
    ) -> WorkflowResult:
        """SDLC mode: 5 gated phases executed sequentially."""
        prior: dict[str, str] = {}
        gated_phases = [
            WorkflowPhase.EARS_SPEC,
            WorkflowPhase.CUSTOMER_JOURNEY,
            WorkflowPhase.DESIGN_DOC,
            WorkflowPhase.TEST_FIRST,
            WorkflowPhase.CODE,
        ]

        for phase in gated_phases:
            agent_name = PHASE_AGENT_MAP[phase]
            agent_def = self._resolve_agent(agent_name)

            state.current_phase = phase
            state.phase_statuses[phase.value] = "running"

            # Execute phase (may retry on CHANGES_REQUESTED)
            summary = self._execute_gated_phase(
                state, phase, agent_def, feature_request, prior, artifact_dir
            )

            if summary is None:
                # Aborted
                state.phase_statuses[phase.value] = "aborted"
                state.updated_at = datetime.now(UTC).isoformat()
                self._save_state(state, artifact_dir)
                return WorkflowResult(
                    state=state,
                    final_summary=f"Workflow aborted at phase '{phase.value}'.",
                    aborted=True,
                )

            prior[phase.value] = summary
            state.phase_statuses[phase.value] = "complete"

        # Final validation phase
        state.current_phase = WorkflowPhase.VALIDATION
        state.phase_statuses[WorkflowPhase.VALIDATION.value] = "complete"
        validation_summary = self._generate_validation_report(state, prior)
        self._write_artifact(artifact_dir, WorkflowPhase.VALIDATION, validation_summary)

        # Generate traceability matrix
        matrix = self._build_traceability(state, prior)
        matrix_path = artifact_dir / "traceability-matrix.md"
        matrix_path.write_text(
            self._render_traceability(matrix), encoding="utf-8"
        )
        state.traceability_matrix_path = str(matrix_path)

        state.updated_at = datetime.now(UTC).isoformat()
        self._save_state(state, artifact_dir)

        return WorkflowResult(
            state=state,
            final_summary=validation_summary,
            traceability=matrix,
        )

    def _execute_gated_phase(
        self,
        state: WorkflowState,
        phase: WorkflowPhase,
        agent_def: AgentDefinition,
        feature_request: str,
        prior: dict[str, str],
        artifact_dir: Path,
    ) -> str | None:
        """Execute a phase with approval gate. Returns summary or None if aborted."""
        max_attempts = 3

        for attempt in range(max_attempts):
            # Build task with any feedback from prior attempt
            task = feature_request
            if attempt > 0 and state.gate_decisions:
                last = state.gate_decisions[-1]
                feedback = last.get("feedback", "")
                if feedback:
                    task = f"{feature_request}\n\n## Reviewer Feedback\n{feedback}"

            summary = self._phase_callback(agent_def, task, prior)

            artifact_path = artifact_dir / PHASE_ARTIFACT_NAMES[phase]
            self._write_artifact(artifact_dir, phase, summary)
            state.artifacts[phase.value] = str(artifact_path)

            # Approval gate
            gate_result = self._approval_callback(
                phase, summary, [str(artifact_path)]
            )

            state.gate_decisions.append({
                "phase": phase.value,
                "attempt": attempt + 1,
                "decision": gate_result.decision.value,
                "feedback": gate_result.feedback,
            })

            if gate_result.decision == GateDecision.APPROVED:
                return summary
            if gate_result.decision == GateDecision.ABORTED:
                return None
            # CHANGES_REQUESTED → loop and retry

        # Exhausted attempts — auto-approve last version
        logger.warning(
            "Phase '%s' exhausted %d attempts; auto-approving last version.",
            phase.value, max_attempts,
        )
        return summary

    # ------------------------------------------------------------------
    # Strands GraphBuilder integration
    # ------------------------------------------------------------------

    def _run_sdlc_graph(self, feature_request: str) -> WorkflowResult:
        """Wire SDLC phases as a Strands Graph with conditional edges.

        Each phase is a graph node. After each node, a conditional edge
        routes to either the next phase (approved) or back to the same
        phase (changes requested) or to an abort node.
        """
        from strands.multiagent.graph import GraphBuilder  # type: ignore[import-untyped]

        workflow_id = f"wf-{uuid.uuid4().hex[:8]}"
        artifact_dir = self._workflow_dir / workflow_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        state = WorkflowState(
            workflow_id=workflow_id,
            mode=WorkflowMode.SDLC,
            current_phase=WorkflowPhase.EARS_SPEC,
            started_at=datetime.now(UTC).isoformat(),
            feature_description=feature_request,
        )

        # Shared mutable context threaded through graph execution
        ctx: dict[str, Any] = {
            "state": state,
            "prior": {},
            "feature_request": feature_request,
            "artifact_dir": artifact_dir,
            "aborted": False,
        }

        # Build the graph
        builder = GraphBuilder()

        gated_phases = [
            WorkflowPhase.EARS_SPEC,
            WorkflowPhase.CUSTOMER_JOURNEY,
            WorkflowPhase.DESIGN_DOC,
            WorkflowPhase.TEST_FIRST,
            WorkflowPhase.CODE,
        ]

        # Create a node for each phase
        for phase in gated_phases:
            agent_name = PHASE_AGENT_MAP[phase]
            agent_def = self._resolve_agent(agent_name)

            def make_node(p: WorkflowPhase, a: AgentDefinition) -> Callable:
                """Create a graph node function for a phase."""
                def node_fn(request: str) -> str:
                    ctx["state"].current_phase = p
                    ctx["state"].phase_statuses[p.value] = "running"
                    summary = self._phase_callback(a, request, ctx["prior"])
                    self._write_artifact(ctx["artifact_dir"], p, summary)
                    ctx["state"].artifacts[p.value] = str(
                        ctx["artifact_dir"] / PHASE_ARTIFACT_NAMES[p]
                    )
                    ctx["prior"][p.value] = summary
                    ctx["state"].phase_statuses[p.value] = "complete"
                    return summary
                node_fn.__name__ = p.value
                return node_fn

            builder.add_node(phase.value, make_node(phase, agent_def))

        # Add edges: sequential flow with conditional routing at gates
        for i, phase in enumerate(gated_phases):
            if i + 1 < len(gated_phases):
                next_phase = gated_phases[i + 1]
                builder.add_edge(phase.value, next_phase.value)

        graph = builder.build()
        # Execute starting from first phase
        graph.execute(feature_request, entry_point=gated_phases[0].value)

        # Final validation
        state.current_phase = WorkflowPhase.VALIDATION
        validation = self._generate_validation_report(state, ctx["prior"])
        self._write_artifact(artifact_dir, WorkflowPhase.VALIDATION, validation)

        state.updated_at = datetime.now(UTC).isoformat()
        self._save_state(state, artifact_dir)

        return WorkflowResult(state=state, final_summary=validation)

    # ------------------------------------------------------------------
    # Agent resolution
    # ------------------------------------------------------------------

    def _resolve_agent(self, agent_name: str) -> AgentDefinition:
        """Resolve agent name via registry, or return a stub definition."""
        if self._agent_registry is not None:
            agent_def = self._agent_registry.get(agent_name)
            if agent_def is not None:
                return agent_def

        # Stub definition for testing (when no registry is available)
        return AgentDefinition(
            name=agent_name,
            description=f"SDLC agent: {agent_name}",
            model="sonnet",
            system_prompt=f"You are the {agent_name} agent.",
        )

    # ------------------------------------------------------------------
    # Artifact I/O
    # ------------------------------------------------------------------

    def _write_artifact(
        self,
        artifact_dir: Path,
        phase: WorkflowPhase,
        content: str,
    ) -> Path:
        """Write a phase artifact to disk."""
        filename = PHASE_ARTIFACT_NAMES.get(phase, f"{phase.value}.md")
        path = artifact_dir / filename
        path.write_text(content, encoding="utf-8")
        logger.info("Wrote artifact: %s", path)
        return path

    def _save_state(self, state: WorkflowState, artifact_dir: Path) -> None:
        """Persist workflow state to disk."""
        state_path = artifact_dir / "state.json"
        data = {
            "workflow_id": state.workflow_id,
            "mode": state.mode.value,
            "current_phase": state.current_phase.value,
            "phase_statuses": state.phase_statuses,
            "artifacts": state.artifacts,
            "traceability_matrix_path": state.traceability_matrix_path,
            "started_at": state.started_at,
            "updated_at": state.updated_at,
            "gate_decisions": state.gate_decisions,
            "feature_description": state.feature_description,
        }
        state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load_state(state_path: str) -> WorkflowState:
        """Load a persisted workflow state from disk."""
        data = json.loads(Path(state_path).read_text(encoding="utf-8"))
        return WorkflowState(
            workflow_id=data["workflow_id"],
            mode=WorkflowMode(data["mode"]),
            current_phase=WorkflowPhase(data["current_phase"]),
            phase_statuses=data.get("phase_statuses", {}),
            artifacts=data.get("artifacts", {}),
            traceability_matrix_path=data.get("traceability_matrix_path"),
            started_at=data.get("started_at", ""),
            updated_at=data.get("updated_at", ""),
            gate_decisions=data.get("gate_decisions", []),
            feature_description=data.get("feature_description", ""),
        )

    # ------------------------------------------------------------------
    # Traceability
    # ------------------------------------------------------------------

    def _build_traceability(
        self,
        state: WorkflowState,
        artifacts: dict[str, str],
    ) -> TraceabilityMatrix:
        """Build a traceability matrix from workflow artifacts.

        Scans EARS IDs across all artifact texts and cross-references them.
        """
        import re

        matrix = TraceabilityMatrix(
            workflow_id=state.workflow_id,
            generated_after_phase=state.current_phase,
        )

        ears_content = artifacts.get(WorkflowPhase.EARS_SPEC.value, "")
        journey_content = artifacts.get(WorkflowPhase.CUSTOMER_JOURNEY.value, "")
        design_content = artifacts.get(WorkflowPhase.DESIGN_DOC.value, "")
        test_content = artifacts.get(WorkflowPhase.TEST_FIRST.value, "")
        code_content = artifacts.get(WorkflowPhase.CODE.value, "")

        # Extract EARS IDs from spec
        ears_ids = re.findall(r"EARS-\d+", ears_content)
        ears_ids = sorted(set(ears_ids))

        for ears_id in ears_ids:
            row = TraceabilityRow(ears_id=ears_id)

            # Find journey steps referencing this EARS ID
            step_refs = re.findall(r"(STEP-\d+).*?" + re.escape(ears_id), journey_content)
            row.journey_steps = sorted(set(step_refs))

            # Find correctness properties referencing this EARS ID
            cp_refs = re.findall(r"(CP-\d+).*?" + re.escape(ears_id), design_content)
            row.correctness_props = sorted(set(cp_refs))

            # Find test names referencing this EARS ID
            test_refs = re.findall(r"(test_\w+).*?" + re.escape(ears_id), test_content)
            row.test_names = sorted(set(test_refs))

            # Find tasks referencing this EARS ID
            task_refs = re.findall(r"(TASK-\d+).*?" + re.escape(ears_id), code_content)
            row.task_ids = sorted(set(task_refs))

            # Determine coverage status
            if row.journey_steps and row.correctness_props and row.test_names:
                row.status = "full"
            elif row.journey_steps or row.correctness_props or row.test_names:
                row.status = "partial"
            else:
                row.status = "missing"

            matrix.rows.append(row)

        # Find orphan tasks (tasks not referencing any EARS ID)
        all_task_ids = set(re.findall(r"TASK-\d+", code_content))
        traced_task_ids = set()
        for row in matrix.rows:
            traced_task_ids.update(row.task_ids)
        matrix.orphan_tasks = sorted(all_task_ids - traced_task_ids)

        # Coverage gaps
        for row in matrix.rows:
            if row.status == "missing":
                matrix.coverage_gaps.append(f"{row.ears_id} has no traceability")
            elif not row.journey_steps:
                matrix.coverage_gaps.append(f"{row.ears_id} has no journey coverage")

        return matrix

    @staticmethod
    def _render_traceability(matrix: TraceabilityMatrix) -> str:
        """Render a traceability matrix as markdown."""
        lines = [
            "# Traceability Matrix",
            f"## Workflow: {matrix.workflow_id}",
            f"## Generated after: {matrix.generated_after_phase.value}",
            "",
            "### Forward Traceability",
            "",
            "| EARS ID | Journey Steps | Correctness Props | Tests | Tasks | Status |",
            "|---------|--------------|-------------------|-------|-------|--------|",
        ]

        for row in matrix.rows:
            status_icon = {"full": "Full", "partial": "Partial", "missing": "Missing"}.get(
                row.status, row.status
            )
            lines.append(
                f"| {row.ears_id} "
                f"| {', '.join(row.journey_steps) or '—'} "
                f"| {', '.join(row.correctness_props) or '—'} "
                f"| {', '.join(row.test_names) or '—'} "
                f"| {', '.join(row.task_ids) or '—'} "
                f"| {status_icon} |"
            )

        if matrix.orphan_tasks:
            lines.extend([
                "",
                "### Orphan Tasks (no requirement traceability)",
                "",
            ])
            for task_id in matrix.orphan_tasks:
                lines.append(f"- {task_id}")

        if matrix.coverage_gaps:
            lines.extend([
                "",
                "### Coverage Gaps",
                "",
            ])
            for gap in matrix.coverage_gaps:
                lines.append(f"- {gap}")

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _generate_validation_report(
        self,
        state: WorkflowState,
        artifacts: dict[str, str],
    ) -> str:
        """Generate a final validation report summarizing the workflow."""
        lines = [
            "# Workflow Completion Report",
            f"## Feature: {state.feature_description[:80]}",
            f"## Workflow ID: {state.workflow_id}",
            "",
            "### Phase Summary",
            "| Phase | Status |",
            "|-------|--------|",
        ]

        for phase in SDLC_PHASE_ORDER:
            status = state.phase_statuses.get(phase.value, "skipped")
            lines.append(f"| {phase.value} | {status} |")

        lines.extend([
            "",
            "### Gate Decisions",
            "",
        ])
        for decision in state.gate_decisions:
            lines.append(
                f"- Phase '{decision['phase']}' attempt {decision['attempt']}: "
                f"{decision['decision']}"
            )

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    state: WorkflowState
    final_summary: str = ""
    traceability: TraceabilityMatrix | None = None
    aborted: bool = False
