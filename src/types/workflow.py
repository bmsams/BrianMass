"""Workflow type definitions for the SDLC Development Workflow Engine.

Defines enums, dataclasses, and type aliases for structured SDLC workflows,
EARS requirements, customer journeys, design correctness properties,
task tracking (Kiro-style R/Y/G), and traceability matrices.

These types are consumed by the workflow engine (src/workflow/engine.py)
and by custom agent templates in .brainmass/agents/.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Workflow mode and phase enums
# ---------------------------------------------------------------------------

class WorkflowMode(Enum):
    """Three development workflow modes."""
    VIBE = "vibe"        # Fast, minimal guardrails, direct coding
    DESIGN = "design"    # Research + high-level design exploration
    SDLC = "sdlc"        # Full 5-phase gated workflow


class WorkflowPhase(Enum):
    """Phases in the structured SDLC workflow."""
    EARS_SPEC = "ears_spec"
    CUSTOMER_JOURNEY = "customer_journey"
    DESIGN_DOC = "design_doc"
    TEST_FIRST = "test_first"
    CODE = "code"
    VALIDATION = "validation"


# Phase ordering for SDLC pipeline
SDLC_PHASE_ORDER: list[WorkflowPhase] = [
    WorkflowPhase.EARS_SPEC,
    WorkflowPhase.CUSTOMER_JOURNEY,
    WorkflowPhase.DESIGN_DOC,
    WorkflowPhase.TEST_FIRST,
    WorkflowPhase.CODE,
    WorkflowPhase.VALIDATION,
]


# ---------------------------------------------------------------------------
# EARS requirements types
# ---------------------------------------------------------------------------

class EARSType(Enum):
    """EARS (Easy Approach to Requirements Syntax) template types."""
    UBIQUITOUS = "ubiquitous"        # "The <system> shall <action>."
    EVENT_DRIVEN = "event_driven"    # "When <trigger>, the <system> shall <action>."
    STATE_DRIVEN = "state_driven"    # "While <precondition>, the <system> shall <action>."
    OPTIONAL = "optional"            # "Where <feature>, the <system> shall <action>."
    UNWANTED = "unwanted"            # "If <condition>, then the <system> shall <action>."


class MoSCoWPriority(Enum):
    """MoSCoW prioritization for requirements."""
    MUST = "must"
    SHOULD = "should"
    COULD = "could"
    WONT = "wont"


@dataclass
class EARSRequirement:
    """A single EARS-formatted requirement."""
    id: str                              # e.g. "EARS-001"
    type: EARSType
    system: str                          # The system/component name
    action: str                          # What the system shall do
    trigger: str | None = None           # EVENT_DRIVEN: "When <trigger>"
    precondition: str | None = None      # STATE_DRIVEN: "While <precondition>"
    feature: str | None = None           # OPTIONAL: "Where <feature>"
    unwanted_condition: str | None = None  # UNWANTED: "If <condition>"
    acceptance_criteria: list[str] = field(default_factory=list)
    priority: MoSCoWPriority = MoSCoWPriority.MUST
    rationale: str = ""
    source_text: str = ""                # Original user request text

    def to_sentence(self) -> str:
        """Render the requirement as a natural-language EARS sentence."""
        match self.type:
            case EARSType.UBIQUITOUS:
                return f"The {self.system} shall {self.action}."
            case EARSType.EVENT_DRIVEN:
                return f"When {self.trigger}, the {self.system} shall {self.action}."
            case EARSType.STATE_DRIVEN:
                return f"While {self.precondition}, the {self.system} shall {self.action}."
            case EARSType.OPTIONAL:
                return f"Where {self.feature}, the {self.system} shall {self.action}."
            case EARSType.UNWANTED:
                return f"If {self.unwanted_condition}, then the {self.system} shall {self.action}."


# ---------------------------------------------------------------------------
# Customer journey types
# ---------------------------------------------------------------------------

@dataclass
class JourneyStep:
    """A single step in a customer journey."""
    id: str                          # e.g. "STEP-001"
    persona: str
    action: str
    touchpoint: str                  # Where in the system (UI, API, etc.)
    expected_outcome: str
    emotion: str = ""
    pain_points: list[str] = field(default_factory=list)
    requirement_refs: list[str] = field(default_factory=list)  # EARS IDs
    path_type: str = "happy"         # "happy" | "error" | "edge"


@dataclass
class CustomerJourney:
    """A complete customer journey for one persona and goal."""
    id: str                          # e.g. "JOURNEY-001"
    persona: str
    goal: str
    steps: list[JourneyStep] = field(default_factory=list)
    entry_point: str = ""
    exit_point: str = ""
    requirement_coverage: list[str] = field(default_factory=list)  # EARS IDs


# ---------------------------------------------------------------------------
# Design doc and correctness property types
# ---------------------------------------------------------------------------

class CorrectnessType(Enum):
    """Types of correctness properties for design components."""
    INVARIANT = "invariant"          # Always true
    PRECONDITION = "precondition"    # Must be true before
    POSTCONDITION = "postcondition"  # Must be true after
    SAFETY = "safety"                # Bad things that must never happen
    LIVENESS = "liveness"            # Good things that must eventually happen


@dataclass
class CorrectnessProperty:
    """A correctness property attached to a design component."""
    id: str                          # e.g. "CP-001"
    component: str
    type: CorrectnessType
    property_text: str
    formal_expression: str | None = None
    test_strategy: str = ""
    requirement_refs: list[str] = field(default_factory=list)
    journey_refs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task tracking types (Kiro-style R/Y/G)
# ---------------------------------------------------------------------------

class TaskColor(Enum):
    """Red/Yellow/Green task status for Kiro-style tracking."""
    RED = "red"          # Tests exist, no code yet (all fail)
    YELLOW = "yellow"    # Code exists, some tests pass
    GREEN = "green"      # All tests pass
    BLOCKED = "blocked"  # Blocked by dependency
    DONE = "done"        # GREEN + alignment verified


@dataclass
class WorkflowTask:
    """A single implementation task with R/Y/G tracking."""
    id: str                          # e.g. "TASK-001"
    title: str
    component: str
    test_names: list[str] = field(default_factory=list)
    requirement_refs: list[str] = field(default_factory=list)
    property_refs: list[str] = field(default_factory=list)
    journey_refs: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # Task IDs
    status: TaskColor = TaskColor.RED
    tests_total: int = 0
    tests_passing: int = 0
    implementation_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Approval gate types
# ---------------------------------------------------------------------------

class GateDecision(Enum):
    """Possible decisions at an approval gate."""
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    ABORTED = "aborted"


@dataclass
class GateResult:
    """Result from an interactive approval gate."""
    decision: GateDecision
    feedback: str | None = None
    modifications: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Traceability types
# ---------------------------------------------------------------------------

@dataclass
class TraceabilityRow:
    """One row in the forward traceability matrix."""
    ears_id: str
    journey_steps: list[str] = field(default_factory=list)
    design_components: list[str] = field(default_factory=list)
    correctness_props: list[str] = field(default_factory=list)
    test_names: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    status: str = "pending"  # "full" | "partial" | "missing" | "pending"


@dataclass
class TraceabilityMatrix:
    """Cross-reference matrix across all SDLC phases."""
    workflow_id: str
    generated_after_phase: WorkflowPhase
    rows: list[TraceabilityRow] = field(default_factory=list)
    orphan_tasks: list[str] = field(default_factory=list)
    coverage_gaps: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Workflow state (persisted to .brainmass/workflow/<id>/state.json)
# ---------------------------------------------------------------------------

@dataclass
class WorkflowState:
    """Persisted state for a running workflow."""
    workflow_id: str
    mode: WorkflowMode
    current_phase: WorkflowPhase
    phase_statuses: dict[str, str] = field(default_factory=dict)  # phase → status
    artifacts: dict[str, str] = field(default_factory=dict)       # phase → file path
    traceability_matrix_path: str | None = None
    started_at: str = ""
    updated_at: str = ""
    gate_decisions: list[dict] = field(default_factory=list)
    feature_description: str = ""

