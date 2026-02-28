# Brainmass v3 — Development Workflow Engine Design Plan

## Table of Contents

1. [Current Codebase Analysis](#1-current-codebase-analysis)
2. [Issues & Gaps Identified](#2-issues--gaps-identified)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Workflow Modes](#4-workflow-modes)
5. [Phase Details — Structured SDLC Mode](#5-phase-details--structured-sdlc-mode)
6. [EARS Requirements System](#6-ears-requirements-system)
7. [Customer Journey Mapper](#7-customer-journey-mapper)
8. [Design Doc with Correctness Properties](#8-design-doc-with-correctness-properties)
9. [TDD Enforcer & UAT Script Generation](#9-tdd-enforcer--uat-script-generation)
10. [Red/Yellow/Green Task Tracker (Kiro-Style)](#10-redyellowgreen-task-tracker-kiro-style)
11. [Traceability Matrix & Alignment Hooks](#11-traceability-matrix--alignment-hooks)
12. [Approval Gates](#12-approval-gates)
13. [Final Validation Loop](#13-final-validation-loop)
14. [Local vs AgentCore Execution](#14-local-vs-agentcore-execution)
15. [File-by-File Implementation Plan](#15-file-by-file-implementation-plan)
16. [Data Structures & Types](#16-data-structures--types)
17. [Hook Integration Map](#17-hook-integration-map)

---

## 1. Current Codebase Analysis

### Architecture Overview

Brainmass v3 is a Python-based enterprise agentic coding system with a three-tier architecture:

| Tier | Components | Role |
|------|-----------|------|
| **Client** | CLI/IDE Extension, Plugin Registry | User interface and plugin management |
| **Control** | Orchestrator, Context Manager, Policy Engine | Request processing, context triage, lifecycle hooks |
| **Data** | MCP Tool Gateway, Worker Pool | Tool execution, subagents, teams, loops |

### Current Flow (8-Step Processing)

```
User Request
  → 1. SessionStart hooks fire
  → 2. UserPromptSubmit hooks fire (can modify input)
  → 3. Classify request complexity → select effort level (Quick/Standard/Deep)
  → 4. Select model tier via Cost_Governor (Opus/Sonnet/Haiku)
  → 5. Decompose into tasks if complex (select topology: hierarchical/teams/loop)
  → 6. Execute tool calls with PreToolUse/PostToolUse hooks
  → 7. Update context via Context_Manager (semantic triage)
  → 8. Fire Stop hooks (can block completion)
```

### Existing Hook System (12 Events)

| Event | When | Blocking? | Key Use |
|-------|------|-----------|---------|
| `SessionStart` | Session starts/resumes | No | Load session state |
| `UserPromptSubmit` | User submits prompt | Can modify | Input transformation |
| `PreToolUse` | Before tool executes | Can block (exit 2) | Permission checks, guardrails |
| `PostToolUse` | After tool succeeds | No | Logging, side effects |
| `PostToolUseFailure` | After tool fails | No | Error tracking |
| `PermissionRequest` | Tool permission requested | Can allow/deny | Auto-approve/reject tools |
| `PreCompact` | Before compaction | No | Backup critical context |
| `Notification` | Alert sent | No | External notifications |
| `Stop` | Agent finishes | Can block | Force agent to continue |
| `SubagentStop` | Subagent finishes | No | Phase transition trigger |
| `SessionEnd` | Session terminates | No | Cleanup |
| `Setup` | Init/maintenance flags | No | Setup tasks |

### Existing Orchestration Patterns

1. **LoopRunner** (Self-improving loop / "Ralph Wiggum" pattern):
   - Fresh agent per iteration with clean context
   - Structured context file (task, acceptance criteria, constraints, learnings, failed approaches)
   - Git commit after each iteration
   - Learning accumulation across iterations
   - AcceptanceCriteriaGate for loop termination

2. **CompoundLoopOrchestrator** (Pipeline):
   - YAML-configurable stages with input/output mapping
   - Each stage is a LoopContext-driven agent
   - SubagentStop hooks fire after each stage
   - Pipeline state persisted to `.brainmass/pipeline-state.json`
   - Stage flow: Analysis → Planning → Execution

3. **Safety Controls**:
   - ErrorMonitor: Auto-pause on 3+ repeated errors
   - DiffSizeChecker: Abort on oversized diffs or out-of-scope files
   - StopFileChecker: `.auto/stop` sentinel file
   - AcceptanceCriteriaGate: Validate all criteria met

### Pluggable Callback Pattern

The codebase consistently uses a **pluggable callback pattern** for local vs production:

```python
# Local mode: inject test/callback stub
runner = LoopRunner(agent_callback=my_local_callback)

# Production mode: uses _production_loop_callback (Strands Agent + BedrockModel)
runner = LoopRunner()  # defaults to production callback
```

This pattern must be preserved and extended for the workflow engine.

---

## 2. Issues & Gaps Identified

### Structural Issues

| # | Issue | Impact | Location |
|---|-------|--------|----------|
| 1 | **No development workflow modes** | Can't enforce structured SDLC, vibe coding, or design research sessions | Missing entirely |
| 2 | **CompoundLoop is generic** | Only chains Analysis → Planning → Execution — no spec/design/test/code phases | `src/agents/compound_loop.py` |
| 3 | **No EARS-based requirements** | No structured requirements framework; requirements are prose in markdown | `.kiro/specs/*/requirements.md` |
| 4 | **No customer journey mapping** | No user flow validation; no way to ensure all user paths are covered | Missing entirely |
| 5 | **No traceability matrix generation** | `TASK_QA_MAP.json` exists but is static/manual, not auto-generated from artifacts | `.kiro/specs/*/TASK_QA_MAP.json` |
| 6 | **No TDD enforcement** | Tests and code co-created; no test-first guarantee or UAT script generation | Missing entirely |
| 7 | **No red/yellow/green task staging** | Tasks are binary `[x]/[-]/[ ]` checkboxes — no color-coded progress | `.kiro/specs/*/tasks.md` |
| 8 | **Hooks don't trigger phase transitions** | `SubagentStop` fires but doesn't trigger traceability checks or phase gates | `src/hooks/hook_engine.py` |
| 9 | **No alignment verification loop** | No mechanism to verify all artifacts (spec → journey → design → test → code) are consistent | Missing entirely |
| 10 | **No final validation loop** | No end-to-end check of user flows, wiring, completeness | Missing entirely |

### Code Quality Issues

| # | Issue | Location |
|---|-------|----------|
| 11 | `_execute_handler` references `self._handler_callback` but it's a class variable, not instance — works but is fragile | `hook_engine.py:503-511` |
| 12 | Step numbering is wrong in `_run_iteration` (two "Step 2" comments) | `loop_runner.py:321-333` |
| 13 | `_production_loop_callback` has unreachable code after `return` (GraphBuilder upgrade path) | `loop_runner.py:88-102` |
| 14 | `_production_stage_callback` has unreachable code after `return` (Workflow upgrade path) | `compound_loop.py:110-118` |
| 15 | No type narrowing on `self._hook_engine` — it's typed as `object | None` | `loop_runner.py:187`, `compound_loop.py:327` |

---

## 3. Proposed Architecture

### Multi-Mode Workflow Engine

```
┌─────────────────────────────────────────────────────────────┐
│                    WorkflowEngine                            │
│                                                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │
│  │  Vibe Mode   │ │ Design Mode  │ │ Structured SDLC Mode │ │
│  │  (fast,      │ │ (research,   │ │ (gated phases,       │ │
│  │   minimal    │ │  HLD,        │ │  EARS → Journey →    │ │
│  │   guardrails)│ │  explore)    │ │  Design → Test →     │ │
│  │              │ │              │ │  Code)               │ │
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘ │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│                  ┌───────▼────────┐                          │
│                  │  PhaseRunner   │                          │
│                  │  (extends      │                          │
│                  │   LoopRunner)  │                          │
│                  └───────┬────────┘                          │
│                          │                                   │
│    ┌─────────────────────┼─────────────────────┐            │
│    │                     │                     │            │
│  ┌─▼──────────┐  ┌──────▼───────┐  ┌─────────▼──────┐     │
│  │ Approval   │  │ Traceability │  │ Alignment      │     │
│  │ Gate       │  │ Matrix Hook  │  │ Verifier       │     │
│  └────────────┘  └──────────────┘  └────────────────┘     │
│                                                              │
│              Sits alongside CompoundLoopOrchestrator         │
│              Uses same hook_engine, cost_governor, etc.      │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **New `WorkflowEngine` alongside `CompoundLoopOrchestrator`** — doesn't replace it, sits next to it. Both use the same hook engine, cost governor, and pluggable callback infrastructure.

2. **Three workflow modes** configured via YAML or CLI argument:
   - `vibe` — Fast, minimal guardrails, direct coding
   - `design` — Research and high-level design exploration
   - `sdlc` — Full 5-phase gated workflow with EARS → Journey → Design → Test → Code

3. **Interactive approval gates** — Agent pauses within the session and prompts for approval before proceeding to the next phase.

4. **Hook-driven traceability** — `SubagentStop` hooks automatically trigger traceability matrix generation after each phase.

5. **Artifact storage** — All workflow artifacts (specs, journeys, designs, tests, traceability) stored in `.brainmass/workflow/<workflow-id>/`.

---

## 4. Workflow Modes

### Mode 1: Vibe Coding (`vibe`)

```
User Request → Quick classify → Code directly → Run tests → Done
```

- Minimal ceremony, maximum speed
- No specs, no design docs, no gates
- Just code, test, iterate
- Uses LoopRunner internally with relaxed acceptance criteria
- Good for prototyping, small features, bug fixes

### Mode 2: Design Session (`design`)

```
User Request → Research (codebase, web, docs)
            → Architecture exploration
            → High-Level Design document
            → Component diagram
            → Decision records
            → [Optional] Gate for approval
            → Done
```

- Focuses on research and high-level thinking
- Explores codebase, reads docs, builds mental model
- Produces HLD, component diagrams, decision records
- Can optionally gate before finishing
- Good for new features, major refactors, architecture decisions

### Mode 3: Structured SDLC (`sdlc`)

```
User Request
  → Phase 1: EARS Spec          ──→ [GATE: Approval] ──┐
  → Phase 2: Customer Journey   ←──────────────────────┘──→ [GATE: Approval] ──┐
  → Phase 3: Design Doc         ←──────────────────────────────────────────────┘──→ [GATE: Approval] ──┐
  → Phase 4: Test-First (TDD)   ←────────────────────────────────────────────────────────────────────┘──→ [GATE: Approval] ──┐
  → Phase 5: Code (R/Y/G Tasks) ←──────────────────────────────────────────────────────────────────────────────────────────┘
  → Final Validation Loop
  → Done

  [After EACH phase]: SubagentStop hook → Traceability Matrix → Alignment Check
```

Full details in Section 5 below.

---

## 5. Phase Details — Structured SDLC Mode

### Phase 1: EARS Spec (Requirements)

**Input**: User's feature request / problem statement
**Output**: `.brainmass/workflow/<id>/01-ears-spec.md`

Agent behavior:
1. Analyze the user's request
2. Identify entities, triggers, preconditions, states
3. Write EARS requirements using the 5 templates (see Section 6)
4. Include acceptance criteria for each requirement
5. Write the spec to disk
6. **GATE**: Present spec to user, ask for approval
7. On approval, SubagentStop fires → traceability hook builds initial matrix

### Phase 2: Customer Journey

**Input**: Approved EARS spec
**Output**: `.brainmass/workflow/<id>/02-customer-journey.md`

Agent behavior:
1. Read the approved EARS spec
2. Identify all user personas from the requirements
3. Map each persona's journey (steps, touchpoints, emotions, pain points)
4. Identify happy paths and error paths
5. Cross-reference every journey step back to an EARS requirement
6. Write the journey map to disk
7. **GATE**: Present journey to user, ask for approval
8. On approval, SubagentStop fires → traceability hook updates matrix

### Phase 3: Design Document

**Input**: Approved EARS spec + Customer Journey
**Output**: `.brainmass/workflow/<id>/03-design-doc.md`

Agent behavior:
1. Read spec and journey
2. Design component architecture
3. Define interfaces and data flow
4. Write **correctness properties** for each component:
   - Invariants (always true)
   - Pre-conditions (must be true before)
   - Post-conditions (must be true after)
   - Safety properties (bad things that must never happen)
   - Liveness properties (good things that must eventually happen)
5. Cross-reference every design decision back to requirements and journeys
6. Write the design doc to disk
7. **GATE**: Present design to user, ask for approval
8. On approval, SubagentStop fires → traceability hook updates matrix

### Phase 4: Test-First (TDD)

**Input**: Approved Design Doc + Correctness Properties
**Output**:
- `.brainmass/workflow/<id>/04-tests/` (test files)
- `.brainmass/workflow/<id>/04-uat-script.md` (UAT script)

Agent behavior:
1. Read design doc and correctness properties
2. Generate test files BEFORE any implementation code:
   - Unit tests for each correctness property
   - Integration tests for each interface
   - Edge case tests from customer journey error paths
3. Generate a UAT script from the test suite:
   - Manual verification steps for each customer journey
   - Expected outcomes tied to acceptance criteria
4. Verify all tests are **RED** (failing, no code yet)
5. Write tests and UAT script to disk
6. **GATE**: Present tests and UAT script for approval
7. On approval, SubagentStop fires → traceability hook updates matrix

### Phase 5: Code (Red/Yellow/Green Tasks)

**Input**: Approved Tests + Design Doc
**Output**:
- Actual implementation code
- `.brainmass/workflow/<id>/05-tasks.md` (Kiro-style task list)

Agent behavior:
1. Generate task list from design doc components (see Section 10)
2. Each task starts as **RED** (test exists, code doesn't)
3. Implement code for each task:
   - Write implementation
   - Run associated tests
   - Task moves to **YELLOW** if some tests pass
   - Task moves to **GREEN** when all tests pass
4. Check off tasks Kiro-style as they complete
5. After all tasks are GREEN, run final validation loop (Section 13)
6. SubagentStop fires → final traceability matrix

---

## 6. EARS Requirements System

### EARS Template Types

EARS (Easy Approach to Requirements Syntax) uses 5 structured templates:

#### 1. Ubiquitous (always active)
```
The <system> shall <action>.
```
Example: "The authentication module shall encrypt all passwords using bcrypt."

#### 2. Event-Driven (triggered by event)
```
When <trigger>, the <system> shall <action>.
```
Example: "When a user submits the login form, the system shall validate credentials within 2 seconds."

#### 3. State-Driven (while in a state)
```
While <precondition>, the <system> shall <action>.
```
Example: "While the user is authenticated, the system shall maintain a session token."

#### 4. Optional Feature (conditional)
```
Where <feature is supported>, the <system> shall <action>.
```
Example: "Where MFA is enabled, the system shall require a second factor after password validation."

#### 5. Unwanted Behavior (negative/safety)
```
If <unwanted condition>, then the <system> shall <action>.
```
Example: "If 5 failed login attempts occur, then the system shall lock the account for 15 minutes."

### EARS Requirement Data Structure

```python
@dataclass
class EARSRequirement:
    id: str                        # e.g., "EARS-001"
    type: EARSType                 # UBIQUITOUS | EVENT_DRIVEN | STATE_DRIVEN | OPTIONAL | UNWANTED
    system: str                    # The system/component name
    action: str                    # What the system shall do
    trigger: str | None            # For EVENT_DRIVEN: "When <trigger>"
    precondition: str | None       # For STATE_DRIVEN: "While <precondition>"
    feature: str | None            # For OPTIONAL: "Where <feature>"
    unwanted_condition: str | None # For UNWANTED: "If <condition>"
    acceptance_criteria: list[str] # Testable acceptance criteria
    priority: str                  # MUST | SHOULD | COULD | WONT (MoSCoW)
    rationale: str                 # Why this requirement exists
    source_text: str               # Original user request text that spawned this
```

### EARS Spec File Format

```markdown
# EARS Requirements Specification
## Feature: <feature name>
## Date: <date>
## Source: <original user request>

---

### EARS-001 [MUST] — Ubiquitous
> The authentication module shall encrypt all passwords using bcrypt.

**Acceptance Criteria:**
- [ ] AC-001.1: Passwords are hashed with bcrypt cost factor ≥ 12
- [ ] AC-001.2: No plaintext passwords stored in database or logs

**Rationale:** Security best practice; prevents credential exposure.

---

### EARS-002 [MUST] — Event-Driven
> When a user submits the login form, the system shall validate credentials within 2 seconds.

**Trigger:** User submits login form
**Acceptance Criteria:**
- [ ] AC-002.1: P95 login validation < 2000ms
- [ ] AC-002.2: Invalid credentials return 401 with generic message

**Rationale:** UX responsiveness; OWASP authentication guidelines.
```

---

## 7. Customer Journey Mapper

### Journey Map Structure

```python
@dataclass
class JourneyStep:
    id: str                     # e.g., "STEP-001"
    persona: str                # e.g., "New User", "Admin"
    action: str                 # What the user does
    touchpoint: str             # Where in the system (UI, API, etc.)
    expected_outcome: str       # What should happen
    emotion: str                # Expected user emotion
    pain_points: list[str]      # Potential frustrations
    requirement_refs: list[str] # Back-references to EARS IDs
    path_type: str              # "happy" | "error" | "edge"

@dataclass
class CustomerJourney:
    id: str                     # e.g., "JOURNEY-001"
    persona: str
    goal: str
    steps: list[JourneyStep]
    entry_point: str
    exit_point: str
    requirement_coverage: list[str]  # All EARS IDs covered by this journey
```

### Journey Map File Format

```markdown
# Customer Journey Map
## Feature: <feature name>

---

### Journey 1: New User Registration (Happy Path)
**Persona:** New User
**Goal:** Create an account and access the dashboard

| Step | Action | Touchpoint | Expected Outcome | Req Refs |
|------|--------|------------|-----------------|----------|
| 1 | Navigate to signup page | Web UI | Signup form displayed | EARS-003 |
| 2 | Fill in email + password | Signup form | Form validates inputs | EARS-004 |
| 3 | Submit form | Signup API | Account created, redirect to dashboard | EARS-005, EARS-001 |
| 4 | See dashboard | Dashboard UI | Personalized welcome shown | EARS-006 |

**Error Paths:**
- Step 2 → Invalid email → Show validation error (EARS-007)
- Step 2 → Weak password → Show strength indicator (EARS-008)
- Step 3 → Duplicate email → Show "already registered" message (EARS-009)
```

---

## 8. Design Doc with Correctness Properties

### Correctness Property Types

```python
@dataclass
class CorrectnessProperty:
    id: str                     # e.g., "CP-001"
    component: str              # Which component this applies to
    type: CorrectnessType       # INVARIANT | PRECONDITION | POSTCONDITION | SAFETY | LIVENESS
    property_text: str          # Natural language description
    formal_expression: str | None  # Optional formal notation
    test_strategy: str          # How to verify this property
    requirement_refs: list[str] # Back-references to EARS IDs
    journey_refs: list[str]     # Back-references to Journey step IDs
```

### Design Doc File Format

```markdown
# Design Document
## Feature: <feature name>

---

## 1. Component Architecture

### AuthenticationService
**Responsibility:** Handle user login, registration, and session management
**Interfaces:** `login(email, password) → Session`, `register(email, password) → User`

#### Correctness Properties

| ID | Type | Property | Test Strategy | Reqs |
|----|------|----------|---------------|------|
| CP-001 | INVARIANT | All stored passwords are bcrypt hashed | DB query check: no plaintext | EARS-001 |
| CP-002 | PRECONDITION | `login()` requires non-empty email and password | Unit test with empty inputs | EARS-002 |
| CP-003 | POSTCONDITION | After `login()` succeeds, session token is valid for 24h | Token expiry test | EARS-002 |
| CP-004 | SAFETY | System never returns password hash in API response | Response body scan | EARS-001 |
| CP-005 | LIVENESS | Account lockout eventually expires after 15 minutes | Time-based test | EARS-010 |
```

---

## 9. TDD Enforcer & UAT Script Generation

### TDD Flow

```
Design Doc (correctness properties)
    │
    ▼
Generate Tests (all RED)
    │
    ├── Unit tests from correctness properties
    ├── Integration tests from interfaces
    ├── Journey tests from customer journeys (error paths)
    │
    ▼
Generate UAT Script (from journeys + acceptance criteria)
    │
    ▼
[GATE: Review tests + UAT script]
    │
    ▼
Phase 5: Write code until all GREEN
```

### UAT Script Format

```markdown
# UAT Script
## Feature: <feature name>
## Tester: <name>
## Date: <date>

---

### Scenario 1: New User Registration (JOURNEY-001)
**Preconditions:** No existing account for test email

| Step | Action | Expected Result | EARS Ref | Pass/Fail |
|------|--------|----------------|----------|-----------|
| 1 | Navigate to /signup | Signup form loads within 1s | EARS-003 | ☐ |
| 2 | Enter "test@example.com" and "SecureP@ss1" | No validation errors | EARS-004 | ☐ |
| 3 | Click "Create Account" | Redirect to /dashboard within 2s | EARS-005 | ☐ |
| 4 | Verify dashboard shows "Welcome, test@example.com" | Personalized greeting | EARS-006 | ☐ |

### Scenario 2: Duplicate Registration (JOURNEY-001, Error Path)
...
```

### TDD Enforcer Data Structure

```python
@dataclass
class TestMapping:
    test_file: str              # Path to test file
    test_name: str              # Test function name
    correctness_property: str   # CP-ID this test verifies
    requirement_refs: list[str] # EARS IDs
    journey_refs: list[str]     # Journey step IDs
    status: TestStatus          # RED | YELLOW | GREEN
```

---

## 10. Red/Yellow/Green Task Tracker (Kiro-Style)

### Task States

| Color | Symbol | Meaning | Criteria |
|-------|--------|---------|----------|
| 🔴 **RED** | `[R]` | Tests exist, no code | All associated tests fail |
| 🟡 **YELLOW** | `[Y]` | Code exists, partial pass | Some tests pass, some fail |
| 🟢 **GREEN** | `[G]` | All tests pass | All associated tests pass |
| ⬜ **BLOCKED** | `[B]` | Blocked by dependency | Dependency task not GREEN |
| ✅ **DONE** | `[✓]` | Verified & checked off | GREEN + alignment verified |

### Kiro-Style Task File Format

```markdown
# Task List — <feature name>
## Workflow: <workflow-id>
## Phase: 5 — Code Implementation

---

### Component: AuthenticationService

- [✓] TASK-001: Implement password hashing utility
  - Tests: `test_password_hashing.py::test_bcrypt_hash`, `::test_cost_factor`
  - Reqs: EARS-001
  - Properties: CP-001
  - Status: GREEN → VERIFIED

- [G] TASK-002: Implement login endpoint
  - Tests: `test_auth.py::test_login_success`, `::test_login_invalid`, `::test_login_timing`
  - Reqs: EARS-002
  - Properties: CP-002, CP-003
  - Status: GREEN (3/3 tests pass)

- [Y] TASK-003: Implement session management
  - Tests: `test_session.py::test_token_creation` ✅, `::test_token_expiry` ❌, `::test_refresh` ❌
  - Reqs: EARS-002
  - Properties: CP-003
  - Status: YELLOW (1/3 tests pass)

- [R] TASK-004: Implement account lockout
  - Tests: `test_lockout.py::test_lock_after_5_failures`, `::test_lockout_expiry`
  - Reqs: EARS-010
  - Properties: CP-005
  - Status: RED (0/2 tests pass)

- [B] TASK-005: Implement MFA integration
  - Tests: `test_mfa.py::test_totp_validation`, `::test_mfa_fallback`
  - Reqs: EARS-011
  - Properties: CP-006
  - Status: BLOCKED (depends on TASK-002)

---

### Progress Summary

| Status | Count | Percentage |
|--------|-------|-----------|
| ✅ DONE | 1 | 20% |
| 🟢 GREEN | 1 | 20% |
| 🟡 YELLOW | 1 | 20% |
| 🔴 RED | 1 | 20% |
| ⬜ BLOCKED | 1 | 20% |
```

### Task Data Structure

```python
@dataclass
class WorkflowTask:
    id: str                       # e.g., "TASK-001"
    title: str
    component: str
    test_files: list[str]         # Test file paths
    test_names: list[str]         # Individual test function names
    requirement_refs: list[str]   # EARS IDs
    property_refs: list[str]      # Correctness property IDs
    journey_refs: list[str]       # Journey step IDs
    dependencies: list[str]       # Task IDs this depends on
    status: TaskColor             # RED | YELLOW | GREEN | BLOCKED | DONE
    tests_total: int
    tests_passing: int
    implementation_files: list[str]  # Code files for this task
```

---

## 11. Traceability Matrix & Alignment Hooks

### Traceability Matrix

The traceability matrix is a **cross-reference document** generated after each phase that maps every artifact backward through all previous phases to ensure nothing is missed.

### Matrix Format

```markdown
# Traceability Matrix
## Workflow: <workflow-id>
## Generated after: Phase <N>

---

### Forward Traceability (Requirement → ... → Code)

| EARS ID | Journey Steps | Design Components | Correctness Props | Tests | Tasks | Status |
|---------|--------------|-------------------|-------------------|-------|-------|--------|
| EARS-001 | STEP-003 | AuthService | CP-001, CP-004 | 3 tests | TASK-001 | ✅ Full |
| EARS-002 | STEP-002, STEP-003 | AuthService | CP-002, CP-003 | 5 tests | TASK-002, TASK-003 | ⚠️ Partial |
| EARS-010 | — | AuthService | CP-005 | 2 tests | TASK-004 | ❌ Missing journey |

### Backward Traceability (Code → ... → Requirement)

| Task | Tests | Properties | Design | Journey | EARS | Status |
|------|-------|-----------|--------|---------|------|--------|
| TASK-001 | 2 tests | CP-001 | AuthService | STEP-003 | EARS-001 | ✅ Full |
| TASK-006 | 1 test | — | — | — | — | ❌ ORPHAN (no requirement!) |

### Coverage Gaps

- ⚠️ EARS-010 has no customer journey coverage
- ❌ TASK-006 is an orphan — not traced to any requirement
- ⚠️ CP-005 (liveness: lockout expiry) has no integration test
```

### How Traceability Hooks Work

```python
# Registered on SubagentStop — fires after each phase completes
def traceability_hook(context: HookContext) -> HookResult:
    """
    1. Read all artifacts generated so far
    2. Build forward + backward traceability
    3. Identify gaps (missing coverage, orphans)
    4. Write matrix to .brainmass/workflow/<id>/traceability-matrix.md
    5. If gaps found, add warnings to HookResult.additional_context
    """
```

### Alignment Verifier

Runs as a SubagentStop hook alongside the traceability matrix:

```python
def alignment_verifier(context: HookContext) -> HookResult:
    """
    Checks:
    1. Every EARS requirement has at least one journey step
    2. Every journey step maps to at least one design component
    3. Every correctness property has at least one test
    4. Every test maps to a task
    5. No orphan artifacts exist
    6. Numbering/IDs are consistent across documents
    7. No broken cross-references
    """
```

---

## 12. Approval Gates

### Interactive Gate Implementation

```python
@dataclass
class ApprovalGate:
    phase: str                   # Which phase just completed
    artifact_paths: list[str]    # Files to review
    summary: str                 # Human-readable summary of what was produced
    approval_prompt: str         # Question to ask the user

class GateResult:
    approved: bool
    feedback: str | None         # User's feedback if not approved
    modifications: list[str]     # Specific change requests
```

### Gate Flow

```
Phase N completes
    │
    ▼
Write artifacts to disk
    │
    ▼
Fire SubagentStop → Traceability Matrix + Alignment Check
    │
    ▼
Present to user:
    "Phase N complete. Artifacts written to:
     - .brainmass/workflow/<id>/01-ears-spec.md
     - .brainmass/workflow/<id>/traceability-matrix.md

     [Approve] [Request Changes] [Abort]"
    │
    ├── Approve → Continue to Phase N+1
    ├── Request Changes → Re-run Phase N with feedback
    └── Abort → Save state, exit gracefully
```

### Implementation Detail

The gate uses the `Stop` hook event's ability to **block completion**:

```python
# When the agent tries to "finish" a phase, the Stop hook blocks it
# and instead presents the approval prompt to the user.
# The agent only proceeds when the user approves.
```

---

## 13. Final Validation Loop

After all Phase 5 tasks are GREEN, a final validation loop runs:

### Validation Checks

1. **User Flow Verification**: Walk through every customer journey and verify:
   - Each step has corresponding implemented code
   - The happy path works end-to-end
   - Error paths are handled

2. **Wiring Check**: Verify all components are properly connected:
   - All interfaces called from the right places
   - No dead code or unused imports
   - All dependencies resolved

3. **Completeness Check**:
   - All EARS requirements have full traceability
   - All tests pass (re-run full suite)
   - No orphan tasks or artifacts
   - UAT script matches implementation

4. **Final Traceability Matrix**: Generate the definitive matrix with all columns filled

5. **Summary Report**: Generate a human-readable summary:

```markdown
# Workflow Completion Report
## Feature: <name>
## Workflow ID: <id>

### Phase Summary
| Phase | Status | Artifacts | Duration |
|-------|--------|-----------|----------|
| 1. EARS Spec | ✅ Approved | 12 requirements | 5 min |
| 2. Customer Journey | ✅ Approved | 4 journeys, 23 steps | 8 min |
| 3. Design Doc | ✅ Approved | 3 components, 15 properties | 12 min |
| 4. Tests | ✅ Approved | 28 tests, 1 UAT script | 10 min |
| 5. Code | ✅ Complete | 8 tasks, all GREEN | 45 min |

### Traceability Summary
- Requirements: 12/12 fully traced (100%)
- Journey steps: 23/23 covered (100%)
- Correctness properties: 15/15 tested (100%)
- Tests: 28/28 passing (100%)
- Orphan artifacts: 0

### Final Validation
- ✅ All user flows verified
- ✅ All components properly wired
- ✅ No dead code detected
- ✅ UAT script matches implementation
```

---

## 14. Local vs AgentCore Execution

### How Each Component Works in Both Modes

| Component | Local Mode | AgentCore Mode |
|-----------|-----------|---------------|
| **WorkflowEngine** | Direct Python execution, callbacks | Strands Agent + BedrockModel |
| **Phase execution** | `_local_phase_callback` | `_production_phase_callback` (Bedrock) |
| **Approval gates** | Interactive CLI prompt | AgentCore Memory + webhook notification |
| **Traceability hook** | Local file I/O + Python analysis | Same + CloudWatch metrics |
| **Test execution** | `subprocess.run(["pytest", ...])` | Same (container has test runner) |
| **Artifact storage** | `.brainmass/workflow/` on disk | Same + S3 backup via AgentCore Memory |
| **State persistence** | `.brainmass/workflow-state.json` | Same + AgentCore Memory short-term |
| **Observability** | Local logging | OpenTelemetry → CloudWatch |

### Pluggable Callback Pattern (Preserved)

```python
class WorkflowEngine:
    def __init__(
        self,
        mode: WorkflowMode,              # VIBE | DESIGN | SDLC
        phase_callback: PhaseCallback | None = None,  # Inject for local
        approval_callback: ApprovalCallback | None = None,  # Inject for local
        hook_engine: BrainmassHookEngine | None = None,
        cost_governor: CostGovernor | None = None,
        ...
    ):
        self._phase_callback = phase_callback or _production_phase_callback
        self._approval_callback = approval_callback or _interactive_approval_callback
```

---

## 15. File-by-File Implementation Plan

### New Files to Create

| # | File | Purpose |
|---|------|---------|
| 1 | `src/types/workflow.py` | All new types: EARSRequirement, JourneyStep, CorrectnessProperty, WorkflowTask, etc. |
| 2 | `src/workflow/engine.py` | Main WorkflowEngine class with mode selection and phase orchestration |
| 3 | `src/workflow/phases.py` | Phase implementations: EARSSpecPhase, JourneyPhase, DesignPhase, TestPhase, CodePhase |
| 4 | `src/workflow/ears_parser.py` | EARS requirements parser and generator |
| 5 | `src/workflow/journey_mapper.py` | Customer journey mapper |
| 6 | `src/workflow/design_generator.py` | Design doc generator with correctness properties |
| 7 | `src/workflow/tdd_enforcer.py` | TDD enforcement: test generation, UAT script, red/yellow/green tracking |
| 8 | `src/workflow/task_tracker.py` | Kiro-style task tracker with R/Y/G status |
| 9 | `src/workflow/traceability.py` | Traceability matrix generation and gap analysis |
| 10 | `src/workflow/alignment.py` | Alignment verifier (cross-artifact consistency) |
| 11 | `src/workflow/approval_gate.py` | Interactive approval gate implementation |
| 12 | `src/workflow/validation_loop.py` | Final validation loop |
| 13 | `src/workflow/artifact_store.py` | Artifact file I/O (read/write spec, journey, design, tests, matrix) |
| 14 | `src/workflow/__init__.py` | Package init with public API |
| 15 | `tests/unit/test_ears_parser.py` | Tests for EARS parser |
| 16 | `tests/unit/test_journey_mapper.py` | Tests for journey mapper |
| 17 | `tests/unit/test_design_generator.py` | Tests for design doc generator |
| 18 | `tests/unit/test_tdd_enforcer.py` | Tests for TDD enforcer |
| 19 | `tests/unit/test_task_tracker.py` | Tests for task tracker |
| 20 | `tests/unit/test_traceability.py` | Tests for traceability matrix |
| 21 | `tests/unit/test_alignment.py` | Tests for alignment verifier |
| 22 | `tests/unit/test_workflow_engine.py` | Tests for workflow engine |
| 23 | `tests/unit/test_approval_gate.py` | Tests for approval gate |
| 24 | `tests/unit/test_validation_loop.py` | Tests for final validation loop |

### Existing Files to Modify

| # | File | Changes |
|---|------|---------|
| 1 | `src/types/core.py` | Add `WorkflowMode` enum, `WorkflowPhase` enum, `WorkflowState` dataclass |
| 2 | `src/hooks/hook_engine.py` | Fix `_handler_callback` class var issue; no structural changes needed (hook system already supports what we need) |
| 3 | `src/agents/loop_runner.py` | Fix step numbering comments; no structural changes needed |
| 4 | `src/agents/compound_loop.py` | Fix unreachable code after return; no structural changes needed |

---

## 16. Data Structures & Types

### New Types in `src/types/workflow.py`

```python
class WorkflowMode(Enum):
    VIBE = "vibe"
    DESIGN = "design"
    SDLC = "sdlc"

class WorkflowPhase(Enum):
    EARS_SPEC = "ears_spec"
    CUSTOMER_JOURNEY = "customer_journey"
    DESIGN_DOC = "design_doc"
    TEST_FIRST = "test_first"
    CODE = "code"
    VALIDATION = "validation"

class EARSType(Enum):
    UBIQUITOUS = "ubiquitous"
    EVENT_DRIVEN = "event_driven"
    STATE_DRIVEN = "state_driven"
    OPTIONAL = "optional"
    UNWANTED = "unwanted"

class CorrectnessType(Enum):
    INVARIANT = "invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    SAFETY = "safety"
    LIVENESS = "liveness"

class TaskColor(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    BLOCKED = "blocked"
    DONE = "done"

class GateDecision(Enum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    ABORTED = "aborted"

@dataclass
class WorkflowState:
    workflow_id: str
    mode: WorkflowMode
    current_phase: WorkflowPhase
    phase_statuses: dict[str, str]
    artifacts: dict[str, str]       # phase → artifact file path
    traceability_matrix_path: str | None
    started_at: str
    updated_at: str
    gate_decisions: list[dict]      # history of approval decisions
    feature_description: str
```

---

## 17. Hook Integration Map

### How Hooks Drive the Workflow

```
Phase Starts
    │
    ├── SessionStart hook (if first phase)
    │   └── Load workflow state from disk
    │
    ├── UserPromptSubmit hook
    │   └── Inject phase context (EARS spec, journey, etc.)
    │
    ├── PreToolUse hooks
    │   └── Enforce TDD (block code writes if tests don't exist yet in Phase 4)
    │   └── Enforce scope (only modify files relevant to current task)
    │
    ├── PostToolUse hooks
    │   └── Track test results (update R/Y/G status)
    │   └── Update task tracker
    │
    ├── Stop hook
    │   └── Gate: Block completion, present artifacts for approval
    │   └── Re-run phase if changes requested
    │
    └── SubagentStop hook (after phase approved)
        ├── Generate/update traceability matrix
        ├── Run alignment verifier
        ├── Write workflow state to disk
        └── Log phase completion metrics
```

### New Hook Registrations

```python
# Registered by WorkflowEngine at startup
workflow_hooks = {
    HookEvent.SESSION_START: [load_workflow_state],
    HookEvent.SUBAGENT_STOP: [
        traceability_matrix_generator,
        alignment_verifier,
        workflow_state_writer,
    ],
    HookEvent.STOP: [approval_gate],
    HookEvent.PRE_TOOL_USE: [tdd_enforcer, scope_guard],
    HookEvent.POST_TOOL_USE: [test_result_tracker, task_status_updater],
}
```

---

## Implementation Order

1. **`src/types/workflow.py`** — All new types (no dependencies)
2. **`src/workflow/artifact_store.py`** — File I/O for artifacts
3. **`src/workflow/ears_parser.py`** — EARS parser (depends on types)
4. **`src/workflow/journey_mapper.py`** — Journey mapper (depends on types, EARS)
5. **`src/workflow/design_generator.py`** — Design doc generator (depends on types, EARS, journey)
6. **`src/workflow/tdd_enforcer.py`** — TDD enforcer (depends on types, design)
7. **`src/workflow/task_tracker.py`** — Task tracker (depends on types, TDD)
8. **`src/workflow/traceability.py`** — Traceability matrix (depends on all artifact types)
9. **`src/workflow/alignment.py`** — Alignment verifier (depends on traceability)
10. **`src/workflow/approval_gate.py`** — Approval gate (depends on types)
11. **`src/workflow/validation_loop.py`** — Final validation loop (depends on all above)
12. **`src/workflow/phases.py`** — Phase implementations (depends on all above)
13. **`src/workflow/engine.py`** — Main engine (depends on all above)
14. **Tests for all components**
15. **Fix existing code issues** (hook_engine, loop_runner, compound_loop)
