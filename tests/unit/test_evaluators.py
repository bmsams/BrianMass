"""Unit tests for src/evals/evaluators.py.

Tests every evaluator with diverse inputs to ensure scoring is accurate
and catches real quality issues. Each evaluator is tested with:
- A high-quality output (should PASS with score ≥ 0.75)
- A low-quality output (should FAIL with score < 0.75)
- An empty output (should FAIL)
- Edge cases specific to the evaluator

These tests validate the evaluators themselves — they ensure the eval
system can distinguish good agent output from bad.
"""

from __future__ import annotations

import pytest

from src.evals.evaluators import (
    CoderEvaluator,
    DesignDocEvaluator,
    EARSSpecEvaluator,
    EvalResult,
    EvalVerdict,
    JourneyMapEvaluator,
    TDDEvaluator,
    TraceabilityEvaluator,
)


# ==========================================================================
# EARS Spec Evaluator
# ==========================================================================

class TestEARSSpecEvaluator:
    """Tests that the EARS evaluator correctly scores spec quality."""

    def setup_method(self):
        self.evaluator = EARSSpecEvaluator()

    def test_high_quality_spec_passes(self):
        """A well-formed EARS spec should score ≥ 0.75."""
        output = """
# EARS Requirements Specification
## Feature: User Authentication

### EARS-001 [MUST] — Ubiquitous
> The authentication module shall encrypt all passwords using bcrypt.

**Acceptance Criteria:**
- [ ] AC-001.1: Passwords hashed with bcrypt cost factor ≥ 12
- [ ] AC-001.2: No plaintext passwords in database within 100ms

**Rationale:** Prevents credential exposure per OWASP guidelines.

### EARS-002 [MUST] — Event-Driven
> When a user submits the login form, the system shall validate credentials within 2 seconds.

**Acceptance Criteria:**
- [ ] AC-002.1: P95 validation under 2000ms
- [ ] AC-002.2: Invalid credentials return 401

**Rationale:** UX responsiveness requirement.

### EARS-003 [SHOULD] — State-Driven
> While the user is authenticated, the system shall maintain a valid session token.

**Acceptance Criteria:**
- [ ] AC-003.1: Token refreshed every 15 minutes
- [ ] AC-003.2: Session expires after 24 hours of inactivity

**Rationale:** Security best practice for session management.

### EARS-004 [MUST] — Unwanted
> If 5 failed login attempts occur within 15 minutes, then the system shall lock the account.

**Acceptance Criteria:**
- [ ] AC-004.1: Account locked after 5 failures in 15 minute window
- [ ] AC-004.2: Lockout expires after 30 minutes

**Rationale:** Brute force attack mitigation.
"""
        result = self.evaluator.evaluate(output, {"case_name": "high_quality"})
        assert result.passed, f"Expected PASS but got {result.summary}"
        assert result.overall_score >= 0.75

    def test_low_quality_spec_fails(self):
        """A vague, poorly structured spec should score < 0.75."""
        output = """
# Requirements
- The system should work properly
- Login should be appropriate and adequate
- Passwords should be handled correctly
- The system should be good enough for users
"""
        result = self.evaluator.evaluate(output, {"case_name": "low_quality"})
        assert not result.passed, f"Expected FAIL but got {result.summary}"
        assert result.overall_score < 0.75

    def test_empty_output_fails(self):
        output = ""
        result = self.evaluator.evaluate(output, {"case_name": "empty"})
        assert not result.passed

    def test_detects_vague_language(self):
        output = """
### EARS-001 [MUST]
> The system shall handle data appropriately and correctly in a reasonable manner.

**Acceptance Criteria:**
- [ ] AC-001.1: Data handled adequately
"""
        result = self.evaluator.evaluate(output, {"case_name": "vague"})
        vague_check = next(
            (c for c in result.checks if c.name == "no_vague_language"), None
        )
        assert vague_check is not None
        assert vague_check.score < 1.0

    def test_detects_missing_unwanted_requirements(self):
        output = """
### EARS-001 [MUST] — Ubiquitous
> The system shall store data.

**Acceptance Criteria:**
- [ ] AC-001.1: Data stored within 100ms
"""
        result = self.evaluator.evaluate(output, {"case_name": "no_safety"})
        safety_check = next(
            (c for c in result.checks if c.name == "safety_requirements"), None
        )
        assert safety_check is not None
        assert safety_check.verdict in (EvalVerdict.WARN, EvalVerdict.FAIL)

    def test_counts_template_diversity(self):
        """Spec using multiple EARS template types scores higher on diversity."""
        output = """
### EARS-001 [MUST] — Ubiquitous
> The system shall do A.
AC-001.1: test

### EARS-002 [MUST] — Event-Driven
> When X happens, the system shall do B.
AC-002.1: test

### EARS-003 [MUST] — State-Driven
> While Y is true, the system shall do C.
AC-003.1: test

**Rationale:** needed.
"""
        result = self.evaluator.evaluate(output, {"case_name": "diverse"})
        diversity_check = next(
            (c for c in result.checks if c.name == "template_diversity"), None
        )
        assert diversity_check is not None
        assert diversity_check.score >= 0.6

    def test_id_gap_detection(self):
        """Sequential ID gaps should be flagged."""
        output = """
### EARS-001 [MUST]
> The system shall do A.
AC-001.1: test

### EARS-003 [MUST]
> The system shall do C.
AC-003.1: test
"""
        result = self.evaluator.evaluate(output, {"case_name": "gaps"})
        id_check = next(
            (c for c in result.checks if c.name == "id_consistency"), None
        )
        assert id_check is not None
        assert id_check.verdict == EvalVerdict.WARN


# ==========================================================================
# Journey Map Evaluator
# ==========================================================================

class TestJourneyMapEvaluator:
    def setup_method(self):
        self.evaluator = JourneyMapEvaluator()

    def test_high_quality_journey_passes(self):
        output = """
# Customer Journey Map
## Feature: User Authentication

### Journey 1: New User — Register Account (Happy Path)
**Persona:** New User
**Goal:** Create account
**Entry Point:** Landing page
**Exit Point:** Dashboard

| Step | ID | Action | Touchpoint | Expected Outcome | Req Refs |
|------|-----|--------|------------|-----------------|----------|
| 1 | STEP-001 | Click signup | Web UI | Form shown | EARS-001 |
| 2 | STEP-002 | Enter details | Signup form | Validated | EARS-001 |
| 3 | STEP-003 | Submit | API | Account created | EARS-002 |
| 4 | STEP-004 | Confirm email | Email | Verified | EARS-003 |

The user feels confident after seeing the confirmation.
A frustrated user who enters invalid data sees clear error messages.

**Error Paths:**
- STEP-002 → Invalid email → Validation error shown (EARS-002)
- STEP-003 → Server error → Error page with retry (EARS-002)

### Requirements Coverage
| EARS ID | Journey Steps | Status |
|---------|--------------|--------|
| EARS-001 | STEP-001, STEP-002 | Covered |
| EARS-004 | — | GAP: No journey coverage |
"""
        result = self.evaluator.evaluate(output, {"case_name": "high_quality"})
        assert result.passed, f"Expected PASS but got {result.summary}"
        assert result.overall_score >= 0.75

    def test_low_quality_journey_fails(self):
        output = """
# Journeys
- User logs in
- User sees dashboard
- Done
"""
        result = self.evaluator.evaluate(output, {"case_name": "low_quality"})
        assert not result.passed

    def test_empty_output_fails(self):
        result = self.evaluator.evaluate("", {"case_name": "empty"})
        assert not result.passed

    def test_detects_missing_error_paths(self):
        output = """
### Journey 1: Happy Path
**Persona:** User
**Entry Point:** Home
**Exit Point:** Dashboard

| Step | ID | Action | Touchpoint | Expected Outcome | Req Refs |
|------|-----|--------|------------|-----------------|----------|
| 1 | STEP-001 | Login | Web UI | Success | EARS-001 |
"""
        result = self.evaluator.evaluate(output, {"case_name": "no_errors"})
        path_check = next(
            (c for c in result.checks if c.name == "path_coverage"), None
        )
        assert path_check is not None
        assert path_check.score < 1.0

    def test_detects_missing_ears_refs(self):
        output = """
### Journey 1
**Persona:** User

| Step | ID | Action | Touchpoint | Expected |
|------|-----|--------|------------|---------|
| 1 | STEP-001 | Login | Web | Success |
| 2 | STEP-002 | View | Dashboard | Loaded |
"""
        result = self.evaluator.evaluate(output, {"case_name": "no_refs"})
        ref_check = next(
            (c for c in result.checks if c.name == "requirement_refs"), None
        )
        assert ref_check is not None
        assert ref_check.verdict == EvalVerdict.FAIL


# ==========================================================================
# Design Doc Evaluator
# ==========================================================================

class TestDesignDocEvaluator:
    def setup_method(self):
        self.evaluator = DesignDocEvaluator()

    def test_high_quality_design_passes(self):
        output = """
# Design Document
## Feature: User Authentication

## 1. Component Architecture

### AuthService
**Responsibility:** Handle authentication flows
**Interfaces:**
- `login(email, password) → Session`
- `validate_token(token) → bool`

#### Correctness Properties

| ID | Type | Property | Test Strategy | Reqs | Journey |
|----|------|----------|---------------|------|---------|
| CP-001 | INVARIANT | All stored passwords are bcrypt hashed | DB query scan | EARS-001 | STEP-001 |
| CP-002 | PRECONDITION | login() requires non-empty email | Unit test | EARS-002 | STEP-002 |
| CP-003 | POSTCONDITION | After login(), session valid 24h | Expiry test | EARS-002 | STEP-003 |
| CP-004 | SAFETY | Never return password hash in response | Response scan | EARS-001 | STEP-001 |
| CP-005 | LIVENESS | Lockout eventually expires | Time-based | EARS-005 | STEP-004 |

## 2. Data Flow
Data moves from the login form → AuthService → UserRepository → SessionStore.
The request → validation → token generation flow ensures security.

## 3. Design Decisions
| Decision | Rationale | Alternatives Considered | Reqs |
|----------|-----------|------------------------|------|
| bcrypt over argon2 | Wider library support | argon2id, scrypt | EARS-001 |
| JWT for sessions | Stateless, horizontally scalable | Server-side sessions | EARS-002 |

We chose bcrypt because of its trade-off between security and compatibility.
"""
        result = self.evaluator.evaluate(output, {"case_name": "high_quality"})
        assert result.passed, f"Expected PASS but got {result.summary}"
        assert result.overall_score >= 0.75

    def test_low_quality_design_fails(self):
        output = """
# Design
We'll use a simple function to handle login.
"""
        result = self.evaluator.evaluate(output, {"case_name": "low_quality"})
        assert not result.passed

    def test_detects_missing_safety_properties(self):
        output = """
### AuthModule
**Interfaces:**
- `login(email, password) → Session`

| ID | Type | Property | Test Strategy | Reqs |
|----|------|----------|---------------|------|
| CP-001 | PRECONDITION | email not empty | unit test | EARS-001 |
| CP-002 | POSTCONDITION | returns session | unit test | EARS-002 |
| CP-003 | INVARIANT | data consistent | integration | EARS-003 |
"""
        result = self.evaluator.evaluate(output, {"case_name": "no_safety"})
        safety_check = next(
            (c for c in result.checks if c.name == "safety_properties"), None
        )
        assert safety_check is not None
        # Has INVARIANT but not SAFETY specifically
        assert safety_check.verdict in (EvalVerdict.WARN, EvalVerdict.FAIL)


# ==========================================================================
# TDD Evaluator
# ==========================================================================

class TestTDDEvaluator:
    def setup_method(self):
        self.evaluator = TDDEvaluator()

    def test_high_quality_tdd_passes(self):
        output = '''
# Test Suite for AuthService

## test_password_hashing.py (Unit Tests)

```python
def test_bcrypt_hash():
    """CP-001, EARS-001: All passwords stored as bcrypt hashes."""
    result = hash_password("test123")
    assert is_bcrypt_hash(result)

def test_cost_factor():
    """CP-001, EARS-001: Bcrypt cost factor >= 12."""
    assert get_bcrypt_cost(hash_password("x")) >= 12
```

## test_login.py (Integration Tests)

```python
def test_login_success():
    """CP-002, CP-003, EARS-002: Valid login returns session."""
    session = login("user@test.com", "valid")
    assert session is not None

def test_login_empty_email():
    """CP-002, EARS-002: Empty email raises ValueError."""
    with pytest.raises(ValueError):
        login("", "pwd")

def test_login_invalid_password():
    """CP-002, EARS-002: Wrong password returns None (negative test)."""
    assert login("user@test.com", "wrong") is None
```

## test_lockout.py (Edge Case / Boundary Tests)

```python
def test_lockout_boundary():
    """CP-005, EARS-005: Account not locked after 4 failures (boundary)."""
    for _ in range(4):
        login("user@test.com", "wrong")
    assert not is_locked("user@test.com")

def test_lockout_trigger():
    """CP-005, EARS-005: Account locked after exactly 5 failures."""
    for _ in range(5):
        login("user@test.com", "wrong")
    assert is_locked("user@test.com")
```

All tests are designed to be RED (failing) — no implementation exists yet.

## UAT Script

### Scenario 1: New User Registration (JOURNEY-001)
**Preconditions:** Clean database

| Step | Action | Expected Result | EARS Ref | Pass/Fail |
|------|--------|----------------|----------|-----------|
| 1 | Navigate to /signup | Signup form loads | EARS-001 | ☐ |
| 2 | Enter valid credentials | Form validates | EARS-002 | ☐ |
'''
        result = self.evaluator.evaluate(output, {"case_name": "high_quality"})
        assert result.passed, f"Expected PASS but got {result.summary}"
        assert result.overall_score >= 0.75

    def test_low_quality_tdd_fails(self):
        output = """
# Tests
def test_something():
    pass

def test_another():
    assert True
"""
        result = self.evaluator.evaluate(output, {"case_name": "low_quality"})
        assert not result.passed

    def test_detects_missing_uat(self):
        output = """
## test_auth.py
```python
def test_login():
    \"\"\"CP-001, EARS-001: Login test.\"\"\"
    assert login("a", "b")
```
"""
        result = self.evaluator.evaluate(output, {"case_name": "no_uat"})
        uat_check = next(
            (c for c in result.checks if c.name == "uat_script"), None
        )
        assert uat_check is not None
        assert uat_check.verdict == EvalVerdict.FAIL

    def test_detects_missing_traceability(self):
        output = """
## test_auth.py
```python
def test_login():
    assert login("a", "b")

def test_register():
    assert register("a", "b")

def test_logout():
    assert logout()
```
UAT: manual testing
"""
        result = self.evaluator.evaluate(output, {"case_name": "no_trace"})
        cp_check = next(
            (c for c in result.checks if c.name == "correctness_property_refs"), None
        )
        assert cp_check is not None
        assert cp_check.verdict == EvalVerdict.FAIL


# ==========================================================================
# Coder Evaluator
# ==========================================================================

class TestCoderEvaluator:
    def setup_method(self):
        self.evaluator = CoderEvaluator()

    def test_high_quality_tasks_pass(self):
        output = """
# Implementation Progress

## Component: AuthService

- [R] TASK-001: Implement password hashing
  Tests: test_password_hashing.py::test_bcrypt_hash (0/2 passing)
  Reqs: EARS-001
  Props: CP-001
  Files: auth_service.py

- [Y] TASK-002: Implement login flow
  Tests: test_login.py::test_login_success ✅, ::test_login_empty ❌ (1/3 passing)
  Reqs: EARS-002
  Props: CP-002, CP-003
  Files: auth_service.py, session.py
  Depends on: TASK-001

- [G] TASK-003: Implement session token
  Tests: test_session.py::test_create ✅, ::test_expiry ✅ (2/2 passing)
  Reqs: EARS-002
  Props: CP-003
  Files: session.py

## Progress: 1/3 GREEN (33%)
"""
        result = self.evaluator.evaluate(output, {"case_name": "high_quality"})
        assert result.passed, f"Expected PASS but got {result.summary}"

    def test_low_quality_tasks_fail(self):
        output = """
# TODO
- Write some code
- Fix bugs
- Deploy
"""
        result = self.evaluator.evaluate(output, {"case_name": "low_quality"})
        assert not result.passed


# ==========================================================================
# Traceability Evaluator
# ==========================================================================

class TestTraceabilityEvaluator:
    def setup_method(self):
        self.evaluator = TraceabilityEvaluator()

    def test_high_quality_matrix_passes(self):
        output = """
# Traceability Matrix

## Forward Traceability

| EARS ID | Journey Steps | CP Props | Tests | Tasks | Status |
|---------|--------------|----------|-------|-------|--------|
| EARS-001 | STEP-001, STEP-002 | CP-001, CP-004 | test_hash, test_cost | TASK-001 | Full |
| EARS-002 | STEP-003, STEP-004 | CP-002, CP-003 | test_login, test_empty | TASK-002 | Full |
| EARS-003 | — | — | — | — | Missing |

## Orphan Tasks
- TASK-099: Not linked to any requirement

## Coverage Gaps
- EARS-003: No journey step coverage, no tests, not covered
"""
        result = self.evaluator.evaluate(output, {"case_name": "high_quality"})
        assert result.passed, f"Expected PASS but got {result.summary}"
        assert result.overall_score >= 0.75

    def test_low_quality_matrix_fails(self):
        output = "Everything looks fine."
        result = self.evaluator.evaluate(output, {"case_name": "low_quality"})
        assert not result.passed


# ==========================================================================
# Cross-cutting tests
# ==========================================================================

class TestEvalResultProperties:
    def test_summary_format_pass(self):
        result = EvalResult(
            evaluator_name="test",
            case_name="test",
            overall_score=0.85,
            passed=True,
        )
        assert "PASS" in result.summary

    def test_summary_format_fail(self):
        from src.evals.evaluators import EvalCheck

        result = EvalResult(
            evaluator_name="test",
            case_name="test",
            overall_score=0.3,
            passed=False,
            checks=[
                EvalCheck(
                    name="bad_check",
                    verdict=EvalVerdict.FAIL,
                    score=0.0,
                    reason="everything is wrong",
                ),
            ],
        )
        assert "FAIL" in result.summary
        assert "bad_check" in result.summary
