"""Custom evaluators for SDLC workflow agent outputs.

Each evaluator implements deterministic scoring rubrics that check structural
completeness, cross-reference integrity, and domain-specific quality criteria.

These evaluators work in two modes:
1. **Deterministic** (default): Pattern-based checks that don't require an LLM
2. **LLM-as-Judge** (optional): Wraps ``strands_evals.OutputEvaluator`` for
   semantic quality assessment when strands-agents-evals is installed

The deterministic mode ensures evals run fast in CI without API calls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Evaluation result types (standalone — no SDK dependency required)
# ---------------------------------------------------------------------------

class EvalVerdict(Enum):
    """Pass/fail verdict for an evaluation check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


@dataclass
class EvalCheck:
    """A single evaluation check result."""
    name: str
    verdict: EvalVerdict
    score: float  # 0.0–1.0
    reason: str
    category: str = ""  # e.g. "completeness", "traceability", "format"


@dataclass
class EvalResult:
    """Aggregated evaluation result for one case."""
    evaluator_name: str
    case_name: str
    overall_score: float  # 0.0–1.0 (weighted average of checks)
    passed: bool
    checks: list[EvalCheck] = field(default_factory=list)
    pass_threshold: float = 0.75

    @property
    def summary(self) -> str:
        failed = [c for c in self.checks if c.verdict == EvalVerdict.FAIL]
        if not failed:
            return f"PASS ({self.overall_score:.2f})"
        reasons = "; ".join(f"{c.name}: {c.reason}" for c in failed[:3])
        return f"FAIL ({self.overall_score:.2f}) — {reasons}"


# ---------------------------------------------------------------------------
# Base evaluator
# ---------------------------------------------------------------------------

class WorkflowEvaluator:
    """Base class for deterministic SDLC workflow evaluators."""

    name: str = "base"
    pass_threshold: float = 0.75

    def evaluate(self, output: str, context: dict | None = None) -> EvalResult:
        """Run all checks against the output and return aggregated result."""
        checks = self._run_checks(output, context or {})
        if not checks:
            return EvalResult(
                evaluator_name=self.name,
                case_name=context.get("case_name", "unknown") if context else "unknown",
                overall_score=0.0,
                passed=False,
                checks=[],
                pass_threshold=self.pass_threshold,
            )

        total_weight = len(checks)
        total_score = sum(c.score for c in checks)
        overall = total_score / total_weight if total_weight > 0 else 0.0

        return EvalResult(
            evaluator_name=self.name,
            case_name=context.get("case_name", "unknown") if context else "unknown",
            overall_score=overall,
            passed=overall >= self.pass_threshold,
            checks=checks,
            pass_threshold=self.pass_threshold,
        )

    def _run_checks(self, output: str, context: dict) -> list[EvalCheck]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# EARS Spec Evaluator
# ---------------------------------------------------------------------------

class EARSSpecEvaluator(WorkflowEvaluator):
    """Evaluates EARS requirements specification quality.

    Checks:
    1. Format: Uses at least one of the 5 EARS templates correctly
    2. Completeness: Has multiple requirements (EARS-001, EARS-002, ...)
    3. Acceptance criteria: Each requirement has testable ACs
    4. Priority: MoSCoW priorities assigned (MUST/SHOULD/COULD/WONT)
    5. Safety: Includes at least one UNWANTED-type requirement
    6. Rationale: Requirements include rationale/justification
    7. ID consistency: Sequential IDs with no gaps
    8. Diversity: Uses multiple EARS template types
    9. Specificity: ACs are concrete (contain numbers, specific values)
    10. No vague language: Avoids "should be good", "appropriate", etc.
    """

    name = "ears_spec"

    def _run_checks(self, output: str, context: dict) -> list[EvalCheck]:
        checks: list[EvalCheck] = []
        text = output.lower()

        # 1. EARS format — uses structured templates
        ears_patterns = [
            r"the\s+\w+\s+shall\s+",              # Ubiquitous
            r"when\s+.+?,\s*the\s+\w+\s+shall\s+", # Event-driven
            r"while\s+.+?,\s*the\s+\w+\s+shall\s+", # State-driven
            r"where\s+.+?,\s*the\s+\w+\s+shall\s+", # Optional
            r"if\s+.+?,\s*then\s+the\s+\w+\s+shall\s+",  # Unwanted
        ]
        template_matches = sum(1 for p in ears_patterns if re.search(p, text))
        checks.append(EvalCheck(
            name="ears_format",
            verdict=EvalVerdict.PASS if template_matches >= 1 else EvalVerdict.FAIL,
            score=min(template_matches / 3.0, 1.0),
            reason=f"Found {template_matches}/5 EARS template types",
            category="format",
        ))

        # 2. Multiple requirements
        req_ids = re.findall(r"EARS-\d+", output)
        unique_ids = sorted(set(req_ids))
        checks.append(EvalCheck(
            name="requirement_count",
            verdict=EvalVerdict.PASS if len(unique_ids) >= 3 else EvalVerdict.FAIL,
            score=min(len(unique_ids) / 5.0, 1.0),
            reason=f"Found {len(unique_ids)} unique requirements (need ≥3)",
            category="completeness",
        ))

        # 3. Acceptance criteria present
        ac_count = len(re.findall(r"AC-\d+", output))
        ac_per_req = ac_count / max(len(unique_ids), 1)
        checks.append(EvalCheck(
            name="acceptance_criteria",
            verdict=EvalVerdict.PASS if ac_per_req >= 1.5 else EvalVerdict.FAIL,
            score=min(ac_per_req / 2.0, 1.0),
            reason=f"Found {ac_count} ACs across {len(unique_ids)} reqs ({ac_per_req:.1f}/req, need ≥1.5)",
            category="completeness",
        ))

        # 4. MoSCoW priorities
        priorities = re.findall(r"\[(?:MUST|SHOULD|COULD|WONT)\]", output, re.IGNORECASE)
        checks.append(EvalCheck(
            name="moscow_priorities",
            verdict=EvalVerdict.PASS if len(priorities) >= len(unique_ids) * 0.5 else EvalVerdict.FAIL,
            score=min(len(priorities) / max(len(unique_ids), 1), 1.0),
            reason=f"Found {len(priorities)} priority tags for {len(unique_ids)} reqs",
            category="format",
        ))

        # 5. Safety — at least one UNWANTED requirement
        has_unwanted = bool(re.search(
            r"(?:unwanted|if\s+.+?,\s*then\s+the\s+\w+\s+shall)", text
        ))
        checks.append(EvalCheck(
            name="safety_requirements",
            verdict=EvalVerdict.PASS if has_unwanted else EvalVerdict.WARN,
            score=1.0 if has_unwanted else 0.3,
            reason="Includes unwanted/safety requirements" if has_unwanted
                   else "Missing UNWANTED-type safety requirements",
            category="safety",
        ))

        # 6. Rationale provided
        rationale_count = len(re.findall(r"\*\*rationale\*\*:?", text))
        checks.append(EvalCheck(
            name="rationale",
            verdict=EvalVerdict.PASS if rationale_count >= len(unique_ids) * 0.5 else EvalVerdict.WARN,
            score=min(rationale_count / max(len(unique_ids), 1), 1.0),
            reason=f"Found {rationale_count} rationale sections",
            category="completeness",
        ))

        # 7. ID consistency — sequential with no gaps
        if unique_ids:
            numbers = sorted(int(re.search(r"\d+", i).group()) for i in unique_ids if re.search(r"\d+", i))
            expected = list(range(numbers[0], numbers[0] + len(numbers))) if numbers else []
            sequential = numbers == expected
            checks.append(EvalCheck(
                name="id_consistency",
                verdict=EvalVerdict.PASS if sequential else EvalVerdict.WARN,
                score=1.0 if sequential else 0.5,
                reason="IDs are sequential" if sequential else f"ID gaps detected: {numbers}",
                category="format",
            ))

        # 8. Template diversity — uses >1 EARS type
        checks.append(EvalCheck(
            name="template_diversity",
            verdict=EvalVerdict.PASS if template_matches >= 2 else EvalVerdict.WARN,
            score=min(template_matches / 3.0, 1.0),
            reason=f"Uses {template_matches} different EARS template types (want ≥2)",
            category="diversity",
        ))

        # 9. Specificity — ACs contain concrete values
        specific_acs = re.findall(
            r"AC-\d+[^:]*:.*?(?:\d+|seconds?|minutes?|ms|%|bytes?|mb|gb)", text
        )
        specificity_ratio = len(specific_acs) / max(ac_count, 1)
        checks.append(EvalCheck(
            name="ac_specificity",
            verdict=EvalVerdict.PASS if specificity_ratio >= 0.3 else EvalVerdict.WARN,
            score=min(specificity_ratio / 0.5, 1.0),
            reason=f"{len(specific_acs)}/{ac_count} ACs contain specific measurable values",
            category="quality",
        ))

        # 10. No vague language
        vague_phrases = re.findall(
            r"\b(?:appropriate|adequate|reasonable|good enough|as needed|properly|correctly)\b",
            text,
        )
        checks.append(EvalCheck(
            name="no_vague_language",
            verdict=EvalVerdict.PASS if len(vague_phrases) <= 2 else EvalVerdict.FAIL,
            score=max(1.0 - len(vague_phrases) * 0.2, 0.0),
            reason=f"Found {len(vague_phrases)} vague phrases" + (
                f": {', '.join(vague_phrases[:3])}" if vague_phrases else ""
            ),
            category="quality",
        ))

        return checks


# ---------------------------------------------------------------------------
# Journey Map Evaluator
# ---------------------------------------------------------------------------

class JourneyMapEvaluator(WorkflowEvaluator):
    """Evaluates customer journey map quality.

    Checks:
    1. Has journey IDs (JOURNEY-001, ...)
    2. Has step IDs (STEP-001, ...)
    3. Multiple personas identified
    4. Happy path + error paths present
    5. Every step references EARS requirements
    6. Requirements coverage table present
    7. Touchpoints specified (UI, API, etc.)
    8. Entry/exit points defined
    9. Emotional journey tracked
    10. Coverage gaps explicitly identified
    """

    name = "journey_map"

    def _run_checks(self, output: str, context: dict) -> list[EvalCheck]:
        checks: list[EvalCheck] = []
        text = output.lower()

        # 1. Journey IDs
        journey_ids = set(re.findall(r"JOURNEY-\d+", output))
        checks.append(EvalCheck(
            name="journey_ids",
            verdict=EvalVerdict.PASS if len(journey_ids) >= 1 else EvalVerdict.FAIL,
            score=min(len(journey_ids) / 2.0, 1.0),
            reason=f"Found {len(journey_ids)} journey(s)",
            category="completeness",
        ))

        # 2. Step IDs
        step_ids = set(re.findall(r"STEP-\d+", output))
        checks.append(EvalCheck(
            name="step_ids",
            verdict=EvalVerdict.PASS if len(step_ids) >= 3 else EvalVerdict.FAIL,
            score=min(len(step_ids) / 5.0, 1.0),
            reason=f"Found {len(step_ids)} journey steps (need ≥3)",
            category="completeness",
        ))

        # 3. Multiple personas
        persona_indicators = re.findall(
            r"\*\*persona\*\*:?\s*(.+?)(?:\n|$)", text
        )
        unique_personas = set(p.strip().rstrip("*").strip() for p in persona_indicators)
        checks.append(EvalCheck(
            name="persona_diversity",
            verdict=EvalVerdict.PASS if len(unique_personas) >= 1 else EvalVerdict.FAIL,
            score=min(len(unique_personas) / 2.0, 1.0),
            reason=f"Found {len(unique_personas)} persona(s): {', '.join(list(unique_personas)[:3])}",
            category="diversity",
        ))

        # 4. Happy + error paths
        has_happy = bool(re.search(r"happy\s*path", text))
        has_error = bool(re.search(r"error\s*path|error\s*case|failure|exception", text))
        path_score = (0.5 if has_happy else 0.0) + (0.5 if has_error else 0.0)
        checks.append(EvalCheck(
            name="path_coverage",
            verdict=EvalVerdict.PASS if has_happy and has_error else EvalVerdict.WARN,
            score=path_score,
            reason=f"Happy: {'yes' if has_happy else 'no'}, Error: {'yes' if has_error else 'no'}",
            category="completeness",
        ))

        # 5. EARS cross-references in steps
        ears_refs = re.findall(r"EARS-\d+", output)
        checks.append(EvalCheck(
            name="requirement_refs",
            verdict=EvalVerdict.PASS if len(ears_refs) >= len(step_ids) * 0.5 else EvalVerdict.FAIL,
            score=min(len(ears_refs) / max(len(step_ids), 1), 1.0),
            reason=f"Found {len(ears_refs)} EARS refs across {len(step_ids)} steps",
            category="traceability",
        ))

        # 6. Coverage table
        has_coverage_table = bool(re.search(
            r"(?:coverage|requirement.*coverage|ears.*id.*(?:journey|step|status))", text
        ))
        checks.append(EvalCheck(
            name="coverage_table",
            verdict=EvalVerdict.PASS if has_coverage_table else EvalVerdict.WARN,
            score=1.0 if has_coverage_table else 0.3,
            reason="Coverage table present" if has_coverage_table else "Missing coverage summary table",
            category="traceability",
        ))

        # 7. Touchpoints
        touchpoint_keywords = re.findall(
            r"\b(?:ui|api|web|mobile|cli|email|sms|webhook|dashboard|form|endpoint|database)\b",
            text,
        )
        checks.append(EvalCheck(
            name="touchpoints",
            verdict=EvalVerdict.PASS if len(touchpoint_keywords) >= 2 else EvalVerdict.WARN,
            score=min(len(set(touchpoint_keywords)) / 3.0, 1.0),
            reason=f"Found {len(set(touchpoint_keywords))} distinct touchpoint types",
            category="completeness",
        ))

        # 8. Entry/exit points
        has_entry = bool(re.search(r"entry\s*point", text))
        has_exit = bool(re.search(r"exit\s*point", text))
        checks.append(EvalCheck(
            name="entry_exit_points",
            verdict=EvalVerdict.PASS if has_entry and has_exit else EvalVerdict.WARN,
            score=(0.5 if has_entry else 0.0) + (0.5 if has_exit else 0.0),
            reason=f"Entry: {'yes' if has_entry else 'no'}, Exit: {'yes' if has_exit else 'no'}",
            category="format",
        ))

        # 9. Emotional journey (optional but valuable)
        emotion_keywords = re.findall(
            r"\b(?:frustrat|confus|satisf|delight|anxious|relieved|happy|annoyed|confident)\w*\b",
            text,
        )
        checks.append(EvalCheck(
            name="emotional_journey",
            verdict=EvalVerdict.PASS if len(emotion_keywords) >= 2 else EvalVerdict.WARN,
            score=min(len(emotion_keywords) / 3.0, 1.0),
            reason=f"Found {len(emotion_keywords)} emotion references",
            category="quality",
        ))

        # 10. Gaps identified
        has_gaps = bool(re.search(r"(?:gap|missing|not covered|no coverage)", text))
        checks.append(EvalCheck(
            name="coverage_gaps_identified",
            verdict=EvalVerdict.PASS if has_gaps else EvalVerdict.WARN,
            score=1.0 if has_gaps else 0.5,
            reason="Coverage gaps explicitly identified" if has_gaps
                   else "No explicit gap analysis (may be OK if full coverage)",
            category="traceability",
        ))

        return checks


# ---------------------------------------------------------------------------
# Design Doc Evaluator
# ---------------------------------------------------------------------------

class DesignDocEvaluator(WorkflowEvaluator):
    """Evaluates design document quality with correctness properties.

    Checks:
    1. Component architecture defined
    2. Correctness property IDs (CP-001, ...)
    3. Multiple property types (INVARIANT, PRECONDITION, etc.)
    4. EARS requirement cross-references
    5. Interface definitions present
    6. Test strategies specified for each property
    7. Data flow described
    8. Safety properties cover UNWANTED requirements
    9. Design decisions with rationale
    10. No orphan components (all traced to requirements)
    """

    name = "design_doc"

    def _run_checks(self, output: str, context: dict) -> list[EvalCheck]:
        checks: list[EvalCheck] = []
        text = output.lower()

        # 1. Component architecture
        component_markers = re.findall(r"(?:###?\s+\w+(?:Service|Manager|Controller|Handler|Module|Component))", output)
        checks.append(EvalCheck(
            name="components_defined",
            verdict=EvalVerdict.PASS if len(component_markers) >= 1 else EvalVerdict.FAIL,
            score=min(len(component_markers) / 2.0, 1.0),
            reason=f"Found {len(component_markers)} component(s)",
            category="completeness",
        ))

        # 2. Correctness property IDs
        cp_ids = set(re.findall(r"CP-\d+", output))
        checks.append(EvalCheck(
            name="correctness_property_count",
            verdict=EvalVerdict.PASS if len(cp_ids) >= 3 else EvalVerdict.FAIL,
            score=min(len(cp_ids) / 5.0, 1.0),
            reason=f"Found {len(cp_ids)} correctness properties (need ≥3)",
            category="completeness",
        ))

        # 3. Property type diversity
        property_types = set()
        for t in ["invariant", "precondition", "postcondition", "safety", "liveness"]:
            if t in text:
                property_types.add(t)
        checks.append(EvalCheck(
            name="property_type_diversity",
            verdict=EvalVerdict.PASS if len(property_types) >= 2 else EvalVerdict.WARN,
            score=min(len(property_types) / 3.0, 1.0),
            reason=f"Found {len(property_types)} property types: {', '.join(sorted(property_types))}",
            category="diversity",
        ))

        # 4. EARS cross-references
        ears_refs = set(re.findall(r"EARS-\d+", output))
        checks.append(EvalCheck(
            name="requirement_refs",
            verdict=EvalVerdict.PASS if len(ears_refs) >= 2 else EvalVerdict.FAIL,
            score=min(len(ears_refs) / 3.0, 1.0),
            reason=f"References {len(ears_refs)} EARS requirements",
            category="traceability",
        ))

        # 5. Interface definitions
        interface_markers = re.findall(
            r"(?:`\w+\([^)]*\)\s*(?:→|->)\s*\w+`|def\s+\w+|method|endpoint|api\s+contract)",
            text,
        )
        checks.append(EvalCheck(
            name="interfaces_defined",
            verdict=EvalVerdict.PASS if len(interface_markers) >= 1 else EvalVerdict.WARN,
            score=min(len(interface_markers) / 2.0, 1.0),
            reason=f"Found {len(interface_markers)} interface definitions",
            category="completeness",
        ))

        # 6. Test strategies
        test_strategy_markers = re.findall(
            r"(?:test\s*strategy|how\s*to\s*(?:verify|test|validate)|unit\s*test|integration\s*test)",
            text,
        )
        checks.append(EvalCheck(
            name="test_strategies",
            verdict=EvalVerdict.PASS if len(test_strategy_markers) >= len(cp_ids) * 0.5 else EvalVerdict.WARN,
            score=min(len(test_strategy_markers) / max(len(cp_ids), 1), 1.0),
            reason=f"Found {len(test_strategy_markers)} test strategy references for {len(cp_ids)} properties",
            category="completeness",
        ))

        # 7. Data flow
        has_data_flow = bool(re.search(r"data\s*flow|data\s*moves|sequence|→|-->", text))
        checks.append(EvalCheck(
            name="data_flow",
            verdict=EvalVerdict.PASS if has_data_flow else EvalVerdict.WARN,
            score=1.0 if has_data_flow else 0.3,
            reason="Data flow described" if has_data_flow else "Missing data flow description",
            category="completeness",
        ))

        # 8. Safety properties present
        has_safety = "safety" in text
        checks.append(EvalCheck(
            name="safety_properties",
            verdict=EvalVerdict.PASS if has_safety else EvalVerdict.WARN,
            score=1.0 if has_safety else 0.3,
            reason="Safety properties present" if has_safety else "Missing safety properties",
            category="safety",
        ))

        # 9. Design decisions with rationale
        decision_markers = re.findall(
            r"(?:decision|rationale|trade.?off|alternative|chose|selected|because)",
            text,
        )
        checks.append(EvalCheck(
            name="design_decisions",
            verdict=EvalVerdict.PASS if len(decision_markers) >= 2 else EvalVerdict.WARN,
            score=min(len(decision_markers) / 3.0, 1.0),
            reason=f"Found {len(decision_markers)} decision/rationale markers",
            category="quality",
        ))

        # 10. Journey step references
        step_refs = set(re.findall(r"STEP-\d+", output))
        checks.append(EvalCheck(
            name="journey_refs",
            verdict=EvalVerdict.PASS if len(step_refs) >= 1 else EvalVerdict.WARN,
            score=min(len(step_refs) / 2.0, 1.0),
            reason=f"References {len(step_refs)} journey steps",
            category="traceability",
        ))

        return checks


# ---------------------------------------------------------------------------
# TDD Evaluator
# ---------------------------------------------------------------------------

class TDDEvaluator(WorkflowEvaluator):
    """Evaluates TDD enforcer output quality.

    Checks:
    1. Test file structure present (test_*.py patterns)
    2. Test functions defined (def test_*)
    3. Correctness property cross-references (CP-XXX)
    4. EARS requirement cross-references
    5. UAT script present
    6. UAT scenarios reference journeys
    7. Tests cover multiple categories (unit, integration, edge)
    8. Assertions present (assert, assertEqual, etc.)
    9. Test docstrings with traceability info
    10. RED state acknowledgment (tests expected to fail)
    """

    name = "tdd_enforcer"

    def _run_checks(self, output: str, context: dict) -> list[EvalCheck]:
        checks: list[EvalCheck] = []
        text = output.lower()

        # 1. Test file patterns
        test_files = re.findall(r"test_\w+\.py", output)
        checks.append(EvalCheck(
            name="test_files",
            verdict=EvalVerdict.PASS if len(test_files) >= 1 else EvalVerdict.FAIL,
            score=min(len(set(test_files)) / 2.0, 1.0),
            reason=f"Found {len(set(test_files))} test file(s)",
            category="completeness",
        ))

        # 2. Test functions
        test_fns = re.findall(r"(?:def\s+test_\w+|test_\w+\()", output)
        checks.append(EvalCheck(
            name="test_functions",
            verdict=EvalVerdict.PASS if len(test_fns) >= 3 else EvalVerdict.FAIL,
            score=min(len(test_fns) / 5.0, 1.0),
            reason=f"Found {len(test_fns)} test functions (need ≥3)",
            category="completeness",
        ))

        # 3. CP cross-references
        cp_refs = set(re.findall(r"CP-\d+", output))
        checks.append(EvalCheck(
            name="correctness_property_refs",
            verdict=EvalVerdict.PASS if len(cp_refs) >= 2 else EvalVerdict.FAIL,
            score=min(len(cp_refs) / 3.0, 1.0),
            reason=f"Tests reference {len(cp_refs)} correctness properties",
            category="traceability",
        ))

        # 4. EARS cross-references
        ears_refs = set(re.findall(r"EARS-\d+", output))
        checks.append(EvalCheck(
            name="requirement_refs",
            verdict=EvalVerdict.PASS if len(ears_refs) >= 2 else EvalVerdict.FAIL,
            score=min(len(ears_refs) / 3.0, 1.0),
            reason=f"Tests reference {len(ears_refs)} EARS requirements",
            category="traceability",
        ))

        # 5. UAT script
        has_uat = bool(re.search(r"(?:uat|user\s*acceptance\s*test|acceptance\s*script)", text))
        checks.append(EvalCheck(
            name="uat_script",
            verdict=EvalVerdict.PASS if has_uat else EvalVerdict.FAIL,
            score=1.0 if has_uat else 0.0,
            reason="UAT script present" if has_uat else "Missing UAT script",
            category="completeness",
        ))

        # 6. UAT references journeys
        journey_refs = set(re.findall(r"JOURNEY-\d+", output))
        checks.append(EvalCheck(
            name="uat_journey_refs",
            verdict=EvalVerdict.PASS if len(journey_refs) >= 1 else EvalVerdict.WARN,
            score=min(len(journey_refs) / 1.0, 1.0),
            reason=f"UAT references {len(journey_refs)} journeys",
            category="traceability",
        ))

        # 7. Test category diversity
        categories_found = set()
        if re.search(r"unit\s*test", text):
            categories_found.add("unit")
        if re.search(r"integration\s*test", text):
            categories_found.add("integration")
        if re.search(r"edge\s*case|boundary|corner\s*case", text):
            categories_found.add("edge")
        if re.search(r"negative\s*test|error\s*case|failure", text):
            categories_found.add("negative")
        checks.append(EvalCheck(
            name="test_category_diversity",
            verdict=EvalVerdict.PASS if len(categories_found) >= 2 else EvalVerdict.WARN,
            score=min(len(categories_found) / 3.0, 1.0),
            reason=f"Found {len(categories_found)} test categories: {', '.join(sorted(categories_found))}",
            category="diversity",
        ))

        # 8. Assertions
        assertions = re.findall(
            r"\b(?:assert\b|assertEqual|assertRaises|assertIn|pytest\.raises|expect\()",
            output,
        )
        checks.append(EvalCheck(
            name="assertions",
            verdict=EvalVerdict.PASS if len(assertions) >= 3 else EvalVerdict.WARN,
            score=min(len(assertions) / 5.0, 1.0),
            reason=f"Found {len(assertions)} assertion statements",
            category="quality",
        ))

        # 9. Docstrings with traceability
        docstring_with_refs = re.findall(
            r'""".*?(?:CP-\d+|EARS-\d+).*?"""', output, re.DOTALL
        )
        checks.append(EvalCheck(
            name="traceable_docstrings",
            verdict=EvalVerdict.PASS if len(docstring_with_refs) >= 1 else EvalVerdict.WARN,
            score=min(len(docstring_with_refs) / 2.0, 1.0),
            reason=f"Found {len(docstring_with_refs)} docstrings with traceability refs",
            category="traceability",
        ))

        # 10. RED state acknowledgment
        has_red = bool(re.search(r"(?:red|fail|failing|expected to fail|no implementation)", text))
        checks.append(EvalCheck(
            name="red_state",
            verdict=EvalVerdict.PASS if has_red else EvalVerdict.WARN,
            score=1.0 if has_red else 0.5,
            reason="Acknowledges RED (failing) test state" if has_red
                   else "Does not explicitly acknowledge tests should be RED initially",
            category="tdd_discipline",
        ))

        return checks


# ---------------------------------------------------------------------------
# Coder Evaluator
# ---------------------------------------------------------------------------

class CoderEvaluator(WorkflowEvaluator):
    """Evaluates code implementation with R/Y/G task tracking.

    Checks:
    1. Task IDs present (TASK-001, ...)
    2. Status markers ([R], [Y], [G], [B])
    3. Test references per task
    4. EARS requirement refs per task
    5. Correctness property refs per task
    6. Progress summary (counts/percentages)
    7. Implementation file references
    8. Tasks in order (RED→YELLOW→GREEN progression)
    9. No orphan tasks (all linked to requirements)
    10. Component grouping
    """

    name = "coder"

    def _run_checks(self, output: str, context: dict) -> list[EvalCheck]:
        checks: list[EvalCheck] = []
        text = output.lower()

        # 1. Task IDs
        task_ids = set(re.findall(r"TASK-\d+", output))
        checks.append(EvalCheck(
            name="task_ids",
            verdict=EvalVerdict.PASS if len(task_ids) >= 2 else EvalVerdict.FAIL,
            score=min(len(task_ids) / 3.0, 1.0),
            reason=f"Found {len(task_ids)} tasks",
            category="completeness",
        ))

        # 2. R/Y/G status markers
        status_markers = re.findall(r"\[(?:R|Y|G|B|✓)\]", output)
        checks.append(EvalCheck(
            name="status_markers",
            verdict=EvalVerdict.PASS if len(status_markers) >= len(task_ids) * 0.5 else EvalVerdict.FAIL,
            score=min(len(status_markers) / max(len(task_ids), 1), 1.0),
            reason=f"Found {len(status_markers)} status markers for {len(task_ids)} tasks",
            category="format",
        ))

        # 3. Test references
        test_refs = re.findall(r"test_\w+", output)
        checks.append(EvalCheck(
            name="test_refs",
            verdict=EvalVerdict.PASS if len(test_refs) >= len(task_ids) else EvalVerdict.WARN,
            score=min(len(test_refs) / max(len(task_ids) * 2, 1), 1.0),
            reason=f"Found {len(test_refs)} test references across {len(task_ids)} tasks",
            category="traceability",
        ))

        # 4. EARS refs
        ears_refs = set(re.findall(r"EARS-\d+", output))
        checks.append(EvalCheck(
            name="requirement_refs",
            verdict=EvalVerdict.PASS if len(ears_refs) >= 1 else EvalVerdict.FAIL,
            score=min(len(ears_refs) / 2.0, 1.0),
            reason=f"Tasks reference {len(ears_refs)} EARS requirements",
            category="traceability",
        ))

        # 5. CP refs
        cp_refs = set(re.findall(r"CP-\d+", output))
        checks.append(EvalCheck(
            name="property_refs",
            verdict=EvalVerdict.PASS if len(cp_refs) >= 1 else EvalVerdict.WARN,
            score=min(len(cp_refs) / 2.0, 1.0),
            reason=f"Tasks reference {len(cp_refs)} correctness properties",
            category="traceability",
        ))

        # 6. Progress summary
        has_progress = bool(re.search(r"(?:progress|summary|percentage|%|\d+/\d+)", text))
        checks.append(EvalCheck(
            name="progress_summary",
            verdict=EvalVerdict.PASS if has_progress else EvalVerdict.WARN,
            score=1.0 if has_progress else 0.3,
            reason="Progress summary present" if has_progress else "Missing progress summary",
            category="format",
        ))

        # 7. Implementation files
        impl_files = re.findall(r"\w+\.py", output)
        non_test_files = [f for f in impl_files if not f.startswith("test_")]
        checks.append(EvalCheck(
            name="implementation_files",
            verdict=EvalVerdict.PASS if len(non_test_files) >= 1 else EvalVerdict.WARN,
            score=min(len(non_test_files) / 2.0, 1.0),
            reason=f"References {len(non_test_files)} implementation files",
            category="completeness",
        ))

        # 8. Status distribution (should have multiple states)
        statuses = {"R": 0, "Y": 0, "G": 0}
        for m in re.findall(r"\[(R|Y|G)\]", output):
            statuses[m] = statuses.get(m, 0) + 1
        active_states = sum(1 for v in statuses.values() if v > 0)
        checks.append(EvalCheck(
            name="status_distribution",
            verdict=EvalVerdict.PASS if active_states >= 1 else EvalVerdict.WARN,
            score=min(active_states / 2.0, 1.0),
            reason=f"Status distribution: R={statuses['R']}, Y={statuses['Y']}, G={statuses['G']}",
            category="quality",
        ))

        # 9. Component grouping
        has_components = bool(re.search(r"(?:component|module|service|class):", text))
        checks.append(EvalCheck(
            name="component_grouping",
            verdict=EvalVerdict.PASS if has_components else EvalVerdict.WARN,
            score=1.0 if has_components else 0.5,
            reason="Tasks grouped by component" if has_components
                   else "Tasks not grouped by component",
            category="format",
        ))

        # 10. Dependencies
        has_deps = bool(re.search(r"(?:depends|blocked|dependency|depends on)", text))
        checks.append(EvalCheck(
            name="dependency_tracking",
            verdict=EvalVerdict.PASS if has_deps else EvalVerdict.WARN,
            score=1.0 if has_deps else 0.5,
            reason="Dependency tracking present" if has_deps else "No dependency tracking",
            category="quality",
        ))

        return checks


# ---------------------------------------------------------------------------
# Traceability Evaluator
# ---------------------------------------------------------------------------

class TraceabilityEvaluator(WorkflowEvaluator):
    """Evaluates traceability matrix completeness.

    Checks:
    1. Has forward traceability section
    2. Has backward traceability or coverage summary
    3. EARS IDs present
    4. Journey step cross-refs
    5. CP cross-refs
    6. Test cross-refs
    7. Task cross-refs
    8. Status column (Full/Partial/Missing)
    9. Orphan detection
    10. Gap analysis section
    """

    name = "traceability"

    def _run_checks(self, output: str, context: dict) -> list[EvalCheck]:
        checks: list[EvalCheck] = []
        text = output.lower()

        ears_ids = set(re.findall(r"EARS-\d+", output))
        step_ids = set(re.findall(r"STEP-\d+", output))
        cp_ids = set(re.findall(r"CP-\d+", output))
        task_ids = set(re.findall(r"TASK-\d+", output))
        test_refs = set(re.findall(r"test_\w+", output))

        checks.append(EvalCheck(
            name="forward_traceability",
            verdict=EvalVerdict.PASS if "forward" in text or "traceability" in text else EvalVerdict.FAIL,
            score=1.0 if "traceability" in text else 0.0,
            reason="Forward traceability section present" if "traceability" in text
                   else "Missing traceability section",
            category="format",
        ))

        checks.append(EvalCheck(
            name="ears_coverage",
            verdict=EvalVerdict.PASS if len(ears_ids) >= 2 else EvalVerdict.FAIL,
            score=min(len(ears_ids) / 3.0, 1.0),
            reason=f"Traces {len(ears_ids)} EARS requirements",
            category="completeness",
        ))

        checks.append(EvalCheck(
            name="journey_step_refs",
            verdict=EvalVerdict.PASS if len(step_ids) >= 1 else EvalVerdict.WARN,
            score=min(len(step_ids) / 2.0, 1.0),
            reason=f"Includes {len(step_ids)} journey step references",
            category="traceability",
        ))

        checks.append(EvalCheck(
            name="correctness_prop_refs",
            verdict=EvalVerdict.PASS if len(cp_ids) >= 1 else EvalVerdict.WARN,
            score=min(len(cp_ids) / 2.0, 1.0),
            reason=f"Includes {len(cp_ids)} correctness property references",
            category="traceability",
        ))

        checks.append(EvalCheck(
            name="test_refs",
            verdict=EvalVerdict.PASS if len(test_refs) >= 1 else EvalVerdict.WARN,
            score=min(len(test_refs) / 2.0, 1.0),
            reason=f"Includes {len(test_refs)} test references",
            category="traceability",
        ))

        checks.append(EvalCheck(
            name="task_refs",
            verdict=EvalVerdict.PASS if len(task_ids) >= 1 else EvalVerdict.WARN,
            score=min(len(task_ids) / 2.0, 1.0),
            reason=f"Includes {len(task_ids)} task references",
            category="traceability",
        ))

        has_status = bool(re.search(r"(?:full|partial|missing|status)", text))
        checks.append(EvalCheck(
            name="status_column",
            verdict=EvalVerdict.PASS if has_status else EvalVerdict.WARN,
            score=1.0 if has_status else 0.3,
            reason="Status indicators present" if has_status else "Missing status indicators",
            category="format",
        ))

        has_orphan = bool(re.search(r"orphan", text))
        checks.append(EvalCheck(
            name="orphan_detection",
            verdict=EvalVerdict.PASS if has_orphan else EvalVerdict.WARN,
            score=1.0 if has_orphan else 0.5,
            reason="Orphan detection present" if has_orphan else "No orphan detection section",
            category="quality",
        ))

        has_gaps = bool(re.search(r"(?:gap|missing coverage|not covered)", text))
        checks.append(EvalCheck(
            name="gap_analysis",
            verdict=EvalVerdict.PASS if has_gaps else EvalVerdict.WARN,
            score=1.0 if has_gaps else 0.5,
            reason="Gap analysis present" if has_gaps else "No explicit gap analysis",
            category="quality",
        ))

        # Cross-phase linkage density
        total_artifacts = len(ears_ids) + len(step_ids) + len(cp_ids) + len(task_ids) + len(test_refs)
        checks.append(EvalCheck(
            name="linkage_density",
            verdict=EvalVerdict.PASS if total_artifacts >= 10 else EvalVerdict.WARN,
            score=min(total_artifacts / 15.0, 1.0),
            reason=f"Total cross-phase artifacts: {total_artifacts}",
            category="completeness",
        ))

        return checks
