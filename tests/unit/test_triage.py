"""Unit tests for the semantic triage classifier.

Tests cover VERBATIM_PATTERNS, STRUCTURED_PATTERNS, classify() logic,
source-based fallback, and determinism.

Requirements: 2.1, 2.2
"""

import re

from src.context.triage import (
    STRUCTURED_PATTERNS,
    VERBATIM_PATTERNS,
    classify,
)
from src.types.core import ContextCategory

# ---------------------------------------------------------------------------
# VERBATIM pattern tests
# ---------------------------------------------------------------------------

class TestVerbatimPatterns:
    """Content matching VERBATIM_PATTERNS → PRESERVE_VERBATIM."""

    def test_stack_trace_js(self):
        content = "Error: Cannot find module 'foo' at Function.Module._resolveFilename"
        assert classify(content) == ContextCategory.PRESERVE_VERBATIM

    def test_stack_trace_python(self):
        content = "Traceback (most recent call last):\n  File \"app.py\", line 42"
        assert classify(content) == ContextCategory.PRESERVE_VERBATIM

    def test_stack_trace_at_line(self):
        content = "    at Object.<anonymous> (/home/user/app.js:10:5)"
        assert classify(content) == ContextCategory.PRESERVE_VERBATIM

    def test_exit_code(self):
        assert classify("Process exit code 1") == ContextCategory.PRESERVE_VERBATIM

    def test_exited_with(self):
        assert classify("Command exited with 127") == ContextCategory.PRESERVE_VERBATIM

    def test_return_code(self):
        assert classify("return code 0") == ContextCategory.PRESERVE_VERBATIM

    def test_unix_file_path(self):
        assert classify("See /home/user/project/src/main.py for details") == ContextCategory.PRESERVE_VERBATIM

    def test_windows_file_path(self):
        assert classify(r"Open C:\Users\dev\project\main.py") == ContextCategory.PRESERVE_VERBATIM

    def test_relative_file_path(self):
        assert classify("Check ~/config/settings.json") == ContextCategory.PRESERVE_VERBATIM

    def test_declaration_def(self):
        assert classify("def process_request(self, req):") == ContextCategory.PRESERVE_VERBATIM

    def test_declaration_class(self):
        assert classify("class ContextManager:") == ContextCategory.PRESERVE_VERBATIM

    def test_declaration_import(self):
        assert classify("from src.types.core import ContextCategory") == ContextCategory.PRESERVE_VERBATIM

    def test_declaration_const(self):
        assert classify("const MAX_RETRIES = 3") == ContextCategory.PRESERVE_VERBATIM

    def test_test_output_passed(self):
        assert classify("12 tests passed, 0 failed") == ContextCategory.PRESERVE_VERBATIM

    def test_test_output_failed(self):
        assert classify("FAILED test_something") == ContextCategory.PRESERVE_VERBATIM

    def test_test_output_checkmark(self):
        assert classify("✓ should return correct value") == ContextCategory.PRESERVE_VERBATIM

    def test_test_output_pytest(self):
        assert classify("pytest: 5 passed, 1 failed") == ContextCategory.PRESERVE_VERBATIM

    def test_version_number_semver(self):
        assert classify("Updated to v3.2.1") == ContextCategory.PRESERVE_VERBATIM

    def test_version_number_bare(self):
        assert classify("Requires 1.0.0 or higher") == ContextCategory.PRESERVE_VERBATIM

    def test_version_constraint(self):
        assert classify("dependency >=2.1") == ContextCategory.PRESERVE_VERBATIM

    def test_env_variable_dollar(self):
        assert classify("Set $HOME to the user directory") == ContextCategory.PRESERVE_VERBATIM

    def test_env_variable_braces(self):
        assert classify("Use ${DATABASE_URL} for connection") == ContextCategory.PRESERVE_VERBATIM

    def test_env_variable_assignment(self):
        assert classify("DATABASE_URL=postgres://localhost/db") == ContextCategory.PRESERVE_VERBATIM


# ---------------------------------------------------------------------------
# STRUCTURED pattern tests
# ---------------------------------------------------------------------------

class TestStructuredPatterns:
    """Content matching STRUCTURED_PATTERNS → PRESERVE_STRUCTURED."""

    def test_decision_record(self):
        assert classify("DECISION: Use PostgreSQL for persistence") == ContextCategory.PRESERVE_STRUCTURED

    def test_todo(self):
        assert classify("TODO: Implement caching layer") == ContextCategory.PRESERVE_STRUCTURED

    def test_task(self):
        assert classify("TASK: Refactor the auth module") == ContextCategory.PRESERVE_STRUCTURED

    def test_adr(self):
        assert classify("ADR-003 Use event sourcing for audit trail") == ContextCategory.PRESERVE_STRUCTURED

    def test_acceptance_criteria_given_when_then(self):
        content = "GIVEN a logged-in user WHEN they click logout THEN the session ends"
        assert classify(content) == ContextCategory.PRESERVE_STRUCTURED

    def test_acceptance_criteria_shall(self):
        content = "The system SHALL validate all inputs before processing"
        assert classify(content) == ContextCategory.PRESERVE_STRUCTURED

    def test_json_schema(self):
        content = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        assert classify(content) == ContextCategory.PRESERVE_STRUCTURED

    def test_sql_schema(self):
        content = "CREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT)"
        assert classify(content) == ContextCategory.PRESERVE_STRUCTURED


# ---------------------------------------------------------------------------
# Source-based fallback and default
# ---------------------------------------------------------------------------

class TestClassifyFallback:
    """Tests for source-based EPHEMERAL fallback and COMPRESS_AGGRESSIVE default."""

    def test_tool_call_no_pattern_is_ephemeral(self):
        content = "Searching for files matching the query..."
        assert classify(content, source="tool_call") == ContextCategory.EPHEMERAL

    def test_tool_result_no_pattern_is_ephemeral(self):
        content = "Found 42 results in the index."
        assert classify(content, source="tool_result") == ContextCategory.EPHEMERAL

    def test_tool_call_with_verbatim_pattern_is_verbatim(self):
        # Verbatim patterns override source-based fallback
        content = "Error: connection refused at socket.connect"
        assert classify(content, source="tool_call") == ContextCategory.PRESERVE_VERBATIM

    def test_tool_result_with_structured_pattern_is_structured(self):
        content = "DECISION: Use Redis for caching"
        assert classify(content, source="tool_result") == ContextCategory.PRESERVE_STRUCTURED

    def test_plain_discussion_is_compress_aggressive(self):
        content = "I think we should consider using a different approach for this feature."
        assert classify(content) == ContextCategory.COMPRESS_AGGRESSIVE

    def test_user_source_no_pattern_is_compress_aggressive(self):
        content = "Let me think about the best way to implement this."
        assert classify(content, source="user") == ContextCategory.COMPRESS_AGGRESSIVE

    def test_assistant_source_no_pattern_is_compress_aggressive(self):
        content = "Here is my analysis of the situation and possible alternatives."
        assert classify(content, source="assistant") == ContextCategory.COMPRESS_AGGRESSIVE

    def test_empty_source_defaults_to_compress(self):
        content = "Some general discussion text without any patterns."
        assert classify(content, source="") == ContextCategory.COMPRESS_AGGRESSIVE


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same (content, source) must always produce the same category."""

    def test_repeated_calls_same_result(self):
        content = "Error: something failed at line 10"
        results = [classify(content, "tool_result") for _ in range(50)]
        assert all(r == results[0] for r in results)

    def test_deterministic_across_categories(self):
        cases = [
            ("Traceback (most recent call last):", ""),
            ("DECISION: use X", ""),
            ("just some chat", "user"),
            ("raw search output", "tool_call"),
        ]
        for content, source in cases:
            first = classify(content, source)
            second = classify(content, source)
            assert first == second, f"Non-deterministic for ({content!r}, {source!r})"


# ---------------------------------------------------------------------------
# Priority: verbatim > structured > ephemeral > compress
# ---------------------------------------------------------------------------

class TestPriority:
    """Verbatim patterns take precedence over structured patterns."""

    def test_verbatim_beats_structured(self):
        # Content that matches both verbatim (file path) and structured (DECISION:)
        content = "DECISION: Fix /home/user/src/main.py immediately"
        # File path is a verbatim pattern, DECISION: is structured.
        # Verbatim is checked first, so it wins.
        assert classify(content) == ContextCategory.PRESERVE_VERBATIM

    def test_structured_beats_ephemeral(self):
        content = "TODO: Clean up the search results"
        assert classify(content, source="tool_result") == ContextCategory.PRESERVE_STRUCTURED


# ---------------------------------------------------------------------------
# Pattern list sanity
# ---------------------------------------------------------------------------

class TestPatternLists:
    """Verify pattern lists are non-empty and well-formed."""

    def test_verbatim_patterns_not_empty(self):
        assert len(VERBATIM_PATTERNS) >= 7  # 7 categories per design

    def test_structured_patterns_not_empty(self):
        assert len(STRUCTURED_PATTERNS) >= 3  # 3 categories per design

    def test_all_patterns_are_tuples(self):
        for name, pat in VERBATIM_PATTERNS:
            assert isinstance(name, str)
            assert isinstance(pat, type(re.compile("")))
        for name, pat in STRUCTURED_PATTERNS:
            assert isinstance(name, str)
            assert isinstance(pat, type(re.compile("")))
