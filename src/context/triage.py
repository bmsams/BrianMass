"""Semantic triage classifier for context items.

Classifies context content into one of four ContextCategory values using
pattern matching against VERBATIM_PATTERNS and STRUCTURED_PATTERNS.

Classification is deterministic: same (content, source) always produces
the same category.

Requirements: 2.1, 2.2
"""

import re

from src.types.core import ContextCategory

# --- VERBATIM_PATTERNS ---
# Content matching these patterns is never summarized.
# Each entry is (name, compiled regex).
VERBATIM_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Stack traces: "Error: ... at ..." or "Traceback (most recent call last):"
    ("stack_trace", re.compile(
        r"(Error:.*at\s+|Traceback \(most recent call last\):|"
        r"^\s+at\s+\S+|^\s+File \".+\", line \d+)",
        re.MULTILINE,
    )),
    # Exit codes: "exit code 1", "exited with 127", "return code 0"
    ("exit_code", re.compile(
        r"(exit\s*(code|status)\s*[=:]?\s*\d+|exited\s+with\s+\d+|return\s*code\s*\d+)",
        re.IGNORECASE,
    )),
    # File paths: Unix or Windows absolute/relative paths
    ("file_path", re.compile(
        r"((?:^|\s)[/~][\w./\-]+\.\w+|(?:^|\s)[A-Z]:\\[\w\\.\-]+\.\w+)",
        re.MULTILINE,
    )),
    # Declarations: function/class/def/const/let/var/import/export
    ("declaration", re.compile(
        r"^\s*(def\s+\w+|class\s+\w+|function\s+\w+|const\s+\w+|"
        r"let\s+\w+|var\s+\w+|import\s+|export\s+|from\s+\S+\s+import)",
        re.MULTILINE,
    )),
    # Test output: PASSED, FAILED, test results, assertions
    ("test_output", re.compile(
        r"(PASS(ED)?|FAIL(ED)?|ERROR|✓|✗|✘|"
        r"\d+\s+(tests?|specs?)\s+(passed|failed|skipped)|"
        r"assert(ion)?\s+(failed|error)|"
        r"(pytest|jest|mocha|rspec|junit).*(\d+\s+(passed|failed)))",
        re.IGNORECASE | re.MULTILINE,
    )),
    # Version numbers: v1.2.3, 1.2.3, ==1.2.3, >=1.2.3
    ("version_number", re.compile(
        r"(v?\d+\.\d+\.\d+[\w.\-]*|[><=!~]+\d+\.\d+)",
    )),
    # Environment variables: $VAR, ${VAR}, ENV_VAR=value
    ("env_variable", re.compile(
        r"(\$\{?\w+\}?|^[A-Z][A-Z0-9_]{2,}=.+)",
        re.MULTILINE,
    )),
    # Git diff hunks: @@ -n,m +n,m @@, diff headers
    ("git_diff", re.compile(
        r"(^@@\s*-\d+.*\+\d+.*@@|^diff --git\s|^index [0-9a-f]+\.\.[0-9a-f]+|"
        r"^---\s+a/|^\+\+\+\s+b/)",
        re.MULTILINE,
    )),
    # API responses: HTTP status codes, JSON error/message patterns
    ("api_response", re.compile(
        r"(HTTP/\d\.\d\s+\d{3}|"
        r"\"status(_code)?\":\s*\d{3}|"
        r"\b(GET|POST|PUT|DELETE|PATCH)\s+https?://|"
        r"\"error\":\s*\{|\"message\":\s*\")",
        re.IGNORECASE,
    )),
]

# --- STRUCTURED_PATTERNS ---
# Content matching these patterns keeps data but may compress formatting.
STRUCTURED_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Decision records: DECISION:, TODO:, TASK:, ADR-NNN
    ("decision_record", re.compile(
        r"(DECISION:|TODO:|TASK:|ADR-\d+|RESOLVED:|ACTION\s*ITEM:)",
        re.IGNORECASE,
    )),
    # Acceptance criteria: "GIVEN ... WHEN ... THEN", "SHALL", numbered criteria
    ("acceptance_criteria", re.compile(
        r"(GIVEN\s+.+WHEN\s+.+THEN|SHALL\s+|"
        r"acceptance\s+criteria|"
        r"^\s*\d+\.\s+(THE|WHEN|IF)\s+)",
        re.IGNORECASE | re.MULTILINE,
    )),
    # Schemas: JSON schema keywords, type definitions, field specs
    ("schema", re.compile(
        r'("type"\s*:\s*"(string|number|integer|boolean|object|array)"|'
        r'"properties"\s*:\s*\{|"required"\s*:\s*\[|'
        r"schema\s*[=:]\s*\{|"
        r"CREATE\s+TABLE|ALTER\s+TABLE)",
        re.IGNORECASE,
    )),
]


def classify(content: str, source: str = "") -> ContextCategory:
    """Classify content into a ContextCategory.

    Classification priority:
    1. Check VERBATIM_PATTERNS → PRESERVE_VERBATIM
    2. Check STRUCTURED_PATTERNS → PRESERVE_STRUCTURED
    3. If source is 'tool_call' or 'tool_result' with no pattern match → EPHEMERAL
    4. Default → COMPRESS_AGGRESSIVE

    Args:
        content: The text content to classify.
        source: Origin of the content (e.g. 'user', 'assistant',
                'tool_call', 'tool_result', 'system').

    Returns:
        A ContextCategory enum value. Classification is deterministic.
    """
    # 1. Verbatim patterns take highest priority
    for _name, pattern in VERBATIM_PATTERNS:
        if pattern.search(content):
            return ContextCategory.PRESERVE_VERBATIM

    # 2. Structured patterns
    for _name, pattern in STRUCTURED_PATTERNS:
        if pattern.search(content):
            return ContextCategory.PRESERVE_STRUCTURED

    # 3. Unmatched tool traffic is ephemeral
    if source in ("tool_call", "tool_result"):
        return ContextCategory.EPHEMERAL

    # 4. Everything else gets aggressively compressed
    return ContextCategory.COMPRESS_AGGRESSIVE
