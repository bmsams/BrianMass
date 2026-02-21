"""Data classification, guardrails, constitutional AI, and nest guard.

Provides the DataClassifier with four security capabilities:
1. Sensitive data detection (PII, PHI, financial, credentials) via regex
2. Bedrock Guardrails integration (pluggable callback pattern)
3. Constitutional AI content filtering
4. Nested session detection (Nest Guard)

Requirements: 14.1, 14.2, 14.3, 14.5
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class Classification:
    """A single sensitive-data finding within a text.

    Attributes:
        category: Top-level class -- 'pii', 'phi', 'financial', 'credentials'.
        subcategory: Specific type, e.g. 'ssn', 'email', 'credit_card', 'api_key'.
        confidence: Score from 0.0 (speculative) to 1.0 (definite match).
        matched_text: The matched content, partially redacted for safe display.
        start: Character offset of the match start.
        end: Character offset of the match end.
    """

    category: str
    subcategory: str
    confidence: float
    matched_text: str
    start: int
    end: int


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

@dataclass
class _PatternDef:
    """Internal definition for a single detection pattern."""

    category: str
    subcategory: str
    pattern: re.Pattern[str]
    confidence: float
    validator: str | None = None  # name of post-match validation function


# PII patterns  (Requirement 14.1)
_PII_PATTERNS: list[_PatternDef] = [
    _PatternDef(
        category="pii",
        subcategory="ssn",
        pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        confidence=0.95,
    ),
    _PatternDef(
        category="pii",
        subcategory="email",
        pattern=re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        ),
        confidence=0.90,
    ),
    _PatternDef(
        category="pii",
        subcategory="phone",
        pattern=re.compile(
            r"(?<!\d)"
            r"(?:\+?1[\s\-.]?)?"
            r"(?:\(?\d{3}\)?[\s\-.]?)"
            r"\d{3}[\s\-.]?\d{4}"
            r"(?!\d)",
        ),
        confidence=0.80,
    ),
    _PatternDef(
        category="pii",
        subcategory="name",
        pattern=re.compile(
            r"(?i)(?:name|patient|customer|employee|applicant|user)"
            r"[\s:=]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        ),
        confidence=0.70,
    ),
]

# PHI patterns  (Requirement 14.1)
_PHI_PATTERNS: list[_PatternDef] = [
    _PatternDef(
        category="phi",
        subcategory="medical_record_number",
        pattern=re.compile(
            r"(?i)(?:MRN|medical\s+record(?:\s+number)?|med\s*rec)"
            r"[\s#:=]*([A-Z0-9\-]{4,20})",
        ),
        confidence=0.90,
    ),
    _PatternDef(
        category="phi",
        subcategory="icd_code",
        pattern=re.compile(
            r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b",
        ),
        confidence=0.85,
    ),
    _PatternDef(
        category="phi",
        subcategory="diagnosis",
        pattern=re.compile(
            r"(?i)\b(?:diagnosis|diagnosed\s+with|dx|prognosis|treatment\s+for)"
            r"[\s:=]+[A-Za-z\s]{3,50}",
        ),
        confidence=0.80,
    ),
    _PatternDef(
        category="phi",
        subcategory="health_term",
        pattern=re.compile(
            r"(?i)\b(?:blood\s+pressure|heart\s+rate|hemoglobin|cholesterol"
            r"|glucose\s+level|bmi|prescription|medication|dosage"
            r"|allergy|surgery|biopsy|radiology|oncology|pathology)\b",
        ),
        confidence=0.75,
    ),
]

# Financial patterns  (Requirement 14.1)
_FINANCIAL_PATTERNS: list[_PatternDef] = [
    _PatternDef(
        category="financial",
        subcategory="credit_card",
        pattern=re.compile(
            r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
            r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b",
        ),
        confidence=0.90,
        validator="luhn",
    ),
    _PatternDef(
        category="financial",
        subcategory="bank_account",
        pattern=re.compile(
            r"(?i)(?:account|acct|routing|aba|iban)[\s#:=]*(\d[\d\s\-]{6,30}\d)",
        ),
        confidence=0.80,
    ),
    _PatternDef(
        category="financial",
        subcategory="currency_amount",
        pattern=re.compile(
            r"(?:[$\u00a3\u20ac\u00a5])\s?\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?"
            r"|\b\d{1,3}(?:[,\s]\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP|JPY)\b",
        ),
        confidence=0.70,
    ),
]

# Credential patterns  (Requirement 14.1)
_CREDENTIAL_PATTERNS: list[_PatternDef] = [
    _PatternDef(
        category="credentials",
        subcategory="api_key",
        pattern=re.compile(
            r"(?i)(?:api[_\-]?key|apikey|api[_\-]?secret)[\s:=]+['\"]?"
            r"([A-Za-z0-9\-_]{20,})['\"]?",
        ),
        confidence=0.95,
    ),
    _PatternDef(
        category="credentials",
        subcategory="token",
        pattern=re.compile(
            r"(?i)(?:bearer|token|auth[_\-]?token|access[_\-]?token"
            r"|refresh[_\-]?token|jwt)[\s:=]+['\"]?"
            r"([A-Za-z0-9\-_\.]{20,})['\"]?",
        ),
        confidence=0.90,
    ),
    _PatternDef(
        category="credentials",
        subcategory="password",
        pattern=re.compile(
            r"(?i)(?:password|passwd|pwd|secret)[\s:=]+['\"]?"
            r"(\S{6,})['\"]?",
        ),
        confidence=0.85,
    ),
    _PatternDef(
        category="credentials",
        subcategory="private_key",
        pattern=re.compile(
            r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
        ),
        confidence=1.0,
    ),
    _PatternDef(
        category="credentials",
        subcategory="aws_key",
        pattern=re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        confidence=0.95,
    ),
    _PatternDef(
        category="credentials",
        subcategory="generic_secret",
        pattern=re.compile(
            r"(?i)(?:client[_\-]?secret|secret[_\-]?key|signing[_\-]?key"
            r"|encryption[_\-]?key|private[_\-]?key|service[_\-]?key)"
            r"[\s:=]+['\"]?(\S{8,})['\"]?",
        ),
        confidence=0.85,
    ),
]

# All patterns combined for iteration
_ALL_PATTERNS: list[_PatternDef] = (
    _PII_PATTERNS + _PHI_PATTERNS + _FINANCIAL_PATTERNS + _CREDENTIAL_PATTERNS
)


# ---------------------------------------------------------------------------
# Constitutional AI patterns  (Requirement 14.3)
# ---------------------------------------------------------------------------

_CONSTITUTIONAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "malware_exploit",
        re.compile(
            r"(?i)(?:how\s+to\s+(?:create|write|build|make|develop)\s+"
            r"(?:a\s+)?(?:malware|virus|trojan|worm|ransomware|rootkit|exploit|"
            r"keylogger|backdoor|botnet|rat\b|remote\s+access\s+tool))"
            r"|(?:exploit\s+(?:code|kit|chain)\s+(?:for|targeting))"
            r"|(?:zero[\s\-]?day\s+(?:exploit|vulnerability)\s+(?:creation|development))"
            r"|(?:buffer\s+overflow\s+(?:exploit|attack)\s+(?:code|tutorial))"
            r"|(?:(?:sql|code)\s+injection\s+(?:payload|attack)\s+(?:to|against))",
        ),
    ),
    (
        "social_engineering",
        re.compile(
            r"(?i)(?:how\s+to\s+(?:phish|spear[\s\-]?phish|impersonate|catfish"
            r"|manipulate\s+(?:someone|people|victims)))"
            r"|(?:(?:craft|create|write)\s+(?:a\s+)?(?:phishing|spear[\s\-]?phishing)"
            r"\s+(?:email|message|campaign))"
            r"|(?:social\s+engineering\s+(?:attack|technique|script)\s+"
            r"(?:to|for|against))",
        ),
    ),
    (
        "harmful_synthesis",
        re.compile(
            r"(?i)(?:(?:how\s+to\s+)?(?:synthesize|produce|manufacture|make|create)"
            r"\s+(?:nerve\s+agent|poison|toxic\s+gas|chemical\s+weapon|"
            r"biological\s+weapon|bioweapon|explosive|bomb|ricin|sarin|anthrax|"
            r"mustard\s+gas|vx\s+gas|fentanyl\s+analog))"
            r"|(?:(?:synthesis|production|manufacturing)\s+(?:route|instructions"
            r"|procedure|protocol)\s+(?:for\s+)?(?:controlled|illegal|illicit"
            r"|prohibited)\s+substance)",
        ),
    ),
    (
        "weaponization",
        re.compile(
            r"(?i)(?:(?:how\s+to\s+)?(?:build|construct|assemble|manufacture|make)"
            r"\s+(?:a\s+)?(?:bomb|ied|improvised\s+explosive|detonator|"
            r"firearm|gun|weapon|silencer|suppressor))"
            r"|(?:(?:3d[\s\-]?print|print)\s+(?:a\s+)?(?:gun|firearm|weapon))"
            r"|(?:(?:weaponize|convert|modify)\s+(?:a\s+)?(?:drone|uav|chemical"
            r"|biological|pathogen|virus))",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _luhn_check(number_str: str) -> bool:
    """Validate a credit card number using the Luhn algorithm.

    Strips whitespace and dashes before validation.

    Returns True if the number passes the Luhn checksum.
    """
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse = digits[::-1]
    for i, d in enumerate(reverse):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


_VALIDATORS: dict[str, type[object] | object] = {
    "luhn": _luhn_check,
}


# ---------------------------------------------------------------------------
# Redaction helper
# ---------------------------------------------------------------------------

def _redact(text: str, keep_start: int = 2, keep_end: int = 2) -> str:
    """Return a redacted version of *text* preserving only the edges.

    For very short strings (length <= keep_start + keep_end + 2), the whole
    string is replaced with asterisks.
    """
    if len(text) <= keep_start + keep_end + 2:
        return "*" * len(text)
    hidden_len = len(text) - keep_start - keep_end
    return text[:keep_start] + "*" * hidden_len + text[-keep_end:]


# ---------------------------------------------------------------------------
# DataClassifier
# ---------------------------------------------------------------------------

class DataClassifier:
    """Classifies text for sensitive data and enforces content safety policies.

    Provides four main capabilities:

    * ``classify`` -- regex-based detection of PII, PHI, financial data,
      and credentials  (Req 14.1).
    * ``apply_guardrails`` -- pluggable Bedrock Guardrails callback
      (Req 14.2).
    * ``check_constitutional_ai`` -- checks for dangerous content patterns
      (Req 14.3).
    * ``check_nest_guard`` -- detects nested Brainmass sessions
      (Req 14.5).
    """

    def __init__(
        self,
        *,
        guardrail_callback: type[object] | object | None = None,
        extra_patterns: list[_PatternDef] | None = None,
    ) -> None:
        """Initialise the classifier.

        Args:
            guardrail_callback: Optional callable for Bedrock Guardrails
                integration.  Signature:
                ``(content, guardrail_id, guardrail_version) -> dict``.
                When *None*, ``apply_guardrails`` falls back to local
                classification.
            extra_patterns: Additional ``_PatternDef`` entries appended
                to the built-in detection patterns.
        """
        self._guardrail_callback = guardrail_callback
        self._patterns: list[_PatternDef] = list(_ALL_PATTERNS)
        if extra_patterns:
            self._patterns.extend(extra_patterns)

    # ------------------------------------------------------------------
    # Data classification  (Requirement 14.1)
    # ------------------------------------------------------------------

    def classify(self, text: str) -> list[Classification]:
        """Classify *text* for sensitive data.

        Scans the input against all registered patterns (PII, PHI,
        financial, credentials) and returns a list of ``Classification``
        findings sorted by character offset.

        Args:
            text: The content to scan.

        Returns:
            A list of ``Classification`` instances.  Empty when no
            sensitive data is detected.
        """
        if not text:
            return []

        # Collect all candidate matches first
        candidates: list[tuple[float, int, int, str, str, str]] = []

        for pdef in self._patterns:
            for match in pdef.pattern.finditer(text):
                start, end = match.start(), match.end()
                matched_text = match.group(0)

                # Run optional post-match validator
                if pdef.validator:
                    validator_fn = _VALIDATORS.get(pdef.validator)
                    if validator_fn and not validator_fn(matched_text):  # type: ignore[operator]
                        logger.debug(
                            "Validator '%s' rejected match: %s",
                            pdef.validator,
                            _redact(matched_text),
                        )
                        continue

                candidates.append((
                    pdef.confidence, start, end,
                    pdef.category, pdef.subcategory, matched_text,
                ))

        # Sort by confidence DESC, then by span length DESC (prefer more specific)
        candidates.sort(key=lambda c: (-c[0], -(c[2] - c[1])))

        # Deduplicate: higher-confidence / longer matches take priority
        findings: list[Classification] = []
        seen_spans: list[tuple[int, int]] = []

        for conf, start, end, category, subcategory, matched_text in candidates:
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            seen_spans.append((start, end))
            findings.append(
                Classification(
                    category=category,
                    subcategory=subcategory,
                    confidence=conf,
                    matched_text=_redact(matched_text),
                    start=start,
                    end=end,
                ),
            )

        findings.sort(key=lambda c: c.start)
        logger.debug(
            "classify: found %d sensitive item(s) across %d category(ies)",
            len(findings),
            len({f.category for f in findings}),
        )
        return findings

    # ------------------------------------------------------------------
    # Bedrock Guardrails  (Requirement 14.2)
    # ------------------------------------------------------------------

    def apply_guardrails(
        self,
        content: str,
        guardrail_id: str = "",
        guardrail_version: str = "",
    ) -> dict:
        """Format and apply Bedrock Guardrails for content filtering.

        When a ``guardrail_callback`` was provided at construction, it is
        invoked with the content and guardrail identifiers.  The callback
        is expected to return a dict with ``{blocked, reason,
        filtered_content}``.

        When no callback is registered the method falls back to local
        classification: content is blocked only if high-confidence
        credential findings are present, and all sensitive spans are
        redacted in ``filtered_content``.

        The returned dict is structured for consumption by Strands /
        Bedrock as the ``guardrail_latest_message`` parameter::

            {
                "blocked": bool,
                "reason": str,
                "filtered_content": str,
                "guardrail_id": str,
                "guardrail_version": str,
                "classifications": [...],
            }

        Args:
            content: The text to evaluate.
            guardrail_id: AWS Guardrail identifier (optional).
            guardrail_version: AWS Guardrail version (optional).

        Returns:
            A dict with blocking decision, reason, and filtered content.
        """
        # Delegate to external callback when available
        if self._guardrail_callback is not None:
            try:
                result = self._guardrail_callback(  # type: ignore[operator]
                    content, guardrail_id, guardrail_version,
                )
                if isinstance(result, dict):
                    return result
                logger.warning(
                    "Guardrail callback returned non-dict; falling back to local",
                )
            except Exception:
                logger.exception("Guardrail callback raised; falling back to local")

        # Local fallback using classification
        findings = self.classify(content)
        blocked = any(
            f.category == "credentials" and f.confidence >= 0.90
            for f in findings
        )

        filtered = self._redact_content(content, findings)
        reason = ""
        if blocked:
            credential_types = sorted(
                {f.subcategory for f in findings if f.category == "credentials"},
            )
            reason = (
                f"Blocked: credential data detected ({', '.join(credential_types)})"
            )
        elif findings:
            reason = (
                f"Passed with {len(findings)} redaction(s) applied"
            )

        return {
            "blocked": blocked,
            "reason": reason,
            "filtered_content": filtered,
            "guardrail_id": guardrail_id,
            "guardrail_version": guardrail_version,
            "classifications": [
                {
                    "category": f.category,
                    "subcategory": f.subcategory,
                    "confidence": f.confidence,
                    "start": f.start,
                    "end": f.end,
                }
                for f in findings
            ],
        }

    @staticmethod
    def _redact_content(
        content: str,
        findings: list[Classification],
    ) -> str:
        """Replace each finding span in *content* with asterisks."""
        if not findings:
            return content
        # Process from end to start to preserve character offsets
        result = content
        for f in sorted(findings, key=lambda c: c.start, reverse=True):
            result = result[:f.start] + "*" * (f.end - f.start) + result[f.end:]
        return result

    # ------------------------------------------------------------------
    # Constitutional AI  (Requirement 14.3)
    # ------------------------------------------------------------------

    def check_constitutional_ai(self, content: str) -> dict:
        """Check *content* against constitutional AI safety filters.

        Scans for patterns indicating requests for:
        - Malware or exploit creation instructions
        - Social engineering guidance
        - Harmful chemical or biological synthesis
        - Weaponization instructions

        Args:
            content: The text to evaluate.

        Returns:
            A dict with keys:
            - ``safe`` (bool): True when no violations are found.
            - ``violations`` (list[str]): Names of violated categories.
        """
        if not content:
            return {"safe": True, "violations": []}

        violations: list[str] = []
        for name, pattern in _CONSTITUTIONAL_PATTERNS:
            if pattern.search(content):
                violations.append(name)
                logger.warning(
                    "Constitutional AI violation detected: %s", name,
                )

        safe = len(violations) == 0
        if safe:
            logger.debug("Constitutional AI check passed")

        return {"safe": safe, "violations": violations}

    # ------------------------------------------------------------------
    # Nest Guard  (Requirement 14.5)
    # ------------------------------------------------------------------

    def check_nest_guard(
        self,
        env_vars: dict[str, str] | None = None,
        process_tree: list[str] | None = None,
    ) -> dict:
        """Detect if Brainmass is being launched inside another session.

        Checks three signals:
        1. ``BRAINMASS_SESSION_ID`` environment variable is set.
        2. ``BRAINMASS_NESTED`` marker environment variable is set.
        3. Process tree contains a brainmass parent process.

        Any single signal is sufficient to flag the session as nested.

        Args:
            env_vars: Environment variable mapping.  Defaults to ``None``
                (caller should pass ``os.environ`` when available).
            process_tree: List of process names / command lines from the
                current process up to PID 1.  Defaults to ``None``.

        Returns:
            A dict with keys:
            - ``nested`` (bool): True when a nesting indicator is found.
            - ``reason`` (str): Human-readable explanation, empty when
              not nested.
        """
        env = env_vars or {}
        procs = process_tree or []
        reasons: list[str] = []

        # 1. Check BRAINMASS_SESSION_ID
        session_id = env.get("BRAINMASS_SESSION_ID", "")
        if session_id:
            reasons.append(
                f"BRAINMASS_SESSION_ID is set (value: {session_id[:8]}...)"
            )

        # 2. Check BRAINMASS_NESTED marker
        nested_marker = env.get("BRAINMASS_NESTED", "")
        if nested_marker:
            reasons.append("BRAINMASS_NESTED marker is set")

        # 3. Scan process tree for brainmass parent processes
        brainmass_pattern = re.compile(r"(?i)\bbrainmass\b")
        for proc in procs:
            if brainmass_pattern.search(proc):
                reasons.append(
                    f"Parent process matches brainmass: {proc[:80]}"
                )
                break  # one match is sufficient

        nested = len(reasons) > 0
        reason = "; ".join(reasons) if nested else ""

        if nested:
            logger.warning("Nest Guard triggered: %s", reason)
        else:
            logger.debug("Nest Guard check passed")

        return {"nested": nested, "reason": reason}
