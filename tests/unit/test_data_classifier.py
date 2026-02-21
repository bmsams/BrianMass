"""Tests for Data Classifier and security subsystem.

Validates Requirements 14.1, 14.2, 14.3, 14.5:
- 14.1: Data classification for PII, PHI, financial data, credentials
- 14.2: Bedrock Guardrails integration
- 14.3: Constitutional AI filter
- 14.5: Nest guard
"""

from __future__ import annotations

import pytest

from src.security.data_classifier import DataClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> DataClassifier:
    return DataClassifier()


# ---------------------------------------------------------------------------
# Req 14.1: Data classification — PII
# ---------------------------------------------------------------------------


class TestPIIClassification:
    """Req 14.1: Classify PII (SSN, email, phone, names)."""

    def test_detects_ssn(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("SSN is 123-45-6789")
        ssn_findings = [f for f in findings if f.subcategory == "ssn"]
        assert len(ssn_findings) == 1
        assert ssn_findings[0].category == "pii"
        assert ssn_findings[0].confidence >= 0.90

    def test_detects_email(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("Contact john.doe@example.com")
        email_findings = [f for f in findings if f.subcategory == "email"]
        assert len(email_findings) == 1
        assert email_findings[0].category == "pii"

    def test_detects_phone(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("Call (555) 123-4567")
        phone_findings = [f for f in findings if f.subcategory == "phone"]
        assert len(phone_findings) >= 1
        assert phone_findings[0].category == "pii"

    def test_detects_name_with_context(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("patient: John Smith")
        name_findings = [f for f in findings if f.subcategory == "name"]
        assert len(name_findings) == 1

    def test_no_false_positive_on_clean_text(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("The quick brown fox jumps over the lazy dog.")
        pii_findings = [f for f in findings if f.category == "pii"]
        assert len(pii_findings) == 0

    def test_matched_text_is_redacted(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("SSN is 123-45-6789")
        ssn = [f for f in findings if f.subcategory == "ssn"][0]
        assert "*" in ssn.matched_text

    def test_offsets_are_correct(self, classifier: DataClassifier) -> None:
        text = "SSN is 123-45-6789"
        findings = classifier.classify(text)
        ssn = [f for f in findings if f.subcategory == "ssn"][0]
        assert text[ssn.start : ssn.end] == "123-45-6789"


# ---------------------------------------------------------------------------
# Req 14.1: Data classification — PHI
# ---------------------------------------------------------------------------


class TestPHIClassification:
    """Req 14.1: Classify PHI (medical records, ICD, diagnosis, health terms)."""

    def test_detects_medical_record_number(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("MRN: ABC-12345")
        mrn_findings = [f for f in findings if f.subcategory == "medical_record_number"]
        assert len(mrn_findings) >= 1
        assert mrn_findings[0].category == "phi"

    def test_detects_icd_code(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("Diagnosis code: J18.9")
        icd_findings = [f for f in findings if f.subcategory == "icd_code"]
        assert len(icd_findings) >= 1

    def test_detects_diagnosis(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("diagnosed with Type 2 Diabetes")
        dx_findings = [f for f in findings if f.subcategory == "diagnosis"]
        assert len(dx_findings) >= 1

    def test_detects_health_terms(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("blood pressure reading was normal")
        health_findings = [f for f in findings if f.subcategory == "health_term"]
        assert len(health_findings) >= 1


# ---------------------------------------------------------------------------
# Req 14.1: Data classification — Financial
# ---------------------------------------------------------------------------


class TestFinancialClassification:
    """Req 14.1: Classify financial data (credit cards, bank accounts, currency)."""

    def test_detects_credit_card_visa(self, classifier: DataClassifier) -> None:
        # Valid Visa with Luhn check: 4111111111111111
        findings = classifier.classify("Card: 4111 1111 1111 1111")
        cc_findings = [f for f in findings if f.subcategory == "credit_card"]
        assert len(cc_findings) >= 1
        assert cc_findings[0].category == "financial"

    def test_rejects_invalid_luhn(self, classifier: DataClassifier) -> None:
        # Invalid Luhn: 4111111111111112
        findings = classifier.classify("Card: 4111 1111 1111 1112")
        cc_findings = [f for f in findings if f.subcategory == "credit_card"]
        assert len(cc_findings) == 0

    def test_detects_bank_account(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("account: 1234567890")
        bank_findings = [f for f in findings if f.subcategory == "bank_account"]
        assert len(bank_findings) >= 1

    def test_detects_currency_amount(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("Payment of $1,500.00")
        currency_findings = [f for f in findings if f.subcategory == "currency_amount"]
        assert len(currency_findings) >= 1


# ---------------------------------------------------------------------------
# Req 14.1: Data classification — Credentials
# ---------------------------------------------------------------------------


class TestCredentialClassification:
    """Req 14.1: Classify credentials (API keys, tokens, passwords, private keys)."""

    def test_detects_api_key(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("api_key: sk_live_abcdefghij1234567890")
        api_findings = [f for f in findings if f.subcategory == "api_key"]
        assert len(api_findings) >= 1
        assert api_findings[0].category == "credentials"
        assert api_findings[0].confidence >= 0.90

    def test_detects_bearer_token(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test")
        token_findings = [f for f in findings if f.subcategory == "token"]
        assert len(token_findings) >= 1

    def test_detects_password(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("password: MyS3cretP@ss!")
        pwd_findings = [f for f in findings if f.subcategory == "password"]
        assert len(pwd_findings) >= 1

    def test_detects_private_key(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("-----BEGIN RSA PRIVATE KEY-----")
        pk_findings = [f for f in findings if f.subcategory == "private_key"]
        assert len(pk_findings) >= 1
        assert pk_findings[0].confidence == 1.0

    def test_detects_aws_key(self, classifier: DataClassifier) -> None:
        findings = classifier.classify("AKIAIOSFODNN7EXAMPLE")
        aws_findings = [f for f in findings if f.subcategory == "aws_key"]
        assert len(aws_findings) >= 1

    def test_empty_text_returns_empty(self, classifier: DataClassifier) -> None:
        assert classifier.classify("") == []


# ---------------------------------------------------------------------------
# Req 14.1: Multiple categories in one text
# ---------------------------------------------------------------------------


class TestMultiCategoryClassification:
    """Multiple sensitive data types in a single text."""

    def test_mixed_findings_sorted_by_offset(self, classifier: DataClassifier) -> None:
        text = "SSN: 123-45-6789, email: test@example.com"
        findings = classifier.classify(text)
        assert len(findings) >= 2
        # Should be sorted by start offset
        for i in range(len(findings) - 1):
            assert findings[i].start <= findings[i + 1].start


# ---------------------------------------------------------------------------
# Req 14.2: Bedrock Guardrails
# ---------------------------------------------------------------------------


class TestGuardrails:
    """Req 14.2: Bedrock Guardrails integration."""

    def test_no_callback_falls_back_to_local(self) -> None:
        classifier = DataClassifier()
        result = classifier.apply_guardrails("Hello world")
        assert result["blocked"] is False
        assert "filtered_content" in result

    def test_blocks_on_credentials(self) -> None:
        classifier = DataClassifier()
        result = classifier.apply_guardrails("api_key: sk_live_abcdefghij1234567890")
        assert result["blocked"] is True

    def test_custom_callback_used(self) -> None:
        def mock_guardrail(content: str, gid: str, gver: str) -> dict:
            return {
                "blocked": True,
                "reason": "Custom block",
                "filtered_content": "[BLOCKED]",
            }

        classifier = DataClassifier(guardrail_callback=mock_guardrail)
        result = classifier.apply_guardrails("test content", "gr-123", "1")
        assert result["blocked"] is True
        assert result["reason"] == "Custom block"

    def test_guardrail_ids_passed_through(self) -> None:
        classifier = DataClassifier()
        result = classifier.apply_guardrails("Hello", "gr-abc", "2")
        assert result["guardrail_id"] == "gr-abc"
        assert result["guardrail_version"] == "2"


# ---------------------------------------------------------------------------
# Req 14.3: Constitutional AI filter
# ---------------------------------------------------------------------------


class TestConstitutionalAI:
    """Req 14.3: Constitutional AI content filtering."""

    def test_safe_content(self, classifier: DataClassifier) -> None:
        result = classifier.check_constitutional_ai("How to bake a cake")
        assert result["safe"] is True
        assert result["violations"] == []

    def test_detects_malware_request(self, classifier: DataClassifier) -> None:
        result = classifier.check_constitutional_ai(
            "how to create a malware that steals data"
        )
        assert result["safe"] is False
        assert any("malware" in v for v in result["violations"])

    def test_detects_social_engineering(self, classifier: DataClassifier) -> None:
        result = classifier.check_constitutional_ai(
            "how to phish someone's credentials"
        )
        assert result["safe"] is False

    def test_detects_harmful_synthesis(self, classifier: DataClassifier) -> None:
        result = classifier.check_constitutional_ai(
            "how to synthesize nerve agent"
        )
        assert result["safe"] is False

    def test_detects_weaponization(self, classifier: DataClassifier) -> None:
        result = classifier.check_constitutional_ai(
            "how to build a bomb"
        )
        assert result["safe"] is False

    def test_empty_content_is_safe(self, classifier: DataClassifier) -> None:
        result = classifier.check_constitutional_ai("")
        assert result["safe"] is True


# ---------------------------------------------------------------------------
# Req 14.5: Nest guard
# ---------------------------------------------------------------------------


class TestNestGuard:
    """Req 14.5: Prevent Brainmass inside Brainmass."""

    def test_no_nesting_detected(self, classifier: DataClassifier) -> None:
        result = classifier.check_nest_guard(
            env_vars={"HOME": "/home/user"},
            process_tree=["bash", "python"],
        )
        assert result["nested"] is False

    def test_detects_session_id_env(self, classifier: DataClassifier) -> None:
        result = classifier.check_nest_guard(
            env_vars={"BRAINMASS_SESSION_ID": "abc-123"},
        )
        assert result["nested"] is True
        assert "BRAINMASS_SESSION_ID" in result["reason"]

    def test_detects_nested_marker(self, classifier: DataClassifier) -> None:
        result = classifier.check_nest_guard(
            env_vars={"BRAINMASS_NESTED": "1"},
        )
        assert result["nested"] is True

    def test_detects_process_tree(self, classifier: DataClassifier) -> None:
        result = classifier.check_nest_guard(
            process_tree=["init", "bash", "brainmass", "python", "brainmass"],
        )
        assert result["nested"] is True

    def test_defaults_when_none(self, classifier: DataClassifier) -> None:
        result = classifier.check_nest_guard()
        assert "nested" in result
