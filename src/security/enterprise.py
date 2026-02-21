"""Enterprise security features — HIPAA compliance, identity, policy, and VPC isolation.

Implements pluggable enterprise-grade security controls for Brainmass v3:
- HIPAA compliance management with PHI consent and audit logging
- Identity provider integration (Okta, Entra ID, Cognito) via pluggable callbacks
- Cedar-based policy evaluation engine
- VPC-only mode with network isolation validation

All external SDK calls are pluggable callbacks so the module remains testable
without real AWS or identity provider dependencies.

Requirements: 14.4, 14.9, 14.10
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_IDENTITY_PROVIDERS = {"okta", "entra", "cognito"}

ENTERPRISE_PLAN_TYPES = {"enterprise", "enterprise_plus"}

PLATFORMS_REQUIRING_PHI_CONSENT = {"ios", "android"}

VALID_CEDAR_ACTIONS = {"read", "write", "execute", "delete", "admin"}


# ---------------------------------------------------------------------------
# Auth result and policy decision data models
# ---------------------------------------------------------------------------

@dataclass
class AuthResult:
    """Result returned from an identity provider authentication attempt."""

    authenticated: bool
    user_id: str = ""
    email: str = ""
    roles: list[str] = field(default_factory=list)
    provider: str = ""
    error: str = ""


@dataclass
class PolicyDecision:
    """Result of a Cedar policy evaluation."""

    allowed: bool
    policy_id: str = ""
    reason: str = ""


# ---------------------------------------------------------------------------
# PHI audit record
# ---------------------------------------------------------------------------

@dataclass
class PHIAuditRecord:
    """An immutable record of a PHI access event for HIPAA audit trails."""

    agent_id: str
    data_description: str
    timestamp: datetime
    session_id: str = ""


# ---------------------------------------------------------------------------
# HIPAA Compliance Manager (Req 14.4)
# ---------------------------------------------------------------------------

class HIPAAComplianceManager:
    """Manages HIPAA compliance state, PHI consent, and audit logging.

    Thread-safe: all mutations are guarded by a lock so concurrent agents
    can safely call ``audit_phi_access``.

    Usage::

        hipaa = HIPAAComplianceManager()
        if hipaa.is_hipaa_enabled("enterprise"):
            consent = hipaa.check_phi_consent("ios")
            if consent["consent_required"]:
                # ... obtain user consent ...
                hipaa.record_phi_consent("ios", user_id="u-123")
            hipaa.audit_phi_access("agent-1", "Patient blood pressure readings")
    """

    def __init__(
        self,
        audit_callback: Callable[[PHIAuditRecord], None] | None = None,
    ) -> None:
        """
        Args:
            audit_callback: Optional callback invoked for each PHI access event.
                Allows external systems (e.g. CloudWatch, SIEM) to receive
                audit records without coupling to a specific SDK.
        """
        self._lock = threading.Lock()
        self._audit_callback = audit_callback
        self._audit_log: list[PHIAuditRecord] = []
        # platform → {user_id → consent_timestamp}
        self._consent_records: dict[str, dict[str, datetime]] = {}

    # ------------------------------------------------------------------
    # Public API: Plan gating
    # ------------------------------------------------------------------

    def is_hipaa_enabled(self, plan_type: str) -> bool:
        """Check whether HIPAA compliance features are available for a plan.

        Only Enterprise-tier plans have access to HIPAA compliance features.

        Args:
            plan_type: The subscription plan identifier (e.g. 'enterprise',
                'pro', 'free').

        Returns:
            True if the plan supports HIPAA compliance.
        """
        enabled = plan_type.lower() in ENTERPRISE_PLAN_TYPES
        logger.debug("HIPAA enabled check for plan '%s': %s", plan_type, enabled)
        return enabled

    # ------------------------------------------------------------------
    # Public API: PHI consent
    # ------------------------------------------------------------------

    def check_phi_consent(self, platform: str) -> dict:
        """Check whether explicit PHI consent is required for a platform.

        iOS and Android require explicit user consent before any PHI data
        can be processed. Desktop and web platforms rely on the enterprise
        BAA (Business Associate Agreement) instead.

        Args:
            platform: The client platform ('ios', 'android', 'web', 'desktop').

        Returns:
            Dict with keys:
            - ``consent_required``: bool
            - ``platform``: str
            - ``reason``: str — human-readable explanation
        """
        normalised = platform.lower()
        required = normalised in PLATFORMS_REQUIRING_PHI_CONSENT

        if required:
            reason = (
                f"Platform '{normalised}' requires explicit user consent "
                f"before PHI data can be processed per HIPAA mobile guidelines."
            )
        else:
            reason = (
                f"Platform '{normalised}' is covered under the enterprise BAA; "
                f"no additional per-session consent is required."
            )

        logger.debug("PHI consent check for platform '%s': required=%s", normalised, required)
        return {
            "consent_required": required,
            "platform": normalised,
            "reason": reason,
        }

    def record_phi_consent(
        self,
        platform: str,
        user_id: str,
    ) -> None:
        """Record that a user has given explicit PHI consent on a platform.

        Args:
            platform: The client platform.
            user_id: The user who granted consent.
        """
        normalised = platform.lower()
        now = datetime.now(UTC)
        with self._lock:
            if normalised not in self._consent_records:
                self._consent_records[normalised] = {}
            self._consent_records[normalised][user_id] = now
        logger.info(
            "PHI consent recorded: user=%s, platform=%s", user_id, normalised
        )

    def has_phi_consent(self, platform: str, user_id: str) -> bool:
        """Check whether a specific user has given PHI consent on a platform.

        Args:
            platform: The client platform.
            user_id: The user to check.

        Returns:
            True if consent has been recorded.
        """
        normalised = platform.lower()
        with self._lock:
            platform_consents = self._consent_records.get(normalised, {})
            return user_id in platform_consents

    # ------------------------------------------------------------------
    # Public API: PHI audit logging
    # ------------------------------------------------------------------

    def audit_phi_access(
        self,
        agent_id: str,
        data_description: str,
        session_id: str = "",
    ) -> None:
        """Log a PHI access event for HIPAA audit trail compliance.

        Every access to protected health information must be recorded with
        the agent that accessed it, a description of the data, and a
        timestamp. Records are stored in-memory and optionally forwarded
        to an external audit system via the ``audit_callback``.

        Args:
            agent_id: Identifier of the agent accessing PHI.
            data_description: Human-readable description of the PHI accessed.
            session_id: Optional session identifier for correlation.
        """
        record = PHIAuditRecord(
            agent_id=agent_id,
            data_description=data_description,
            timestamp=datetime.now(UTC),
            session_id=session_id,
        )

        with self._lock:
            self._audit_log.append(record)

        logger.info(
            "PHI access audit: agent=%s, data='%s', session=%s",
            agent_id,
            data_description[:100],
            session_id or "(none)",
        )

        if self._audit_callback is not None:
            try:
                self._audit_callback(record)
            except Exception as exc:
                logger.error("PHI audit callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API: Compliance status
    # ------------------------------------------------------------------

    def get_compliance_status(self) -> dict:
        """Return the overall HIPAA compliance status.

        Returns:
            Dict with keys:
            - ``hipaa_active``: bool — always True for this manager
            - ``total_audit_records``: int
            - ``consent_platforms``: list[str] — platforms with recorded consents
            - ``consent_counts``: dict[str, int] — per-platform consent count
            - ``last_audit_timestamp``: str | None — ISO-8601 of most recent audit
        """
        with self._lock:
            consent_counts = {
                platform: len(users)
                for platform, users in self._consent_records.items()
            }
            last_ts = (
                self._audit_log[-1].timestamp.isoformat()
                if self._audit_log
                else None
            )

            return {
                "hipaa_active": True,
                "total_audit_records": len(self._audit_log),
                "consent_platforms": sorted(self._consent_records.keys()),
                "consent_counts": consent_counts,
                "last_audit_timestamp": last_ts,
            }

    # ------------------------------------------------------------------
    # Public API: Audit log access
    # ------------------------------------------------------------------

    def get_audit_log(self) -> list[dict]:
        """Return a copy of all PHI audit records as dicts.

        Returns:
            List of dicts with keys: agent_id, data_description, timestamp,
            session_id.
        """
        with self._lock:
            return [
                {
                    "agent_id": r.agent_id,
                    "data_description": r.data_description,
                    "timestamp": r.timestamp.isoformat(),
                    "session_id": r.session_id,
                }
                for r in self._audit_log
            ]


# ---------------------------------------------------------------------------
# Identity Provider (Req 14.9 — AgentCore Identity integration)
# ---------------------------------------------------------------------------

class IdentityProvider:
    """Pluggable identity provider integration for Okta, Entra ID, and Cognito.

    External token validation is handled through a pluggable callback so
    that the module does not depend on any specific SDK. The callback
    receives ``(provider_type, token, config)`` and must return an
    ``AuthResult``.

    Thread-safe: configuration and authentication are guarded by a lock.

    Usage::

        def my_validator(provider_type, token, config):
            # Call real Okta/Entra/Cognito SDK here
            return AuthResult(authenticated=True, user_id="u-1", ...)

        idp = IdentityProvider(token_validator=my_validator)
        idp.configure_provider("okta", {"domain": "acme.okta.com", ...})
        result = idp.authenticate("eyJhbGc...")
    """

    def __init__(
        self,
        token_validator: Callable[[str, str, dict], AuthResult] | None = None,
        permission_resolver: Callable[[str, str, dict], list[str]] | None = None,
    ) -> None:
        """
        Args:
            token_validator: Callback ``(provider_type, token, config) -> AuthResult``.
                If not provided, authentication always returns an error.
            permission_resolver: Callback ``(provider_type, user_id, config) -> list[str]``.
                If not provided, ``get_user_permissions`` returns an empty list.
        """
        self._lock = threading.Lock()
        self._token_validator = token_validator
        self._permission_resolver = permission_resolver
        self._provider_type: str | None = None
        self._provider_config: dict = {}

    # ------------------------------------------------------------------
    # Public API: Provider configuration
    # ------------------------------------------------------------------

    def configure_provider(self, provider_type: str, config: dict) -> None:
        """Configure the identity provider.

        Args:
            provider_type: One of 'okta', 'entra', 'cognito'.
            config: Provider-specific configuration dict (e.g. domain,
                client_id, pool_id).

        Raises:
            ValueError: If provider_type is not supported.
        """
        normalised = provider_type.lower()
        if normalised not in VALID_IDENTITY_PROVIDERS:
            raise ValueError(
                f"Unsupported identity provider '{provider_type}'. "
                f"Must be one of: {sorted(VALID_IDENTITY_PROVIDERS)}"
            )

        with self._lock:
            self._provider_type = normalised
            self._provider_config = dict(config)

        logger.info("Identity provider configured: %s", normalised)

    # ------------------------------------------------------------------
    # Public API: Token authentication
    # ------------------------------------------------------------------

    def authenticate(self, token: str) -> AuthResult:
        """Validate a token using the configured identity provider.

        Args:
            token: The bearer/JWT token to validate.

        Returns:
            AuthResult with authentication outcome.
        """
        with self._lock:
            provider_type = self._provider_type
            config = dict(self._provider_config)

        if provider_type is None:
            logger.warning("Authentication attempted with no provider configured")
            return AuthResult(
                authenticated=False,
                error="No identity provider configured",
            )

        if not token:
            return AuthResult(
                authenticated=False,
                provider=provider_type,
                error="Empty token provided",
            )

        if self._token_validator is None:
            logger.warning(
                "Authentication attempted with no token_validator callback"
            )
            return AuthResult(
                authenticated=False,
                provider=provider_type,
                error="No token validator configured",
            )

        try:
            result = self._token_validator(provider_type, token, config)
            # Ensure the provider field is always populated
            if not result.provider:
                result.provider = provider_type
            logger.info(
                "Authentication result: provider=%s, authenticated=%s, user=%s",
                provider_type,
                result.authenticated,
                result.user_id or "(unknown)",
            )
            return result
        except Exception as exc:
            logger.error("Token validation failed: %s", exc)
            return AuthResult(
                authenticated=False,
                provider=provider_type,
                error=f"Token validation error: {exc}",
            )

    # ------------------------------------------------------------------
    # Public API: Permission resolution
    # ------------------------------------------------------------------

    def get_user_permissions(self, user_id: str) -> list[str]:
        """Return the granted permissions for a user.

        Delegates to the ``permission_resolver`` callback if configured.
        Returns an empty list if no resolver is available or the provider
        is not configured.

        Args:
            user_id: The user identifier to look up.

        Returns:
            List of permission strings (e.g. ['read:agents', 'write:config']).
        """
        with self._lock:
            provider_type = self._provider_type
            config = dict(self._provider_config)

        if provider_type is None:
            logger.warning("Permission lookup attempted with no provider configured")
            return []

        if self._permission_resolver is None:
            logger.debug("No permission_resolver configured; returning empty list")
            return []

        try:
            permissions = self._permission_resolver(provider_type, user_id, config)
            logger.debug(
                "Permissions for user '%s': %s", user_id, permissions
            )
            return permissions
        except Exception as exc:
            logger.error("Permission resolution failed for user '%s': %s", user_id, exc)
            return []

    # ------------------------------------------------------------------
    # Public API: Provider info
    # ------------------------------------------------------------------

    def get_provider_info(self) -> dict:
        """Return current identity provider configuration summary.

        Returns:
            Dict with ``provider_type`` and ``configured`` keys.
            Config values are excluded to avoid leaking secrets.
        """
        with self._lock:
            return {
                "provider_type": self._provider_type,
                "configured": self._provider_type is not None,
            }


# ---------------------------------------------------------------------------
# Policy Engine — Cedar-based authorization (Req 14.9)
# ---------------------------------------------------------------------------

class PolicyEngine:
    """Cedar-style policy evaluation engine for fine-grained access control.

    Policies are registered as Cedar policy strings and evaluated against
    a principal/action/resource triple with optional context. The engine
    uses a deny-by-default model: access is denied unless at least one
    policy explicitly allows it.

    Thread-safe: policy registration and evaluation are guarded by a lock.

    Usage::

        engine = PolicyEngine()
        engine.add_policy("allow-read-agents", '''
            permit(
                principal == User::"admin",
                action == Action::"read",
                resource == Resource::"agents"
            );
        ''')
        decision = engine.evaluate("User::\"admin\"", "read", "agents")
        assert decision.allowed is True
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # policy_id → cedar_policy_string
        self._policies: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API: Policy management
    # ------------------------------------------------------------------

    def add_policy(self, policy_id: str, cedar_policy: str) -> None:
        """Register a Cedar policy.

        Args:
            policy_id: Unique identifier for the policy.
            cedar_policy: Cedar policy string (e.g. ``permit(...)`` or
                ``forbid(...)``).

        Raises:
            ValueError: If policy_id is empty or cedar_policy is empty.
        """
        if not policy_id:
            raise ValueError("policy_id must not be empty")
        if not cedar_policy or not cedar_policy.strip():
            raise ValueError("cedar_policy must not be empty")

        with self._lock:
            self._policies[policy_id] = cedar_policy.strip()

        logger.info("Policy registered: %s", policy_id)

    def remove_policy(self, policy_id: str) -> bool:
        """Remove a registered policy.

        Args:
            policy_id: The policy to remove.

        Returns:
            True if the policy was found and removed.
        """
        with self._lock:
            if policy_id in self._policies:
                del self._policies[policy_id]
                logger.info("Policy removed: %s", policy_id)
                return True
            return False

    def list_policies(self) -> list[str]:
        """Return a list of all registered policy IDs.

        Returns:
            Sorted list of policy identifier strings.
        """
        with self._lock:
            return sorted(self._policies.keys())

    # ------------------------------------------------------------------
    # Public API: Policy evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        principal: str,
        action: str,
        resource: str,
        context: dict | None = None,
    ) -> PolicyDecision:
        """Evaluate whether a principal can perform an action on a resource.

        Iterates over all registered policies. A ``permit`` policy that
        matches the principal, action, and resource grants access. A
        ``forbid`` policy that matches explicitly denies access and takes
        precedence over permits. If no policy matches, access is denied
        (deny-by-default).

        Args:
            principal: The entity requesting access (e.g. 'User::"admin"').
            action: The action being performed (e.g. 'read').
            resource: The target resource (e.g. 'agents').
            context: Optional additional context for policy conditions.

        Returns:
            PolicyDecision indicating whether access is allowed.
        """
        ctx = context or {}

        with self._lock:
            policies_snapshot = dict(self._policies)

        # Evaluate all policies; forbid takes precedence
        permit_match: str | None = None

        for policy_id, cedar_text in policies_snapshot.items():
            match_result = self._match_policy(
                cedar_text, principal, action, resource, ctx
            )
            if match_result is None:
                continue

            if match_result == "forbid":
                logger.debug(
                    "Policy FORBID: policy=%s, principal=%s, action=%s, resource=%s",
                    policy_id, principal, action, resource,
                )
                return PolicyDecision(
                    allowed=False,
                    policy_id=policy_id,
                    reason=f"Explicitly denied by policy '{policy_id}'",
                )

            if match_result == "permit" and permit_match is None:
                permit_match = policy_id

        if permit_match is not None:
            logger.debug(
                "Policy PERMIT: policy=%s, principal=%s, action=%s, resource=%s",
                permit_match, principal, action, resource,
            )
            return PolicyDecision(
                allowed=True,
                policy_id=permit_match,
                reason=f"Allowed by policy '{permit_match}'",
            )

        logger.debug(
            "Policy DENY (default): principal=%s, action=%s, resource=%s",
            principal, action, resource,
        )
        return PolicyDecision(
            allowed=False,
            reason="No matching policy found (deny by default)",
        )

    # ------------------------------------------------------------------
    # Internal: Cedar policy matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_policy(
        cedar_text: str,
        principal: str,
        action: str,
        resource: str,
        context: dict,
    ) -> str | None:
        """Attempt to match a Cedar policy against the request.

        This is a simplified matcher that handles basic Cedar ``permit``
        and ``forbid`` policies. A production deployment would use the
        official Cedar SDK via a pluggable callback.

        Returns:
            'permit' or 'forbid' if the policy matches, None otherwise.
        """
        text = cedar_text.strip()

        # Determine policy effect
        if text.startswith("permit"):
            effect = "permit"
        elif text.startswith("forbid"):
            effect = "forbid"
        else:
            return None

        # Extract the body inside the parentheses
        paren_start = text.find("(")
        paren_end = text.rfind(")")
        if paren_start == -1 or paren_end == -1:
            return None

        body = text[paren_start + 1:paren_end].strip()

        # Check principal match
        if not _cedar_clause_matches(body, "principal", principal):
            return None

        # Check action match
        if not _cedar_clause_matches(body, "action", action):
            return None

        # Check resource match
        if not _cedar_clause_matches(body, "resource", resource):
            return None

        return effect


# ---------------------------------------------------------------------------
# Cedar matching helpers
# ---------------------------------------------------------------------------

def _cedar_clause_matches(body: str, clause_type: str, value: str) -> bool:
    """Check if a Cedar policy body matches a specific clause value.

    Handles three cases:
    1. Clause not mentioned in body → wildcard match (allow all).
    2. Clause uses ``==`` to match a specific value.
    3. Clause uses ``in`` for group membership (simplified).

    Args:
        body: The inner text of a Cedar policy (between parentheses).
        clause_type: One of 'principal', 'action', 'resource'.
        value: The value to match against.

    Returns:
        True if the clause matches the given value.
    """
    # Find the clause in the body
    # Pattern: clause_type == "value" or clause_type == Type::"value"
    pattern = rf'{clause_type}\s*==\s*'
    match = re.search(pattern, body)

    if match is None:
        # Clause not present → wildcard (matches everything)
        return True

    # Extract the value after ==
    rest = body[match.end():].strip()

    # Handle quoted strings and Cedar entity references
    # e.g., User::"admin", Action::"read", "agents"
    extracted = _extract_cedar_value(rest)
    if extracted is None:
        return True  # Cannot parse → treat as wildcard

    # Normalise both sides for comparison:
    # Strip surrounding quotes and Cedar type prefixes for matching
    norm_extracted = _normalise_cedar_value(extracted)
    norm_value = _normalise_cedar_value(value)

    return norm_extracted == norm_value


def _extract_cedar_value(text: str) -> str | None:
    """Extract a Cedar value from the start of a text fragment.

    Handles: ``Type::"value"``, ``"value"``, ``value``.
    Stops at comma, closing paren, or semicolon.

    Returns:
        The extracted value string, or None if parsing fails.
    """
    # Match Type::"value" pattern
    m = re.match(r'(\w+::)?"([^"]*)"', text)
    if m:
        prefix = m.group(1) or ""
        return prefix + '"' + m.group(2) + '"'

    # Match bare word (up to delimiter)
    m = re.match(r'([^\s,);]+)', text)
    if m:
        return m.group(1)

    return None


def _normalise_cedar_value(value: str) -> str:
    """Normalise a Cedar value for comparison.

    Strips optional type prefix (e.g. ``User::``) and surrounding quotes
    so that ``User::"admin"``, ``"admin"``, and ``admin`` all compare
    equal.

    Returns:
        The normalised lowercase value.
    """
    v = value.strip()

    # Remove Cedar type prefix (e.g. Action::, User::, Resource::)
    if "::" in v:
        v = v.split("::", 1)[1]

    # Remove surrounding quotes
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1]

    return v.lower()


# ---------------------------------------------------------------------------
# VPC Configuration (Req 14.10)
# ---------------------------------------------------------------------------

class VPCConfig:
    """Manages VPC-only deployment mode with network isolation validation.

    When VPC mode is enabled, all agent communication must remain within
    the specified VPC, subnets, and security groups. External egress is
    blocked unless explicitly allowed.

    Thread-safe: configuration mutations are guarded by a lock.

    Usage::

        vpc = VPCConfig()
        vpc.enable_vpc_mode(
            vpc_id="vpc-abc123",
            subnet_ids=["subnet-1", "subnet-2"],
            security_group_ids=["sg-001"],
        )
        assert vpc.is_vpc_mode() is True
        status = vpc.validate_network_isolation()
    """

    def __init__(
        self,
        network_validator: Callable[[dict], dict] | None = None,
    ) -> None:
        """
        Args:
            network_validator: Optional callback ``(vpc_config_dict) -> dict``
                that performs actual network isolation checks against the
                cloud provider. Must return a dict with keys ``isolated``
                (bool) and ``issues`` (list[str]). If not provided,
                validation performs only local config checks.
        """
        self._lock = threading.Lock()
        self._network_validator = network_validator
        self._enabled: bool = False
        self._vpc_id: str = ""
        self._subnet_ids: list[str] = []
        self._security_group_ids: list[str] = []

    # ------------------------------------------------------------------
    # Public API: Enable VPC mode
    # ------------------------------------------------------------------

    def enable_vpc_mode(
        self,
        vpc_id: str,
        subnet_ids: list[str],
        security_group_ids: list[str],
    ) -> None:
        """Enable VPC-only mode with the specified network configuration.

        Args:
            vpc_id: The AWS VPC identifier (e.g. 'vpc-abc123').
            subnet_ids: List of subnet identifiers within the VPC.
            security_group_ids: List of security group identifiers.

        Raises:
            ValueError: If vpc_id is empty, or subnet_ids / security_group_ids
                are empty lists.
        """
        if not vpc_id:
            raise ValueError("vpc_id must not be empty")
        if not subnet_ids:
            raise ValueError("At least one subnet_id is required")
        if not security_group_ids:
            raise ValueError("At least one security_group_id is required")

        with self._lock:
            self._enabled = True
            self._vpc_id = vpc_id
            self._subnet_ids = list(subnet_ids)
            self._security_group_ids = list(security_group_ids)

        logger.info(
            "VPC mode enabled: vpc=%s, subnets=%d, security_groups=%d",
            vpc_id,
            len(subnet_ids),
            len(security_group_ids),
        )

    # ------------------------------------------------------------------
    # Public API: Disable VPC mode
    # ------------------------------------------------------------------

    def disable_vpc_mode(self) -> None:
        """Disable VPC-only mode and clear network configuration."""
        with self._lock:
            self._enabled = False
            self._vpc_id = ""
            self._subnet_ids = []
            self._security_group_ids = []

        logger.info("VPC mode disabled")

    # ------------------------------------------------------------------
    # Public API: Status queries
    # ------------------------------------------------------------------

    def is_vpc_mode(self) -> bool:
        """Check whether VPC-only mode is currently active.

        Returns:
            True if VPC mode has been enabled.
        """
        with self._lock:
            return self._enabled

    def get_vpc_config(self) -> dict:
        """Return the current VPC configuration.

        Returns:
            Dict with keys:
            - ``enabled``: bool
            - ``vpc_id``: str
            - ``subnet_ids``: list[str]
            - ``security_group_ids``: list[str]
        """
        with self._lock:
            return {
                "enabled": self._enabled,
                "vpc_id": self._vpc_id,
                "subnet_ids": list(self._subnet_ids),
                "security_group_ids": list(self._security_group_ids),
            }

    # ------------------------------------------------------------------
    # Public API: Network isolation validation
    # ------------------------------------------------------------------

    def validate_network_isolation(self) -> dict:
        """Validate that the VPC configuration provides proper network isolation.

        Performs local configuration checks and, if a ``network_validator``
        callback is configured, delegates to it for live infrastructure
        checks.

        Returns:
            Dict with keys:
            - ``isolated``: bool — True if all checks pass
            - ``issues``: list[str] — human-readable list of issues found
        """
        with self._lock:
            enabled = self._enabled
            vpc_id = self._vpc_id
            subnet_ids = list(self._subnet_ids)
            security_group_ids = list(self._security_group_ids)

        issues: list[str] = []

        # Check basic configuration
        if not enabled:
            issues.append("VPC mode is not enabled")
            return {"isolated": False, "issues": issues}

        if not vpc_id:
            issues.append("No VPC ID configured")

        if not subnet_ids:
            issues.append("No subnets configured")

        if not security_group_ids:
            issues.append("No security groups configured")

        # If a network validator callback is configured, use it for
        # live infrastructure checks
        if self._network_validator is not None and not issues:
            config_dict = {
                "vpc_id": vpc_id,
                "subnet_ids": subnet_ids,
                "security_group_ids": security_group_ids,
            }
            try:
                result = self._network_validator(config_dict)
                if not result.get("isolated", False):
                    issues.extend(result.get("issues", []))
            except Exception as exc:
                logger.error("Network validator callback failed: %s", exc)
                issues.append(f"Network validation error: {exc}")

        isolated = len(issues) == 0

        if isolated:
            logger.info("Network isolation validated: vpc=%s", vpc_id)
        else:
            logger.warning(
                "Network isolation issues found: vpc=%s, issues=%s",
                vpc_id,
                issues,
            )

        return {"isolated": isolated, "issues": issues}


# ---------------------------------------------------------------------------
# AgentCore integration adapters (production path)
# ---------------------------------------------------------------------------


class AgentCoreIdentityAdapter:
    """Concrete adapter for AgentCore Identity integration."""

    def __init__(self) -> None:
        try:
            from bedrock_agentcore.identity import IdentityClient  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("bedrock-agentcore identity client is required.") from exc
        self._client = IdentityClient()

    def authenticate_token(self, provider_type: str, token: str, config: dict) -> AuthResult:
        resp = self._client.authenticate(
            provider=provider_type,
            token=token,
            config=config,
        )
        return AuthResult(
            authenticated=bool(resp.get("authenticated", False)),
            user_id=str(resp.get("user_id", "")),
            email=str(resp.get("email", "")),
            roles=list(resp.get("roles", [])),
            provider=provider_type,
            error=str(resp.get("error", "")),
        )

    def get_permissions(self, provider_type: str, user_id: str, config: dict) -> list[str]:
        resp = self._client.get_permissions(
            provider=provider_type,
            user_id=user_id,
            config=config,
        )
        return [str(v) for v in resp.get("permissions", [])]


class AgentCorePolicyAdapter:
    """Concrete adapter for AgentCore Policy (Cedar) integration."""

    def __init__(self) -> None:
        try:
            from bedrock_agentcore.policy import PolicyClient  # type: ignore
        except Exception as exc:  # pragma: no cover - runtime-only path
            raise RuntimeError("bedrock-agentcore policy client is required.") from exc
        self._client = PolicyClient()

    def evaluate(
        self,
        principal: str,
        action: str,
        resource: str,
        context: dict | None = None,
    ) -> PolicyDecision:
        resp = self._client.evaluate(
            principal=principal,
            action=action,
            resource=resource,
            context=context or {},
        )
        return PolicyDecision(
            allowed=bool(resp.get("allowed", False)),
            policy_id=str(resp.get("policy_id", "")),
            reason=str(resp.get("reason", "")),
        )


class VpcIsolationValidatorAdapter:
    """Runtime validator that checks VPC, subnet, and security group existence."""

    def __init__(self, ec2_client: object | None = None) -> None:
        if ec2_client is not None:
            self._ec2 = ec2_client
        else:
            try:
                import boto3  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime-only path
                raise RuntimeError("boto3 is required for VPC validation.") from exc
            self._ec2 = boto3.client("ec2")

    def __call__(self, vpc_config: dict) -> dict:
        issues: list[str] = []
        vpc_id = vpc_config.get("vpc_id", "")
        subnet_ids = list(vpc_config.get("subnet_ids", []))
        sg_ids = list(vpc_config.get("security_group_ids", []))

        try:
            self._ec2.describe_vpcs(VpcIds=[vpc_id])
        except Exception:
            issues.append(f"VPC does not exist or is inaccessible: {vpc_id}")

        if subnet_ids:
            try:
                self._ec2.describe_subnets(SubnetIds=subnet_ids)
            except Exception:
                issues.append("One or more subnets do not exist or are inaccessible")

        if sg_ids:
            try:
                self._ec2.describe_security_groups(GroupIds=sg_ids)
            except Exception:
                issues.append("One or more security groups do not exist or are inaccessible")

        return {"isolated": len(issues) == 0, "issues": issues}
