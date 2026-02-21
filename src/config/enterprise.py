"""Enterprise Managed Settings — organization-level policy enforcement.

Loads enterprise settings from ~/.brainmass/enterprise.json and enforces
organization-level policies that override all user/project/plugin settings.
Supports self-serve deployment, Git-based policy distribution, and pluggable
callbacks for external integrations (audit logging, git sync).

Policy precedence: enterprise > local > project > user.

All external calls (git sync, audit log delivery) are injected as pluggable
callbacks so the module remains fully testable without external dependencies.

Requirements: 23.1, 23.2, 23.3, 23.4
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.types.core import EnterpriseSettings, HookDefinition, SessionState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ENTERPRISE_CONFIG_DIR = Path.home() / ".brainmass"

ENTERPRISE_CONFIG_FILENAME = "enterprise.json"

# Hook scopes that are blocked when allow_managed_hooks_only is True
_NON_MANAGED_SCOPES = frozenset({
    "user_wide",
    "project_shared",
    "project_local",
    "skill_frontmatter",
    "subagent_frontmatter",
    "plugin",
})

# The only scope allowed when managed-only mode is active
_MANAGED_SCOPE = "enterprise_managed"


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class PolicyValidationResult:
    """Result of an enterprise policy validation check."""

    valid: bool
    violations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Git sync result
# ---------------------------------------------------------------------------

@dataclass
class GitSyncResult:
    """Result of a Git-based policy sync operation."""

    success: bool
    message: str = ""
    settings: EnterpriseSettings | None = None


# ---------------------------------------------------------------------------
# Pluggable callback type aliases
# ---------------------------------------------------------------------------

# Syncs enterprise policies from a git repository; returns GitSyncResult
GitSyncCallback = Callable[
    [str],  # repo_url
    GitSyncResult,
]

# Delivers audit log events to an external endpoint
AuditLogCallback = Callable[
    [str, dict],  # (endpoint_url, event_dict)
    None,
]


# ---------------------------------------------------------------------------
# Default no-op callbacks
# ---------------------------------------------------------------------------

def _default_git_sync(repo_url: str) -> GitSyncResult:
    """Default no-op git sync callback.

    --- Production integration point ---
    In production, this would:
        import subprocess
        subprocess.run(["git", "clone", "--depth=1", repo_url, target_dir])
        settings = _load_settings_from_dir(target_dir)
        return GitSyncResult(success=True, settings=settings)
    """
    logger.debug("Git sync requested for %s (no-op callback)", repo_url)
    return GitSyncResult(
        success=True,
        message=f"No-op sync from {repo_url}",
    )


def _default_audit_log(endpoint: str, event: dict) -> None:
    """Default no-op audit log callback.

    --- Production integration point ---
    In production, this would POST the event to the audit log endpoint:
        import requests
        requests.post(endpoint, json=event, timeout=5)
    """
    logger.debug("Audit log event sent to %s (no-op callback)", endpoint)


# ---------------------------------------------------------------------------
# Settings loading helpers
# ---------------------------------------------------------------------------

def _load_enterprise_json(path: Path) -> dict[str, Any]:
    """Load enterprise.json, returning an empty dict on failure."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _parse_enterprise_settings(raw: dict[str, Any]) -> EnterpriseSettings:
    """Parse a raw dict into an EnterpriseSettings dataclass."""
    return EnterpriseSettings(
        allow_managed_hooks_only=bool(raw.get("allow_managed_hooks_only", False)),
        allowed_models=list(raw.get("allowed_models", [])),
        max_session_budget_usd=raw.get("max_session_budget_usd"),
        max_team_budget_usd=raw.get("max_team_budget_usd"),
        required_skills=list(raw.get("required_skills", [])),
        blocked_tools=list(raw.get("blocked_tools", [])),
        audit_log_endpoint=raw.get("audit_log_endpoint"),
        git_config_repo_url=raw.get("git_config_repo_url"),
    )


# ---------------------------------------------------------------------------
# EnterprisePolicyManager
# ---------------------------------------------------------------------------

class EnterprisePolicyManager:
    """Enforces organization-level enterprise policies.

    Loads enterprise settings from ~/.brainmass/enterprise.json and provides
    methods to enforce policies across hooks, models, tools, and budgets.
    When ``allow_managed_hooks_only`` is True, all non-enterprise hooks are
    stripped, enforcing that only managed hooks run (Req 23.2).

    Policy precedence: enterprise > local > project > user (Req 23.1).

    Thread-safe: all mutations are guarded by a lock so that concurrent
    agents can safely call enforcement methods.

    Usage::

        manager = EnterprisePolicyManager()
        manager.load()

        # Enforce hooks policy
        result = manager.enforce_hooks_policy(scope="project_local")

        # Validate a model selection
        result = manager.enforce_model_policy("claude-sonnet-4-5-20250929")

        # Check tool access
        result = manager.enforce_tool_policy("Bash")

        # Cap budgets
        result = manager.enforce_budget_policy(requested_budget_usd=5.0)

        # Sync from git config repo
        manager.sync_from_git(callback=my_sync_fn)

    Parameters
    ----------
    config_dir:
        Path to the directory containing enterprise.json (defaults to
        ``~/.brainmass``).
    config_path:
        Explicit path to the enterprise.json file. If provided, overrides
        *config_dir*. Kept for backward compatibility.
    settings:
        Pre-built ``EnterpriseSettings`` to use directly (skips file loading).
    git_sync_callback:
        Pluggable callback for Git-based policy distribution (Req 23.3).
    audit_log_callback:
        Pluggable callback for delivering audit log events.
    """

    def __init__(
        self,
        config_dir: str | Path | None = None,
        config_path: Path | str | None = None,
        settings: EnterpriseSettings | None = None,
        git_sync_callback: GitSyncCallback | None = None,
        audit_log_callback: AuditLogCallback | None = None,
    ) -> None:
        self._lock = threading.Lock()

        # Resolve config file path from config_dir or config_path
        if config_path is not None:
            self._config_path = Path(config_path)
        elif config_dir is not None:
            self._config_path = Path(config_dir) / ENTERPRISE_CONFIG_FILENAME
        else:
            self._config_path = DEFAULT_ENTERPRISE_CONFIG_DIR / ENTERPRISE_CONFIG_FILENAME

        self._git_sync_cb = git_sync_callback or _default_git_sync
        self._audit_log_cb = audit_log_callback or _default_audit_log

        # If settings provided directly, use them; otherwise load from disk
        if settings is not None:
            self._settings = settings
            self._loaded = True
        else:
            self._settings = self._load_from_disk()
            self._loaded = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return True if enterprise settings have been loaded."""
        with self._lock:
            return self._loaded

    # ------------------------------------------------------------------
    # Settings access
    # ------------------------------------------------------------------

    def get_settings(self) -> EnterpriseSettings:
        """Return the current enterprise settings.

        Returns
        -------
        The active ``EnterpriseSettings`` instance.
        """
        with self._lock:
            return self._settings

    @property
    def settings(self) -> EnterpriseSettings:
        """Return the current enterprise settings (property alias)."""
        return self.get_settings()

    # ------------------------------------------------------------------
    # Loading (Req 23.1)
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> EnterpriseSettings:
        """Load enterprise settings from the config file on disk.

        Returns safe defaults if the file is missing or invalid.
        """
        raw = _load_enterprise_json(self._config_path)
        settings = _parse_enterprise_settings(raw)
        self._log_settings_info(settings)
        logger.debug("Enterprise settings loaded from %s", self._config_path)
        return settings

    def load(self, raw: dict[str, Any] | None = None) -> EnterpriseSettings:
        """Load enterprise settings from disk or a provided dict.

        When *raw* is provided, it is used directly instead of reading
        from the config file. This is useful for testing and for applying
        settings received via ``sync_from_git``.

        Returns the loaded ``EnterpriseSettings``.
        """
        if raw is None:
            raw = _load_enterprise_json(self._config_path)

        settings = _parse_enterprise_settings(raw)

        with self._lock:
            self._settings = settings
            self._loaded = True

        self._log_settings_info(settings)
        logger.debug("Enterprise settings loaded from %s", self._config_path)
        return settings

    @staticmethod
    def _log_settings_info(settings: EnterpriseSettings) -> None:
        """Log key settings for operational visibility."""
        if settings.allow_managed_hooks_only:
            logger.info(
                "Enterprise managed-only hooks mode ENABLED — "
                "all non-managed hooks will be blocked."
            )
        if settings.allowed_models:
            logger.info(
                "Enterprise model allowlist active: %s",
                settings.allowed_models,
            )
        if settings.blocked_tools:
            logger.info(
                "Enterprise blocked tools: %s",
                settings.blocked_tools,
            )
        if settings.max_session_budget_usd is not None:
            logger.info(
                "Enterprise session budget cap: $%.2f",
                settings.max_session_budget_usd,
            )

    # ------------------------------------------------------------------
    # Hook enforcement (Req 23.2)
    # ------------------------------------------------------------------

    def enforce_hooks_policy(
        self,
        hooks: dict[str, list[HookDefinition]] | None = None,
        scope: str = "project_local",
    ) -> dict:
        """Evaluate whether hooks from a given scope are allowed.

        When ``allow_managed_hooks_only`` is True, hooks from any scope
        other than ``enterprise_managed`` are blocked entirely (Req 23.2).

        Parameters
        ----------
        hooks:
            Optional mapping of event name to list of ``HookDefinition``
            objects. When provided and allowed, the hooks are returned
            in the ``hooks`` key. When not provided, only the ``allowed``
            decision is returned.
        scope:
            The scope of the hooks being evaluated (e.g. 'project_local',
            'user_wide', 'enterprise_managed').

        Returns
        -------
        Dict with keys:
        - ``allowed``: bool — whether hooks from this scope are permitted
        - ``reason``: str — explanation of the decision
        - ``hooks``: dict | None — the original hooks if allowed, else empty dict
        """
        with self._lock:
            managed_only = self._settings.allow_managed_hooks_only

        if not managed_only:
            return {
                "allowed": True,
                "reason": "Managed-only mode is not active; all scopes are allowed.",
                "hooks": hooks if hooks is not None else {},
            }

        # When managed-only is active, only enterprise_managed scope is allowed
        if scope == _MANAGED_SCOPE:
            return {
                "allowed": True,
                "reason": "Scope 'enterprise_managed' is always allowed.",
                "hooks": hooks if hooks is not None else {},
            }

        if scope in _NON_MANAGED_SCOPES:
            logger.debug(
                "Hooks blocked: scope '%s' is not allowed in managed-only mode.",
                scope,
            )
            return {
                "allowed": False,
                "reason": (
                    f"Scope '{scope}' is blocked — only managed hooks are "
                    f"allowed when allow_managed_hooks_only is enabled."
                ),
                "hooks": {},
            }

        # Unknown scope — block to be safe
        logger.warning(
            "Hooks blocked: unknown scope '%s' in managed-only mode.",
            scope,
        )
        return {
            "allowed": False,
            "reason": (
                f"Unknown scope '{scope}' — blocked in managed-only mode."
            ),
            "hooks": {},
        }

    # ------------------------------------------------------------------
    # Model enforcement (Req 23.1)
    # ------------------------------------------------------------------

    def enforce_model_policy(self, model_id: str) -> dict:
        """Check whether a model is permitted by the enterprise allowlist.

        When ``allowed_models`` is empty, all models are permitted.

        Parameters
        ----------
        model_id:
            The model identifier to validate (e.g. 'claude-sonnet-4-5').

        Returns
        -------
        Dict with keys:
        - ``allowed``: bool — whether the model is permitted
        - ``reason``: str — explanation of the decision
        """
        with self._lock:
            allowed = list(self._settings.allowed_models)

        # Empty allowlist means all models are permitted
        if not allowed:
            return {
                "allowed": True,
                "reason": "No model allowlist configured; all models are permitted.",
            }

        if model_id in allowed:
            logger.debug("Model '%s' is on the enterprise allowlist.", model_id)
            return {
                "allowed": True,
                "reason": f"Model '{model_id}' is on the enterprise allowlist.",
            }

        logger.warning(
            "Model '%s' is NOT on the enterprise allowlist: %s",
            model_id,
            allowed,
        )
        return {
            "allowed": False,
            "reason": (
                f"Model '{model_id}' is not in the enterprise allowed models "
                f"list. Allowed: {allowed}"
            ),
        }

    # ------------------------------------------------------------------
    # Tool enforcement (Req 23.1)
    # ------------------------------------------------------------------

    def enforce_tool_policy(self, tool_name: str) -> dict:
        """Check whether a tool is blocked by enterprise policy.

        Parameters
        ----------
        tool_name:
            The tool name to validate (e.g. 'Bash', 'Write').

        Returns
        -------
        Dict with keys:
        - ``allowed``: bool — whether the tool is permitted
        - ``reason``: str — explanation of the decision
        """
        with self._lock:
            blocked = list(self._settings.blocked_tools)

        if not blocked:
            return {
                "allowed": True,
                "reason": "No tools are blocked by enterprise policy.",
            }

        if tool_name in blocked:
            logger.warning(
                "Tool '%s' is blocked by enterprise policy.", tool_name
            )
            return {
                "allowed": False,
                "reason": (
                    f"Tool '{tool_name}' is blocked by enterprise policy. "
                    f"Blocked tools: {blocked}"
                ),
            }

        return {
            "allowed": True,
            "reason": f"Tool '{tool_name}' is not on the enterprise block list.",
        }

    # ------------------------------------------------------------------
    # Budget enforcement (Req 23.1)
    # ------------------------------------------------------------------

    def enforce_budget_policy(
        self,
        budget_usd: float | None = None,
        requested_budget_usd: float | None = None,
    ) -> dict:
        """Cap a budget at the enterprise maximum.

        If the enterprise has a ``max_session_budget_usd`` configured, the
        returned effective budget will not exceed that cap. If no cap is
        configured, the original budget is returned unchanged.

        Parameters
        ----------
        budget_usd:
            The requested session budget in USD (positional-friendly).
        requested_budget_usd:
            Alias for *budget_usd* (keyword-friendly). If both are
            provided, *requested_budget_usd* takes precedence.

        Returns
        -------
        Dict with keys:
        - ``effective_budget_usd``: float — the capped budget
        - ``capped``: bool — whether the budget was reduced
        - ``reason``: str — explanation of the decision
        """
        amount = requested_budget_usd if requested_budget_usd is not None else budget_usd
        if amount is None:
            amount = 0.0

        with self._lock:
            cap = self._settings.max_session_budget_usd

        if cap is None:
            return {
                "effective_budget_usd": amount,
                "capped": False,
                "reason": "No enterprise session budget cap configured.",
            }

        if amount > cap:
            logger.info(
                "Budget $%.2f capped to enterprise maximum $%.2f.",
                amount,
                cap,
            )
            return {
                "effective_budget_usd": cap,
                "capped": True,
                "reason": (
                    f"Budget ${amount:.2f} capped to enterprise maximum ${cap:.2f}."
                ),
            }

        return {
            "effective_budget_usd": amount,
            "capped": False,
            "reason": f"Budget ${amount:.2f} is within enterprise limit ${cap:.2f}.",
        }

    def enforce_team_budget_policy(
        self,
        budget_usd: float | None = None,
        requested_budget_usd: float | None = None,
    ) -> dict:
        """Cap a team budget at the enterprise maximum.

        Parameters
        ----------
        budget_usd:
            The requested team budget in USD.
        requested_budget_usd:
            Alias for *budget_usd*. If both are provided,
            *requested_budget_usd* takes precedence.

        Returns
        -------
        Dict with keys:
        - ``effective_budget_usd``: float — the capped budget
        - ``capped``: bool — whether the budget was reduced
        - ``reason``: str — explanation of the decision
        """
        amount = requested_budget_usd if requested_budget_usd is not None else budget_usd
        if amount is None:
            amount = 0.0

        with self._lock:
            cap = self._settings.max_team_budget_usd

        if cap is None:
            return {
                "effective_budget_usd": amount,
                "capped": False,
                "reason": "No enterprise team budget cap configured.",
            }

        if amount > cap:
            logger.info(
                "Team budget $%.2f capped to enterprise maximum $%.2f.",
                amount,
                cap,
            )
            return {
                "effective_budget_usd": cap,
                "capped": True,
                "reason": (
                    f"Team budget ${amount:.2f} capped to enterprise maximum ${cap:.2f}."
                ),
            }

        return {
            "effective_budget_usd": amount,
            "capped": False,
            "reason": f"Team budget ${amount:.2f} is within enterprise limit ${cap:.2f}.",
        }

    # ------------------------------------------------------------------
    # Git-based policy distribution (Req 23.3)
    # ------------------------------------------------------------------

    def sync_from_git(
        self,
        repo_url: str | None = None,
        callback: Callable[[str], None] | None = None,
    ) -> GitSyncResult:
        """Sync enterprise policies from a Git config repository.

        Pulls the latest policy definitions from the configured (or
        provided) Git repository URL and reloads settings. Uses a
        pluggable callback for the actual Git interaction.

        --- Production integration point ---
        In production, the ``git_sync_callback`` would:
        1. Clone/pull the repo to a temp directory
        2. Read the enterprise.json from the repo
        3. Return a GitSyncResult with the parsed settings

        Parameters
        ----------
        repo_url:
            The Git repository URL. If not provided, falls back to the
            ``git_config_repo_url`` from current settings.
        callback:
            Optional per-call callback ``(url) -> None`` invoked with the
            resolved URL. Useful for tests to verify the URL was passed.
            This is separate from the constructor-level ``git_sync_callback``.

        Returns
        -------
        A ``GitSyncResult`` indicating success/failure.
        """
        with self._lock:
            url = repo_url or self._settings.git_config_repo_url

        if not url:
            logger.warning("No git config repo URL configured for sync.")
            return GitSyncResult(
                success=False,
                message="No git_config_repo_url configured.",
            )

        logger.info("Syncing enterprise policies from git: %s", url)

        # Invoke the per-call callback if provided
        if callback is not None:
            try:
                callback(url)
            except Exception as exc:
                logger.error("Git sync per-call callback failed: %s", exc)

        # Invoke the constructor-level git sync callback
        try:
            result = self._git_sync_cb(url)
        except Exception as exc:
            logger.error("Git sync failed: %s", exc)
            return GitSyncResult(
                success=False,
                message=f"Git sync error: {exc}",
            )

        # If sync returned new settings, reload them
        if result.success and result.settings is not None:
            with self._lock:
                self._settings = result.settings
                self._loaded = True
            logger.info("Enterprise settings updated from git sync.")

        return result

    # ------------------------------------------------------------------
    # Session validation (Req 23.1)
    # ------------------------------------------------------------------

    def validate_session(
        self,
        session_state: SessionState | None = None,
        *,
        model: str | None = None,
        budget_usd: float | None = None,
        tools: list[str] | None = None,
    ) -> dict:
        """Validate a session against all enterprise policies.

        Can be called with a full ``SessionState`` object, or with
        individual keyword arguments for simpler validation scenarios.

        Parameters
        ----------
        session_state:
            The complete session state to validate. If provided, the
            keyword arguments below are ignored.
        model:
            The model ID to validate against the allowlist.
        budget_usd:
            The session budget to validate against the budget cap.
        tools:
            List of tool names to check against the blocked tools list.

        Returns
        -------
        Dict with keys:
        - ``valid``: bool — True if no violations found
        - ``violations``: list[str] — list of violation descriptions
        """
        violations: list[str] = []

        with self._lock:
            s = self._settings

        if session_state is not None:
            # Full session state validation
            violations.extend(self._validate_from_session_state(s, session_state))
        else:
            # Keyword-based validation
            if model is not None:
                model_result = self.enforce_model_policy(model)
                if not model_result["allowed"]:
                    violations.append(model_result["reason"])

            if budget_usd is not None:
                budget_result = self.enforce_budget_policy(budget_usd=budget_usd)
                if budget_result["capped"]:
                    violations.append(budget_result["reason"])

            if tools is not None:
                for tool_name in tools:
                    tool_result = self.enforce_tool_policy(tool_name)
                    if not tool_result["allowed"]:
                        violations.append(tool_result["reason"])

        if violations:
            logger.warning(
                "Session validation found %d violation(s): %s",
                len(violations),
                violations,
            )
        else:
            logger.debug("Session validation passed — no enterprise policy violations.")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
        }

    @staticmethod
    def _validate_from_session_state(
        settings: EnterpriseSettings,
        session_state: SessionState,
    ) -> list[str]:
        """Validate a full SessionState against enterprise policies.

        Returns a list of violation strings.
        """
        violations: list[str] = []

        # Check required skills
        if settings.required_skills:
            registered_scopes = set(session_state.hook_registrations.keys())
            for skill in settings.required_skills:
                if skill not in registered_scopes:
                    violations.append(
                        f"Required skill '{skill}' is not loaded in the session."
                    )

        # Check blocked tools
        if settings.blocked_tools:
            for tool_name, allowed in session_state.tool_permissions.items():
                if allowed and tool_name in settings.blocked_tools:
                    violations.append(
                        f"Blocked tool '{tool_name}' is permitted in session."
                    )

        # Check budget limits
        if settings.max_session_budget_usd is not None:
            current_cost = session_state.cost_tracking.get("current_cost_usd", 0.0)
            if current_cost > settings.max_session_budget_usd:
                violations.append(
                    f"Session cost ${current_cost:.2f} exceeds enterprise "
                    f"limit ${settings.max_session_budget_usd:.2f}."
                )

        return violations

    # ------------------------------------------------------------------
    # Required skills (Req 23.1)
    # ------------------------------------------------------------------

    def get_required_skills(self) -> list[str]:
        """Return the list of enterprise-required skills.

        These skills must be loaded for every session regardless of
        project or user configuration.

        Returns
        -------
        List of skill names that are required by enterprise policy.
        """
        with self._lock:
            return list(self._settings.required_skills)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_policy_summary(self) -> dict:
        """Return a summary of the current enterprise policy state.

        Returns
        -------
        Dict with:
        - ``loaded``: bool
        - ``managed_hooks_only``: bool
        - ``allowed_models_count``: int
        - ``blocked_tools_count``: int
        - ``required_skills_count``: int
        - ``session_budget_cap``: float | None
        - ``team_budget_cap``: float | None
        - ``git_config_repo``: str | None
        - ``audit_log_endpoint``: str | None
        """
        with self._lock:
            s = self._settings
            return {
                "loaded": self._loaded,
                "managed_hooks_only": s.allow_managed_hooks_only,
                "allowed_models_count": len(s.allowed_models),
                "blocked_tools_count": len(s.blocked_tools),
                "required_skills_count": len(s.required_skills),
                "session_budget_cap": s.max_session_budget_usd,
                "team_budget_cap": s.max_team_budget_usd,
                "git_config_repo": s.git_config_repo_url,
                "audit_log_endpoint": s.audit_log_endpoint,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_audit_event(self, endpoint: str, event: dict) -> None:
        """Send an audit event via the pluggable callback.

        Failures are logged but never propagated to the caller.
        """
        try:
            self._audit_log_cb(endpoint, event)
        except Exception as exc:
            logger.error("Audit log delivery failed: %s", exc)
