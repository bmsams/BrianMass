"""Unit tests for EnterprisePolicyManager.

Covers: default settings loading, hook policy enforcement, model policy
enforcement, tool policy enforcement, budget policy capping, git sync,
and session validation.

Requirements: 23.1, 23.2, 23.3, 23.4
"""

from __future__ import annotations

from src.config.enterprise import EnterprisePolicyManager
from src.types.core import EnterpriseSettings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(
    allow_managed_hooks_only: bool = False,
    allowed_models: list[str] | None = None,
    max_session_budget_usd: float | None = None,
    max_team_budget_usd: float | None = None,
    blocked_tools: list[str] | None = None,
    git_config_repo_url: str | None = None,
) -> EnterpriseSettings:
    return EnterpriseSettings(
        allow_managed_hooks_only=allow_managed_hooks_only,
        allowed_models=allowed_models or [],
        max_session_budget_usd=max_session_budget_usd,
        max_team_budget_usd=max_team_budget_usd,
        blocked_tools=blocked_tools or [],
        git_config_repo_url=git_config_repo_url,
    )


# ===================================================================
# Default settings loading
# ===================================================================

class TestLoadsDefaultSettings:
    """When no enterprise.json is present, defaults are used."""

    def test_loads_default_settings(self, tmp_path):
        """EnterprisePolicyManager should load safe defaults when
        no enterprise.json file exists in the config directory."""
        manager = EnterprisePolicyManager(config_dir=str(tmp_path))
        settings = manager.get_settings()

        assert settings.allow_managed_hooks_only is False
        assert settings.allowed_models == []
        assert settings.max_session_budget_usd is None
        assert settings.blocked_tools == []


# ===================================================================
# Hook policy enforcement  (Requirement 23.2)
# ===================================================================

class TestEnforceHooksPolicy:
    """enforce_hooks_policy blocks or allows based on managed-only flag."""

    def test_enforce_hooks_policy_blocks_when_managed_only(self):
        settings = _make_settings(allow_managed_hooks_only=True)
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_hooks_policy(scope="user_wide")
        assert result["allowed"] is False
        assert "managed" in result.get("reason", "").lower()

    def test_enforce_hooks_policy_allows_when_not_managed_only(self):
        settings = _make_settings(allow_managed_hooks_only=False)
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_hooks_policy(scope="user_wide")
        assert result["allowed"] is True

    def test_enforce_hooks_policy_always_allows_enterprise_managed(self):
        """enterprise_managed scope hooks should always be allowed,
        even when allow_managed_hooks_only is True."""
        settings = _make_settings(allow_managed_hooks_only=True)
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_hooks_policy(scope="enterprise_managed")
        assert result["allowed"] is True


# ===================================================================
# Model policy enforcement
# ===================================================================

class TestEnforceModelPolicy:
    """enforce_model_policy validates model against allowlist."""

    def test_enforce_model_policy_allowed_model(self):
        settings = _make_settings(allowed_models=["claude-sonnet-4-5", "claude-haiku-4-5"])
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_model_policy("claude-sonnet-4-5")
        assert result["allowed"] is True

    def test_enforce_model_policy_blocked_model(self):
        settings = _make_settings(allowed_models=["claude-haiku-4-5"])
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_model_policy("claude-opus-4-6")
        assert result["allowed"] is False

    def test_enforce_model_policy_empty_allowlist_allows_all(self):
        settings = _make_settings(allowed_models=[])
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_model_policy("claude-opus-4-6")
        assert result["allowed"] is True


# ===================================================================
# Tool policy enforcement
# ===================================================================

class TestEnforceToolPolicy:
    """enforce_tool_policy blocks blacklisted tools."""

    def test_enforce_tool_policy_blocks_tool(self):
        settings = _make_settings(blocked_tools=["bash", "shell"])
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_tool_policy("bash")
        assert result["allowed"] is False

    def test_enforce_tool_policy_allows_tool(self):
        settings = _make_settings(blocked_tools=["bash"])
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_tool_policy("file_read")
        assert result["allowed"] is True


# ===================================================================
# Budget policy enforcement
# ===================================================================

class TestEnforceBudgetPolicy:
    """enforce_budget_policy caps budgets at enterprise limits."""

    def test_enforce_budget_policy_caps_at_limit(self):
        settings = _make_settings(max_session_budget_usd=50.0)
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_budget_policy(requested_budget_usd=100.0)
        assert result["effective_budget_usd"] <= 50.0

    def test_enforce_budget_policy_no_limit(self):
        settings = _make_settings(max_session_budget_usd=None)
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.enforce_budget_policy(requested_budget_usd=100.0)
        assert result["effective_budget_usd"] == 100.0


# ===================================================================
# Git-based policy sync  (Requirement 23.3)
# ===================================================================

class TestSyncFromGit:
    """sync_from_git triggers the provided callback with the repo URL."""

    def test_sync_from_git_calls_callback(self):
        settings = _make_settings(
            git_config_repo_url="https://git.example.com/policies.git"
        )
        manager = EnterprisePolicyManager(settings=settings)

        synced_urls: list[str] = []

        def _sync_cb(url: str) -> None:
            synced_urls.append(url)

        manager.sync_from_git(callback=_sync_cb)
        assert synced_urls == ["https://git.example.com/policies.git"]


# ===================================================================
# Session validation
# ===================================================================

class TestValidateSession:
    """validate_session checks that a session config satisfies all policies."""

    def test_validate_session_passes_valid(self):
        settings = _make_settings(
            allowed_models=["claude-sonnet-4-5"],
            max_session_budget_usd=100.0,
            blocked_tools=[],
        )
        manager = EnterprisePolicyManager(settings=settings)

        result = manager.validate_session(
            model="claude-sonnet-4-5",
            budget_usd=50.0,
            tools=["file_read", "file_write"],
        )
        assert result["valid"] is True
        assert result.get("violations", []) == []
