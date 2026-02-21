"""Configuration loader with 3-scope merging and MCP server parsing.

Loads settings from:
  1. User scope:          ~/.brainmass/settings.json
  2. Project scope:       .brainmass/settings.json  (version-controlled)
  3. Project-local scope: .brainmass/settings.local.json  (gitignored)

Precedence: project-local > project > user (higher overrides lower).

Also parses .mcp.json for MCP server configuration with ${ENV_VAR} substitution.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.types.core import HookDefinition, HookHandler, HookHandlerType, McpServerConfig

# ---------------------------------------------------------------------------
# BrainmassConfig — unified configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BrainmassConfig:
    """Unified configuration produced by ConfigLoader."""

    hooks: dict[str, list[HookDefinition]] = field(default_factory=dict)
    """Per-event hook definitions keyed by HookEvent value (e.g. 'PreToolUse')."""

    feature_flags: dict[str, str] = field(default_factory=dict)
    """Feature flags such as BRAINMASS_EXPERIMENTAL_AGENT_TEAMS."""

    env_overrides: dict[str, str] = field(default_factory=dict)
    """Environment overrides such as BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE."""

    mcp_servers: dict[str, McpServerConfig] = field(default_factory=dict)
    """MCP server configurations keyed by server name."""

    raw: dict[str, Any] = field(default_factory=dict)
    """The fully-merged raw settings dict (for forward-compat access)."""


# ---------------------------------------------------------------------------
# Well-known keys
# ---------------------------------------------------------------------------

_FEATURE_FLAG_KEYS = frozenset({
    "BRAINMASS_EXPERIMENTAL_AGENT_TEAMS",
})

_ENV_OVERRIDE_KEYS = frozenset({
    "BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE",
    "ENABLE_TOOL_SEARCH",
    "MCP_TIMEOUT",
})

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    - dict values are merged recursively.
    - list values in *override* replace the base list entirely.
    - scalar values in *override* replace the base value.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file, returning an empty dict if the file is missing or invalid."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _substitute_env_vars(value: str) -> str:
    """Replace ``${ENV_VAR}`` placeholders with their environment values.

    Unset variables are replaced with the empty string.
    """
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)


def _parse_hook_handler(raw: dict[str, Any]) -> HookHandler:
    """Convert a raw handler dict into a ``HookHandler``."""
    handler_type_str = raw.get("type", "command")
    try:
        handler_type = HookHandlerType(handler_type_str)
    except ValueError:
        handler_type = HookHandlerType.COMMAND

    return HookHandler(
        type=handler_type,
        command=raw.get("command"),
        prompt=raw.get("prompt"),
        agent_config=raw.get("agent_config"),
        is_async=bool(raw.get("async", False)),
        timeout=int(raw.get("timeout", 600000)),
    )


def _parse_hook_definition(raw: dict[str, Any]) -> HookDefinition:
    """Convert a raw hook definition dict into a ``HookDefinition``."""
    matcher = raw.get("matcher")
    hooks_raw = raw.get("hooks", [])
    handlers = [_parse_hook_handler(h) for h in hooks_raw if isinstance(h, dict)]
    return HookDefinition(matcher=matcher, hooks=handlers)


def _parse_hooks(settings: dict[str, Any]) -> dict[str, list[HookDefinition]]:
    """Extract hooks from the merged settings dict.

    Expected shape::

        {
          "hooks": {
            "PreToolUse": [ { "matcher": "...", "hooks": [...] }, ... ],
            ...
          }
        }
    """
    hooks_section = settings.get("hooks", {})
    if not isinstance(hooks_section, dict):
        return {}

    result: dict[str, list[HookDefinition]] = {}
    for event_name, definitions in hooks_section.items():
        if not isinstance(definitions, list):
            continue
        parsed = [_parse_hook_definition(d) for d in definitions if isinstance(d, dict)]
        if parsed:
            result[event_name] = parsed
    return result


def _parse_feature_flags(settings: dict[str, Any]) -> dict[str, str]:
    """Extract known feature flags from the merged settings."""
    return {
        k: str(v)
        for k, v in settings.items()
        if k in _FEATURE_FLAG_KEYS
    }


def _parse_env_overrides(settings: dict[str, Any]) -> dict[str, str]:
    """Extract known environment overrides from the merged settings."""
    return {
        k: str(v)
        for k, v in settings.items()
        if k in _ENV_OVERRIDE_KEYS
    }


def _parse_mcp_servers(mcp_data: dict[str, Any]) -> dict[str, McpServerConfig]:
    """Parse .mcp.json content into ``McpServerConfig`` objects.

    Expected shape::

        {
          "mcpServers": {
            "server-name": {
              "command": "...",
              "args": ["..."],
              "env": { "KEY": "${ENV_VAR}" }
            }
          }
        }
    """
    servers_section = mcp_data.get("mcpServers", {})
    if not isinstance(servers_section, dict):
        return {}

    result: dict[str, McpServerConfig] = {}
    for name, cfg in servers_section.items():
        if not isinstance(cfg, dict):
            continue
        command = cfg.get("command", "")
        args = cfg.get("args", [])
        if not isinstance(args, list):
            args = []
        raw_env = cfg.get("env", {})
        if not isinstance(raw_env, dict):
            raw_env = {}
        env = {k: _substitute_env_vars(str(v)) for k, v in raw_env.items()}
        scope = cfg.get("scope", "project")
        result[name] = McpServerConfig(command=command, args=args, env=env, scope=scope)
    return result


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Loads and merges Brainmass configuration from multiple scopes.

    Parameters
    ----------
    project_dir:
        The project root directory (defaults to cwd).
    user_dir:
        The user-level config directory (defaults to ``~/.brainmass``).
    """

    def __init__(
        self,
        project_dir: Path | str | None = None,
        user_dir: Path | str | None = None,
    ) -> None:
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.user_dir = Path(user_dir) if user_dir else Path.home() / ".brainmass"

    # -- scope paths --------------------------------------------------------

    @property
    def user_settings_path(self) -> Path:
        return self.user_dir / "settings.json"

    @property
    def project_settings_path(self) -> Path:
        return self.project_dir / ".brainmass" / "settings.json"

    @property
    def project_local_settings_path(self) -> Path:
        return self.project_dir / ".brainmass" / "settings.local.json"

    @property
    def project_mcp_path(self) -> Path:
        return self.project_dir / ".mcp.json"

    @property
    def user_mcp_path(self) -> Path:
        return self.user_dir / ".mcp.json"

    # -- public API ---------------------------------------------------------

    def load(self) -> BrainmassConfig:
        """Load and merge settings from all scopes, returning a ``BrainmassConfig``."""
        user = _load_json(self.user_settings_path)
        project = _load_json(self.project_settings_path)
        local = _load_json(self.project_local_settings_path)

        # Merge: user < project < local
        merged = _deep_merge(user, project)
        merged = _deep_merge(merged, local)

        # Parse MCP servers from both user and project .mcp.json
        user_mcp = _load_json(self.user_mcp_path)
        project_mcp = _load_json(self.project_mcp_path)
        mcp_servers = _parse_mcp_servers(user_mcp)
        mcp_servers.update(_parse_mcp_servers(project_mcp))

        return BrainmassConfig(
            hooks=_parse_hooks(merged),
            feature_flags=_parse_feature_flags(merged),
            env_overrides=_parse_env_overrides(merged),
            mcp_servers=mcp_servers,
            raw=merged,
        )
