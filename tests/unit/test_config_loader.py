"""Unit tests for the configuration loader."""

import json
from pathlib import Path

from src.config.config_loader import (
    ConfigLoader,
    _deep_merge,
    _substitute_env_vars,
)
from src.types.core import HookHandlerType

# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_empty_dicts(self):
        assert _deep_merge({}, {}) == {}

    def test_override_replaces_scalar(self):
        assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_override_adds_new_key(self):
        assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_nested_dict_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        over = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, over) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_list_replaced_not_merged(self):
        base = {"items": [1, 2]}
        over = {"items": [3]}
        assert _deep_merge(base, over) == {"items": [3]}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        over = {"x": {"b": 2}}
        _deep_merge(base, over)
        assert base == {"x": {"a": 1}}
        assert over == {"x": {"b": 2}}


# ---------------------------------------------------------------------------
# _substitute_env_vars
# ---------------------------------------------------------------------------

class TestSubstituteEnvVars:
    def test_single_var(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "secret123")
        assert _substitute_env_vars("${MY_TOKEN}") == "secret123"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "5432")
        assert _substitute_env_vars("${HOST}:${PORT}") == "localhost:5432"

    def test_missing_var_becomes_empty(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        assert _substitute_env_vars("${NONEXISTENT_VAR_XYZ}") == ""

    def test_no_placeholders(self):
        assert _substitute_env_vars("plain text") == "plain text"


# ---------------------------------------------------------------------------
# ConfigLoader — 3-scope merging
# ---------------------------------------------------------------------------

class TestConfigLoaderMerging:
    """Test that settings from user < project < local are merged correctly."""

    def _write_json(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_loads_empty_when_no_files(self, tmp_path):
        loader = ConfigLoader(project_dir=tmp_path, user_dir=tmp_path / "user")
        cfg = loader.load()
        assert cfg.hooks == {}
        assert cfg.feature_flags == {}
        assert cfg.env_overrides == {}
        assert cfg.mcp_servers == {}

    def test_user_scope_only(self, tmp_path):
        user_dir = tmp_path / "user"
        self._write_json(user_dir / "settings.json", {
            "BRAINMASS_EXPERIMENTAL_AGENT_TEAMS": "1",
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=user_dir)
        cfg = loader.load()
        assert cfg.feature_flags["BRAINMASS_EXPERIMENTAL_AGENT_TEAMS"] == "1"

    def test_project_overrides_user(self, tmp_path):
        user_dir = tmp_path / "user"
        self._write_json(user_dir / "settings.json", {
            "ENABLE_TOOL_SEARCH": "false",
        })
        self._write_json(tmp_path / ".brainmass" / "settings.json", {
            "ENABLE_TOOL_SEARCH": "auto",
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=user_dir)
        cfg = loader.load()
        assert cfg.env_overrides["ENABLE_TOOL_SEARCH"] == "auto"

    def test_local_overrides_project(self, tmp_path):
        user_dir = tmp_path / "user"
        self._write_json(tmp_path / ".brainmass" / "settings.json", {
            "ENABLE_TOOL_SEARCH": "auto",
        })
        self._write_json(tmp_path / ".brainmass" / "settings.local.json", {
            "ENABLE_TOOL_SEARCH": "false",
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=user_dir)
        cfg = loader.load()
        assert cfg.env_overrides["ENABLE_TOOL_SEARCH"] == "false"

    def test_three_scope_precedence(self, tmp_path):
        user_dir = tmp_path / "user"
        self._write_json(user_dir / "settings.json", {
            "BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE": "70",
            "ENABLE_TOOL_SEARCH": "false",
        })
        self._write_json(tmp_path / ".brainmass" / "settings.json", {
            "BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE": "80",
        })
        self._write_json(tmp_path / ".brainmass" / "settings.local.json", {
            "BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE": "90",
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=user_dir)
        cfg = loader.load()
        # local wins over project wins over user
        assert cfg.env_overrides["BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE"] == "90"
        # user value preserved when not overridden
        assert cfg.env_overrides["ENABLE_TOOL_SEARCH"] == "false"


# ---------------------------------------------------------------------------
# ConfigLoader — hooks parsing
# ---------------------------------------------------------------------------

class TestConfigLoaderHooks:
    def _write_json(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_parses_command_hook(self, tmp_path):
        self._write_json(tmp_path / ".brainmass" / "settings.json", {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {"type": "command", "command": "/usr/local/bin/security-check"}
                        ],
                    }
                ]
            }
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=tmp_path / "user")
        cfg = loader.load()
        assert "PreToolUse" in cfg.hooks
        defn = cfg.hooks["PreToolUse"][0]
        assert defn.matcher == "Bash"
        assert len(defn.hooks) == 1
        assert defn.hooks[0].type == HookHandlerType.COMMAND
        assert defn.hooks[0].command == "/usr/local/bin/security-check"

    def test_parses_prompt_hook(self, tmp_path):
        self._write_json(tmp_path / ".brainmass" / "settings.json", {
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {"type": "prompt", "prompt": "Check if $ARGUMENTS is complete"}
                        ],
                    }
                ]
            }
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=tmp_path / "user")
        cfg = loader.load()
        handler = cfg.hooks["Stop"][0].hooks[0]
        assert handler.type == HookHandlerType.PROMPT
        assert "$ARGUMENTS" in handler.prompt

    def test_parses_async_hook(self, tmp_path):
        self._write_json(tmp_path / ".brainmass" / "settings.json", {
            "hooks": {
                "PostToolUse": [
                    {
                        "hooks": [
                            {"type": "command", "command": "echo done", "async": True}
                        ],
                    }
                ]
            }
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=tmp_path / "user")
        cfg = loader.load()
        handler = cfg.hooks["PostToolUse"][0].hooks[0]
        assert handler.is_async is True


# ---------------------------------------------------------------------------
# ConfigLoader — MCP server parsing with env substitution
# ---------------------------------------------------------------------------

class TestConfigLoaderMcp:
    def _write_json(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_parses_mcp_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "ghp_abc123")
        self._write_json(tmp_path / ".mcp.json", {
            "mcpServers": {
                "github": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "${GH_TOKEN}"},
                }
            }
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=tmp_path / "user")
        cfg = loader.load()
        assert "github" in cfg.mcp_servers
        srv = cfg.mcp_servers["github"]
        assert srv.command == "npx"
        assert srv.args == ["-y", "@modelcontextprotocol/server-github"]
        assert srv.env["GITHUB_TOKEN"] == "ghp_abc123"

    def test_user_and_project_mcp_merged(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "user"
        self._write_json(user_dir / ".mcp.json", {
            "mcpServers": {
                "user-server": {"command": "user-cmd", "args": []},
            }
        })
        self._write_json(tmp_path / ".mcp.json", {
            "mcpServers": {
                "project-server": {"command": "proj-cmd", "args": []},
            }
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=user_dir)
        cfg = loader.load()
        assert "user-server" in cfg.mcp_servers
        assert "project-server" in cfg.mcp_servers

    def test_project_mcp_overrides_user_same_name(self, tmp_path):
        user_dir = tmp_path / "user"
        self._write_json(user_dir / ".mcp.json", {
            "mcpServers": {
                "shared": {"command": "user-cmd"},
            }
        })
        self._write_json(tmp_path / ".mcp.json", {
            "mcpServers": {
                "shared": {"command": "project-cmd"},
            }
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=user_dir)
        cfg = loader.load()
        assert cfg.mcp_servers["shared"].command == "project-cmd"

    def test_missing_env_var_substituted_as_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MISSING_VAR_XYZ", raising=False)
        self._write_json(tmp_path / ".mcp.json", {
            "mcpServers": {
                "test": {
                    "command": "test-cmd",
                    "env": {"TOKEN": "${MISSING_VAR_XYZ}"},
                }
            }
        })
        loader = ConfigLoader(project_dir=tmp_path, user_dir=tmp_path / "user")
        cfg = loader.load()
        assert cfg.mcp_servers["test"].env["TOKEN"] == ""
