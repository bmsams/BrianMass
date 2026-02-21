"""Unit tests for AgentRegistry.

Covers:
- Scanning agent directories with correct precedence (Req 8.5)
- Project > user > plugin precedence on name collision (Req 8.5)
- Tool definition generation (name='agent:{name}') (Req 8.10)
- Plugin namespace isolation (Req 9.5)
- Hot-reload via filesystem polling (Req 8.9)
- Error resilience (invalid files skipped, valid ones loaded)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.agents.agent_registry import AgentRegistry
from src.types.core import AgentDefinition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_agent(directory: Path, name: str, description: str, model: str = "sonnet") -> Path:
    """Write a minimal agent .md file to *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    content = (
        f"---\nname: {name}\ndescription: {description}\nmodel: {model}\n---\n\nSystem prompt.\n"
    )
    path = directory / f"{name}.md"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Return a temporary project root directory."""
    return tmp_path / "project"


@pytest.fixture
def tmp_user(tmp_path: Path) -> Path:
    """Return a temporary user home directory."""
    return tmp_path / "user"


@pytest.fixture
def registry(tmp_project: Path, tmp_user: Path) -> AgentRegistry:
    return AgentRegistry(
        project_dir=str(tmp_project),
        user_dir=str(tmp_user),
        load_builtins=True,
    )


# ---------------------------------------------------------------------------
# Tests: Basic loading
# ---------------------------------------------------------------------------


class TestBasicLoading:
    def test_empty_directories_loads_builtin_agents(self, registry: AgentRegistry) -> None:
        """With no project/user agents, the 5 built-in templates are always present."""
        registry.load_all()
        agents = registry.list_agents()
        names = {a.name for a in agents}
        assert len(agents) == 5
        assert names == {
            "code-reviewer",
            "security-auditor",
            "implementer-tester",
            "researcher",
            "architect",
        }

    def test_loads_project_agents(
        self, registry: AgentRegistry, tmp_project: Path
    ) -> None:
        _write_agent(tmp_project / ".brainmass" / "agents", "my-agent", "Does things.")
        registry.load_all()
        agent = registry.get("my-agent")
        assert agent is not None
        assert agent.name == "my-agent"
        assert agent.source == "project"

    def test_loads_user_agents(
        self, registry: AgentRegistry, tmp_user: Path
    ) -> None:
        _write_agent(tmp_user / ".brainmass" / "agents", "user-agent", "User-level agent.")
        registry.load_all()
        agent = registry.get("user-agent")
        assert agent is not None
        assert agent.source == "user"

    def test_loads_multiple_agents(
        self, registry: AgentRegistry, tmp_project: Path
    ) -> None:
        agents_dir = tmp_project / ".brainmass" / "agents"
        _write_agent(agents_dir, "agent-a", "Agent A.")
        _write_agent(agents_dir, "agent-b", "Agent B.")
        registry.load_all()
        # 2 project agents + 5 built-in templates
        assert len(registry.list_agents()) == 7
        assert registry.get("agent-a") is not None
        assert registry.get("agent-b") is not None

    def test_invalid_file_skipped(
        self, registry: AgentRegistry, tmp_project: Path
    ) -> None:
        agents_dir = tmp_project / ".brainmass" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        # Write an invalid agent (missing required fields)
        (agents_dir / "bad-agent.md").write_text("---\nno_name: true\n---\n", encoding="utf-8")
        _write_agent(agents_dir, "good-agent", "Valid agent.")
        registry.load_all()
        # Only the valid agent should be loaded
        assert registry.get("good-agent") is not None
        assert registry.get("bad-agent") is None
        # 1 valid project agent + 5 built-in templates
        assert len(registry.list_agents()) == 6


# ---------------------------------------------------------------------------
# Tests: Precedence (project > user > plugin)
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_project_beats_user_on_collision(
        self,
        registry: AgentRegistry,
        tmp_project: Path,
        tmp_user: Path,
    ) -> None:
        _write_agent(
            tmp_project / ".brainmass" / "agents",
            "shared-agent",
            "Project version.",
        )
        _write_agent(
            tmp_user / ".brainmass" / "agents",
            "shared-agent",
            "User version.",
        )
        registry.load_all()
        agent = registry.get("shared-agent")
        assert agent is not None
        assert agent.description == "Project version."
        assert agent.source == "project"

    def test_user_beats_plugin_on_collision(
        self,
        registry: AgentRegistry,
        tmp_user: Path,
        tmp_path: Path,
    ) -> None:
        plugin_dir = tmp_path / "plugin" / "agents"
        _write_agent(plugin_dir, "shared-agent", "Plugin version.")
        _write_agent(
            tmp_user / ".brainmass" / "agents",
            "shared-agent",
            "User version.",
        )
        registry.add_plugin_agents(str(plugin_dir), "my-plugin")
        registry.load_all()

        # User agent wins under its own name
        user_agent = registry.get("shared-agent")
        assert user_agent is not None
        assert user_agent.description == "User version."

        # Plugin agent is accessible under namespaced key
        plugin_agent = registry.get("my-plugin:shared-agent")
        assert plugin_agent is not None
        assert plugin_agent.description == "Plugin version."

    def test_project_beats_plugin_on_collision(
        self,
        registry: AgentRegistry,
        tmp_project: Path,
        tmp_path: Path,
    ) -> None:
        plugin_dir = tmp_path / "plugin" / "agents"
        _write_agent(plugin_dir, "shared-agent", "Plugin version.")
        _write_agent(
            tmp_project / ".brainmass" / "agents",
            "shared-agent",
            "Project version.",
        )
        registry.add_plugin_agents(str(plugin_dir), "my-plugin")
        registry.load_all()

        project_agent = registry.get("shared-agent")
        assert project_agent is not None
        assert project_agent.description == "Project version."

    def test_plugin_agents_namespaced(
        self,
        registry: AgentRegistry,
        tmp_path: Path,
    ) -> None:
        plugin_dir = tmp_path / "plugin" / "agents"
        _write_agent(plugin_dir, "code-reviewer", "Plugin code reviewer.")
        registry.add_plugin_agents(str(plugin_dir), "acme")
        registry.load_all()

        # Plugin agent accessible under namespace
        assert registry.get("acme:code-reviewer") is not None
        assert registry.get("acme:code-reviewer").description == "Plugin code reviewer."
        # Bare 'code-reviewer' resolves to the built-in template, not the plugin
        builtin = registry.get("code-reviewer")
        assert builtin is not None
        assert builtin.source == "cli"  # built-in, not the plugin

    def test_multiple_plugins_no_collision(
        self,
        registry: AgentRegistry,
        tmp_path: Path,
    ) -> None:
        plugin_a_dir = tmp_path / "plugin-a" / "agents"
        plugin_b_dir = tmp_path / "plugin-b" / "agents"
        _write_agent(plugin_a_dir, "helper", "Plugin A helper.")
        _write_agent(plugin_b_dir, "helper", "Plugin B helper.")
        registry.add_plugin_agents(str(plugin_a_dir), "plugin-a")
        registry.add_plugin_agents(str(plugin_b_dir), "plugin-b")
        registry.load_all()

        assert registry.get("plugin-a:helper") is not None
        assert registry.get("plugin-b:helper") is not None
        assert registry.get("plugin-a:helper").description == "Plugin A helper."
        assert registry.get("plugin-b:helper").description == "Plugin B helper."


# ---------------------------------------------------------------------------
# Tests: Tool definitions
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    def test_tool_definitions_format(
        self, registry: AgentRegistry, tmp_project: Path
    ) -> None:
        agents_dir = tmp_project / ".brainmass" / "agents"
        _write_agent(agents_dir, "code-reviewer", "Reviews code for quality.")
        registry.load_all()

        tools = registry.get_tool_definitions()
        # Project code-reviewer overrides the builtin; 4 other builtins remain = 5 total
        assert len(tools) == 5
        tool = next(t for t in tools if t["name"] == "agent:code-reviewer")
        # Project definition wins over builtin
        assert tool["description"] == "Reviews code for quality."

    def test_tool_definitions_all_agents(
        self, registry: AgentRegistry, tmp_project: Path
    ) -> None:
        agents_dir = tmp_project / ".brainmass" / "agents"
        _write_agent(agents_dir, "agent-a", "Agent A.")
        _write_agent(agents_dir, "agent-b", "Agent B.")
        registry.load_all()

        tools = registry.get_tool_definitions()
        names = {t["name"] for t in tools}
        assert "agent:agent-a" in names
        assert "agent:agent-b" in names

    def test_empty_registry_returns_builtin_tools(self, registry: AgentRegistry) -> None:
        """With no project/user agents, tool definitions for the 5 builtins are returned."""
        registry.load_all()
        tools = registry.get_tool_definitions()
        names = {t["name"] for t in tools}
        assert len(tools) == 5
        assert "agent:code-reviewer" in names
        assert "agent:security-auditor" in names
        assert "agent:implementer-tester" in names
        assert "agent:researcher" in names
        assert "agent:architect" in names


# ---------------------------------------------------------------------------
# Tests: Manual registration
# ---------------------------------------------------------------------------


class TestManualRegistration:
    def test_register_agent_manually(self, registry: AgentRegistry) -> None:
        agent_def = AgentDefinition(
            name="inline-agent",
            description="Registered inline.",
            source="cli",
        )
        registry.register_agent(agent_def)
        assert registry.get("inline-agent") is not None

    def test_manual_registration_overwrites_loaded(
        self, registry: AgentRegistry, tmp_project: Path
    ) -> None:
        _write_agent(
            tmp_project / ".brainmass" / "agents",
            "my-agent",
            "Loaded from file.",
        )
        registry.load_all()

        override = AgentDefinition(
            name="my-agent",
            description="Overridden inline.",
            source="cli",
        )
        registry.register_agent(override)
        assert registry.get("my-agent").description == "Overridden inline."


# ---------------------------------------------------------------------------
# Tests: Hot-reload
# ---------------------------------------------------------------------------


class TestHotReload:
    def test_hot_reload_detects_new_file(
        self, registry: AgentRegistry, tmp_project: Path
    ) -> None:
        registry.load_all()
        # Built-in templates are always present even with empty dirs
        assert len(registry.list_agents()) == 5

        reload_events = []
        registry.start_hot_reload(on_reload=lambda: reload_events.append(1), )

        try:
            # Write a new agent file
            agents_dir = tmp_project / ".brainmass" / "agents"
            _write_agent(agents_dir, "new-agent", "Newly added agent.")

            # Wait for the watcher to detect the change (poll interval is 2s by default,
            # but we use a short interval for tests)
            deadline = time.time() + 5.0
            while time.time() < deadline and len(reload_events) == 0:
                time.sleep(0.1)

            assert len(reload_events) >= 1
            assert registry.get("new-agent") is not None
        finally:
            registry.stop_hot_reload()

    def test_hot_reload_can_be_stopped(self, registry: AgentRegistry) -> None:
        registry.start_hot_reload()
        registry.stop_hot_reload()
        # Should not raise; watcher thread should be stopped
        assert registry._watcher_thread is None or not registry._watcher_thread.is_alive()

    def test_hot_reload_short_interval(
        self, tmp_project: Path, tmp_user: Path
    ) -> None:
        """Use a short poll interval for faster test execution."""
        registry = AgentRegistry(
            project_dir=str(tmp_project),
            user_dir=str(tmp_user),
            poll_interval=0.1,
            load_builtins=True,
        )
        registry.load_all()

        reload_events = []
        registry.start_hot_reload(on_reload=lambda: reload_events.append(1))

        try:
            agents_dir = tmp_project / ".brainmass" / "agents"
            _write_agent(agents_dir, "fast-agent", "Fast reload test.")

            deadline = time.time() + 3.0
            while time.time() < deadline and len(reload_events) == 0:
                time.sleep(0.05)

            assert len(reload_events) >= 1
            assert registry.get("fast-agent") is not None
        finally:
            registry.stop_hot_reload()
