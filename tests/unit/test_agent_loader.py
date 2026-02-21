"""Unit tests for AgentLoader.

Covers:
- Loading valid .md files with YAML frontmatter (Req 8.1)
- Required field validation: name regex, description non-empty (Req 8.2)
- Optional field normalization: tools, model, permissionMode, color (Req 8.3)
- Model alias resolution via MODEL_MAP (Req 8.4)
- tools/disallowedTools mutual exclusivity (Req 8.8)
- Encoding edge cases: UTF-8 BOM, CRLF line endings (Req 8.1)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.agents.agent_loader import MODEL_MAP, AgentLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md(frontmatter: str, body: str = "You are a test agent.") -> str:
    """Build a minimal .md string with YAML frontmatter."""
    return f"---\n{frontmatter.strip()}\n---\n\n{body}\n"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loader() -> AgentLoader:
    return AgentLoader()


# ---------------------------------------------------------------------------
# Tests: Required fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_valid_minimal_definition(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Does something useful.")
        agent = loader.load_agent_string(content)
        assert agent.name == "my-agent"
        assert agent.description == "Does something useful."

    def test_missing_name_raises(self, loader: AgentLoader) -> None:
        content = _md("description: Does something.")
        with pytest.raises(ValueError, match="Missing required field 'name'"):
            loader.load_agent_string(content)

    def test_missing_description_raises(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent")
        with pytest.raises(ValueError, match="Missing required field 'description'"):
            loader.load_agent_string(content)

    def test_empty_name_raises(self, loader: AgentLoader) -> None:
        content = _md("name: ''\ndescription: Does something.")
        with pytest.raises(ValueError, match="must not be empty"):
            loader.load_agent_string(content)

    def test_empty_description_raises(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: ''")
        with pytest.raises(ValueError, match="must not be empty"):
            loader.load_agent_string(content)


# ---------------------------------------------------------------------------
# Tests: Name validation (regex ^[a-z][a-z0-9-]*$)
# ---------------------------------------------------------------------------


class TestNameValidation:
    @pytest.mark.parametrize("name", [
        "a",
        "my-agent",
        "code-reviewer",
        "agent123",
        "a1b2c3",
        "x-y-z",
    ])
    def test_valid_names(self, loader: AgentLoader, name: str) -> None:
        content = _md(f"name: {name}\ndescription: Valid agent.")
        agent = loader.load_agent_string(content)
        assert agent.name == name

    @pytest.mark.parametrize("name", [
        "MyAgent",        # uppercase
        "1agent",         # starts with digit
        "-agent",         # starts with hyphen
        "my agent",       # space
        "my_agent",       # underscore
        "my.agent",       # dot
        "MY-AGENT",       # uppercase
    ])
    def test_invalid_names_raise(self, loader: AgentLoader, name: str) -> None:
        content = _md(f"name: {name}\ndescription: Invalid name.")
        with pytest.raises(ValueError, match="Invalid name"):
            loader.load_agent_string(content)


# ---------------------------------------------------------------------------
# Tests: Model alias resolution
# ---------------------------------------------------------------------------


class TestModelAliasResolution:
    @pytest.mark.parametrize("alias,expected", [
        ("sonnet", "claude-sonnet-4-5-20250929"),
        ("opus", "claude-opus-4-6"),
        ("haiku", "claude-haiku-4-5-20251001"),
        ("inherit", "inherit"),
    ])
    def test_model_aliases_resolved(
        self, loader: AgentLoader, alias: str, expected: str
    ) -> None:
        content = _md(f"name: my-agent\ndescription: Test.\nmodel: {alias}")
        agent = loader.load_agent_string(content)
        assert agent.model == expected

    def test_default_model_is_inherit(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.")
        agent = loader.load_agent_string(content)
        assert agent.model == "inherit"

    def test_invalid_model_raises(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\nmodel: gpt-4")
        with pytest.raises(ValueError, match="Invalid model"):
            loader.load_agent_string(content)

    def test_model_map_completeness(self) -> None:
        """MODEL_MAP must contain all four required aliases."""
        assert "sonnet" in MODEL_MAP
        assert "opus" in MODEL_MAP
        assert "haiku" in MODEL_MAP
        assert "inherit" in MODEL_MAP


# ---------------------------------------------------------------------------
# Tests: Tool list normalization
# ---------------------------------------------------------------------------


class TestToolListNormalization:
    def test_comma_separated_string(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\ntools: Read,Glob,Grep")
        agent = loader.load_agent_string(content)
        assert agent.tools == ["Read", "Glob", "Grep"]

    def test_comma_separated_with_spaces(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\ntools: Read, Glob, Grep")
        agent = loader.load_agent_string(content)
        assert agent.tools == ["Read", "Glob", "Grep"]

    def test_yaml_list(self, loader: AgentLoader) -> None:
        content = _md(
            "name: my-agent\ndescription: Test.\ntools:\n  - Read\n  - Glob\n  - Grep"
        )
        agent = loader.load_agent_string(content)
        assert agent.tools == ["Read", "Glob", "Grep"]

    def test_no_tools_is_none(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.")
        agent = loader.load_agent_string(content)
        assert agent.tools is None

    def test_disallowed_tools_normalized(self, loader: AgentLoader) -> None:
        content = _md(
            "name: my-agent\ndescription: Test.\ndisallowedTools: Bash,Write"
        )
        agent = loader.load_agent_string(content)
        assert agent.disallowed_tools == ["Bash", "Write"]

    def test_tools_and_disallowed_tools_mutually_exclusive(
        self, loader: AgentLoader
    ) -> None:
        content = _md(
            "name: my-agent\ndescription: Test.\n"
            "tools: Read\ndisallowedTools: Bash"
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            loader.load_agent_string(content)


# ---------------------------------------------------------------------------
# Tests: Optional fields
# ---------------------------------------------------------------------------


class TestOptionalFields:
    def test_permission_mode_default(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.")
        agent = loader.load_agent_string(content)
        assert agent.permission_mode == "default"

    @pytest.mark.parametrize("mode", ["default", "bypassPermissions", "plan"])
    def test_valid_permission_modes(self, loader: AgentLoader, mode: str) -> None:
        content = _md(f"name: my-agent\ndescription: Test.\npermissionMode: {mode}")
        agent = loader.load_agent_string(content)
        assert agent.permission_mode == mode

    def test_invalid_permission_mode_raises(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\npermissionMode: admin")
        with pytest.raises(ValueError, match="Invalid permissionMode"):
            loader.load_agent_string(content)

    @pytest.mark.parametrize("color", ["purple", "cyan", "green", "orange", "blue", "red"])
    def test_valid_colors(self, loader: AgentLoader, color: str) -> None:
        content = _md(f"name: my-agent\ndescription: Test.\ncolor: {color}")
        agent = loader.load_agent_string(content)
        assert agent.color == color

    def test_invalid_color_raises(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\ncolor: pink")
        with pytest.raises(ValueError, match="Invalid color"):
            loader.load_agent_string(content)

    def test_max_turns(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\nmaxTurns: 10")
        agent = loader.load_agent_string(content)
        assert agent.max_turns == 10

    def test_invalid_max_turns_raises(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\nmaxTurns: -1")
        with pytest.raises(ValueError, match="maxTurns"):
            loader.load_agent_string(content)

    def test_skills_normalized(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\nskills: owasp,adr")
        agent = loader.load_agent_string(content)
        assert agent.skills == ["owasp", "adr"]

    def test_mcp_servers_parsed(self, loader: AgentLoader) -> None:
        content = _md(textwrap.dedent("""\
            name: my-agent
            description: Test.
            mcpServers:
              github:
                command: npx
                args:
                  - "@modelcontextprotocol/server-github"
        """))
        agent = loader.load_agent_string(content)
        assert "github" in agent.mcp_servers
        assert agent.mcp_servers["github"]["command"] == "npx"

    def test_memory_field(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\nmemory: Remember user preferences.")
        agent = loader.load_agent_string(content)
        assert agent.memory == "Remember user preferences."

    def test_system_prompt_from_body(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.", body="You are a helpful assistant.")
        agent = loader.load_agent_string(content)
        assert agent.system_prompt == "You are a helpful assistant."

    def test_source_and_plugin_namespace(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.")
        agent = loader.load_agent_string(
            content, source="plugin", plugin_namespace="my-plugin"
        )
        assert agent.source == "plugin"
        assert agent.plugin_namespace == "my-plugin"


# ---------------------------------------------------------------------------
# Tests: Encoding edge cases
# ---------------------------------------------------------------------------


class TestEncodingEdgeCases:
    def test_utf8_bom_stripped(self, loader: AgentLoader) -> None:
        """UTF-8 BOM (\xef\xbb\xbf) at start of file must be stripped."""
        content = _md("name: my-agent\ndescription: Test.")
        # Prepend BOM as string (unicode BOM character)
        bom_content = "\ufeff" + content
        agent = loader.load_agent_string(bom_content)
        assert agent.name == "my-agent"

    def test_crlf_normalized(self, loader: AgentLoader) -> None:
        """CRLF line endings must be normalized to LF."""
        content = "---\r\nname: my-agent\r\ndescription: Test.\r\n---\r\n\r\nBody text.\r\n"
        agent = loader.load_agent_string(content)
        assert agent.name == "my-agent"
        assert agent.system_prompt == "Body text."

    def test_utf8_bom_in_file(self, loader: AgentLoader, tmp_path: Path) -> None:
        """UTF-8 BOM in actual file bytes must be stripped."""
        content = _md("name: my-agent\ndescription: Test.")
        bom_bytes = b"\xef\xbb\xbf" + content.encode("utf-8")
        md_file = tmp_path / "my-agent.md"
        md_file.write_bytes(bom_bytes)
        agent = loader.load_agent_file(str(md_file))
        assert agent.name == "my-agent"

    def test_crlf_in_file(self, loader: AgentLoader, tmp_path: Path) -> None:
        """CRLF in actual file must be normalized."""
        content = "---\r\nname: my-agent\r\ndescription: Test.\r\n---\r\n\r\nBody.\r\n"
        md_file = tmp_path / "my-agent.md"
        md_file.write_bytes(content.encode("utf-8"))
        agent = loader.load_agent_file(str(md_file))
        assert agent.name == "my-agent"
        assert agent.system_prompt == "Body."

    def test_unicode_in_body(self, loader: AgentLoader) -> None:
        """Unicode characters in the body must be preserved."""
        body = "You are an agent. 你好世界 🚀"
        content = _md("name: my-agent\ndescription: Test.", body=body)
        agent = loader.load_agent_string(content)
        assert agent.system_prompt == body


# ---------------------------------------------------------------------------
# Tests: File loading
# ---------------------------------------------------------------------------


class TestFileLoading:
    def test_load_from_file(self, loader: AgentLoader, tmp_path: Path) -> None:
        content = _md("name: file-agent\ndescription: Loaded from file.")
        md_file = tmp_path / "file-agent.md"
        md_file.write_text(content, encoding="utf-8")
        agent = loader.load_agent_file(str(md_file))
        assert agent.name == "file-agent"
        assert agent.file_path == str(md_file)

    def test_file_not_found_raises(self, loader: AgentLoader) -> None:
        with pytest.raises(FileNotFoundError):
            loader.load_agent_file("/nonexistent/path/agent.md")


# ---------------------------------------------------------------------------
# Tests: Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_basic_round_trip(self, loader: AgentLoader) -> None:
        content = _md(
            "name: my-agent\ndescription: Does something useful.\nmodel: sonnet\ncolor: purple",
            body="You are a helpful agent.",
        )
        original = loader.load_agent_string(content)
        serialized = AgentLoader.serialize_to_frontmatter(original)
        restored = loader.load_agent_string(serialized)

        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.model == original.model
        assert restored.color == original.color
        assert restored.system_prompt == original.system_prompt

    def test_tools_round_trip(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\ntools: Read,Glob,Grep")
        original = loader.load_agent_string(content)
        serialized = AgentLoader.serialize_to_frontmatter(original)
        restored = loader.load_agent_string(serialized)
        assert restored.tools == original.tools

    def test_disallowed_tools_round_trip(self, loader: AgentLoader) -> None:
        content = _md("name: my-agent\ndescription: Test.\ndisallowedTools: Bash,Write")
        original = loader.load_agent_string(content)
        serialized = AgentLoader.serialize_to_frontmatter(original)
        restored = loader.load_agent_string(serialized)
        assert restored.disallowed_tools == original.disallowed_tools


# ---------------------------------------------------------------------------
# Tests: Built-in templates
# ---------------------------------------------------------------------------


class TestBuiltinTemplates:
    """Verify all 5 built-in templates load without errors."""

    TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "agents" / "templates"

    @pytest.mark.parametrize("template_name", [
        "code-reviewer.md",
        "security-auditor.md",
        "implementer-tester.md",
        "researcher.md",
        "architect.md",
    ])
    def test_template_loads(self, loader: AgentLoader, template_name: str) -> None:
        template_path = self.TEMPLATES_DIR / template_name
        assert template_path.exists(), f"Template not found: {template_path}"
        agent = loader.load_agent_file(str(template_path))
        assert agent.name
        assert agent.description
        assert agent.system_prompt

    def test_code_reviewer_config(self, loader: AgentLoader) -> None:
        path = self.TEMPLATES_DIR / "code-reviewer.md"
        agent = loader.load_agent_file(str(path))
        assert agent.name == "code-reviewer"
        assert agent.model == MODEL_MAP["sonnet"]
        assert agent.tools is not None
        assert "Read" in agent.tools
        assert agent.color == "purple"

    def test_security_auditor_config(self, loader: AgentLoader) -> None:
        path = self.TEMPLATES_DIR / "security-auditor.md"
        agent = loader.load_agent_file(str(path))
        assert agent.name == "security-auditor"
        assert agent.model == MODEL_MAP["opus"]
        assert agent.tools is not None
        assert "Bash" not in agent.tools  # read-only
        assert agent.color == "red"
        assert "owasp" in agent.skills

    def test_implementer_tester_config(self, loader: AgentLoader) -> None:
        path = self.TEMPLATES_DIR / "implementer-tester.md"
        agent = loader.load_agent_file(str(path))
        assert agent.name == "implementer-tester"
        assert agent.model == MODEL_MAP["sonnet"]
        assert agent.color == "green"

    def test_researcher_config(self, loader: AgentLoader) -> None:
        path = self.TEMPLATES_DIR / "researcher.md"
        agent = loader.load_agent_file(str(path))
        assert agent.name == "researcher"
        assert agent.model == MODEL_MAP["haiku"]
        assert agent.permission_mode == "plan"
        assert agent.color == "cyan"

    def test_architect_config(self, loader: AgentLoader) -> None:
        path = self.TEMPLATES_DIR / "architect.md"
        agent = loader.load_agent_file(str(path))
        assert agent.name == "architect"
        assert agent.model == MODEL_MAP["opus"]
        assert "github" in agent.mcp_servers
        assert "adr" in agent.skills
        assert agent.color == "blue"
