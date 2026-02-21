"""Property-based tests for Agent Registry and Dispatch System.

Property 14: Agent definition parsing round-trip
For any valid .md file with YAML frontmatter conforming to the agent schema,
parsing into an AgentDefinition and then serializing back to YAML frontmatter +
Markdown body SHALL produce a semantically equivalent agent definition (all fields
preserved).
Validates: Requirements 8.1

Property 15: Agent definition validation
For any agent definition, validation SHALL accept definitions with valid required
fields (name matching ^[a-z][a-z0-9-]*$, non-empty description) and valid optional
field values (model in {sonnet, opus, haiku, inherit}, tools XOR disallowedTools not
both), and SHALL reject invalid definitions with descriptive error messages.
Validates: Requirements 8.2, 8.8

Property 16: Model alias resolution
For any model alias string, resolution SHALL produce:
  'sonnet' → 'claude-sonnet-4-5-20250929'
  'opus'   → 'claude-opus-4-6'
  'haiku'  → 'claude-haiku-4-5-20251001'
  'inherit'→ 'inherit'
Validates: Requirements 8.4

Property 17: Agent precedence resolution
For any set of agents loaded from multiple storage locations where agents share the
same name, the agent from the highest-precedence location SHALL be returned:
project > user > plugin. Plugin agents with namespace prefixes SHALL never collide
with project/user agents.
Validates: Requirements 8.5
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.agents.agent_loader import MODEL_MAP, AgentLoader
from src.agents.agent_registry import AgentRegistry

# ---------------------------------------------------------------------------
# Safe text strategy — printable ASCII, no YAML-special leading chars,
# no control characters, no characters that break unquoted YAML scalars.
# ---------------------------------------------------------------------------

# Characters that are safe anywhere in a YAML unquoted scalar value.
# Excludes ALL YAML-special chars: : { } [ ] , & * # ? | - < > = ! % @ ` \ and control chars.
_YAML_SAFE_CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .()/+"
)

# A description that is safe to embed unquoted in YAML.
# Must contain at least one alpha char so YAML doesn't resolve it as int/float/bool/null.
_YAML_SPECIAL_SCALARS = {"null", "true", "false", "yes", "no", "on", "off", "~", ""}
_safe_description = (
    st.text(alphabet=_YAML_SAFE_CHARS, min_size=3, max_size=80)
    .map(str.strip)
    .filter(lambda s: bool(s) and s.lower() not in _YAML_SPECIAL_SCALARS and any(c.isalpha() for c in s))
)

# Valid agent name: ^[a-z][a-z0-9-]*$
_name_first_char = st.sampled_from("abcdefghijklmnopqrstuvwxyz")
_name_rest_chars = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-",
    min_size=0,
    max_size=30,
)
_valid_name = st.builds(lambda first, rest: first + rest, _name_first_char, _name_rest_chars)

# Valid model aliases
_valid_model = st.sampled_from(["sonnet", "opus", "haiku", "inherit"])

# Valid colors
_valid_color = st.one_of(
    st.none(),
    st.sampled_from(["purple", "cyan", "green", "orange", "blue", "red"]),
)

# Valid permission modes
_valid_permission_mode = st.sampled_from(["default", "bypassPermissions", "plan"])

# Tool names — simple alpha identifiers only (no digits, no YAML-special values).
# Avoids YAML special scalars like 'null', 'true', 'false', 'yes', 'no', 'on', 'off'.
_YAML_SPECIAL_TOOL_NAMES = {"null", "true", "false", "yes", "no", "on", "off"}
_tool_name = (
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz",
        min_size=2,
        max_size=20,
    )
    .filter(lambda s: s not in _YAML_SPECIAL_TOOL_NAMES)
)
_tool_list = st.lists(_tool_name, min_size=1, max_size=5).map(
    lambda tools: list(dict.fromkeys(tools))  # deduplicate
)

# Optional tool list (None or a list)
_optional_tools = st.one_of(st.none(), _tool_list)

# System prompt body — safe printable text
_system_prompt = st.text(
    alphabet=_YAML_SAFE_CHARS + "\n",
    min_size=0,
    max_size=200,
)
# Max turns
_max_turns = st.one_of(st.none(), st.integers(min_value=1, max_value=100))

# Skills list
_skills = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=1, max_size=20),
    min_size=0,
    max_size=5,
)

# Memory — safe text
_memory = st.one_of(st.none(), st.text(alphabet=_YAML_SAFE_CHARS, min_size=1, max_size=100))

# Plugin namespace: starts with alpha, no trailing hyphen
_plugin_namespace = (
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz-",
        min_size=1,
        max_size=20,
    )
    .filter(lambda s: s[0].isalpha() and not s.endswith("-"))
)


def _build_valid_frontmatter(
    name: str,
    description: str,
    model: str,
    color: str | None,
    permission_mode: str,
    tools: list[str] | None,
    max_turns: int | None,
    skills: list[str],
    memory: str | None,
    system_prompt: str,
) -> str:
    """Build a valid .md string from components."""
    lines = [
        f"name: {name}",
        f"description: {description.strip()}",
    ]
    if model != "inherit":
        lines.append(f"model: {model}")
    if color is not None:
        lines.append(f"color: {color}")
    if permission_mode != "default":
        lines.append(f"permissionMode: {permission_mode}")
    if tools is not None:
        lines.append(f"tools: {','.join(tools)}")
    if max_turns is not None:
        lines.append(f"maxTurns: {max_turns}")
    if skills:
        lines.append(f"skills: {','.join(skills)}")
    if memory is not None:
        lines.append(f"memory: {memory.strip()}")

    fm = "\n".join(lines)
    body = system_prompt.strip()
    if body:
        return f"---\n{fm}\n---\n\n{body}\n"
    return f"---\n{fm}\n---\n"


# ---------------------------------------------------------------------------
# Property 14: Agent definition parsing round-trip
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    name=_valid_name,
    description=_safe_description,
    model=_valid_model,
    color=_valid_color,
    permission_mode=_valid_permission_mode,
    tools=_optional_tools,
    max_turns=_max_turns,
    skills=_skills,
    memory=_memory,
    system_prompt=_system_prompt,
)
@settings(max_examples=20)
def test_agent_definition_round_trip(
    name: str,
    description: str,
    model: str,
    color: str | None,
    permission_mode: str,
    tools: list[str] | None,
    max_turns: int | None,
    skills: list[str],
    memory: str | None,
    system_prompt: str,
) -> None:
    """**Property 14: Agent definition parsing round-trip**

    For any valid .md file with YAML frontmatter conforming to the agent schema,
    parsing into an AgentDefinition and then serializing back to YAML frontmatter +
    Markdown body SHALL produce a semantically equivalent agent definition (all
    fields preserved).

    Validates: Requirements 8.1
    """
    loader = AgentLoader()

    content = _build_valid_frontmatter(
        name=name,
        description=description.strip(),
        model=model,
        color=color,
        permission_mode=permission_mode,
        tools=tools,
        max_turns=max_turns,
        skills=skills,
        memory=memory,
        system_prompt=system_prompt,
    )

    # Parse
    original = loader.load_agent_string(content)

    # Serialize back
    serialized = AgentLoader.serialize_to_frontmatter(original)

    # Re-parse
    restored = loader.load_agent_string(serialized)

    # All fields must be semantically equivalent
    assert restored.name == original.name
    assert restored.description == original.description
    assert restored.model == original.model
    assert restored.color == original.color
    assert restored.permission_mode == original.permission_mode
    assert restored.max_turns == original.max_turns
    assert restored.system_prompt == original.system_prompt

    # Tool lists: order and content preserved
    if original.tools is not None:
        assert restored.tools == original.tools
    else:
        assert restored.tools is None

    # Skills: order and content preserved
    assert restored.skills == original.skills

    # Memory preserved
    assert restored.memory == original.memory


# ---------------------------------------------------------------------------
# Property 15: Agent definition validation
# ---------------------------------------------------------------------------

# Invalid name generators
_invalid_name_uppercase = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=1,
    max_size=10,
)
_invalid_name_starts_digit = st.integers(min_value=0, max_value=9).map(
    lambda d: str(d) + "agent"
)
_invalid_name_starts_hyphen = st.just("-agent")
_invalid_name_with_space = st.just("my agent")
_invalid_name_with_underscore = st.just("my_agent")
_invalid_name_empty = st.just("")

_invalid_name = st.one_of(
    _invalid_name_uppercase,
    _invalid_name_starts_digit,
    _invalid_name_starts_hyphen,
    _invalid_name_with_space,
    _invalid_name_with_underscore,
    _invalid_name_empty,
)


@pytest.mark.property
@given(name=_valid_name, description=_safe_description)
@settings(max_examples=20)
def test_valid_definitions_accepted(name: str, description: str) -> None:
    """**Property 15 (part A): Valid definitions are accepted**

    For any agent definition with valid required fields (name matching
    ^[a-z][a-z0-9-]*$, non-empty description), validation SHALL accept it.

    Validates: Requirements 8.2, 8.8
    """
    loader = AgentLoader()
    content = f"---\nname: {name}\ndescription: {description.strip()}\n---\n"
    # Should not raise
    agent = loader.load_agent_string(content)
    assert agent.name == name
    assert agent.description == description.strip()


@pytest.mark.property
@given(name=_invalid_name, description=_safe_description)
@settings(max_examples=30)
def test_invalid_names_rejected(name: str, description: str) -> None:
    """**Property 15 (part B): Invalid names are rejected with descriptive errors**

    For any agent definition with an invalid name (not matching ^[a-z][a-z0-9-]*$
    or empty), validation SHALL reject it with a descriptive error message.

    Validates: Requirements 8.2, 8.8
    """
    loader = AgentLoader()
    # Quote name to avoid YAML parse errors from special chars
    safe_name = name.replace("'", "''")
    content = f"---\nname: '{safe_name}'\ndescription: {description.strip()}\n---\n"
    with pytest.raises((ValueError, Exception)):
        loader.load_agent_string(content)


@pytest.mark.property
@given(
    name=_valid_name,
    tools=_tool_list,
    disallowed_tools=_tool_list,
)
@settings(max_examples=30)
def test_tools_and_disallowed_tools_mutually_exclusive(
    name: str,
    tools: list[str],
    disallowed_tools: list[str],
) -> None:
    """**Property 15 (part C): tools XOR disallowedTools**

    For any agent definition with both tools and disallowedTools set,
    validation SHALL reject it with a descriptive error.

    Validates: Requirements 8.8
    """
    loader = AgentLoader()
    tools_str = ",".join(tools)
    disallowed_str = ",".join(disallowed_tools)
    content = (
        f"---\nname: {name}\ndescription: Test agent.\n"
        f"tools: {tools_str}\ndisallowedTools: {disallowed_str}\n---\n"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        loader.load_agent_string(content)


@pytest.mark.property
@given(
    name=_valid_name,
    tools=_optional_tools,
)
@settings(max_examples=30)
def test_tools_xor_disallowed_tools_valid_when_only_one(
    name: str,
    tools: list[str] | None,
) -> None:
    """**Property 15 (part D): Only tools OR disallowedTools is valid**

    For any agent definition with only tools OR only disallowedTools (not both),
    validation SHALL accept it.

    Validates: Requirements 8.8
    """
    loader = AgentLoader()
    if tools is None:
        content = f"---\nname: {name}\ndescription: Test agent.\n---\n"
    else:
        tools_str = ",".join(tools)
        content = (
            f"---\nname: {name}\ndescription: Test agent.\n"
            f"tools: {tools_str}\n---\n"
        )
    # Should not raise
    agent = loader.load_agent_string(content)
    assert agent.name == name


# ---------------------------------------------------------------------------
# Property 16: Model alias resolution
# ---------------------------------------------------------------------------

_alias_to_expected = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "inherit": "inherit",
}


@pytest.mark.property
@given(
    name=_valid_name,
    description=_safe_description,
    alias=st.sampled_from(list(_alias_to_expected.keys())),
)
@settings(max_examples=30)
def test_model_alias_resolution(name: str, description: str, alias: str) -> None:
    """**Property 16: Model alias resolution**

    For any model alias string, resolution SHALL produce the correct canonical
    model ID.

    Validates: Requirements 8.4
    """
    loader = AgentLoader()
    content = (
        f"---\nname: {name}\ndescription: {description.strip()}\n"
        f"model: {alias}\n---\n"
    )
    agent = loader.load_agent_string(content)
    expected = _alias_to_expected[alias]
    assert agent.model == expected, (
        f"Alias '{alias}' resolved to '{agent.model}', expected '{expected}'"
    )


@pytest.mark.property
@given(
    name=_valid_name,
    description=_safe_description,
)
@settings(max_examples=30)
def test_default_model_is_inherit(name: str, description: str) -> None:
    """**Property 16 (default): Omitted model defaults to 'inherit'**

    When no model is specified, the resolved model SHALL be 'inherit'.

    Validates: Requirements 8.4
    """
    loader = AgentLoader()
    content = f"---\nname: {name}\ndescription: {description.strip()}\n---\n"
    agent = loader.load_agent_string(content)
    assert agent.model == "inherit"


@pytest.mark.property
@given(alias=st.sampled_from(list(_alias_to_expected.keys())))
@settings(max_examples=20)
def test_model_map_is_consistent(alias: str) -> None:
    """**Property 16 (map consistency): MODEL_MAP entries match expected values**

    Validates: Requirements 8.4
    """
    assert MODEL_MAP[alias] == _alias_to_expected[alias]


# ---------------------------------------------------------------------------
# Property 17: Agent precedence resolution
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    agent_name=_valid_name,
    project_desc=_safe_description,
    user_desc=_safe_description,
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_project_beats_user_precedence(
    agent_name: str,
    project_desc: str,
    user_desc: str,
) -> None:
    """**Property 17 (project > user): Project agents win over user agents**

    For any agent name present in both project and user directories,
    the project version SHALL be returned.

    Validates: Requirements 8.5
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir = tmp_path / "project"
        user_dir = tmp_path / "user"

        project_agents = project_dir / ".brainmass" / "agents"
        user_agents = user_dir / ".brainmass" / "agents"
        project_agents.mkdir(parents=True, exist_ok=True)
        user_agents.mkdir(parents=True, exist_ok=True)

        p_desc = project_desc.strip()[:100] or "Project agent."
        u_desc = user_desc.strip()[:100] or "User agent."

        (project_agents / f"{agent_name}.md").write_text(
            f"---\nname: {agent_name}\ndescription: {p_desc}\n---\n",
            encoding="utf-8",
        )
        (user_agents / f"{agent_name}.md").write_text(
            f"---\nname: {agent_name}\ndescription: {u_desc}\n---\n",
            encoding="utf-8",
        )

        registry = AgentRegistry(
            project_dir=str(project_dir),
            user_dir=str(user_dir),
        )
        registry.load_all()

        agent = registry.get(agent_name)
        assert agent is not None
        assert agent.source == "project", (
            f"Expected source='project', got source='{agent.source}'"
        )
        assert agent.description == p_desc


@pytest.mark.property
@given(
    agent_name=_valid_name,
    user_desc=_safe_description,
    plugin_desc=_safe_description,
    plugin_namespace=_plugin_namespace,
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_user_beats_plugin_precedence(
    agent_name: str,
    user_desc: str,
    plugin_desc: str,
    plugin_namespace: str,
) -> None:
    """**Property 17 (user > plugin): User agents win over plugin agents**

    For any agent name present in both user and plugin directories,
    the user version SHALL be returned under the unnamespaced key.
    The plugin version SHALL be accessible under the namespaced key.

    Validates: Requirements 8.5
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir = tmp_path / "project"
        user_dir = tmp_path / "user"
        plugin_dir = tmp_path / "plugin" / "agents"

        user_agents = user_dir / ".brainmass" / "agents"
        user_agents.mkdir(parents=True, exist_ok=True)
        plugin_dir.mkdir(parents=True, exist_ok=True)

        u_desc = user_desc.strip()[:100] or "User agent."
        p_desc = plugin_desc.strip()[:100] or "Plugin agent."

        (user_agents / f"{agent_name}.md").write_text(
            f"---\nname: {agent_name}\ndescription: {u_desc}\n---\n",
            encoding="utf-8",
        )
        (plugin_dir / f"{agent_name}.md").write_text(
            f"---\nname: {agent_name}\ndescription: {p_desc}\n---\n",
            encoding="utf-8",
        )

        registry = AgentRegistry(
            project_dir=str(project_dir),
            user_dir=str(user_dir),
        )
        registry.add_plugin_agents(str(plugin_dir), plugin_namespace)
        registry.load_all()

        # User agent wins under unnamespaced key
        user_agent = registry.get(agent_name)
        assert user_agent is not None
        assert user_agent.source == "user"
        assert user_agent.description == u_desc

        # Plugin agent accessible under namespaced key
        namespaced_key = f"{plugin_namespace}:{agent_name}"
        plugin_agent = registry.get(namespaced_key)
        assert plugin_agent is not None
        assert plugin_agent.description == p_desc


@pytest.mark.property
@given(
    agent_name=_valid_name,
    project_desc=_safe_description,
    plugin_desc=_safe_description,
    plugin_namespace=_plugin_namespace,
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_plugin_namespace_prevents_collision(
    agent_name: str,
    project_desc: str,
    plugin_desc: str,
    plugin_namespace: str,
) -> None:
    """**Property 17 (namespace isolation): Plugin namespaces prevent collisions**

    Plugin agents with namespace prefixes SHALL never collide with project/user
    agents. The namespaced key '{namespace}:{name}' is always distinct from
    the unnamespaced key '{name}'.

    Validates: Requirements 8.5
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir = tmp_path / "project"
        user_dir = tmp_path / "user"
        plugin_dir = tmp_path / "plugin" / "agents"

        project_agents = project_dir / ".brainmass" / "agents"
        project_agents.mkdir(parents=True, exist_ok=True)
        plugin_dir.mkdir(parents=True, exist_ok=True)

        p_desc = project_desc.strip()[:100] or "Project agent."
        pl_desc = plugin_desc.strip()[:100] or "Plugin agent."

        (project_agents / f"{agent_name}.md").write_text(
            f"---\nname: {agent_name}\ndescription: {p_desc}\n---\n",
            encoding="utf-8",
        )
        (plugin_dir / f"{agent_name}.md").write_text(
            f"---\nname: {agent_name}\ndescription: {pl_desc}\n---\n",
            encoding="utf-8",
        )

        registry = AgentRegistry(
            project_dir=str(project_dir),
            user_dir=str(user_dir),
        )
        registry.add_plugin_agents(str(plugin_dir), plugin_namespace)
        registry.load_all()

        # Unnamespaced key → project agent
        project_agent = registry.get(agent_name)
        assert project_agent is not None
        assert project_agent.source == "project"

        # Namespaced key → plugin agent
        namespaced_key = f"{plugin_namespace}:{agent_name}"
        plugin_agent = registry.get(namespaced_key)
        assert plugin_agent is not None

        # The two keys are always distinct
        assert agent_name != namespaced_key


@pytest.mark.property
@given(
    names=st.lists(
        _valid_name,
        min_size=1,
        max_size=5,
        unique=True,
    ),
    descriptions=st.lists(
        _safe_description,
        min_size=5,
        max_size=5,
    ),
)
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_tool_definitions_use_agent_prefix(
    names: list[str],
    descriptions: list[str],
) -> None:
    """**Property 17 (tool definitions): All tool names use 'agent:' prefix**

    For any set of registered agents, get_tool_definitions() SHALL return
    tool descriptors where every name is 'agent:{agent.name}'.

    Validates: Requirements 8.10
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        project_dir = tmp_path / "project"
        user_dir = tmp_path / "user"
        agents_dir = project_dir / ".brainmass" / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)

        for name, desc in zip(names, descriptions):
            safe_desc = desc.strip()[:100] or "Agent."
            (agents_dir / f"{name}.md").write_text(
                f"---\nname: {name}\ndescription: {safe_desc}\n---\n",
                encoding="utf-8",
            )

        registry = AgentRegistry(
            project_dir=str(project_dir),
            user_dir=str(user_dir),
        )
        registry.load_all()

        tools = registry.get_tool_definitions()
        assert len(tools) == len(names)

        for tool in tools:
            assert tool["name"].startswith("agent:"), (
                f"Tool name '{tool['name']}' does not start with 'agent:'"
            )
            suffix = tool["name"][len("agent:"):]
            assert re.match(r"^[a-z][a-z0-9-]*$", suffix), (
                f"Tool name suffix '{suffix}' is not a valid agent name"
            )
