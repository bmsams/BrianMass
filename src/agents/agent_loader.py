"""Agent Loader — Parse .md files with YAML frontmatter into AgentDefinition objects.

Loads custom agent definitions from Markdown files where the YAML frontmatter
defines agent metadata and the Markdown body becomes the system prompt.

Handles encoding edge cases (UTF-8 BOM, CRLF), validates required fields,
normalizes tool lists, resolves model aliases, and enforces mutual exclusivity
of tools/disallowedTools.

Requirements: 8.1, 8.2, 8.3, 8.8
"""

from __future__ import annotations

import re
from pathlib import Path

import frontmatter  # python-frontmatter

from src.types.core import AgentDefinition, HookDefinition

# ---------------------------------------------------------------------------
# Model alias map
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "inherit": "inherit",
}

# Valid model aliases (including resolved IDs for pass-through)
_VALID_MODEL_ALIASES = set(MODEL_MAP.keys())

# Valid permission modes
_VALID_PERMISSION_MODES = {"default", "bypassPermissions", "plan"}

# Valid colors
_VALID_COLORS = {"purple", "cyan", "green", "orange", "blue", "red"}

# Name validation regex: lowercase-with-hyphens
_NAME_REGEX = re.compile(r"^[a-z][a-z0-9-]*$")


# ---------------------------------------------------------------------------
# Tool name → strands_tools callable map (Requirements: 14.2)
# ---------------------------------------------------------------------------

# Lazily populated on first call to _resolve_strands_tools.
_STRANDS_TOOL_MAP: dict[str, object] | None = None


def _resolve_strands_tools(tool_names: list[str] | None) -> list[object]:
    """Map agent tool names to strands_tools callables.

    Imports strands_tools lazily so the module loads without the SDK.
    Unknown tool names are silently skipped (the agent will have no tools
    for those names rather than failing at load time).

    --- Production integration point ---
    # from strands_tools import file_read, file_write, editor, ...
    # Returns the callable objects that Strands Agent accepts as tools.

    Args:
        tool_names: List of tool name strings from AgentDefinition.tools,
                    or None to return all default tools.

    Returns:
        List of strands_tools callable objects.
    """
    global _STRANDS_TOOL_MAP

    if _STRANDS_TOOL_MAP is None:
        try:
            import importlib
            import importlib.util

            if importlib.util.find_spec("strands_tools") is None:
                raise ImportError("strands_tools not found")

            _STRANDS_TOOL_MAP = {}
            # Import each tool module individually to avoid platform-specific failures
            # (e.g. strands_tools.shell requires termios on Linux only)
            _safe_tool_names = [
                "file_read", "file_write", "editor", "current_time",
                "calculator", "environment", "file_metadata",
            ]
            for _name in _safe_tool_names:
                try:
                    _mod = importlib.import_module(f"strands_tools.{_name}")
                    _fn = getattr(_mod, _name, None)
                    if _fn is not None:
                        _STRANDS_TOOL_MAP[_name] = _fn
                except Exception:
                    pass
        except ImportError:
            _STRANDS_TOOL_MAP = {}

    if tool_names is None:
        # Default tool set for agents with no explicit tool list
        return list(_STRANDS_TOOL_MAP.values())

    result: list[object] = []
    for name in tool_names:
        tool = _STRANDS_TOOL_MAP.get(name)
        if tool is not None:
            result.append(tool)
    return result


# ---------------------------------------------------------------------------
# AgentLoader
# ---------------------------------------------------------------------------


class AgentLoader:
    """Loads and validates agent definitions from .md files with YAML frontmatter.

    Usage::

        loader = AgentLoader()
        agent_def = loader.load_agent_file("/path/to/code-reviewer.md")
    """

    def load_agent_file(
        self,
        filepath: str,
        source: str = "project",
        plugin_namespace: str | None = None,
    ) -> AgentDefinition:
        """Parse a .md file with YAML frontmatter into an AgentDefinition.

        Args:
            filepath: Absolute or relative path to the .md file.
            source: Origin of the agent — 'project' | 'user' | 'plugin' | 'cli'.
            plugin_namespace: Namespace prefix for plugin agents (e.g. 'my-plugin').

        Returns:
            A validated AgentDefinition.

        Raises:
            ValueError: If required fields are missing or invalid.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Agent file not found: {filepath}")

        raw = path.read_bytes()

        # Strip UTF-8 BOM if present
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]

        # Decode and normalize CRLF → LF
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")

        post = frontmatter.loads(text)
        fm: dict = post.metadata
        body: str = post.content.strip()

        return self._build_definition(fm, body, str(filepath), source, plugin_namespace)

    def load_agent_string(
        self,
        content: str,
        filepath: str = "<string>",
        source: str = "project",
        plugin_namespace: str | None = None,
    ) -> AgentDefinition:
        """Parse agent definition from a string (for testing / CLI inline agents).

        Args:
            content: Raw .md content with YAML frontmatter.
            filepath: Logical path for error messages.
            source: Origin of the agent.
            plugin_namespace: Namespace prefix for plugin agents.

        Returns:
            A validated AgentDefinition.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Normalize CRLF → LF
        text = content.replace("\r\n", "\n").replace("\r", "\n")

        # Strip UTF-8 BOM if present (can appear in strings too)
        text = text.lstrip("\ufeff")

        post = frontmatter.loads(text)
        fm: dict = post.metadata
        body: str = post.content.strip()

        return self._build_definition(fm, body, filepath, source, plugin_namespace)

    # ------------------------------------------------------------------
    # Internal: build and validate
    # ------------------------------------------------------------------

    def _build_definition(
        self,
        fm: dict,
        body: str,
        filepath: str,
        source: str,
        plugin_namespace: str | None,
    ) -> AgentDefinition:
        """Validate frontmatter fields and construct an AgentDefinition."""
        # --- Required fields ---
        name = self._validate_name(fm, filepath)
        description = self._validate_description(fm, filepath)

        # --- Optional: model ---
        model_raw = str(fm.get("model", "inherit")).strip().lower()
        if model_raw not in _VALID_MODEL_ALIASES:
            raise ValueError(
                f"[{filepath}] Invalid model '{model_raw}'. "
                f"Must be one of: {sorted(_VALID_MODEL_ALIASES)}"
            )
        # Resolve alias to canonical model ID (keep 'inherit' as-is)
        model = MODEL_MAP[model_raw]

        # --- Optional: tools / disallowedTools (mutually exclusive) ---
        tools = self._normalize_tool_list(fm.get("tools"))
        disallowed_tools = self._normalize_tool_list(fm.get("disallowedTools"))

        if tools is not None and disallowed_tools is not None:
            raise ValueError(
                f"[{filepath}] 'tools' and 'disallowedTools' are mutually exclusive. "
                "Specify only one."
            )

        # --- Optional: permissionMode ---
        permission_mode = str(fm.get("permissionMode", "default")).strip()
        if permission_mode not in _VALID_PERMISSION_MODES:
            raise ValueError(
                f"[{filepath}] Invalid permissionMode '{permission_mode}'. "
                f"Must be one of: {sorted(_VALID_PERMISSION_MODES)}"
            )

        # --- Optional: color ---
        color: str | None = None
        if "color" in fm:
            color = str(fm["color"]).strip().lower()
            if color not in _VALID_COLORS:
                raise ValueError(
                    f"[{filepath}] Invalid color '{color}'. "
                    f"Must be one of: {sorted(_VALID_COLORS)}"
                )

        # --- Optional: maxTurns ---
        max_turns: int | None = None
        if "maxTurns" in fm:
            try:
                max_turns = int(fm["maxTurns"])
                if max_turns <= 0:
                    raise ValueError("maxTurns must be a positive integer")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"[{filepath}] Invalid maxTurns: {exc}") from exc

        # --- Optional: hooks ---
        hooks = self._parse_hooks(fm.get("hooks", {}), filepath)

        # --- Optional: skills ---
        skills = self._normalize_tool_list(fm.get("skills")) or []

        # --- Optional: mcpServers ---
        mcp_servers: dict[str, dict] = {}
        if "mcpServers" in fm and isinstance(fm["mcpServers"], dict):
            mcp_servers = fm["mcpServers"]

        # --- Optional: memory ---
        memory: str | None = None
        if "memory" in fm:
            memory = str(fm["memory"]).strip() or None

        return AgentDefinition(
            name=name,
            description=description,
            model=model,
            tools=tools,
            disallowed_tools=disallowed_tools,
            permission_mode=permission_mode,
            color=color,
            max_turns=max_turns,
            hooks=hooks,
            skills=skills,
            mcp_servers=mcp_servers,
            memory=memory,
            system_prompt=body,
            source=source,
            plugin_namespace=plugin_namespace,
            file_path=filepath,
        )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_name(fm: dict, filepath: str) -> str:
        """Validate and return the agent name."""
        if "name" not in fm:
            raise ValueError(f"[{filepath}] Missing required field 'name'.")
        name = str(fm["name"]).strip()
        if not name:
            raise ValueError(f"[{filepath}] Field 'name' must not be empty.")
        if not _NAME_REGEX.match(name):
            raise ValueError(
                f"[{filepath}] Invalid name '{name}'. "
                "Must match ^[a-z][a-z0-9-]*$ (lowercase letters, digits, hyphens; "
                "must start with a letter)."
            )
        return name

    @staticmethod
    def _validate_description(fm: dict, filepath: str) -> str:
        """Validate and return the agent description."""
        if "description" not in fm:
            raise ValueError(f"[{filepath}] Missing required field 'description'.")
        description = str(fm["description"]).strip()
        if not description:
            raise ValueError(f"[{filepath}] Field 'description' must not be empty.")
        return description

    @staticmethod
    def _normalize_tool_list(value: object) -> list[str] | None:
        """Normalize a tools field to a list of strings, or None if absent.

        Accepts:
        - None / missing → None
        - A comma-separated string → list of stripped, non-empty strings
        - A list of strings → cleaned list
        """
        if value is None:
            return None
        if isinstance(value, str):
            parts = [t.strip() for t in value.split(",")]
            result = [t for t in parts if t]
            return result if result else None
        if isinstance(value, list):
            result = [str(t).strip() for t in value if t is not None and str(t).strip()]
            return result if result else None
        return None

    @staticmethod
    def _parse_hooks(
        hooks_raw: object,
        filepath: str,
    ) -> dict[str, list[HookDefinition]]:
        """Parse the hooks frontmatter field into a dict of HookDefinition lists.

        The hooks field can be a dict mapping event names to lists of handler dicts.
        """
        if not hooks_raw or not isinstance(hooks_raw, dict):
            return {}

        result: dict[str, list[HookDefinition]] = {}
        for event_name, handlers in hooks_raw.items():
            if not isinstance(handlers, list):
                handlers = [handlers]
            defs: list[HookDefinition] = []
            for h in handlers:
                if isinstance(h, dict):
                    defs.append(HookDefinition(
                        matcher=h.get("matcher"),
                        hooks=[],  # Handler details stored in raw dict for now
                    ))
                else:
                    defs.append(HookDefinition(matcher=None, hooks=[]))
            result[str(event_name)] = defs

        return result

    # ------------------------------------------------------------------
    # Strands Agent construction (Requirements: 14.1, 14.2, 14.3, 14.4)
    # ------------------------------------------------------------------

    def to_strands_agent(self, agent_def: AgentDefinition) -> object:
        """Construct a live Strands Agent from an AgentDefinition.

        Resolves the model alias to a BedrockModel, maps the agent's tool
        list to strands_tools callables, and returns a configured Agent
        ready for invocation.

        --- Production integration point ---
        # from strands import Agent
        # from strands.models.bedrock import BedrockModel
        # from strands_tools import file_read, file_write, editor
        # model_id = MODEL_MAP.get(agent_def.model, agent_def.model)
        # bedrock_id = _BEDROCK_MODEL_IDS.get(model_id, model_id)
        # model = BedrockModel(model_id=bedrock_id)
        # tools = _resolve_strands_tools(agent_def.tools)
        # return Agent(
        #     name=agent_def.name,
        #     model=model,
        #     system_prompt=agent_def.system_prompt,
        #     tools=tools,
        # )

        Args:
            agent_def: A validated AgentDefinition.

        Returns:
            A configured strands.Agent instance.

        Raises:
            RuntimeError: If the strands package is not installed.
        """
        try:
            from strands import Agent  # type: ignore
            from strands.models.bedrock import BedrockModel  # type: ignore

            from src.agents._strands_utils import _BEDROCK_MODEL_IDS
        except ImportError as exc:
            raise RuntimeError(
                "strands package is required for to_strands_agent(). "
                "Install it with: pip install strands-agents"
            ) from exc

        # Resolve model alias → canonical ID → cross-region profile ID
        canonical = MODEL_MAP.get(agent_def.model, agent_def.model)
        bedrock_id = _BEDROCK_MODEL_IDS.get(canonical, canonical)
        model = BedrockModel(model_id=bedrock_id)

        tools = _resolve_strands_tools(agent_def.tools)

        return Agent(
            name=agent_def.name,
            model=model,
            system_prompt=agent_def.system_prompt or "",
            tools=tools,
        )

    # ------------------------------------------------------------------
    # Serialization helper (for round-trip testing)
    # ------------------------------------------------------------------

    @staticmethod
    def serialize_to_frontmatter(agent_def: AgentDefinition) -> str:
        """Serialize an AgentDefinition back to YAML frontmatter + Markdown body.

        This is the inverse of load_agent_string / load_agent_file.
        Used for round-trip testing (Property 14).
        """
        import yaml  # PyYAML is a transitive dependency of python-frontmatter

        fm: dict = {
            "name": agent_def.name,
            "description": agent_def.description,
        }

        # Resolve model back to alias
        reverse_map = {v: k for k, v in MODEL_MAP.items()}
        model_alias = reverse_map.get(agent_def.model, agent_def.model)
        if model_alias != "inherit":
            fm["model"] = model_alias

        if agent_def.tools is not None:
            fm["tools"] = ",".join(agent_def.tools)
        if agent_def.disallowed_tools is not None:
            fm["disallowedTools"] = ",".join(agent_def.disallowed_tools)
        if agent_def.permission_mode != "default":
            fm["permissionMode"] = agent_def.permission_mode
        if agent_def.color is not None:
            fm["color"] = agent_def.color
        if agent_def.max_turns is not None:
            fm["maxTurns"] = agent_def.max_turns
        if agent_def.skills:
            fm["skills"] = ",".join(agent_def.skills)
        if agent_def.mcp_servers:
            fm["mcpServers"] = agent_def.mcp_servers
        if agent_def.memory is not None:
            fm["memory"] = agent_def.memory

        yaml_str = yaml.dump(fm, default_flow_style=False, allow_unicode=True).strip()
        body = agent_def.system_prompt.strip()

        if body:
            return f"---\n{yaml_str}\n---\n\n{body}\n"
        return f"---\n{yaml_str}\n---\n"
