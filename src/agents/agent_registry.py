"""Agent Registry — Discover, load, and manage custom agent definitions.

Scans agent directories in precedence order (project > user > plugin),
loads .md files via AgentLoader, generates tool definitions for the model,
and supports hot-reload via filesystem polling.

Requirements: 8.5, 8.9, 8.10
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from src.agents.agent_loader import AgentLoader
from src.agents.builtin_templates import get_builtin_agent_templates
from src.types.core import AgentDefinition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Precedence order (lower index = higher precedence)
_SOURCE_PRECEDENCE = ["project", "user", "plugin"]

# Default polling interval for hot-reload (seconds)
_DEFAULT_POLL_INTERVAL = 2.0


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class AgentRegistry:
    """Discovers, loads, and manages custom agent definitions.

    Storage locations scanned (in precedence order):
    1. Project: ``.brainmass/agents/*.md`` (highest precedence)
    2. User: ``~/.brainmass/agents/*.md``
    3. Plugin: provided via ``add_plugin_agents()``

    On name collision, the highest-precedence source wins.

    Usage::

        registry = AgentRegistry(project_dir="/my/project")
        registry.load_all()
        agent = registry.get("code-reviewer")
        tools = registry.get_tool_definitions()
    """

    def __init__(
        self,
        project_dir: str | None = None,
        user_dir: str | None = None,
        loader: AgentLoader | None = None,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        load_builtins: bool | None = None,
    ) -> None:
        """
        Args:
            project_dir: Root of the project (contains .brainmass/agents/).
                         Defaults to current working directory.
            user_dir: User home directory (contains .brainmass/agents/).
                      Defaults to ``~``.
            loader: AgentLoader instance. A default one is created if not provided.
            poll_interval: Seconds between filesystem polls for hot-reload.
            load_builtins: Whether to include built-in agent templates. Defaults to
                           ``True`` when using the default project/user directories
                           (i.e. neither was explicitly provided), ``False`` otherwise.
                           Pass ``True`` to force built-ins even with explicit dirs.
        """
        self._project_dir = Path(project_dir) if project_dir else Path.cwd()
        self._user_dir = Path(user_dir) if user_dir else Path.home()
        # Default: load builtins only when using implicit (default) directories.
        if load_builtins is None:
            self._load_builtins = project_dir is None and user_dir is None
        else:
            self._load_builtins = load_builtins
        self._loader = loader or AgentLoader()
        self._poll_interval = poll_interval

        # name → AgentDefinition (highest-precedence wins)
        self._agents: dict[str, AgentDefinition] = {}

        # Plugin agent directories: list of (directory_path, plugin_namespace)
        self._plugin_dirs: list[tuple[Path, str]] = []

        # Hot-reload state
        self._watcher_thread: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._file_mtimes: dict[str, float] = {}

        # Optional reload callback (called after each reload cycle)
        self._on_reload: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Scan all agent directories and load definitions.

        Precedence: project > user > plugin > builtin.
        On name collision, the higher-precedence definition wins.
        Built-in templates are registered at lowest precedence so any
        user/project definition with the same name overrides them.

        Requirements: 8.5, 8.12
        """
        # Load in reverse precedence order so higher-precedence overwrites
        loaded: dict[str, AgentDefinition] = {}

        # 4. Built-in templates (lowest precedence — Req 8.12)
        # Only loaded when using default (implicit) directories or explicitly requested.
        if self._load_builtins:
            for agent_def in get_builtin_agent_templates():
                loaded[agent_def.name] = agent_def

        # 3. Plugin agents
        for plugin_dir, namespace in self._plugin_dirs:
            self._load_directory(plugin_dir, "plugin", namespace, loaded)

        # 2. User agents
        user_agents_dir = self._user_dir / ".brainmass" / "agents"
        self._load_directory(user_agents_dir, "user", None, loaded)

        # 1. Project agents (highest precedence — overwrites user/plugin/builtin)
        project_agents_dir = self._project_dir / ".brainmass" / "agents"
        self._load_directory(project_agents_dir, "project", None, loaded)

        self._agents = loaded
        logger.info("AgentRegistry: loaded %d agents.", len(self._agents))

    def get(self, name: str) -> AgentDefinition | None:
        """Return the AgentDefinition for *name*, or None if not found.

        For plugin agents, *name* may include the namespace prefix
        (e.g. ``'my-plugin:code-reviewer'``).
        """
        return self._agents.get(name)

    def list_agents(self) -> list[AgentDefinition]:
        """Return all registered agent definitions."""
        return list(self._agents.values())

    def get_tool_definitions(self) -> list[dict]:
        """Generate tool definitions for all registered agents.

        Each agent becomes a tool with:
        - ``name``: ``'agent:{agent.name}'``
        - ``description``: ``agent.description``

        Requirements: 8.10
        """
        return [
            {
                "name": f"agent:{agent.name}",
                "description": agent.description,
            }
            for agent in self._agents.values()
        ]

    def add_plugin_agents(self, plugin_dir: str, plugin_namespace: str) -> None:
        """Register a plugin's agent directory.

        Plugin agents are namespaced as ``'{plugin_namespace}:{agent_name}'``
        to avoid collisions with project/user agents.

        Args:
            plugin_dir: Path to the plugin's agents directory.
            plugin_namespace: Plugin name used as namespace prefix.
        """
        self._plugin_dirs.append((Path(plugin_dir), plugin_namespace))

    def register_agent(self, agent_def: AgentDefinition) -> None:
        """Manually register an AgentDefinition (e.g. from CLI --agents flag).

        Project-source agents registered this way take highest precedence.
        """
        self._agents[agent_def.name] = agent_def

    def to_strands_agent(self, name: str) -> object:
        """Look up an agent by name and return a live Strands Agent instance.

        Delegates to ``AgentLoader.to_strands_agent`` after resolving the
        AgentDefinition from the registry.

        --- Production integration point ---
        # agent_def = self.get(name)
        # return self._loader.to_strands_agent(agent_def)

        Args:
            name: Registered agent name (or namespaced plugin name).

        Returns:
            A configured strands.Agent instance.

        Raises:
            KeyError: If no agent with *name* is registered.
            RuntimeError: If the strands package is not installed.
        """
        agent_def = self._agents.get(name)
        if agent_def is None:
            raise KeyError(f"No agent registered with name '{name}'.")
        return self._loader.to_strands_agent(agent_def)

    # ------------------------------------------------------------------
    # Hot-reload
    # ------------------------------------------------------------------

    def start_hot_reload(
        self,
        on_reload: Callable[[], None] | None = None,
    ) -> None:
        """Start background filesystem polling for hot-reload.

        When .md files are added, modified, or removed, ``load_all()`` is
        called automatically.

        Args:
            on_reload: Optional callback invoked after each reload cycle.

        Requirements: 8.9
        """
        if self._watcher_thread is not None and self._watcher_thread.is_alive():
            return  # Already running

        self._on_reload = on_reload
        self._watcher_stop.clear()
        self._file_mtimes = self._snapshot_mtimes()

        self._watcher_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="agent-registry-watcher",
        )
        self._watcher_thread.start()
        logger.info(
            "AgentRegistry: hot-reload watcher started (interval=%.1fs).",
            self._poll_interval,
        )

    def stop_hot_reload(self) -> None:
        """Stop the background filesystem watcher."""
        self._watcher_stop.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=self._poll_interval * 2)
            self._watcher_thread = None
        logger.info("AgentRegistry: hot-reload watcher stopped.")

    # ------------------------------------------------------------------
    # Internal: directory loading
    # ------------------------------------------------------------------

    def _load_directory(
        self,
        directory: Path,
        source: str,
        plugin_namespace: str | None,
        target: dict[str, AgentDefinition],
    ) -> None:
        """Load all .md files from *directory* into *target*.

        Errors loading individual files are logged and skipped.
        """
        if not directory.exists() or not directory.is_dir():
            return

        for md_file in sorted(directory.glob("*.md")):
            try:
                agent_def = self._loader.load_agent_file(
                    str(md_file),
                    source=source,
                    plugin_namespace=plugin_namespace,
                )
                # Plugin agents are stored under namespaced key
                key = (
                    f"{plugin_namespace}:{agent_def.name}"
                    if plugin_namespace
                    else agent_def.name
                )
                target[key] = agent_def
                logger.debug("Loaded agent '%s' from %s", key, md_file)
            except Exception as exc:
                logger.error("Failed to load agent from %s: %s", md_file, exc)

    # ------------------------------------------------------------------
    # Internal: hot-reload polling
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread: poll for file changes and reload if needed."""
        while not self._watcher_stop.is_set():
            time.sleep(self._poll_interval)
            if self._watcher_stop.is_set():
                break
            try:
                current = self._snapshot_mtimes()
                if current != self._file_mtimes:
                    logger.info("AgentRegistry: file changes detected, reloading.")
                    self._file_mtimes = current
                    self.load_all()
                    if self._on_reload is not None:
                        self._on_reload()
            except Exception as exc:
                logger.error("AgentRegistry: error during hot-reload poll: %s", exc)

    def _snapshot_mtimes(self) -> dict[str, float]:
        """Snapshot modification times of all .md files in watched directories."""
        mtimes: dict[str, float] = {}

        dirs_to_watch: list[Path] = [
            self._project_dir / ".brainmass" / "agents",
            self._user_dir / ".brainmass" / "agents",
        ]
        for plugin_dir, _ in self._plugin_dirs:
            dirs_to_watch.append(plugin_dir)

        for directory in dirs_to_watch:
            if not directory.exists():
                continue
            for md_file in directory.glob("*.md"):
                try:
                    mtimes[str(md_file)] = md_file.stat().st_mtime
                except OSError:
                    pass

        return mtimes
