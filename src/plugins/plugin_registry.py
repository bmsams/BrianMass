"""Plugin Registry for plugin discovery, installation, and lifecycle.

Manages:
- Marketplace registration and plugin discovery
- Plugin installation via "{name}@{marketplace}" specs
- Capability registration (commands, agents, skills, hooks, MCP, LSP)
- Per-project enable/disable toggling
- Uninstallation
- Agent namespacing as "plugin-name:agent-name"

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.types.core import PluginManifest

logger = logging.getLogger(__name__)

DEFAULT_PLUGINS_DIR = ".brainmass/plugins"
DEFAULT_MARKETPLACES_FILE = ".brainmass/marketplaces.json"
OFFICIAL_MARKETPLACE = "brainmass/official-plugins"


@dataclass
class PluginCapabilities:
    """Capabilities bundled in a plugin."""

    commands: list[str] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    lsp_servers: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstalledPlugin:
    """Represents an installed plugin and its runtime state."""

    manifest: PluginManifest
    marketplace: str
    install_path: str
    capabilities: PluginCapabilities = field(default_factory=PluginCapabilities)
    enabled: bool = True


@dataclass
class MarketplaceEntry:
    """One plugin entry in a marketplace catalog."""

    name: str
    description: str
    version: str
    repo: str


@dataclass
class Marketplace:
    """A plugin marketplace."""

    owner_repo: str
    plugins: list[MarketplaceEntry] = field(default_factory=list)
    catalog_path: str | None = None


def parse_plugin_manifest(plugin_dir: str) -> PluginManifest:
    """Parse `.brainmass-plugin/plugin.json` from a plugin directory."""

    manifest_path = Path(plugin_dir) / ".brainmass-plugin" / "plugin.json"
    if not manifest_path.is_file():
        raise ValueError(f"No plugin.json found at {manifest_path}")

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid plugin.json at {manifest_path}: {exc}") from exc

    for req_field in ("name", "description", "version", "author"):
        if req_field not in data:
            raise ValueError(f"Missing required field '{req_field}' in {manifest_path}")

    author = data["author"]
    if not isinstance(author, dict) or "name" not in author:
        raise ValueError(f"'author' must be an object with 'name' in {manifest_path}")

    return PluginManifest(
        name=data["name"],
        description=data["description"],
        version=data["version"],
        author=author,
        homepage=data.get("homepage"),
        license=data.get("license"),
    )


def scan_plugin_capabilities(plugin_dir: str) -> PluginCapabilities:
    """Scan a plugin directory for bundled capabilities."""

    root = Path(plugin_dir)
    caps = PluginCapabilities()

    commands_dir = root / "commands"
    if commands_dir.is_dir():
        for file_path in sorted(commands_dir.iterdir()):
            if file_path.is_file():
                caps.commands.append(file_path.stem)

    agents_dir = root / "agents"
    if agents_dir.is_dir():
        for file_path in sorted(agents_dir.glob("*.md")):
            caps.agents.append(file_path.stem)

    skills_dir = root / "skills"
    if skills_dir.is_dir():
        for file_path in sorted(skills_dir.rglob("SKILL.md")):
            if file_path.parent == skills_dir:
                caps.skills.append(file_path.stem)
            else:
                caps.skills.append(file_path.parent.name)

    hooks_dir = root / "hooks"
    if hooks_dir.is_dir():
        for file_path in sorted(hooks_dir.iterdir()):
            if file_path.is_file():
                caps.hooks.append(file_path.stem)

    mcp_path = root / ".mcp.json"
    if mcp_path.is_file():
        try:
            caps.mcp_servers = json.loads(mcp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to parse .mcp.json in %s", plugin_dir)

    lsp_path = root / ".lsp.json"
    if lsp_path.is_file():
        try:
            caps.lsp_servers = json.loads(lsp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to parse .lsp.json in %s", plugin_dir)

    return caps


def parse_marketplace_catalog(catalog_path: str) -> list[MarketplaceEntry]:
    """Parse a `marketplace.json` file into marketplace entries."""

    path = Path(catalog_path)
    if not path.is_file():
        raise ValueError(f"No marketplace catalog found at {catalog_path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Invalid marketplace catalog at {catalog_path}: {exc}") from exc

    if not isinstance(data, dict) or "plugins" not in data:
        raise ValueError(f"Marketplace catalog must have a 'plugins' array at {catalog_path}")

    entries: list[MarketplaceEntry] = []
    for item in data["plugins"]:
        if not isinstance(item, dict):
            continue
        for req_field in ("name", "description", "version", "repo"):
            if req_field not in item:
                logger.warning("Skipping marketplace entry missing '%s'", req_field)
                break
        else:
            entries.append(
                MarketplaceEntry(
                    name=item["name"],
                    description=item["description"],
                    version=item["version"],
                    repo=item["repo"],
                )
            )
    return entries


RepoResolver = Callable[[str], str]


def _default_repo_resolver(repo: str) -> str:
    """Resolve repo identifiers to a local path.

    Default behavior accepts an existing path only.
    """

    path = Path(repo)
    if path.exists():
        return str(path.resolve())
    raise ValueError(f"Cannot resolve repo '{repo}' to a local path")


class PluginRegistry:
    """Registry for plugin marketplaces and installed plugins."""

    def __init__(
        self,
        plugins_dir: str = DEFAULT_PLUGINS_DIR,
        marketplaces_file: str = DEFAULT_MARKETPLACES_FILE,
        repo_resolver: RepoResolver | None = None,
    ) -> None:
        self._plugins_dir = Path(plugins_dir)
        self._marketplaces_file = Path(marketplaces_file)
        self._repo_resolver = repo_resolver or _default_repo_resolver

        self._marketplaces: dict[str, Marketplace] = {}
        self._installed_plugins: dict[str, InstalledPlugin] = {}
        self._project_disabled: dict[str, set[str]] = {}

        self._capability_index: dict[str, set[str]] = {
            "commands": set(),
            "agents": set(),
            "skills": set(),
            "hooks": set(),
            "mcp_servers": set(),
            "lsp_servers": set(),
        }

        self._plugins_dir.mkdir(parents=True, exist_ok=True)
        self._load_marketplaces()
        self.discover_installed_plugins()

    @property
    def marketplaces(self) -> dict[str, Marketplace]:
        return dict(self._marketplaces)

    @property
    def installed_plugins(self) -> dict[str, InstalledPlugin]:
        return dict(self._installed_plugins)

    def add_marketplace(
        self,
        owner_repo: str,
        catalog_path: str | None = None,
    ) -> Marketplace:
        """Register a marketplace.

        If `catalog_path` is omitted, the marketplace repo is resolved and
        `.brainmass-plugin/marketplace.json` is used.
        """

        if catalog_path is None:
            repo_root = Path(self._repo_resolver(owner_repo))
            catalog = repo_root / ".brainmass-plugin" / "marketplace.json"
            catalog_path = str(catalog)

        plugins = parse_marketplace_catalog(catalog_path)
        marketplace = Marketplace(
            owner_repo=owner_repo,
            plugins=plugins,
            catalog_path=catalog_path,
        )
        self._marketplaces[owner_repo] = marketplace
        self._save_marketplaces()
        return marketplace

    def remove_marketplace(self, owner_repo: str) -> bool:
        if owner_repo not in self._marketplaces:
            return False
        del self._marketplaces[owner_repo]
        self._save_marketplaces()
        return True

    def get_marketplace(self, owner_repo: str) -> Marketplace | None:
        return self._marketplaces.get(owner_repo)

    def list_marketplace_plugins(self, owner_repo: str) -> list[MarketplaceEntry]:
        marketplace = self._marketplaces.get(owner_repo)
        if marketplace is None:
            return []
        return list(marketplace.plugins)

    def install(self, plugin_spec: str) -> InstalledPlugin:
        """Install a plugin from a `{name}@{marketplace}` spec."""

        plugin_name, marketplace_name = self._parse_install_spec(plugin_spec)

        marketplace = self._marketplaces.get(marketplace_name)
        if marketplace is None:
            marketplace = self.add_marketplace(marketplace_name)

        entry = next((p for p in marketplace.plugins if p.name == plugin_name), None)
        if entry is None:
            raise ValueError(
                f"Plugin '{plugin_name}' not found in marketplace '{marketplace_name}'"
            )

        source_dir = Path(self._repo_resolver(entry.repo))
        if not source_dir.is_dir():
            raise ValueError(f"Resolved plugin source is not a directory: {source_dir}")

        install_dir = self._plugins_dir / plugin_name
        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.copytree(source_dir, install_dir)

        manifest = parse_plugin_manifest(str(install_dir))
        capabilities = scan_plugin_capabilities(str(install_dir))
        capabilities = self._namespace_agents(manifest.name, capabilities)

        installed = InstalledPlugin(
            manifest=manifest,
            marketplace=marketplace_name,
            install_path=str(install_dir),
            capabilities=capabilities,
            enabled=True,
        )
        self._installed_plugins[manifest.name] = installed
        self._rebuild_capability_index()
        return installed

    def discover_installed_plugins(self) -> int:
        """Load installed plugins from disk and register their capabilities."""

        count = 0
        self._installed_plugins.clear()
        if not self._plugins_dir.is_dir():
            self._rebuild_capability_index()
            return 0

        for plugin_dir in sorted(self._plugins_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue
            try:
                manifest = parse_plugin_manifest(str(plugin_dir))
                capabilities = scan_plugin_capabilities(str(plugin_dir))
                capabilities = self._namespace_agents(manifest.name, capabilities)
            except ValueError as exc:
                logger.warning("Skipping invalid plugin at %s: %s", plugin_dir, exc)
                continue

            self._installed_plugins[manifest.name] = InstalledPlugin(
                manifest=manifest,
                marketplace="local",
                install_path=str(plugin_dir),
                capabilities=capabilities,
                enabled=True,
            )
            count += 1

        self._rebuild_capability_index()
        return count

    def register_capabilities(self, plugin: InstalledPlugin) -> None:
        """Register capabilities for a single installed plugin."""

        self._capability_index["commands"].update(plugin.capabilities.commands)
        self._capability_index["agents"].update(plugin.capabilities.agents)
        self._capability_index["skills"].update(plugin.capabilities.skills)
        self._capability_index["hooks"].update(plugin.capabilities.hooks)

        mcp = plugin.capabilities.mcp_servers.get("mcpServers", {})
        if isinstance(mcp, dict):
            self._capability_index["mcp_servers"].update(mcp.keys())

        lsp = plugin.capabilities.lsp_servers.get("lspServers", {})
        if isinstance(lsp, dict):
            self._capability_index["lsp_servers"].update(lsp.keys())

    def get_registered_capabilities(self) -> dict[str, list[str]]:
        """Return aggregated capability index across enabled plugins."""

        return {
            key: sorted(values)
            for key, values in self._capability_index.items()
        }

    def enable_plugin(
        self,
        plugin_name: str,
        project_id: str | None = None,
    ) -> bool:
        """Enable plugin globally or for a specific project."""

        plugin = self._installed_plugins.get(plugin_name)
        if plugin is None:
            return False

        if project_id is None:
            plugin.enabled = True
            self._rebuild_capability_index()
            return True

        disabled = self._project_disabled.setdefault(project_id, set())
        disabled.discard(plugin_name)
        return True

    def disable_plugin(
        self,
        plugin_name: str,
        project_id: str | None = None,
    ) -> bool:
        """Disable plugin globally or for a specific project."""

        plugin = self._installed_plugins.get(plugin_name)
        if plugin is None:
            return False

        if project_id is None:
            plugin.enabled = False
            self._rebuild_capability_index()
            return True

        disabled = self._project_disabled.setdefault(project_id, set())
        disabled.add(plugin_name)
        return True

    def is_enabled(self, plugin_name: str, project_id: str | None = None) -> bool:
        plugin = self._installed_plugins.get(plugin_name)
        if plugin is None or not plugin.enabled:
            return False

        if project_id is None:
            return True
        return plugin_name not in self._project_disabled.get(project_id, set())

    def uninstall(self, plugin_name: str) -> bool:
        """Uninstall a plugin and remove all registered capabilities."""

        plugin = self._installed_plugins.pop(plugin_name, None)
        if plugin is None:
            return False

        install_path = Path(plugin.install_path)
        if install_path.exists():
            shutil.rmtree(install_path)

        for disabled in self._project_disabled.values():
            disabled.discard(plugin_name)

        self._rebuild_capability_index()
        return True

    @staticmethod
    def get_namespaced_agent_name(plugin_name: str, agent_name: str) -> str:
        """Return agent namespace `plugin-name:agent-name`."""

        return f"{plugin_name}:{agent_name}"

    @staticmethod
    def _namespace_agents(
        plugin_name: str,
        capabilities: PluginCapabilities,
    ) -> PluginCapabilities:
        return PluginCapabilities(
            commands=list(capabilities.commands),
            agents=[
                PluginRegistry.get_namespaced_agent_name(plugin_name, name)
                for name in capabilities.agents
            ],
            skills=list(capabilities.skills),
            hooks=list(capabilities.hooks),
            mcp_servers=dict(capabilities.mcp_servers),
            lsp_servers=dict(capabilities.lsp_servers),
        )

    @staticmethod
    def _parse_install_spec(plugin_spec: str) -> tuple[str, str]:
        if "@" not in plugin_spec:
            raise ValueError(
                f"Invalid install spec '{plugin_spec}'. Expected 'name@marketplace'"
            )
        plugin_name, marketplace = plugin_spec.split("@", 1)
        plugin_name = plugin_name.strip()
        marketplace = marketplace.strip()
        if not plugin_name or not marketplace:
            raise ValueError(
                f"Invalid install spec '{plugin_spec}'. Expected 'name@marketplace'"
            )
        return plugin_name, marketplace

    def _rebuild_capability_index(self) -> None:
        for values in self._capability_index.values():
            values.clear()

        for plugin in self._installed_plugins.values():
            if plugin.enabled:
                self.register_capabilities(plugin)

    def _save_marketplaces(self) -> None:
        self._marketplaces_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "marketplaces": [
                {
                    "owner_repo": marketplace.owner_repo,
                    "catalog_path": marketplace.catalog_path,
                }
                for marketplace in self._marketplaces.values()
            ]
        }
        self._marketplaces_file.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _load_marketplaces(self) -> None:
        if not self._marketplaces_file.is_file():
            return

        try:
            raw = json.loads(self._marketplaces_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Ignoring invalid marketplaces file: %s", self._marketplaces_file)
            return

        for entry in raw.get("marketplaces", []):
            if not isinstance(entry, dict):
                continue
            owner_repo = entry.get("owner_repo")
            catalog_path = entry.get("catalog_path")
            if not isinstance(owner_repo, str) or not owner_repo:
                continue
            try:
                self.add_marketplace(owner_repo, catalog_path=catalog_path)
            except ValueError as exc:
                logger.warning("Failed to load marketplace %s: %s", owner_repo, exc)

