"""Unit tests for plugin registry and marketplace workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.plugins.plugin_registry import (
    MarketplaceEntry,
    PluginRegistry,
    parse_marketplace_catalog,
    parse_plugin_manifest,
    scan_plugin_capabilities,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _create_plugin_repo(base: Path, plugin_name: str) -> Path:
    repo = base / plugin_name
    _write_json(
        repo / ".brainmass-plugin" / "plugin.json",
        {
            "name": plugin_name,
            "description": "Security toolkit plugin",
            "version": "1.0.0",
            "author": {"name": "Community Org"},
        },
    )
    (repo / "commands").mkdir(parents=True, exist_ok=True)
    (repo / "commands" / "security-scan.py").write_text("print('scan')\n", encoding="utf-8")
    (repo / "agents").mkdir(parents=True, exist_ok=True)
    (repo / "agents" / "vuln-scanner.md").write_text("# Agent\n", encoding="utf-8")
    (repo / "skills" / "owasp").mkdir(parents=True, exist_ok=True)
    (repo / "skills" / "owasp" / "SKILL.md").write_text(
        "---\nname: owasp\ndescription: OWASP checks\n---\nbody\n",
        encoding="utf-8",
    )
    (repo / "hooks").mkdir(parents=True, exist_ok=True)
    (repo / "hooks" / "pre_tool_use.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    _write_json(
        repo / ".mcp.json",
        {"mcpServers": {"dependency-checker": {"command": "dep-check"}}},
    )
    _write_json(
        repo / ".lsp.json",
        {"lspServers": {"python": {"command": "pylsp"}}},
    )
    return repo


def _create_marketplace_repo(
    base: Path,
    owner_repo: str,
    plugin_name: str,
    plugin_repo_ref: str,
) -> Path:
    repo = base / owner_repo.replace("/", "__")
    _write_json(
        repo / ".brainmass-plugin" / "marketplace.json",
        {
            "plugins": [
                {
                    "name": plugin_name,
                    "description": "Community security tools",
                    "version": "1.0.0",
                    "repo": plugin_repo_ref,
                }
            ]
        },
    )
    return repo


class TestPluginParsers:
    def test_parse_plugin_manifest(self, tmp_path: Path) -> None:
        repo = _create_plugin_repo(tmp_path, "security-toolkit")
        manifest = parse_plugin_manifest(str(repo))
        assert manifest.name == "security-toolkit"
        assert manifest.version == "1.0.0"

    def test_scan_plugin_capabilities(self, tmp_path: Path) -> None:
        repo = _create_plugin_repo(tmp_path, "security-toolkit")
        caps = scan_plugin_capabilities(str(repo))
        assert "security-scan" in caps.commands
        assert "vuln-scanner" in caps.agents
        assert "owasp" in caps.skills
        assert "pre_tool_use" in caps.hooks
        assert "dependency-checker" in caps.mcp_servers["mcpServers"]
        assert "python" in caps.lsp_servers["lspServers"]

    def test_parse_marketplace_catalog(self, tmp_path: Path) -> None:
        catalog = tmp_path / "marketplace.json"
        _write_json(
            catalog,
            {
                "plugins": [
                    {
                        "name": "security-toolkit",
                        "description": "Security helpers",
                        "version": "1.0.0",
                        "repo": "community-org/security-toolkit",
                    }
                ]
            },
        )
        entries = parse_marketplace_catalog(str(catalog))
        assert len(entries) == 1
        assert isinstance(entries[0], MarketplaceEntry)
        assert entries[0].repo == "community-org/security-toolkit"


class TestPluginRegistry:
    def test_add_marketplace_and_list_plugins(self, tmp_path: Path) -> None:
        plugin_repo = _create_plugin_repo(tmp_path, "security-toolkit")
        marketplace_repo = _create_marketplace_repo(
            tmp_path,
            "community-org/brainmass-plugins",
            "security-toolkit",
            "community-org/security-toolkit",
        )
        resolver_map = {
            "community-org/brainmass-plugins": str(marketplace_repo),
            "community-org/security-toolkit": str(plugin_repo),
        }

        registry = PluginRegistry(
            plugins_dir=str(tmp_path / "installed"),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
            repo_resolver=lambda repo: resolver_map[repo],
        )

        marketplace = registry.add_marketplace("community-org/brainmass-plugins")
        assert marketplace.owner_repo == "community-org/brainmass-plugins"
        assert len(registry.list_marketplace_plugins("community-org/brainmass-plugins")) == 1

    def test_install_plugin_from_marketplace(self, tmp_path: Path) -> None:
        plugin_repo = _create_plugin_repo(tmp_path, "security-toolkit")
        marketplace_repo = _create_marketplace_repo(
            tmp_path,
            "community-org/brainmass-plugins",
            "security-toolkit",
            "community-org/security-toolkit",
        )
        resolver_map = {
            "community-org/brainmass-plugins": str(marketplace_repo),
            "community-org/security-toolkit": str(plugin_repo),
        }

        registry = PluginRegistry(
            plugins_dir=str(tmp_path / "installed"),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
            repo_resolver=lambda repo: resolver_map[repo],
        )
        registry.add_marketplace("community-org/brainmass-plugins")
        installed = registry.install("security-toolkit@community-org/brainmass-plugins")

        assert installed.manifest.name == "security-toolkit"
        assert Path(installed.install_path).exists()
        assert "security-toolkit:vuln-scanner" in installed.capabilities.agents

        caps = registry.get_registered_capabilities()
        assert "security-scan" in caps["commands"]
        assert "security-toolkit:vuln-scanner" in caps["agents"]
        assert "owasp" in caps["skills"]
        assert "pre_tool_use" in caps["hooks"]
        assert "dependency-checker" in caps["mcp_servers"]
        assert "python" in caps["lsp_servers"]

    def test_disable_enable_plugin_globally(self, tmp_path: Path) -> None:
        plugin_repo = _create_plugin_repo(tmp_path, "security-toolkit")
        marketplace_repo = _create_marketplace_repo(
            tmp_path,
            "community-org/brainmass-plugins",
            "security-toolkit",
            "community-org/security-toolkit",
        )
        resolver_map = {
            "community-org/brainmass-plugins": str(marketplace_repo),
            "community-org/security-toolkit": str(plugin_repo),
        }
        registry = PluginRegistry(
            plugins_dir=str(tmp_path / "installed"),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
            repo_resolver=lambda repo: resolver_map[repo],
        )
        registry.add_marketplace("community-org/brainmass-plugins")
        registry.install("security-toolkit@community-org/brainmass-plugins")

        assert registry.is_enabled("security-toolkit") is True
        assert registry.disable_plugin("security-toolkit") is True
        assert registry.is_enabled("security-toolkit") is False

        caps = registry.get_registered_capabilities()
        assert "security-scan" not in caps["commands"]

        assert registry.enable_plugin("security-toolkit") is True
        assert registry.is_enabled("security-toolkit") is True
        caps = registry.get_registered_capabilities()
        assert "security-scan" in caps["commands"]

    def test_disable_enable_plugin_per_project(self, tmp_path: Path) -> None:
        plugin_repo = _create_plugin_repo(tmp_path, "security-toolkit")
        marketplace_repo = _create_marketplace_repo(
            tmp_path,
            "community-org/brainmass-plugins",
            "security-toolkit",
            "community-org/security-toolkit",
        )
        resolver_map = {
            "community-org/brainmass-plugins": str(marketplace_repo),
            "community-org/security-toolkit": str(plugin_repo),
        }
        registry = PluginRegistry(
            plugins_dir=str(tmp_path / "installed"),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
            repo_resolver=lambda repo: resolver_map[repo],
        )
        registry.add_marketplace("community-org/brainmass-plugins")
        registry.install("security-toolkit@community-org/brainmass-plugins")

        project_id = "project-a"
        assert registry.is_enabled("security-toolkit", project_id=project_id) is True
        assert registry.disable_plugin("security-toolkit", project_id=project_id) is True
        assert registry.is_enabled("security-toolkit", project_id=project_id) is False
        assert registry.enable_plugin("security-toolkit", project_id=project_id) is True
        assert registry.is_enabled("security-toolkit", project_id=project_id) is True

    def test_uninstall_plugin(self, tmp_path: Path) -> None:
        plugin_repo = _create_plugin_repo(tmp_path, "security-toolkit")
        marketplace_repo = _create_marketplace_repo(
            tmp_path,
            "community-org/brainmass-plugins",
            "security-toolkit",
            "community-org/security-toolkit",
        )
        resolver_map = {
            "community-org/brainmass-plugins": str(marketplace_repo),
            "community-org/security-toolkit": str(plugin_repo),
        }
        registry = PluginRegistry(
            plugins_dir=str(tmp_path / "installed"),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
            repo_resolver=lambda repo: resolver_map[repo],
        )
        registry.add_marketplace("community-org/brainmass-plugins")
        installed = registry.install("security-toolkit@community-org/brainmass-plugins")

        assert registry.uninstall("security-toolkit") is True
        assert registry.is_enabled("security-toolkit") is False
        assert not Path(installed.install_path).exists()

    def test_install_invalid_spec_raises(self, tmp_path: Path) -> None:
        registry = PluginRegistry(
            plugins_dir=str(tmp_path / "installed"),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
        )
        with pytest.raises(ValueError, match="Expected 'name@marketplace'"):
            registry.install("security-toolkit")

    def test_install_unknown_plugin_raises(self, tmp_path: Path) -> None:
        marketplace_repo = _create_marketplace_repo(
            tmp_path,
            "community-org/brainmass-plugins",
            "security-toolkit",
            "community-org/security-toolkit",
        )
        resolver_map = {"community-org/brainmass-plugins": str(marketplace_repo)}
        registry = PluginRegistry(
            plugins_dir=str(tmp_path / "installed"),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
            repo_resolver=lambda repo: resolver_map[repo],
        )
        registry.add_marketplace("community-org/brainmass-plugins")
        with pytest.raises(ValueError, match="not found in marketplace"):
            registry.install("unknown@community-org/brainmass-plugins")

    def test_discover_installed_plugins(self, tmp_path: Path) -> None:
        plugin_repo = _create_plugin_repo(tmp_path, "security-toolkit")
        install_root = tmp_path / "installed"
        shutil_target = install_root / "security-toolkit"
        shutil_target.parent.mkdir(parents=True, exist_ok=True)
        # Copy using plain file operations to avoid extra registry logic.
        for path in plugin_repo.rglob("*"):
            rel = path.relative_to(plugin_repo)
            dest = shutil_target / rel
            if path.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

        registry = PluginRegistry(
            plugins_dir=str(install_root),
            marketplaces_file=str(tmp_path / "marketplaces.json"),
        )
        count = registry.discover_installed_plugins()
        assert count == 1
        assert "security-toolkit" in registry.installed_plugins
        assert "security-toolkit:vuln-scanner" in registry.get_registered_capabilities()["agents"]
