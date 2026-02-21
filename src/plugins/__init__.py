"""Plugin ecosystem and marketplace."""

from src.plugins.plugin_registry import (
    DEFAULT_MARKETPLACES_FILE,
    DEFAULT_PLUGINS_DIR,
    OFFICIAL_MARKETPLACE,
    InstalledPlugin,
    Marketplace,
    MarketplaceEntry,
    PluginCapabilities,
    PluginRegistry,
    parse_marketplace_catalog,
    parse_plugin_manifest,
    scan_plugin_capabilities,
)

__all__ = [
    "DEFAULT_PLUGINS_DIR",
    "DEFAULT_MARKETPLACES_FILE",
    "OFFICIAL_MARKETPLACE",
    "PluginCapabilities",
    "InstalledPlugin",
    "MarketplaceEntry",
    "Marketplace",
    "PluginRegistry",
    "parse_plugin_manifest",
    "scan_plugin_capabilities",
    "parse_marketplace_catalog",
]
