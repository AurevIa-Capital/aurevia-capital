"""Plugin manager for loading and managing asset plugins."""

from typing import Dict, List, Optional, Type
import logging
import importlib

from ..plugins.base import AssetPlugin
from ..plugins.watch_plugin import WatchPlugin

logger = logging.getLogger(__name__)


class PluginManager:
    """Manages asset-specific plugins for the dashboard."""
    
    def __init__(self):
        self._plugins: Dict[str, AssetPlugin] = {}
        self._plugin_classes: Dict[str, Type[AssetPlugin]] = {}
        self._initialize_plugins()
    
    def _initialize_plugins(self):
        """Initialize available plugins."""
        # Register built-in plugins
        self.register_plugin("watch", WatchPlugin)
        
        # Future plugins can be registered here
        # self.register_plugin("gold", GoldPlugin)
        # self.register_plugin("crypto", CryptoPlugin)
        
        logger.info(f"Initialized {len(self._plugin_classes)} plugins")
    
    def register_plugin(self, asset_type: str, plugin_class: Type[AssetPlugin]):
        """Register a plugin class for an asset type."""
        self._plugin_classes[asset_type] = plugin_class
        logger.info(f"Registered plugin for asset type: {asset_type}")
    
    def get_plugin(self, asset_type: str) -> Optional[AssetPlugin]:
        """Get plugin instance for an asset type."""
        if asset_type not in self._plugins:
            if asset_type in self._plugin_classes:
                # Create plugin instance
                plugin_class = self._plugin_classes[asset_type]
                self._plugins[asset_type] = plugin_class()
                logger.info(f"Created plugin instance for: {asset_type}")
            else:
                logger.warning(f"No plugin found for asset type: {asset_type}")
                return None
        
        return self._plugins[asset_type]
    
    def get_available_asset_types(self) -> List[str]:
        """Get list of available asset types."""
        return list(self._plugin_classes.keys())
    
    def get_plugin_info(self, asset_type: str) -> Optional[Dict[str, str]]:
        """Get plugin information."""
        plugin = self.get_plugin(asset_type)
        if plugin:
            return {
                "asset_type": asset_type,
                "display_name": plugin.get_display_name(),
                "icon": plugin.get_icon(),
                "supported_pages": plugin.get_supported_pages(),
                "custom_metrics": plugin.get_custom_metrics()
            }
        return None
    
    def reload_plugin(self, asset_type: str):
        """Reload a plugin (useful for development)."""
        if asset_type in self._plugins:
            del self._plugins[asset_type]
            logger.info(f"Reloaded plugin for: {asset_type}")
    
    def load_external_plugin(self, asset_type: str, module_path: str):
        """Load an external plugin from a module path."""
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, f"{asset_type.title()}Plugin")
            self.register_plugin(asset_type, plugin_class)
            logger.info(f"Loaded external plugin: {asset_type} from {module_path}")
        except Exception as e:
            logger.error(f"Failed to load external plugin {asset_type}: {e}")
    
    def unregister_plugin(self, asset_type: str):
        """Unregister a plugin."""
        if asset_type in self._plugin_classes:
            del self._plugin_classes[asset_type]
        
        if asset_type in self._plugins:
            del self._plugins[asset_type]
        
        logger.info(f"Unregistered plugin: {asset_type}")
    
    def get_plugin_stats(self) -> Dict[str, int]:
        """Get plugin statistics."""
        return {
            "total_registered": len(self._plugin_classes),
            "total_loaded": len(self._plugins),
            "available_types": len(self.get_available_asset_types())
        }