"""Dashboard configuration management."""

from typing import Dict, Any, Optional
import os
import json
import logging

logger = logging.getLogger(__name__)


class DashboardConfig:
    """Configuration manager for the dashboard."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/dashboard.json"
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            "api": {
                "base_url": "http://localhost:8000/api/v1",
                "timeout": 30
            },
            "dashboard": {
                "title": "Asset Forecasting Dashboard",
                "theme": {
                    "primary_color": "#1f77b4",
                    "background_color": "#ffffff",
                    "sidebar_color": "#f0f2f6"
                },
                "default_asset_type": "watch",
                "items_per_page": 50
            },
            "plugins": {
                "enabled": ["watch"],
                "auto_load": True
            },
            "cache": {
                "enabled": True,
                "ttl_seconds": 300
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    config = {**default_config, **file_config}
                    logger.info(f"Loaded configuration from {self.config_file}")
                    return config
            else:
                logger.info("Using default configuration")
                return default_config
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Set config {key} = {value}")
    
    def save(self) -> bool:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            return False
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
        logger.info("Configuration reloaded")
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return self.get('api', {})
    
    def get_theme_config(self) -> Dict[str, Any]:
        """Get theme configuration."""
        return self.get('dashboard.theme', {})
    
    def get_plugin_config(self) -> Dict[str, Any]:
        """Get plugin configuration.""" 
        return self.get('plugins', {})
    
    def is_plugin_enabled(self, plugin_name: str) -> bool:
        """Check if a plugin is enabled."""
        enabled_plugins = self.get('plugins.enabled', [])
        return plugin_name in enabled_plugins
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self.get('cache', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()