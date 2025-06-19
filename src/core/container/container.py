"""Dependency injection container for the forecasting platform."""

from typing import Dict, Any, Type, Optional
import logging


logger = logging.getLogger(__name__)


class Container:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, callable] = {}
        self._config: Dict[str, Any] = {}
    
    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance."""
        self._singletons[name] = instance
        logger.info(f"Registered singleton: {name}")
    
    def register_factory(self, name: str, factory: callable) -> None:
        """Register a factory function."""
        self._factories[name] = factory
        logger.info(f"Registered factory: {name}")
    
    def register_service(self, name: str, service_class: Type, **kwargs) -> None:
        """Register a service class with configuration."""
        self._services[name] = {
            'class': service_class,
            'config': kwargs
        }
        logger.info(f"Registered service: {name}")
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set global configuration."""
        self._config.update(config)
        logger.info("Configuration updated")
    
    def get_config(self, key: Optional[str] = None) -> Any:
        """Get configuration value."""
        if key:
            return self._config.get(key)
        return self._config
    
    def resolve(self, name: str) -> Any:
        """Resolve a dependency by name."""
        # Check singletons first
        if name in self._singletons:
            return self._singletons[name]
        
        # Check factories
        if name in self._factories:
            instance = self._factories[name]()
            return instance
        
        # Check services
        if name in self._services:
            service_info = self._services[name]
            service_class = service_info['class']
            config = service_info['config']
            
            # Inject configuration if needed
            if 'config' in config:
                config['config'] = self._config.get(config['config'], {})
            
            instance = service_class(**config)
            return instance
        
        raise ValueError(f"Service '{name}' not found in container")
    
    def list_services(self) -> Dict[str, str]:
        """List all registered services."""
        services = {}
        services.update({k: 'singleton' for k in self._singletons.keys()})
        services.update({k: 'factory' for k in self._factories.keys()})
        services.update({k: 'service' for k in self._services.keys()})
        return services