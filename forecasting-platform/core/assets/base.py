"""Base classes for all asset types and data collectors."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd


class BaseAsset(ABC):
    """Abstract base class for all tradeable assets."""
    
    def __init__(self, asset_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.asset_id = asset_id
        self.metadata = metadata or {}
    
    @abstractmethod
    def get_asset_type(self) -> str:
        """Return asset type (watch, gold, crypto, etc.)"""
        pass
    
    @abstractmethod
    def get_identifier(self) -> str:
        """Return unique identifier for the asset."""
        pass
    
    @abstractmethod
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate asset-specific data requirements."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return asset metadata."""
        return self.metadata
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update asset metadata."""
        self.metadata[key] = value


class BaseDataCollector(ABC):
    """Abstract base class for data collectors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def collect(self, asset: BaseAsset) -> pd.DataFrame:
        """Collect data for given asset."""
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        """Return data source name."""
        pass
    
    def validate_config(self) -> bool:
        """Validate collector configuration."""
        return True
    
    def get_rate_limit(self) -> int:
        """Get rate limit for requests (requests per minute)."""
        return self.config.get('rate_limit', 60)
    
    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration."""
        return self.config.get('retry_strategy', {
            'max_retries': 3,
            'backoff': 'exponential'
        })