"""
Consolidated configuration for the data pipeline.

This module contains all configuration classes for the pipeline,
replacing the previous modelling/config/ structure.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class DataPaths:
    """Configuration for data file paths."""
    
    base_data_dir: str = "data"
    scrape_dir: str = "scrape/prices"
    output_dir: str = "output"
    
    @property
    def scrape_path(self) -> Path:
        return Path(self.base_data_dir) / self.scrape_dir
    
    @property
    def output_path(self) -> Path:
        return Path(self.base_data_dir) / self.output_dir


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    
    frequency: str = "D"  # Daily frequency
    interpolation_method: str = "backfill"  # backfill, forward, linear, spline, seasonal, hybrid
    fill_limit: Optional[int] = None
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0
    min_data_points: int = 30  # Minimum points required for processing


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 7, 14])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 30])
    include_temporal_features: bool = True
    include_momentum_features: bool = True
    include_volatility_features: bool = True
    include_technical_indicators: bool = True
    target_shift: int = -1  # Days to shift for target variable


@dataclass
class ModelConfig:
    """Configuration for modeling pipeline."""
    
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    cross_validation_folds: int = 5
    scale_features: bool = True
    feature_selection: bool = True
    max_features: Optional[int] = None


@dataclass
class AssetConfig:
    """Base asset configuration."""
    
    asset_type: str
    price_column: str
    date_column: str
    volume_column: Optional[str] = None
    id_pattern: str = r"([^-]+)-([^-]+)-(\d+)"  # Default pattern
    
    def get_column_mapping(self) -> Dict[str, str]:
        """Returns standardized column names."""
        mapping = {
            "price": self.price_column,
            "date": self.date_column,
        }
        if self.volume_column:
            mapping["volume"] = self.volume_column
        return mapping


@dataclass
class WatchConfig(AssetConfig):
    """Watch-specific configuration."""
    
    asset_type: str = "watch"
    price_column: str = "price(SGD)"
    date_column: str = "date"
    id_pattern: str = r"([^-]+)-([^-]+)-(\d+)"  # Brand-Model-ID pattern
    
    # Watch-specific settings
    luxury_tiers: Dict[str, Dict] = field(default_factory=lambda: {
        "entry_luxury": {"min_price": 0, "max_price": 5000, "volatility_factor": 1.5},
        "mid_luxury": {"min_price": 5000, "max_price": 20000, "volatility_factor": 1.2},
        "high_luxury": {"min_price": 20000, "max_price": 100000, "volatility_factor": 1.0},
        "ultra_luxury": {"min_price": 100000, "max_price": float('inf'), "volatility_factor": 0.8}
    })
    
    brand_tiers: Dict[str, List[str]] = field(default_factory=lambda: {
        "ultra_luxury": ["patek_philippe", "audemars_piguet", "vacheron_constantin", "richard_mille"],
        "high_luxury": ["rolex", "omega", "cartier", "jaeger_lecoultre"],
        "mid_luxury": ["tudor", "longines", "tag_heuer", "breitling"]
    })


@dataclass
class StockConfig(AssetConfig):
    """Stock-specific configuration."""
    
    asset_type: str = "stock"
    price_column: str = "close"
    date_column: str = "date"
    volume_column: str = "volume"
    id_pattern: str = r"([A-Z]+)"  # Stock symbol pattern


@dataclass
class CryptoConfig(AssetConfig):
    """Cryptocurrency-specific configuration."""
    
    asset_type: str = "crypto"
    price_column: str = "close"
    date_column: str = "timestamp"
    volume_column: str = "volume"
    id_pattern: str = r"([A-Z]+)-([A-Z]+)"  # BTC-USD pattern


@dataclass
class PipelineConfig:
    """Main configuration class combining all settings."""
    
    data_paths: DataPaths = field(default_factory=DataPaths)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    modeling: ModelConfig = field(default_factory=ModelConfig)
    
    # Asset-specific configurations
    watch: WatchConfig = field(default_factory=WatchConfig)
    stock: StockConfig = field(default_factory=StockConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    
    def get_asset_config(self, asset_type: str) -> AssetConfig:
        """Get configuration for specific asset type."""
        asset_configs = {
            "watch": self.watch,
            "stock": self.stock,
            "crypto": self.crypto
        }
        
        if asset_type not in asset_configs:
            raise ValueError(f"Unknown asset type: {asset_type}")
            
        return asset_configs[asset_type]
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        return cls(
            data_paths=DataPaths(**config_dict.get('data_paths', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {})),
            features=FeatureConfig(**config_dict.get('features', {})),
            modeling=ModelConfig(**config_dict.get('modeling', {})),
            watch=WatchConfig(**config_dict.get('watch', {})),
            stock=StockConfig(**config_dict.get('stock', {})),
            crypto=CryptoConfig(**config_dict.get('crypto', {}))
        )


# Asset type registry for factory pattern
ASSET_CONFIGS = {
    "watch": WatchConfig,
    "stock": StockConfig,
    "crypto": CryptoConfig
}


def create_asset_config(asset_type: str, **kwargs) -> AssetConfig:
    """Factory function to create asset configuration."""
    if asset_type not in ASSET_CONFIGS:
        raise ValueError(f"Unknown asset type: {asset_type}")
    
    return ASSET_CONFIGS[asset_type](**kwargs)