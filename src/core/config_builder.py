"""
Configuration Builder Pattern for the AurevIa Pipeline.

This module implements the Builder Pattern to centralize configuration logic,
eliminate scattered configuration handling, and provide a fluent API for
building complex configurations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import asdict

# Import only the config classes, avoiding circular imports with full pipeline
try:
    from ..pipeline.config import (
        PipelineConfig, DataPaths, ProcessingConfig, FeatureConfig, ModelConfig,
        WatchConfig, StockConfig, CryptoConfig, AssetConfig
    )
except ImportError:
    # Fallback for when sklearn is not available
    import sys
    import importlib.util
    
    # Direct import of just the config module
    spec = importlib.util.spec_from_file_location(
        "config", 
        Path(__file__).parent.parent / "pipeline" / "config.py"
    )
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    PipelineConfig = config_module.PipelineConfig
    DataPaths = config_module.DataPaths
    ProcessingConfig = config_module.ProcessingConfig
    FeatureConfig = config_module.FeatureConfig
    ModelConfig = config_module.ModelConfig
    WatchConfig = config_module.WatchConfig
    StockConfig = config_module.StockConfig
    CryptoConfig = config_module.CryptoConfig
    AssetConfig = config_module.AssetConfig

logger = logging.getLogger(__name__)


class PipelineConfigBuilder:
    """Builder class for creating pipeline configurations using fluent API."""
    
    def __init__(self):
        self.config = PipelineConfig()
        self._errors = []
    
    def with_data_paths(self, 
                       base_data_dir: str = "data",
                       scrape_dir: str = "scrape/prices",
                       output_dir: str = "output") -> 'PipelineConfigBuilder':
        """Configure data paths."""
        self.config.data_paths = DataPaths(
            base_data_dir=base_data_dir,
            scrape_dir=scrape_dir,
            output_dir=output_dir
        )
        return self
    
    def with_processing_options(self,
                               frequency: str = "D",
                               interpolation_method: str = "backfill",
                               fill_limit: Optional[int] = None,
                               outlier_method: str = "iqr",
                               outlier_threshold: float = 3.0,
                               min_data_points: int = 30) -> 'PipelineConfigBuilder':
        """Configure data processing options."""
        # Validate interpolation method
        valid_interpolation = ["backfill", "forward", "linear", "spline", "seasonal", "hybrid"]
        if interpolation_method not in valid_interpolation:
            self._errors.append(f"Invalid interpolation method: {interpolation_method}. Valid: {valid_interpolation}")
        
        # Validate outlier method
        valid_outlier = ["iqr", "zscore", "isolation_forest"]
        if outlier_method not in valid_outlier:
            self._errors.append(f"Invalid outlier method: {outlier_method}. Valid: {valid_outlier}")
        
        self.config.processing = ProcessingConfig(
            frequency=frequency,
            interpolation_method=interpolation_method,
            fill_limit=fill_limit,
            outlier_method=outlier_method,
            outlier_threshold=outlier_threshold,
            min_data_points=min_data_points
        )
        return self
    
    def with_feature_engineering(self,
                               lag_periods: List[int] = None,
                               rolling_windows: List[int] = None,
                               include_temporal_features: bool = True,
                               include_momentum_features: bool = True,
                               include_volatility_features: bool = True,
                               include_technical_indicators: bool = True,
                               target_shift: int = -1) -> 'PipelineConfigBuilder':
        """Configure feature engineering options."""
        if lag_periods is None:
            lag_periods = [1, 2, 3, 7, 14]
        
        if rolling_windows is None:
            rolling_windows = [3, 7, 14, 30]
        
        # Validate lag periods
        if not all(isinstance(x, int) and x > 0 for x in lag_periods):
            self._errors.append("All lag periods must be positive integers")
        
        # Validate rolling windows
        if not all(isinstance(x, int) and x > 0 for x in rolling_windows):
            self._errors.append("All rolling windows must be positive integers")
        
        self.config.features = FeatureConfig(
            lag_periods=lag_periods,
            rolling_windows=rolling_windows,
            include_temporal_features=include_temporal_features,
            include_momentum_features=include_momentum_features,
            include_volatility_features=include_volatility_features,
            include_technical_indicators=include_technical_indicators,
            target_shift=target_shift
        )
        return self
    
    def with_modeling_options(self,
                            test_size: float = 0.2,
                            validation_size: float = 0.1,
                            random_state: int = 42,
                            cross_validation_folds: int = 5,
                            scale_features: bool = True,
                            feature_selection: bool = True,
                            max_features: Optional[int] = None) -> 'PipelineConfigBuilder':
        """Configure modeling options."""
        # Validate test size
        if not 0 < test_size < 1:
            self._errors.append("Test size must be between 0 and 1")
        
        # Validate validation size
        if not 0 < validation_size < 1:
            self._errors.append("Validation size must be between 0 and 1")
        
        # Validate combined sizes
        if test_size + validation_size >= 1:
            self._errors.append("Combined test and validation size must be less than 1")
        
        # Validate CV folds
        if cross_validation_folds < 2:
            self._errors.append("Cross validation folds must be at least 2")
        
        self.config.modeling = ModelConfig(
            test_size=test_size,
            validation_size=validation_size,
            random_state=random_state,
            cross_validation_folds=cross_validation_folds,
            scale_features=scale_features,
            feature_selection=feature_selection,
            max_features=max_features
        )
        return self
    
    def with_watch_config(self,
                         price_column: str = "price(SGD)",
                         date_column: str = "date",
                         id_pattern: str = r"([^-]+)-([^-]+)-(\d+)",
                         luxury_tiers: Optional[Dict] = None,
                         brand_tiers: Optional[Dict] = None) -> 'PipelineConfigBuilder':
        """Configure watch-specific settings."""
        config_dict = {
            "price_column": price_column,
            "date_column": date_column,
            "id_pattern": id_pattern
        }
        
        if luxury_tiers is not None:
            config_dict["luxury_tiers"] = luxury_tiers
        
        if brand_tiers is not None:
            config_dict["brand_tiers"] = brand_tiers
        
        self.config.watch = WatchConfig(**config_dict)
        return self
    
    def with_stock_config(self,
                         price_column: str = "close",
                         date_column: str = "date",
                         volume_column: str = "volume",
                         id_pattern: str = r"([A-Z]+)") -> 'PipelineConfigBuilder':
        """Configure stock-specific settings."""
        self.config.stock = StockConfig(
            price_column=price_column,
            date_column=date_column,
            volume_column=volume_column,
            id_pattern=id_pattern
        )
        return self
    
    def with_crypto_config(self,
                          price_column: str = "close",
                          date_column: str = "timestamp",
                          volume_column: str = "volume",
                          id_pattern: str = r"([A-Z]+)-([A-Z]+)") -> 'PipelineConfigBuilder':
        """Configure cryptocurrency-specific settings."""
        self.config.crypto = CryptoConfig(
            price_column=price_column,
            date_column=date_column,
            volume_column=volume_column,
            id_pattern=id_pattern
        )
        return self
    
    def from_args(self, args: argparse.Namespace) -> 'PipelineConfigBuilder':
        """Build configuration from command line arguments."""
        # Data paths
        if hasattr(args, 'data_dir') and args.data_dir:
            self.with_data_paths(base_data_dir=args.data_dir)
        
        # Processing options
        processing_kwargs = {}
        if hasattr(args, 'interpolation_method') and args.interpolation_method:
            processing_kwargs['interpolation_method'] = args.interpolation_method
        if hasattr(args, 'outlier_method') and args.outlier_method:
            processing_kwargs['outlier_method'] = args.outlier_method
        if hasattr(args, 'min_data_points') and args.min_data_points:
            processing_kwargs['min_data_points'] = args.min_data_points
        
        if processing_kwargs:
            self.with_processing_options(**processing_kwargs)
        
        # Modeling options
        modeling_kwargs = {}
        if hasattr(args, 'test_size') and args.test_size:
            modeling_kwargs['test_size'] = args.test_size
        if hasattr(args, 'random_state') and args.random_state:
            modeling_kwargs['random_state'] = args.random_state
        if hasattr(args, 'cv_folds') and args.cv_folds:
            modeling_kwargs['cross_validation_folds'] = args.cv_folds
        if hasattr(args, 'no_scaling'):
            modeling_kwargs['scale_features'] = not args.no_scaling
        
        if modeling_kwargs:
            self.with_modeling_options(**modeling_kwargs)
        
        return self
    
    def from_json(self, json_path: Union[str, Path]) -> 'PipelineConfigBuilder':
        """Load configuration from JSON file."""
        json_path = Path(json_path)
        
        if not json_path.exists():
            self._errors.append(f"Configuration file not found: {json_path}")
            return self
        
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            
            return self.from_dict(config_dict)
        
        except json.JSONDecodeError as e:
            self._errors.append(f"Invalid JSON in configuration file: {str(e)}")
            return self
        except Exception as e:
            self._errors.append(f"Error loading configuration: {str(e)}")
            return self
    
    def from_dict(self, config_dict: Dict[str, Any]) -> 'PipelineConfigBuilder':
        """Build configuration from dictionary."""
        # Data paths
        if 'data_paths' in config_dict:
            data_paths = config_dict['data_paths']
            self.with_data_paths(**data_paths)
        
        # Processing
        if 'processing' in config_dict:
            processing = config_dict['processing']
            self.with_processing_options(**processing)
        
        # Features
        if 'features' in config_dict:
            features = config_dict['features']
            self.with_feature_engineering(**features)
        
        # Modeling
        if 'modeling' in config_dict:
            modeling = config_dict['modeling']
            self.with_modeling_options(**modeling)
        
        # Asset configs
        if 'watch' in config_dict:
            watch = config_dict['watch']
            self.with_watch_config(**watch)
        
        if 'stock' in config_dict:
            stock = config_dict['stock']
            self.with_stock_config(**stock)
        
        if 'crypto' in config_dict:
            crypto = config_dict['crypto']
            self.with_crypto_config(**crypto)
        
        return self
    
    def validate(self) -> bool:
        """Validate the configuration."""
        # Check for errors collected during building
        if self._errors:
            return False
        
        # Additional validation logic
        try:
            # Validate data paths exist or can be created
            data_paths = self.config.data_paths
            base_path = Path(data_paths.base_data_dir)
            
            if not base_path.exists():
                logger.warning(f"Base data directory does not exist: {base_path}")
            
            # Validate feature configuration consistency
            features = self.config.features
            if max(features.lag_periods) > min(features.rolling_windows):
                logger.warning("Some lag periods are larger than rolling windows")
            
            # Validate modeling configuration
            modeling = self.config.modeling
            if modeling.test_size + modeling.validation_size >= 0.9:
                logger.warning("Very little data left for training after test/validation split")
            
            return True
            
        except Exception as e:
            self._errors.append(f"Validation error: {str(e)}")
            return False
    
    def build(self, validate: bool = True) -> PipelineConfig:
        """Build the final configuration."""
        if validate and not self.validate():
            error_msg = "Configuration validation failed:\n" + "\n".join(self._errors)
            raise ValueError(error_msg)
        
        logger.info("Pipeline configuration built successfully")
        return self.config
    
    def reset(self) -> 'PipelineConfigBuilder':
        """Reset the builder to start fresh."""
        self.config = PipelineConfig()
        self._errors.clear()
        return self
    
    def get_errors(self) -> List[str]:
        """Get validation errors."""
        return self._errors.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current configuration to dictionary."""
        return asdict(self.config)
    
    def save_to_json(self, json_path: Union[str, Path]) -> None:
        """Save current configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {json_path}")


class TrainingConfigBuilder:
    """Specialized builder for training configurations."""
    
    def __init__(self):
        self.base_builder = PipelineConfigBuilder()
        self.horizons: List[int] = []
        self.models: List[str] = []
        self.max_assets: Optional[int] = None
        self.model_configs: Dict[str, Dict[str, Any]] = {}
    
    def with_horizons(self, horizons: List[int]) -> 'TrainingConfigBuilder':
        """Set prediction horizons."""
        if not all(isinstance(h, int) and h > 0 for h in horizons):
            raise ValueError("All horizons must be positive integers")
        
        self.horizons = sorted(set(horizons))  # Remove duplicates and sort
        return self
    
    def with_models(self, models: List[str]) -> 'TrainingConfigBuilder':
        """Set model types to train."""
        from .model_factory import ModelFactory
        
        # Validate model types
        available_models = ModelFactory.get_available_models()
        invalid_models = [m for m in models if m not in available_models]
        if invalid_models:
            raise ValueError(f"Unknown model types: {invalid_models}. Available: {available_models}")
        
        self.models = list(set(models))  # Remove duplicates
        return self
    
    def with_max_assets(self, max_assets: Optional[int]) -> 'TrainingConfigBuilder':
        """Set maximum number of assets to process."""
        if max_assets is not None and max_assets <= 0:
            raise ValueError("Max assets must be positive")
        
        self.max_assets = max_assets
        return self
    
    def with_model_config(self, model_type: str, **kwargs) -> 'TrainingConfigBuilder':
        """Add configuration for a specific model type."""
        from .model_factory import ModelFactory
        
        if not ModelFactory.validate_model_type(model_type):
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_configs[model_type] = kwargs
        return self
    
    def from_base_config(self, base_config: PipelineConfig) -> 'TrainingConfigBuilder':
        """Initialize from an existing base configuration."""
        self.base_builder.config = base_config
        return self
    
    def build_training_config(self) -> Dict[str, Any]:
        """Build training-specific configuration."""
        # Validate required parameters
        if not self.horizons:
            raise ValueError("Must specify prediction horizons")
        
        if not self.models:
            raise ValueError("Must specify model types")
        
        base_config = self.base_builder.build()
        
        return {
            'base_config': base_config,
            'horizons': self.horizons,
            'models': self.models,
            'max_assets': self.max_assets,
            'model_configs': self.model_configs
        }


def create_default_config() -> PipelineConfig:
    """Create a default pipeline configuration."""
    return (PipelineConfigBuilder()
            .with_data_paths()
            .with_processing_options()
            .with_feature_engineering()
            .with_modeling_options()
            .with_watch_config()
            .build())


def create_config_from_cli_args(args: argparse.Namespace) -> PipelineConfig:
    """Create configuration from CLI arguments."""
    return (PipelineConfigBuilder()
            .from_args(args)
            .build())


def create_training_config_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Create training configuration from CLI arguments."""
    # Build base config
    base_builder = PipelineConfigBuilder().from_args(args)
    
    # Build training config
    training_builder = TrainingConfigBuilder().from_base_config(base_builder.build())
    
    # Add training-specific parameters
    if hasattr(args, 'horizons') and args.horizons:
        training_builder.with_horizons(args.horizons)
    else:
        training_builder.with_horizons([7, 14, 30])  # Default horizons
    
    if hasattr(args, 'models') and args.models:
        training_builder.with_models(args.models)
    else:
        training_builder.with_models(['linear', 'xgboost'])  # Default models
    
    if hasattr(args, 'max_assets') and args.max_assets:
        training_builder.with_max_assets(args.max_assets)
    
    return training_builder.build_training_config()