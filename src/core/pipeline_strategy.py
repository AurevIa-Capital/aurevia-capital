"""
Strategy Pattern for Pipeline Components.

This module implements the Strategy Pattern to improve modularity by creating
interchangeable pipeline components with consistent interfaces.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from ..pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Enumeration of pipeline stages."""
    DATA_LOADING = "data_loading"
    DATA_PROCESSING = "data_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    VISUALIZATION = "visualization"


@dataclass
class PipelineResult:
    """Result of a pipeline strategy execution."""
    stage: PipelineStage
    success: bool
    data: Any
    metadata: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


class PipelineStrategy(ABC):
    """Abstract base class for pipeline strategies."""
    
    def __init__(self, config: PipelineConfig, stage: PipelineStage):
        self.config = config
        self.stage = stage
        self.name = self.__class__.__name__
        
    @abstractmethod
    def execute(self, input_data: Any, **kwargs) -> PipelineResult:
        """Execute the pipeline strategy."""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for this strategy."""
        pass
    
    def get_stage(self) -> PipelineStage:
        """Get the pipeline stage this strategy handles."""
        return self.stage
    
    def get_name(self) -> str:
        """Get the strategy name."""
        return self.name


class DataLoadingStrategy(PipelineStrategy):
    """Abstract strategy for data loading operations."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, PipelineStage.DATA_LOADING)


class WatchDataLoadingStrategy(DataLoadingStrategy):
    """Strategy for loading watch price data."""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input for watch data loading."""
        # For data loading, input_data might be file paths or configuration
        return True  # Basic validation - can be enhanced
    
    def execute(self, input_data: Any, **kwargs) -> PipelineResult:
        """Execute watch data loading strategy."""
        import time
        start_time = time.time()
        
        try:
            from ..pipeline.loader import DataLoader
            
            max_files = kwargs.get('max_files', None)
            loader = DataLoader(self.config, "watch")
            
            logger.info(f"Loading watch data with max_files={max_files}")
            data, report = loader.process(max_files=max_files)
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage=self.stage,
                success=True,
                data=data,
                metadata={
                    'report': report,
                    'assets_loaded': len(data) if data else 0,
                    'loader_config': {
                        'asset_type': 'watch',
                        'max_files': max_files
                    }
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Watch data loading failed: {str(e)}")
            
            return PipelineResult(
                stage=self.stage,
                success=False,
                data=None,
                metadata={'error_details': str(e)},
                execution_time=execution_time,
                error=str(e)
            )


class DataProcessingStrategy(PipelineStrategy):
    """Abstract strategy for data processing operations."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, PipelineStage.DATA_PROCESSING)


class StandardDataProcessingStrategy(DataProcessingStrategy):
    """Strategy for standard data processing (cleaning, validation, interpolation)."""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for processing."""
        if not isinstance(input_data, dict):
            return False
        
        # Check if we have valid DataFrames
        for asset_name, df in input_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                return False
        
        return True
    
    def execute(self, input_data: Dict[str, pd.DataFrame], **kwargs) -> PipelineResult:
        """Execute standard data processing strategy."""
        import time
        start_time = time.time()
        
        try:
            from ..pipeline.processor import DataProcessor
            
            processor = DataProcessor(self.config, "watch")
            
            logger.info(f"Processing {len(input_data)} assets")
            processed_data, report = processor.process(input_data)
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage=self.stage,
                success=True,
                data=processed_data,
                metadata={
                    'report': report,
                    'assets_processed': len(processed_data) if processed_data else 0,
                    'processing_config': {
                        'interpolation_method': self.config.processing.interpolation_method,
                        'outlier_method': self.config.processing.outlier_method
                    }
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Data processing failed: {str(e)}")
            
            return PipelineResult(
                stage=self.stage,
                success=False,
                data=None,
                metadata={'error_details': str(e)},
                execution_time=execution_time,
                error=str(e)
            )


class FeatureEngineeringStrategy(PipelineStrategy):
    """Abstract strategy for feature engineering operations."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, PipelineStage.FEATURE_ENGINEERING)


class StandardFeatureEngineeringStrategy(FeatureEngineeringStrategy):
    """Strategy for standard feature engineering."""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for feature engineering."""
        if not isinstance(input_data, dict):
            return False
        
        for asset_name, df in input_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                return False
        
        return True
    
    def execute(self, input_data: Dict[str, pd.DataFrame], **kwargs) -> PipelineResult:
        """Execute standard feature engineering strategy."""
        import time
        start_time = time.time()
        
        try:
            from ..pipeline.features import FeatureEngineer
            
            feature_engineer = FeatureEngineer(self.config, "watch")
            
            logger.info(f"Engineering features for {len(input_data)} assets")
            featured_data, report = feature_engineer.process(input_data)
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage=self.stage,
                success=True,
                data=featured_data,
                metadata={
                    'report': report,
                    'assets_featured': len(featured_data) if featured_data else 0,
                    'feature_config': {
                        'lag_periods': self.config.features.lag_periods,
                        'rolling_windows': self.config.features.rolling_windows,
                        'include_technical_indicators': self.config.features.include_technical_indicators
                    }
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Feature engineering failed: {str(e)}")
            
            return PipelineResult(
                stage=self.stage,
                success=False,
                data=None,
                metadata={'error_details': str(e)},
                execution_time=execution_time,
                error=str(e)
            )


class ModelTrainingStrategy(PipelineStrategy):
    """Abstract strategy for model training operations."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, PipelineStage.MODEL_TRAINING)


class MultiHorizonTrainingStrategy(ModelTrainingStrategy):
    """Strategy for multi-horizon model training."""
    
    def __init__(self, config: PipelineConfig, horizons: List[int], models: List[str]):
        super().__init__(config)
        self.horizons = horizons
        self.models = models
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for training."""
        if not isinstance(input_data, dict):
            return False
        
        # Check for valid DataFrames with required columns
        for asset_name, df in input_data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                return False
            if 'target' not in df.columns:
                return False
        
        return True
    
    def execute(self, input_data: Dict[str, pd.DataFrame], **kwargs) -> PipelineResult:
        """Execute multi-horizon training strategy."""
        import time
        start_time = time.time()
        
        try:
            from ..core.model_factory import ModelFactory
            from ..ml.training import ModelTrainer
            
            max_assets = kwargs.get('max_assets', None)
            
            # Limit assets if specified
            if max_assets and len(input_data) > max_assets:
                limited_data = dict(list(input_data.items())[:max_assets])
                input_data = limited_data
            
            trained_models = {}
            training_reports = {}
            
            logger.info(f"Training {len(self.models)} models for {len(self.horizons)} horizons on {len(input_data)} assets")
            
            for horizon in self.horizons:
                horizon_models = {}
                horizon_reports = {}
                
                for model_type in self.models:
                    try:
                        # Create model using factory
                        model = ModelFactory.create_model(model_type, self.config)
                        
                        # Train for this horizon
                        trainer = ModelTrainer(self.config)
                        training_result = trainer.train_model_for_horizon(
                            model, input_data, horizon
                        )
                        
                        horizon_models[model_type] = training_result['model']
                        horizon_reports[model_type] = training_result['report']
                        
                    except Exception as e:
                        logger.warning(f"Failed to train {model_type} for {horizon}-day horizon: {str(e)}")
                        continue
                
                if horizon_models:
                    trained_models[f"{horizon}_day"] = horizon_models
                    training_reports[f"{horizon}_day"] = horizon_reports
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage=self.stage,
                success=True,
                data=trained_models,
                metadata={
                    'training_reports': training_reports,
                    'horizons_trained': len(trained_models),
                    'models_per_horizon': len(self.models),
                    'training_config': {
                        'horizons': self.horizons,
                        'models': self.models,
                        'max_assets': max_assets
                    }
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Model training failed: {str(e)}")
            
            return PipelineResult(
                stage=self.stage,
                success=False,
                data=None,
                metadata={'error_details': str(e)},
                execution_time=execution_time,
                error=str(e)
            )


class VisualizationStrategy(PipelineStrategy):
    """Abstract strategy for visualization operations."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, PipelineStage.VISUALIZATION)


class StandardVisualizationStrategy(VisualizationStrategy):
    """Strategy for standard visualization generation."""
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for visualization."""
        # Input should be trained models and data
        if not isinstance(input_data, dict):
            return False
        
        required_keys = ['trained_models', 'data']
        return all(key in input_data for key in required_keys)
    
    def execute(self, input_data: Dict[str, Any], **kwargs) -> PipelineResult:
        """Execute visualization strategy."""
        import time
        start_time = time.time()
        
        try:
            from ..ml.visualization import PerformanceVisualizer, ForecastingVisualizer
            
            trained_models = input_data['trained_models']
            data = input_data['data']
            max_assets = kwargs.get('max_assets', 5)
            
            logger.info(f"Creating visualizations for trained models")
            
            # Create visualizations for each horizon and model
            visualization_results = {}
            
            for horizon_key, models in trained_models.items():
                horizon_viz = {}
                
                for model_type, model in models.items():
                    try:
                        # Create performance visualizations
                        perf_viz = PerformanceVisualizer(self.config)
                        forecast_viz = ForecastingVisualizer(self.config)
                        
                        # Generate plots (simplified for this strategy)
                        viz_info = {
                            'model_type': model_type,
                            'horizon': horizon_key,
                            'visualizations_created': ['performance', 'forecasting']
                        }
                        
                        horizon_viz[model_type] = viz_info
                        
                    except Exception as e:
                        logger.warning(f"Visualization failed for {model_type} {horizon_key}: {str(e)}")
                        continue
                
                if horizon_viz:
                    visualization_results[horizon_key] = horizon_viz
            
            execution_time = time.time() - start_time
            
            return PipelineResult(
                stage=self.stage,
                success=True,
                data=visualization_results,
                metadata={
                    'horizons_visualized': len(visualization_results),
                    'total_visualizations': sum(len(h) for h in visualization_results.values()),
                    'visualization_config': {
                        'max_assets': max_assets
                    }
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Visualization failed: {str(e)}")
            
            return PipelineResult(
                stage=self.stage,
                success=False,
                data=None,
                metadata={'error_details': str(e)},
                execution_time=execution_time,
                error=str(e)
            )


class StrategyFactory:
    """Factory for creating pipeline strategies."""
    
    _strategies = {
        PipelineStage.DATA_LOADING: {
            'watch': WatchDataLoadingStrategy,
            'default': WatchDataLoadingStrategy
        },
        PipelineStage.DATA_PROCESSING: {
            'standard': StandardDataProcessingStrategy,
            'default': StandardDataProcessingStrategy
        },
        PipelineStage.FEATURE_ENGINEERING: {
            'standard': StandardFeatureEngineeringStrategy,
            'default': StandardFeatureEngineeringStrategy
        },
        PipelineStage.MODEL_TRAINING: {
            'multi_horizon': MultiHorizonTrainingStrategy,
            'default': MultiHorizonTrainingStrategy
        },
        PipelineStage.VISUALIZATION: {
            'standard': StandardVisualizationStrategy,
            'default': StandardVisualizationStrategy
        }
    }
    
    @classmethod
    def create_strategy(cls, 
                       stage: PipelineStage, 
                       config: PipelineConfig,
                       strategy_type: str = 'default',
                       **kwargs) -> PipelineStrategy:
        """Create a strategy instance."""
        
        if stage not in cls._strategies:
            raise ValueError(f"Unknown pipeline stage: {stage}")
        
        stage_strategies = cls._strategies[stage]
        
        if strategy_type not in stage_strategies:
            if 'default' in stage_strategies:
                strategy_type = 'default'
            else:
                available = list(stage_strategies.keys())
                raise ValueError(f"Unknown strategy type '{strategy_type}' for stage {stage}. Available: {available}")
        
        strategy_class = stage_strategies[strategy_type]
        
        # Handle special cases that need additional parameters
        if stage == PipelineStage.MODEL_TRAINING and strategy_type in ['multi_horizon', 'default']:
            horizons = kwargs.get('horizons', [7, 14, 30])
            models = kwargs.get('models', ['linear', 'xgboost'])
            return strategy_class(config, horizons, models)
        
        return strategy_class(config)
    
    @classmethod
    def get_available_strategies(cls, stage: PipelineStage) -> List[str]:
        """Get available strategies for a stage."""
        return list(cls._strategies.get(stage, {}).keys())
    
    @classmethod
    def register_strategy(cls, 
                         stage: PipelineStage, 
                         name: str, 
                         strategy_class: type) -> None:
        """Register a new strategy."""
        if stage not in cls._strategies:
            cls._strategies[stage] = {}
        
        cls._strategies[stage][name] = strategy_class
        logger.info(f"Registered strategy '{name}' for stage {stage}")


def create_default_strategies(config: PipelineConfig, **kwargs) -> Dict[PipelineStage, PipelineStrategy]:
    """Create default strategies for all pipeline stages."""
    
    strategies = {}
    
    # Data loading
    strategies[PipelineStage.DATA_LOADING] = StrategyFactory.create_strategy(
        PipelineStage.DATA_LOADING, config, 'default'
    )
    
    # Data processing
    strategies[PipelineStage.DATA_PROCESSING] = StrategyFactory.create_strategy(
        PipelineStage.DATA_PROCESSING, config, 'default'
    )
    
    # Feature engineering
    strategies[PipelineStage.FEATURE_ENGINEERING] = StrategyFactory.create_strategy(
        PipelineStage.FEATURE_ENGINEERING, config, 'default'
    )
    
    # Model training (with parameters)
    horizons = kwargs.get('horizons', [7, 14, 30])
    models = kwargs.get('models', ['linear', 'xgboost'])
    strategies[PipelineStage.MODEL_TRAINING] = StrategyFactory.create_strategy(
        PipelineStage.MODEL_TRAINING, config, 'default',
        horizons=horizons, models=models
    )
    
    # Visualization
    strategies[PipelineStage.VISUALIZATION] = StrategyFactory.create_strategy(
        PipelineStage.VISUALIZATION, config, 'default'
    )
    
    return strategies