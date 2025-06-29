"""
Machine learning components for time series forecasting.

This module provides comprehensive ML training, evaluation, and prediction
capabilities integrated with the data pipeline.
"""

from .base import BaseTimeSeriesModel, SklearnTimeSeriesModel, ModelMetadata, PredictionResult
from .training import ModelTrainer, TrainingResult, TimeSeriesValidator, ValidationResult
from .models import (
    LinearRegressionModel, RidgeModel, LassoModel,
    RandomForestModel, XGBoostModel,
    ARIMAModel, SARIMAModel
)

# Main training function
def train_models(config, featured_data, model_names=None, **kwargs):
    """
    Main entry point for model training.
    
    Parameters:
    ----------
    config : PipelineConfig
        Pipeline configuration
    featured_data : Dict[str, pd.DataFrame]
        Featured data from pipeline
    model_names : List[str], optional
        Models to train (default: all available)
    **kwargs : additional arguments
        
    Returns:
    -------
    Dict[str, Dict[str, TrainingResult]]
        Training results by asset and model
    """
    if model_names is None:
        model_names = ['linear', 'ridge', 'random_forest', 'xgboost', 'arima']
    
    trainer = ModelTrainer(config)
    
    return trainer.train_asset_models(
        featured_data=featured_data,
        model_names=model_names,
        **kwargs
    )

__version__ = "1.0.0"

__all__ = [
    'BaseTimeSeriesModel',
    'SklearnTimeSeriesModel', 
    'ModelMetadata',
    'PredictionResult',
    'ModelTrainer',
    'TrainingResult',
    'TimeSeriesValidator',
    'ValidationResult',
    'LinearRegressionModel',
    'RidgeModel',
    'LassoModel', 
    'RandomForestModel',
    'XGBoostModel',
    'ARIMAModel',
    'SARIMAModel',
    'train_models'
]