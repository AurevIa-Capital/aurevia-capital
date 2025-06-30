"""
Model Factory for creating time series forecasting models.

This module implements the Factory Pattern to centralize model creation,
eliminate duplication, and improve maintainability.
"""

import logging
from typing import Dict, Type, List, Any, Optional
from abc import ABC, abstractmethod

from ..pipeline.config import PipelineConfig
from ..ml.base import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating time series forecasting models."""
    
    # Registry of available models
    _models: Dict[str, Type[BaseTimeSeriesModel]] = {}
    _model_categories: Dict[str, List[str]] = {
        'linear': ['linear', 'ridge', 'lasso', 'polynomial'],
        'ensemble': ['random_forest', 'xgboost', 'gradient_boosting'],
        'time_series': ['arima', 'sarima']
    }
    
    @classmethod
    def register_model(cls, 
                      name: str, 
                      model_class: Type[BaseTimeSeriesModel],
                      category: str = None) -> None:
        """
        Register a new model class with the factory.
        
        Parameters:
        ----------
        name : str
            Model identifier
        model_class : Type[BaseTimeSeriesModel]
            Model class to register
        category : str, optional
            Model category for organization
        """
        if not issubclass(model_class, BaseTimeSeriesModel):
            raise ValueError(f"Model class {model_class} must inherit from BaseTimeSeriesModel")
        
        cls._models[name] = model_class
        logger.info(f"Registered model: {name} -> {model_class.__name__}")
        
        # Add to category if specified
        if category:
            if category not in cls._model_categories:
                cls._model_categories[category] = []
            if name not in cls._model_categories[category]:
                cls._model_categories[category].append(name)
    
    @classmethod
    def create_model(cls, 
                    model_type: str, 
                    config: PipelineConfig,
                    **kwargs) -> BaseTimeSeriesModel:
        """
        Create a model instance using the factory.
        
        Parameters:
        ----------
        model_type : str
            Type of model to create
        config : PipelineConfig
            Pipeline configuration
        **kwargs
            Additional arguments for model constructor
            
        Returns:
        -------
        BaseTimeSeriesModel
            Instantiated model
        """
        if model_type not in cls._models:
            available_models = ', '.join(cls._models.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")
        
        model_class = cls._models[model_type]
        
        try:
            logger.info(f"Creating {model_type} model with config and kwargs: {list(kwargs.keys())}")
            return model_class(config, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {str(e)}")
            raise RuntimeError(f"Model creation failed for {model_type}: {str(e)}")
    
    @classmethod
    def create_multiple_models(cls, 
                             model_types: List[str], 
                             config: PipelineConfig,
                             model_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, BaseTimeSeriesModel]:
        """
        Create multiple model instances.
        
        Parameters:
        ----------
        model_types : List[str]
            List of model types to create
        config : PipelineConfig
            Pipeline configuration
        model_configs : Dict[str, Dict[str, Any]], optional
            Model-specific configurations
            
        Returns:
        -------
        Dict[str, BaseTimeSeriesModel]
            Dictionary of model instances
        """
        models = {}
        model_configs = model_configs or {}
        
        for model_type in model_types:
            try:
                model_kwargs = model_configs.get(model_type, {})
                models[model_type] = cls.create_model(model_type, config, **model_kwargs)
            except Exception as e:
                logger.error(f"Failed to create {model_type}: {str(e)}")
                continue
        
        logger.info(f"Created {len(models)} models: {list(models.keys())}")
        return models
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_categories(cls) -> Dict[str, List[str]]:
        """Get model categories and their associated models."""
        return cls._model_categories.copy()
    
    @classmethod
    def get_models_by_category(cls, category: str) -> List[str]:
        """Get models in a specific category."""
        return cls._model_categories.get(category, [])
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model type."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = cls._models[model_type]
        
        # Find category
        category = None
        for cat, models in cls._model_categories.items():
            if model_type in models:
                category = cat
                break
        
        return {
            'name': model_type,
            'class': model_class.__name__,
            'category': category,
            'module': model_class.__module__,
            'docstring': model_class.__doc__
        }
    
    @classmethod
    def validate_model_type(cls, model_type: str) -> bool:
        """Validate if a model type is available."""
        return model_type in cls._models
    
    @classmethod
    def get_default_config(cls, model_type: str) -> Dict[str, Any]:
        """Get default configuration for a model type."""
        defaults = {
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'polynomial': {'degree': 2},
            'random_forest': {'n_estimators': 100, 'max_depth': None},
            'xgboost': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1},
            'arima': {'order': (1, 1, 1), 'auto_order': True},
            'sarima': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12), 'auto_order': True}
        }
        
        return defaults.get(model_type, {})


class ModelConfigBuilder:
    """Builder class for model configurations."""
    
    def __init__(self):
        self.configs: Dict[str, Dict[str, Any]] = {}
    
    def add_model_config(self, model_type: str, **kwargs) -> 'ModelConfigBuilder':
        """Add configuration for a specific model type."""
        if not ModelFactory.validate_model_type(model_type):
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Start with defaults and override with provided kwargs
        default_config = ModelFactory.get_default_config(model_type)
        self.configs[model_type] = {**default_config, **kwargs}
        
        return self
    
    def add_linear_model_config(self, 
                               fit_intercept: bool = True,
                               normalize: bool = False) -> 'ModelConfigBuilder':
        """Add configuration for linear regression model."""
        return self.add_model_config('linear', 
                                   fit_intercept=fit_intercept,
                                   normalize=normalize)
    
    def add_ridge_model_config(self, 
                              alpha: float = 1.0,
                              fit_intercept: bool = True,
                              max_iter: int = 1000) -> 'ModelConfigBuilder':
        """Add configuration for Ridge regression model."""
        return self.add_model_config('ridge',
                                   alpha=alpha,
                                   fit_intercept=fit_intercept,
                                   max_iter=max_iter)
    
    def add_xgboost_model_config(self,
                                n_estimators: int = 100,
                                max_depth: int = 6,
                                learning_rate: float = 0.1,
                                subsample: float = 1.0,
                                colsample_bytree: float = 1.0) -> 'ModelConfigBuilder':
        """Add configuration for XGBoost model."""
        return self.add_model_config('xgboost',
                                   n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   learning_rate=learning_rate,
                                   subsample=subsample,
                                   colsample_bytree=colsample_bytree)
    
    def add_random_forest_config(self,
                               n_estimators: int = 100,
                               max_depth: Optional[int] = None,
                               min_samples_split: int = 2,
                               max_features: str = 'sqrt') -> 'ModelConfigBuilder':
        """Add configuration for Random Forest model."""
        return self.add_model_config('random_forest',
                                   n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   max_features=max_features)
    
    def build(self) -> Dict[str, Dict[str, Any]]:
        """Build the model configurations."""
        return self.configs.copy()
    
    def clear(self) -> 'ModelConfigBuilder':
        """Clear all configurations."""
        self.configs.clear()
        return self


def _register_default_models():
    """Register default models with the factory."""
    try:
        # Import model classes
        from ..ml.models import (
            LinearRegressionModel, RidgeModel, LassoModel, PolynomialRegressionModel,
            RandomForestModel, XGBoostModel, GradientBoostingModel,
            ARIMAModel, SARIMAModel
        )
        
        # Register linear models
        ModelFactory.register_model('linear', LinearRegressionModel, 'linear')
        ModelFactory.register_model('ridge', RidgeModel, 'linear')
        ModelFactory.register_model('lasso', LassoModel, 'linear')
        ModelFactory.register_model('polynomial', PolynomialRegressionModel, 'linear')
        
        # Register ensemble models
        ModelFactory.register_model('random_forest', RandomForestModel, 'ensemble')
        ModelFactory.register_model('xgboost', XGBoostModel, 'ensemble')
        ModelFactory.register_model('gradient_boosting', GradientBoostingModel, 'ensemble')
        
        # Register time series models
        ModelFactory.register_model('arima', ARIMAModel, 'time_series')
        ModelFactory.register_model('sarima', SARIMAModel, 'time_series')
        
        logger.info("Default models registered successfully")
        
    except ImportError as e:
        logger.error(f"Failed to register default models: {str(e)}")


# Register default models when module is imported
_register_default_models()