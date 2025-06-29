"""
Base classes for machine learning models.

This module defines the core abstractions for time series forecasting models
that integrate with the data pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    
    model_name: str
    model_type: str
    asset_type: str
    features: List[str]
    target_column: str
    training_date: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    data_info: Dict[str, Any]


@dataclass
class PredictionResult:
    """Result of model prediction."""
    
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    prediction_dates: Optional[pd.DatetimeIndex] = None
    model_name: Optional[str] = None
    metadata: Optional[Dict] = None


class BaseTimeSeriesModel(ABC):
    """Abstract base class for time series forecasting models."""
    
    def __init__(self, config: PipelineConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self.model_type = self.__class__.__name__
        self.is_fitted = False
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_names = None
        self.metadata = None
        
    @abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the underlying model. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the underlying model. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available."""
        pass
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> 'BaseTimeSeriesModel':
        """
        Fit the model to training data.
        
        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        validation_data : Tuple[pd.DataFrame, pd.Series], optional
            Validation data for early stopping/monitoring
            
        Returns:
        -------
        BaseTimeSeriesModel
            Fitted model instance
        """
        logger.info(f"Training {self.model_name} with {len(X)} samples and {len(X.columns)} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Prepare data
        X_processed, y_processed = self._prepare_training_data(X, y)
        
        # Fit the model
        self._fit_model(X_processed, y_processed)
        
        # Create metadata
        self.metadata = ModelMetadata(
            model_name=self.model_name,
            model_type=self.model_type,
            asset_type=getattr(self.config, 'asset_type', 'unknown'),
            features=self.feature_names,
            target_column=y.name or 'target',
            training_date=pd.Timestamp.now().isoformat(),
            performance_metrics={},
            hyperparameters=self.get_hyperparameters(),
            data_info={
                'n_samples': len(X),
                'n_features': len(X.columns),
                'target_mean': float(y.mean()),
                'target_std': float(y.std()),
                'date_range': f"{X.index.min()} to {X.index.max()}"
            }
        )
        
        self.is_fitted = True
        logger.info(f"Model {self.model_name} training completed")
        
        return self
    
    def predict(self, 
                X: pd.DataFrame,
                return_confidence: bool = False) -> Union[np.ndarray, PredictionResult]:
        """
        Make predictions on new data.
        
        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix
        return_confidence : bool
            Whether to return confidence intervals
            
        Returns:
        -------
        Union[np.ndarray, PredictionResult]
            Predictions or full prediction result
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Prepare data
        X_processed = self._prepare_prediction_data(X)
        
        # Make predictions
        predictions = self._predict_model(X_processed)
        
        # Inverse transform if needed
        if self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        if not return_confidence:
            return predictions
            
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(X_processed, predictions)
        
        return PredictionResult(
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            prediction_dates=X.index if isinstance(X.index, pd.DatetimeIndex) else None,
            model_name=self.model_name,
            metadata={'model_type': self.model_type}
        )
    
    def _prepare_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training (scaling, etc.)."""
        
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Scale features if configured
        if self.config.modeling.scale_features:
            self.feature_scaler = StandardScaler()
            X_processed = pd.DataFrame(
                self.feature_scaler.fit_transform(X_processed),
                index=X_processed.index,
                columns=X_processed.columns
            )
            
        return X_processed, y_processed
    
    def _prepare_prediction_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for prediction."""
        
        X_processed = X.copy()
        
        # Ensure same features as training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X_processed.columns)
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X_processed[feature] = 0
            
            # Reorder columns to match training
            X_processed = X_processed[self.feature_names]
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Scale features if scaler exists
        if self.feature_scaler is not None:
            X_processed = pd.DataFrame(
                self.feature_scaler.transform(X_processed),
                index=X_processed.index,
                columns=X_processed.columns
            )
            
        return X_processed
    
    def _calculate_confidence_intervals(self, 
                                     X: pd.DataFrame, 
                                     predictions: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Calculate confidence intervals for predictions."""
        
        # Default implementation - can be overridden by subclasses
        # Simple approach: use historical error standard deviation
        if hasattr(self, '_training_residuals'):
            residual_std = np.std(self._training_residuals)
            lower = predictions - 1.96 * residual_std
            upper = predictions + 1.96 * residual_std
            return lower, upper
        
        return None
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {}
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Parameters:
        ----------
        file_path : Union[str, Path]
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
            
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model data
        model_data = {
            'model': self,
            'metadata': self.metadata,
            'feature_names': self.feature_names,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> 'BaseTimeSeriesModel':
        """
        Load a trained model from disk.
        
        Parameters:
        ----------
        file_path : Union[str, Path]
            Path to the saved model
            
        Returns:
        -------
        BaseTimeSeriesModel
            Loaded model instance
        """
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            
        model = model_data['model']
        model.metadata = model_data['metadata']
        model.feature_names = model_data['feature_names']
        model.feature_scaler = model_data['feature_scaler']
        model.target_scaler = model_data['target_scaler']
        
        logger.info(f"Model loaded from {file_path}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        
        info = {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'hyperparameters': self.get_hyperparameters()
        }
        
        if self.metadata:
            info['metadata'] = self.metadata.__dict__
            
        return info


class SklearnTimeSeriesModel(BaseTimeSeriesModel):
    """Base class for sklearn-based time series models."""
    
    def __init__(self, config: PipelineConfig, model_name: str, estimator: BaseEstimator):
        super().__init__(config, model_name)
        self.estimator = estimator
        self._training_residuals = None
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the sklearn estimator."""
        self.estimator.fit(X, y)
        
        # Store training residuals for confidence intervals
        train_predictions = self.estimator.predict(X)
        self._training_residuals = y.values - train_predictions
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with sklearn estimator."""
        return self.estimator.predict(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from sklearn estimator."""
        if hasattr(self.estimator, 'feature_importances_'):
            return dict(zip(self.feature_names, self.estimator.feature_importances_))
        elif hasattr(self.estimator, 'coef_'):
            return dict(zip(self.feature_names, np.abs(self.estimator.coef_)))
        return None
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get sklearn estimator parameters."""
        return self.estimator.get_params()


def create_model_factory(config: PipelineConfig) -> Dict[str, type]:
    """
    Create a factory of available model types.
    
    Parameters:
    ----------
    config : PipelineConfig
        Pipeline configuration
        
    Returns:
    -------
    Dict[str, type]
        Dictionary mapping model names to model classes
    """
    from .models.linear import LinearRegressionModel, RidgeModel, LassoModel
    from .models.ensemble import RandomForestModel, XGBoostModel
    from .models.time_series import ARIMAModel, SARIMAModel
    
    return {
        'linear': LinearRegressionModel,
        'ridge': RidgeModel,
        'lasso': LassoModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'arima': ARIMAModel,
        'sarima': SARIMAModel
    }