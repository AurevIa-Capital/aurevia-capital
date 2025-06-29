"""
Time series cross-validation and model validation.

This module provides time-aware validation techniques specifically
designed for time series forecasting models.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import TimeSeriesSplit

from ...pipeline.config import PipelineConfig
from ..base import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of model validation."""
    
    metrics: Dict[str, float]
    predictions: np.ndarray
    actuals: np.ndarray
    prediction_dates: Optional[pd.DatetimeIndex]
    cv_scores: Optional[Dict[str, List[float]]]
    directional_accuracy: Optional[float]
    residuals: np.ndarray
    validation_metadata: Dict[str, Any]


class TimeSeriesValidator:
    """Time series specific validation and cross-validation."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize time series validator.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        """
        self.config = config
        
    def calculate_metrics(self, 
                         y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         return_all: bool = True) -> Dict[str, float]:
        """
        Calculate comprehensive time series metrics.
        
        Parameters:
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        return_all : bool
            Whether to return all metrics or just core ones
            
        Returns:
        -------
        Dict[str, float]
            Dictionary of metric name to value
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions for metric calculation")
            return {}
        
        metrics = {}
        
        # Core regression metrics
        metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
        metrics['mse'] = mean_squared_error(y_true_clean, y_pred_clean)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
        
        # Mean Absolute Percentage Error (handle division by zero)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean)
                if not np.isnan(mape) and not np.isinf(mape):
                    metrics['mape'] = mape * 100  # Convert to percentage
            except:
                pass
        
        if return_all:
            # Additional time series specific metrics
            
            # Symmetric Mean Absolute Percentage Error
            smape = self._calculate_smape(y_true_clean, y_pred_clean)
            if not np.isnan(smape):
                metrics['smape'] = smape
            
            # Mean Absolute Scaled Error (if we have naive forecast baseline)
            mase = self._calculate_mase(y_true_clean, y_pred_clean)
            if not np.isnan(mase):
                metrics['mase'] = mase
            
            # Directional accuracy
            directional_acc = self._calculate_directional_accuracy(y_true_clean, y_pred_clean)
            if not np.isnan(directional_acc):
                metrics['directional_accuracy'] = directional_acc
            
            # Theil's U statistic
            theil_u = self._calculate_theil_u(y_true_clean, y_pred_clean)
            if not np.isnan(theil_u):
                metrics['theil_u'] = theil_u
            
            # Residual-based metrics
            residuals = y_true_clean - y_pred_clean
            metrics['residual_mean'] = np.mean(residuals)
            metrics['residual_std'] = np.std(residuals)
            metrics['residual_skewness'] = self._calculate_skewness(residuals)
            metrics['residual_kurtosis'] = self._calculate_kurtosis(residuals)
        
        return metrics
    
    def _calculate_smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not mask.any():
            return np.nan
            
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        return smape
    
    def _calculate_mase(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Scaled Error."""
        if len(y_true) < 2:
            return np.nan
            
        # Use naive forecast (previous value) as baseline
        naive_forecast = y_true[:-1]
        actual_next = y_true[1:]
        
        naive_mae = np.mean(np.abs(actual_next - naive_forecast))
        
        if naive_mae == 0:
            return np.nan
            
        forecast_mae = np.mean(np.abs(y_true - y_pred))
        mase = forecast_mae / naive_mae
        
        return mase
    
    def _calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy (percentage of correct direction predictions)."""
        if len(y_true) < 2:
            return np.nan
            
        # Calculate actual and predicted directions
        actual_direction = np.diff(y_true) > 0
        predicted_direction = np.diff(y_pred) > 0
        
        # Calculate accuracy
        correct_directions = np.sum(actual_direction == predicted_direction)
        total_directions = len(actual_direction)
        
        return (correct_directions / total_directions) * 100 if total_directions > 0 else np.nan
    
    def _calculate_theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Theil's U statistic."""
        if len(y_true) < 2:
            return np.nan
            
        # Theil's U = sqrt(MSE) / sqrt(MSE_naive)
        mse_forecast = np.mean((y_true - y_pred) ** 2)
        
        # Naive forecast MSE (using previous value)
        naive_forecast = y_true[:-1]
        actual_next = y_true[1:]
        mse_naive = np.mean((actual_next - naive_forecast) ** 2)
        
        if mse_naive == 0:
            return np.nan
            
        theil_u = np.sqrt(mse_forecast) / np.sqrt(mse_naive)
        return theil_u
    
    def _calculate_skewness(self, x: np.ndarray) -> float:
        """Calculate skewness of array."""
        if len(x) < 3:
            return np.nan
            
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)
        
        if std_x == 0:
            return np.nan
            
        skewness = np.mean(((x - mean_x) / std_x) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis of array."""
        if len(x) < 4:
            return np.nan
            
        mean_x = np.mean(x)
        std_x = np.std(x, ddof=1)
        
        if std_x == 0:
            return np.nan
            
        kurtosis = np.mean(((x - mean_x) / std_x) ** 4) - 3  # Excess kurtosis
        return kurtosis
    
    def validate_model(self, 
                      model: BaseTimeSeriesModel,
                      X_val: pd.DataFrame,
                      y_val: pd.Series,
                      return_predictions: bool = True) -> ValidationResult:
        """
        Validate a trained model on validation data.
        
        Parameters:
        ----------
        model : BaseTimeSeriesModel
            Trained model to validate
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation targets
        return_predictions : bool
            Whether to return full predictions
            
        Returns:
        -------
        ValidationResult
            Comprehensive validation results
        """
        logger.info(f"Validating {model.model_name} on {len(X_val)} samples")
        
        # Make predictions
        predictions = model.predict(X_val)
        actuals = y_val.values
        
        # Calculate metrics
        metrics = self.calculate_metrics(actuals, predictions, return_all=True)
        
        # Calculate directional accuracy separately
        directional_accuracy = self._calculate_directional_accuracy(actuals, predictions)
        
        # Calculate residuals
        residuals = actuals - predictions
        
        # Create validation metadata
        validation_metadata = {
            'validation_samples': len(X_val),
            'date_range': f"{X_val.index.min()} to {X_val.index.max()}",
            'model_type': model.model_type,
            'feature_count': len(X_val.columns)
        }
        
        result = ValidationResult(
            metrics=metrics,
            predictions=predictions if return_predictions else np.array([]),
            actuals=actuals if return_predictions else np.array([]),
            prediction_dates=X_val.index if isinstance(X_val.index, pd.DatetimeIndex) else None,
            cv_scores=None,  # Set by cross_validate if used
            directional_accuracy=directional_accuracy,
            residuals=residuals,
            validation_metadata=validation_metadata
        )
        
        logger.info(f"Validation complete - MAE: {metrics.get('mae', 'N/A'):.4f}, "
                   f"RMSE: {metrics.get('rmse', 'N/A'):.4f}, "
                   f"R²: {metrics.get('r2', 'N/A'):.4f}")
        
        return result
    
    def cross_validate(self, 
                      model_class: type,
                      X: pd.DataFrame,
                      y: pd.Series,
                      n_splits: int = None,
                      model_params: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Perform time series cross-validation.
        
        Parameters:
        ----------
        model_class : type
            Model class to instantiate
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        n_splits : int, optional
            Number of CV splits
        model_params : Dict[str, Any], optional
            Parameters for model instantiation
            
        Returns:
        -------
        ValidationResult
            Cross-validation results with CV scores
        """
        n_splits = n_splits or self.config.modeling.cross_validation_folds
        model_params = model_params or {}
        
        logger.info(f"Performing {n_splits}-fold time series cross-validation")
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None)
        
        cv_predictions = []
        cv_actuals = []
        cv_scores = {'mae': [], 'rmse': [], 'r2': [], 'mape': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.debug(f"CV fold {fold + 1}/{n_splits}")
            
            # Get fold data
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Create and train model
            model = model_class(self.config, **model_params)
            model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            fold_predictions = model.predict(X_val_fold)
            fold_actuals = y_val_fold.values
            
            # Calculate fold metrics
            fold_metrics = self.calculate_metrics(fold_actuals, fold_predictions, return_all=False)
            
            # Store results
            cv_predictions.extend(fold_predictions)
            cv_actuals.extend(fold_actuals)
            
            for metric in cv_scores.keys():
                if metric in fold_metrics:
                    cv_scores[metric].append(fold_metrics[metric])
        
        # Calculate overall metrics
        cv_predictions = np.array(cv_predictions)
        cv_actuals = np.array(cv_actuals)
        
        overall_metrics = self.calculate_metrics(cv_actuals, cv_predictions, return_all=True)
        
        # Add CV statistics
        for metric, scores in cv_scores.items():
            if scores:
                overall_metrics[f'{metric}_mean'] = np.mean(scores)
                overall_metrics[f'{metric}_std'] = np.std(scores)
        
        # Calculate directional accuracy
        directional_accuracy = self._calculate_directional_accuracy(cv_actuals, cv_predictions)
        
        validation_metadata = {
            'cv_folds': n_splits,
            'total_samples': len(X),
            'cv_method': 'TimeSeriesSplit'
        }
        
        result = ValidationResult(
            metrics=overall_metrics,
            predictions=cv_predictions,
            actuals=cv_actuals,
            prediction_dates=None,
            cv_scores=cv_scores,
            directional_accuracy=directional_accuracy,
            residuals=cv_actuals - cv_predictions,
            validation_metadata=validation_metadata
        )
        
        logger.info(f"Cross-validation complete - Mean MAE: {overall_metrics.get('mae_mean', 'N/A'):.4f} "
                   f"(±{overall_metrics.get('mae_std', 'N/A'):.4f})")
        
        return result
    
    def walk_forward_validation(self, 
                               model_class: type,
                               X: pd.DataFrame,
                               y: pd.Series,
                               min_train_size: int = 100,
                               step_size: int = 1,
                               model_params: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Perform walk-forward validation for time series.
        
        Parameters:
        ----------
        model_class : type
            Model class to instantiate
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        min_train_size : int
            Minimum training set size
        step_size : int
            Number of samples to advance in each step
        model_params : Dict[str, Any], optional
            Parameters for model instantiation
            
        Returns:
        -------
        ValidationResult
            Walk-forward validation results
        """
        model_params = model_params or {}
        
        logger.info(f"Performing walk-forward validation with min_train_size={min_train_size}")
        
        predictions = []
        actuals = []
        prediction_dates = []
        
        # Walk forward through the data
        for i in range(min_train_size, len(X), step_size):
            # Training data: all data up to current point
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            
            # Test data: next step_size points
            end_idx = min(i + step_size, len(X))
            X_test = X.iloc[i:end_idx]
            y_test = y.iloc[i:end_idx]
            
            if len(X_test) == 0:
                break
            
            try:
                # Train model
                model = model_class(self.config, **model_params)
                model.fit(X_train, y_train)
                
                # Make predictions
                step_predictions = model.predict(X_test)
                step_actuals = y_test.values
                
                predictions.extend(step_predictions)
                actuals.extend(step_actuals)
                
                if isinstance(X_test.index, pd.DatetimeIndex):
                    prediction_dates.extend(X_test.index)
                    
            except Exception as e:
                logger.warning(f"Walk-forward step failed at index {i}: {str(e)}")
                continue
        
        if not predictions:
            raise ValueError("Walk-forward validation failed - no successful predictions")
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        metrics = self.calculate_metrics(actuals, predictions, return_all=True)
        directional_accuracy = self._calculate_directional_accuracy(actuals, predictions)
        
        validation_metadata = {
            'method': 'walk_forward',
            'min_train_size': min_train_size,
            'step_size': step_size,
            'total_predictions': len(predictions),
            'total_steps': len(predictions) // step_size
        }
        
        result = ValidationResult(
            metrics=metrics,
            predictions=predictions,
            actuals=actuals,
            prediction_dates=pd.DatetimeIndex(prediction_dates) if prediction_dates else None,
            cv_scores=None,
            directional_accuracy=directional_accuracy,
            residuals=actuals - predictions,
            validation_metadata=validation_metadata
        )
        
        logger.info(f"Walk-forward validation complete - {len(predictions)} predictions made")
        
        return result