"""
Consolidated ML training, validation, and hyperparameter tuning.

This module contains all training-related functionality including
model training, time series validation, and hyperparameter optimization.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from itertools import product
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ParameterGrid, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)

from ..pipeline.config import PipelineConfig
from .base import BaseTimeSeriesModel, create_model_factory

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION CLASSES AND FUNCTIONS
# =============================================================================

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


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""
    
    best_params: Dict[str, Any]
    best_score: float
    best_model: Optional[BaseTimeSeriesModel]
    all_results: List[Dict[str, Any]]
    tuning_metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class HyperparameterTuner:
    """Hyperparameter tuning for time series models."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize hyperparameter tuner.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        """
        self.config = config
        self.validator = TimeSeriesValidator(config)
        
    def get_default_param_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get default parameter grids for different model types.
        
        Returns:
        -------
        Dict[str, Dict[str, List]]
            Parameter grids by model type
        """
        return {
            'linear': {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'fit_intercept': [True, False],
                'max_iter': [1000, 2000]
            },
            'lasso': {
                'alpha': [0.01, 0.1, 1.0, 10.0],
                'fit_intercept': [True, False],
                'max_iter': [1000, 2000, 5000],
                'tol': [1e-4, 1e-3]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'reg_alpha': [0, 0.1, 1.0],
                'reg_lambda': [1, 1.5, 2.0]
            },
            'arima': {
                'order': [(1,1,1), (2,1,1), (1,1,2), (2,1,2), (3,1,1), (1,1,3)],
                'auto_order': [False],  # Use manual orders for grid search
                'seasonal': [False, True]
            },
            'sarima': {
                'order': [(1,1,1), (2,1,1), (1,1,2)],
                'seasonal_order': [(1,1,1,12), (1,1,1,7), (0,1,1,12)],
                'auto_order': [False]  # Use manual orders for grid search
            }
        }
    
    def grid_search(self, 
                   model_class: type,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_val: pd.DataFrame,
                   y_val: pd.Series,
                   param_grid: Dict[str, List],
                   scoring: str = 'rmse',
                   cv_folds: Optional[int] = None) -> TuningResult:
        """
        Perform grid search hyperparameter tuning.
        
        Parameters:
        ----------
        model_class : type
            Model class to tune
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        param_grid : Dict[str, List]
            Parameter grid to search
        scoring : str
            Scoring metric (lower is better)
        cv_folds : int, optional
            Number of CV folds (if None, uses validation set)
            
        Returns:
        -------
        TuningResult
            Tuning results with best parameters
        """
        logger.info(f"Starting grid search with {len(ParameterGrid(param_grid))} parameter combinations")
        
        try:
            all_results = []
            best_score = float('inf')
            best_params = None
            best_model = None
            
            for params in ParameterGrid(param_grid):
                logger.debug(f"Testing parameters: {params}")
                
                try:
                    # Create and train model
                    model = model_class(self.config, **params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    if cv_folds:
                        # Use cross-validation
                        val_result = self.validator.cross_validate(
                            model_class, X_train, y_train, n_splits=cv_folds, model_params=params
                        )
                    else:
                        # Use validation set
                        val_result = self.validator.validate_model(model, X_val, y_val)
                    
                    score = val_result.metrics.get(scoring, float('inf'))
                    
                    result = {
                        'params': params,
                        'score': score,
                        'metrics': val_result.metrics
                    }
                    
                    all_results.append(result)
                    
                    # Track best model
                    if score < best_score:
                        best_score = score
                        best_params = params
                        best_model = model
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate parameters {params}: {str(e)}")
                    all_results.append({
                        'params': params,
                        'score': float('inf'),
                        'error': str(e)
                    })
                    continue
            
            if best_params is None:
                raise ValueError("No valid parameter combinations found")
            
            tuning_metadata = {
                'total_combinations': len(all_results),
                'successful_combinations': sum(1 for r in all_results if 'error' not in r),
                'scoring_metric': scoring,
                'search_method': 'grid_search'
            }
            
            logger.info(f"Grid search complete. Best {scoring}: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return TuningResult(
                best_params=best_params,
                best_score=best_score,
                best_model=best_model,
                all_results=all_results,
                tuning_metadata=tuning_metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Grid search failed: {str(e)}")
            return TuningResult(
                best_params={},
                best_score=float('inf'),
                best_model=None,
                all_results=[],
                tuning_metadata={'error': str(e)},
                success=False,
                error_message=str(e)
            )
    
    def random_search(self, 
                     model_class: type,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     param_distributions: Dict[str, Any],
                     n_iter: int = 20,
                     scoring: str = 'rmse',
                     random_state: Optional[int] = None) -> TuningResult:
        """
        Perform random search hyperparameter tuning.
        
        Parameters:
        ----------
        model_class : type
            Model class to tune
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        param_distributions : Dict[str, Any]
            Parameter distributions to sample from
        n_iter : int
            Number of parameter settings to sample
        scoring : str
            Scoring metric
        random_state : int, optional
            Random state for reproducibility
            
        Returns:
        -------
        TuningResult
            Tuning results with best parameters
        """
        logger.info(f"Starting random search with {n_iter} iterations")
        
        try:
            if random_state:
                np.random.seed(random_state)
            
            all_results = []
            best_score = float('inf')
            best_params = None
            best_model = None
            
            for i in range(n_iter):
                # Sample parameters
                params = {}
                for param_name, param_dist in param_distributions.items():
                    if isinstance(param_dist, list):
                        params[param_name] = np.random.choice(param_dist)
                    elif hasattr(param_dist, 'rvs'):  # scipy distribution
                        params[param_name] = param_dist.rvs()
                    else:
                        params[param_name] = param_dist
                
                logger.debug(f"Iteration {i+1}/{n_iter}: {params}")
                
                try:
                    # Create and train model
                    model = model_class(self.config, **params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    val_result = self.validator.validate_model(model, X_val, y_val)
                    score = val_result.metrics.get(scoring, float('inf'))
                    
                    result = {
                        'params': params,
                        'score': score,
                        'metrics': val_result.metrics
                    }
                    
                    all_results.append(result)
                    
                    # Track best model
                    if score < best_score:
                        best_score = score
                        best_params = params
                        best_model = model
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate iteration {i+1}: {str(e)}")
                    all_results.append({
                        'params': params,
                        'score': float('inf'),
                        'error': str(e)
                    })
                    continue
            
            if best_params is None:
                raise ValueError("No valid parameter combinations found")
            
            tuning_metadata = {
                'total_iterations': n_iter,
                'successful_iterations': sum(1 for r in all_results if 'error' not in r),
                'scoring_metric': scoring,
                'search_method': 'random_search',
                'random_state': random_state
            }
            
            logger.info(f"Random search complete. Best {scoring}: {best_score:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return TuningResult(
                best_params=best_params,
                best_score=best_score,
                best_model=best_model,
                all_results=all_results,
                tuning_metadata=tuning_metadata,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Random search failed: {str(e)}")
            return TuningResult(
                best_params={},
                best_score=float('inf'),
                best_model=None,
                all_results=[],
                tuning_metadata={'error': str(e)},
                success=False,
                error_message=str(e)
            )
    
    def tune_model(self, 
                  model_name: str,
                  model_class: type,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: pd.DataFrame,
                  y_val: pd.Series,
                  method: str = 'grid_search',
                  param_grid: Optional[Dict[str, List]] = None,
                  scoring: str = 'rmse',
                  **kwargs) -> TuningResult:
        """
        Tune hyperparameters for a specific model.
        
        Parameters:
        ----------
        model_name : str
            Name of the model
        model_class : type
            Model class to tune
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        method : str
            Tuning method ('grid_search', 'random_search')
        param_grid : Dict[str, List], optional
            Parameter grid (uses default if None)
        scoring : str
            Scoring metric
        **kwargs : additional arguments for tuning method
            
        Returns:
        -------
        TuningResult
            Tuning results
        """
        # Get parameter grid if not provided
        if param_grid is None:
            default_grids = self.get_default_param_grids()
            param_grid = default_grids.get(model_name, {})
            
            if not param_grid:
                logger.warning(f"No default parameter grid for {model_name}, using empty grid")
                return TuningResult(
                    best_params={},
                    best_score=float('inf'),
                    best_model=None,
                    all_results=[],
                    tuning_metadata={'error': 'No parameter grid provided'},
                    success=False,
                    error_message="No parameter grid available for this model"
                )
        
        # Perform tuning based on method
        if method == 'grid_search':
            return self.grid_search(
                model_class, X_train, y_train, X_val, y_val, 
                param_grid, scoring, **kwargs
            )
        elif method == 'random_search':
            return self.random_search(
                model_class, X_train, y_train, X_val, y_val, 
                param_grid, scoring=scoring, **kwargs
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")


# =============================================================================
# TRAINING ORCHESTRATION
# =============================================================================

@dataclass
class TrainingResult:
    """Result of model training process."""
    
    model_name: str
    model: BaseTimeSeriesModel
    training_time: float
    validation_result: Optional[ValidationResult]
    feature_importance: Optional[Dict[str, float]]
    hyperparameters: Dict[str, Any]
    training_metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class ModelTrainer:
    """Main training orchestrator for time series models."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize model trainer.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        """
        self.config = config
        self.validator = TimeSeriesValidator(config)
        self.model_factory = create_model_factory(config)
        self.trained_models = {}
        
    def prepare_time_series_split(self, 
                                 df: pd.DataFrame,
                                 target_column: str = 'target',
                                 test_size: float = None,
                                 validation_size: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                                       pd.Series, pd.Series, pd.Series]:
        """
        Create time-aware train/validation/test splits.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Feature DataFrame with datetime index
        target_column : str
            Name of target column
        test_size : float, optional
            Proportion for test set
        validation_size : float, optional
            Proportion for validation set
            
        Returns:
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        test_size = test_size or self.config.modeling.test_size
        validation_size = validation_size or self.config.modeling.validation_size
        
        # Ensure datetime index is sorted
        df_sorted = df.sort_index()
        
        # Remove rows with missing target
        df_clean = df_sorted.dropna(subset=[target_column])
        
        if len(df_clean) == 0:
            raise ValueError("No valid samples after removing missing targets")
        
        n_samples = len(df_clean)
        
        # Calculate split indices (time-based)
        test_start_idx = int(n_samples * (1 - test_size))
        val_start_idx = int(n_samples * (1 - test_size - validation_size))
        
        # Create splits
        train_data = df_clean.iloc[:val_start_idx]
        val_data = df_clean.iloc[val_start_idx:test_start_idx] 
        test_data = df_clean.iloc[test_start_idx:]
        
        # Separate features and target
        feature_cols = [col for col in df_clean.columns if col != target_column]
        
        X_train = train_data[feature_cols]
        X_val = val_data[feature_cols]
        X_test = test_data[feature_cols]
        
        y_train = train_data[target_column]
        y_val = val_data[target_column]
        y_test = test_data[target_column]
        
        logger.info(f"Time series split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        logger.info(f"Date ranges - Train: {X_train.index.min()} to {X_train.index.max()}")
        logger.info(f"             Val: {X_val.index.min()} to {X_val.index.max()}")
        logger.info(f"             Test: {X_test.index.min()} to {X_test.index.max()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_single_model(self, 
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          model_params: Optional[Dict[str, Any]] = None,
                          validate: bool = True) -> TrainingResult:
        """
        Train a single model.
        
        Parameters:
        ----------
        model_name : str
            Name of model to train
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        model_params : Dict[str, Any], optional
            Model-specific parameters
        validate : bool
            Whether to perform validation
            
        Returns:
        -------
        TrainingResult
            Training result with model and metrics
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Training {model_name} model...")
            
            # Get model class
            if model_name not in self.model_factory:
                raise ValueError(f"Unknown model type: {model_name}")
            
            model_class = self.model_factory[model_name]
            
            # Create model instance with parameters
            if model_params:
                model = model_class(self.config, **model_params)
            else:
                model = model_class(self.config)
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Fit the model
            model.fit(X_train, y_train, validation_data=validation_data)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Perform validation if requested
            validation_result = None
            if validate and X_val is not None and y_val is not None:
                validation_result = self.validator.validate_model(model, X_val, y_val)
            
            # Get feature importance
            feature_importance = model.get_feature_importance()
            
            # Create training metadata
            training_metadata = {
                'training_samples': len(X_train),
                'validation_samples': len(X_val) if X_val is not None else 0,
                'training_date': start_time.isoformat(),
                'training_time_seconds': training_time,
                'feature_count': len(X_train.columns)
            }
            
            logger.info(f"Successfully trained {model_name} in {training_time:.2f} seconds")
            
            return TrainingResult(
                model_name=model_name,
                model=model,
                training_time=training_time,
                validation_result=validation_result,
                feature_importance=feature_importance,
                hyperparameters=model.get_hyperparameters(),
                training_metadata=training_metadata,
                success=True
            )
            
        except Exception as e:
            training_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to train {model_name}: {str(e)}")
            
            return TrainingResult(
                model_name=model_name,
                model=None,
                training_time=training_time,
                validation_result=None,
                feature_importance=None,
                hyperparameters={},
                training_metadata={'error': str(e)},
                success=False,
                error_message=str(e)
            )
    
    def train_multiple_models(self, 
                             model_names: List[str],
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_val: Optional[pd.DataFrame] = None,
                             y_val: Optional[pd.Series] = None,
                             model_params: Optional[Dict[str, Dict[str, Any]]] = None,
                             validate: bool = True) -> Dict[str, TrainingResult]:
        """
        Train multiple models with the same data.
        
        Parameters:
        ----------
        model_names : List[str]
            List of model names to train
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features
        y_val : pd.Series, optional
            Validation target
        model_params : Dict[str, Dict[str, Any]], optional
            Model-specific parameters for each model
        validate : bool
            Whether to perform validation
            
        Returns:
        -------
        Dict[str, TrainingResult]
            Dictionary of training results by model name
        """
        results = {}
        model_params = model_params or {}
        
        for model_name in model_names:
            params = model_params.get(model_name, {})
            
            result = self.train_single_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_params=params,
                validate=validate
            )
            
            results[model_name] = result
            
            # Store successful models
            if result.success:
                self.trained_models[model_name] = result.model
        
        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Training complete: {successful}/{len(model_names)} models successful")
        
        return results
    
    def train_asset_models(self, 
                          featured_data: Dict[str, pd.DataFrame],
                          model_names: List[str],
                          target_column: str = 'target',
                          model_params: Optional[Dict[str, Dict[str, Any]]] = None,
                          max_assets: Optional[int] = None) -> Dict[str, Dict[str, TrainingResult]]:
        """
        Train models for multiple assets.
        
        Parameters:
        ----------
        featured_data : Dict[str, pd.DataFrame]
            Dictionary of featured data by asset
        model_names : List[str]
            List of model names to train
        target_column : str
            Name of target column
        model_params : Dict[str, Dict[str, Any]], optional
            Model-specific parameters
        max_assets : int, optional
            Maximum number of assets to process
            
        Returns:
        -------
        Dict[str, Dict[str, TrainingResult]]
            Nested dictionary: {asset_name: {model_name: TrainingResult}}
        """
        asset_results = {}
        assets_to_process = list(featured_data.keys())
        
        if max_assets:
            assets_to_process = assets_to_process[:max_assets]
        
        for asset_name in assets_to_process:
            logger.info(f"Training models for asset: {asset_name}")
            
            df = featured_data[asset_name]
            
            # Check if we have enough data
            if len(df) < 50:  # Minimum samples for meaningful splits
                logger.warning(f"Insufficient data for {asset_name}: {len(df)} samples")
                continue
            
            try:
                # Create time series splits
                X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_time_series_split(
                    df, target_column
                )
                
                # Train models for this asset
                asset_results[asset_name] = self.train_multiple_models(
                    model_names=model_names,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_params=model_params,
                    validate=True
                )
                
            except Exception as e:
                logger.error(f"Failed to train models for {asset_name}: {str(e)}")
                asset_results[asset_name] = {}
        
        logger.info(f"Completed training for {len(asset_results)} assets")
        return asset_results
    
    def get_best_model(self, 
                      training_results: Dict[str, TrainingResult],
                      metric: str = 'rmse') -> Optional[TrainingResult]:
        """
        Get the best performing model based on validation metric.
        
        Parameters:
        ----------
        training_results : Dict[str, TrainingResult]
            Dictionary of training results
        metric : str
            Metric to use for comparison
            
        Returns:
        -------
        Optional[TrainingResult]
            Best performing model result
        """
        best_result = None
        best_score = float('inf')
        
        for result in training_results.values():
            if not result.success or result.validation_result is None:
                continue
                
            score = result.validation_result.metrics.get(metric)
            if score is not None and score < best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def save_training_results(self, 
                             training_results: Union[Dict[str, TrainingResult], 
                                                   Dict[str, Dict[str, TrainingResult]]],
                             output_dir: str = "data/output/models") -> None:
        """
        Save training results and models to disk.
        
        Parameters:
        ----------
        training_results : Union[Dict[str, TrainingResult], Dict[str, Dict[str, TrainingResult]]]
            Training results to save
        output_dir : str
            Output directory for models
        """
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Handle nested results (multi-asset)
        if isinstance(next(iter(training_results.values())), dict):
            # Multi-asset results
            for asset_name, asset_results in training_results.items():
                asset_dir = output_path / asset_name
                asset_dir.mkdir(exist_ok=True)
                
                for model_name, result in asset_results.items():
                    if result.success and result.model:
                        model_file = asset_dir / f"{model_name}.pkl"
                        result.model.save_model(model_file)
                        
                        # Save training summary
                        summary_file = asset_dir / f"{model_name}_summary.json"
                        summary = {
                            'model_name': result.model_name,
                            'training_time': result.training_time,
                            'hyperparameters': result.hyperparameters,
                            'training_metadata': result.training_metadata,
                            'validation_metrics': result.validation_result.metrics if result.validation_result else None
                        }
                        
                        with open(summary_file, 'w') as f:
                            json.dump(summary, f, indent=2, default=str)
        else:
            # Single asset results
            for model_name, result in training_results.items():
                if result.success and result.model:
                    model_file = output_path / f"{model_name}.pkl"
                    result.model.save_model(model_file)
        
        logger.info(f"Training results saved to {output_path}")
    
    def generate_training_report(self, 
                               training_results: Dict[str, TrainingResult]) -> Dict[str, Any]:
        """
        Generate comprehensive training report.
        
        Parameters:
        ----------
        training_results : Dict[str, TrainingResult]
            Training results to summarize
            
        Returns:
        -------
        Dict[str, Any]
            Training report with summaries and comparisons
        """
        successful_models = {name: result for name, result in training_results.items() if result.success}
        
        report = {
            'summary': {
                'total_models': len(training_results),
                'successful_models': len(successful_models),
                'failed_models': len(training_results) - len(successful_models),
                'training_date': datetime.now().isoformat()
            },
            'model_performance': {},
            'model_timings': {},
            'feature_importance_summary': {}
        }
        
        # Performance comparison
        for name, result in successful_models.items():
            if result.validation_result:
                report['model_performance'][name] = result.validation_result.metrics
            
            report['model_timings'][name] = result.training_time
            
            if result.feature_importance:
                # Get top 10 most important features
                sorted_features = sorted(
                    result.feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:10]
                report['feature_importance_summary'][name] = dict(sorted_features)
        
        # Find best model by validation MAE
        best_model = self.get_best_model(successful_models, 'mae')
        if best_model:
            report['best_model'] = {
                'name': best_model.model_name,
                'mae': best_model.validation_result.metrics.get('mae') if best_model.validation_result else None,
                'training_time': best_model.training_time
            }
        
        return report