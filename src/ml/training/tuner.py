"""
Hyperparameter tuning for time series models.

This module provides automated hyperparameter optimization
using various search strategies.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from ...pipeline.config import PipelineConfig
from ..base import BaseTimeSeriesModel
from .validator import TimeSeriesValidator

logger = logging.getLogger(__name__)


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
                   scoring: str = 'mae',
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
                     scoring: str = 'mae',
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
    
    def bayesian_optimization(self, 
                            model_class: type,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: pd.DataFrame,
                            y_val: pd.Series,
                            param_bounds: Dict[str, Tuple[float, float]],
                            n_calls: int = 20,
                            scoring: str = 'mae') -> TuningResult:
        """
        Perform Bayesian optimization (requires scikit-optimize).
        
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
        param_bounds : Dict[str, Tuple[float, float]]
            Parameter bounds for optimization
        n_calls : int
            Number of function evaluations
        scoring : str
            Scoring metric
            
        Returns:
        -------
        TuningResult
            Tuning results with best parameters
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("scikit-optimize is required for Bayesian optimization")
            return TuningResult(
                best_params={},
                best_score=float('inf'),
                best_model=None,
                all_results=[],
                tuning_metadata={'error': 'scikit-optimize not available'},
                success=False,
                error_message="scikit-optimize is required for Bayesian optimization"
            )
        
        logger.info(f"Starting Bayesian optimization with {n_calls} calls")
        
        try:
            # Define search space
            dimensions = []
            param_names = []
            
            for param_name, (low, high) in param_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    dimensions.append(Integer(low, high, name=param_name))
                else:
                    dimensions.append(Real(low, high, name=param_name))
                param_names.append(param_name)
            
            all_results = []
            best_model = None
            
            @use_named_args(dimensions)
            def objective(**params):
                try:
                    # Create and train model
                    model = model_class(self.config, **params)
                    model.fit(X_train, y_train)
                    
                    # Evaluate model
                    val_result = self.validator.validate_model(model, X_val, y_val)
                    score = val_result.metrics.get(scoring, float('inf'))
                    
                    # Store results
                    result = {
                        'params': params,
                        'score': score,
                        'metrics': val_result.metrics
                    }
                    all_results.append(result)
                    
                    # Update best model
                    nonlocal best_model
                    if best_model is None or score < min(r['score'] for r in all_results[:-1]):
                        best_model = model
                    
                    return score
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate parameters {params}: {str(e)}")
                    all_results.append({
                        'params': params,
                        'score': float('inf'),
                        'error': str(e)
                    })
                    return float('inf')
            
            # Run optimization
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_calls,
                random_state=self.config.modeling.random_state
            )
            
            if not all_results or all(r.get('score') == float('inf') for r in all_results):
                raise ValueError("No valid parameter combinations found")
            
            # Get best results
            best_result = min(all_results, key=lambda x: x.get('score', float('inf')))
            best_params = best_result['params']
            best_score = best_result['score']
            
            tuning_metadata = {
                'total_calls': n_calls,
                'successful_calls': sum(1 for r in all_results if 'error' not in r),
                'scoring_metric': scoring,
                'search_method': 'bayesian_optimization',
                'convergence': result.func_vals if hasattr(result, 'func_vals') else None
            }
            
            logger.info(f"Bayesian optimization complete. Best {scoring}: {best_score:.4f}")
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
            logger.error(f"Bayesian optimization failed: {str(e)}")
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
                  scoring: str = 'mae',
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
            Tuning method ('grid_search', 'random_search', 'bayesian')
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
        elif method == 'bayesian':
            return self.bayesian_optimization(
                model_class, X_train, y_train, X_val, y_val, 
                param_grid, scoring=scoring, **kwargs
            )
        else:
            raise ValueError(f"Unknown tuning method: {method}")
    
    def tune_multiple_models(self, 
                           model_configs: Dict[str, Dict],
                           X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_val: pd.DataFrame,
                           y_val: pd.Series,
                           method: str = 'grid_search',
                           scoring: str = 'mae') -> Dict[str, TuningResult]:
        """
        Tune multiple models with their respective configurations.
        
        Parameters:
        ----------
        model_configs : Dict[str, Dict]
            Dictionary of model configurations
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame
            Validation features
        y_val : pd.Series
            Validation target
        method : str
            Tuning method
        scoring : str
            Scoring metric
            
        Returns:
        -------
        Dict[str, TuningResult]
            Tuning results for each model
        """
        from ..base import create_model_factory
        
        model_factory = create_model_factory(self.config)
        results = {}
        
        for model_name, config in model_configs.items():
            if model_name not in model_factory:
                logger.warning(f"Unknown model: {model_name}")
                continue
                
            logger.info(f"Tuning {model_name} using {method}")
            
            model_class = model_factory[model_name]
            param_grid = config.get('param_grid')
            
            result = self.tune_model(
                model_name=model_name,
                model_class=model_class,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                method=method,
                param_grid=param_grid,
                scoring=scoring
            )
            
            results[model_name] = result
        
        return results