"""
Model training orchestration and coordination.

This module provides the main training interface that coordinates
model fitting, validation, and evaluation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ...pipeline.config import PipelineConfig
from ..base import BaseTimeSeriesModel, create_model_factory
from .validator import TimeSeriesValidator, ValidationResult

logger = logging.getLogger(__name__)


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
                      metric: str = 'mae') -> Optional[TrainingResult]:
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