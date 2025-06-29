"""
Comprehensive time series evaluation metrics.

This module provides specialized metrics for evaluating time series
forecasting model performance.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TimeSeriesMetrics:
    """Comprehensive time series evaluation metrics."""
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                             y_pred: np.ndarray,
                             sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all available time series metrics.
        
        Parameters:
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        sample_weight : np.ndarray, optional
            Sample weights for weighted metrics
            
        Returns:
        -------
        Dict[str, float]
            Dictionary of all calculated metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if sample_weight is not None:
            sample_weight_clean = sample_weight[mask]
        else:
            sample_weight_clean = None
        
        if len(y_true_clean) == 0:
            logger.warning("No valid predictions for metric calculation")
            return {}
        
        metrics = {}
        
        # Basic metrics
        metrics.update(TimeSeriesMetrics._calculate_basic_metrics(
            y_true_clean, y_pred_clean, sample_weight_clean
        ))
        
        # Percentage-based metrics
        metrics.update(TimeSeriesMetrics._calculate_percentage_metrics(
            y_true_clean, y_pred_clean, sample_weight_clean
        ))
        
        # Scaled metrics
        metrics.update(TimeSeriesMetrics._calculate_scaled_metrics(
            y_true_clean, y_pred_clean, sample_weight_clean
        ))
        
        # Directional metrics
        metrics.update(TimeSeriesMetrics._calculate_directional_metrics(
            y_true_clean, y_pred_clean
        ))
        
        # Distribution metrics
        metrics.update(TimeSeriesMetrics._calculate_distribution_metrics(
            y_true_clean, y_pred_clean
        ))
        
        # Threshold-based metrics
        metrics.update(TimeSeriesMetrics._calculate_threshold_metrics(
            y_true_clean, y_pred_clean
        ))
        
        return metrics
    
    @staticmethod
    def _calculate_basic_metrics(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate basic regression metrics."""
        from sklearn.metrics import (
            mean_absolute_error, mean_squared_error, r2_score,
            mean_absolute_percentage_error
        )
        
        metrics = {}
        
        # Core metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['mse'] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)
        
        # Mean Absolute Percentage Error
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mape = mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
                if not np.isnan(mape) and not np.isinf(mape):
                    metrics['mape'] = mape * 100
            except:
                pass
        
        # Additional basic metrics
        residuals = y_true - y_pred
        
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        metrics['median_absolute_error'] = np.median(np.abs(residuals))
        
        # Relative metrics
        if np.mean(np.abs(y_true)) > 0:
            metrics['relative_mae'] = metrics['mae'] / np.mean(np.abs(y_true))
            metrics['relative_rmse'] = metrics['rmse'] / np.mean(np.abs(y_true))
        
        return metrics
    
    @staticmethod
    def _calculate_percentage_metrics(y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate percentage-based metrics."""
        metrics = {}
        
        # Symmetric Mean Absolute Percentage Error
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if mask.any():
            if sample_weight is not None:
                weights = sample_weight[mask]
                smape = np.average(
                    np.abs(y_true[mask] - y_pred[mask]) / denominator[mask],
                    weights=weights
                ) * 100
            else:
                smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
            
            metrics['smape'] = smape
        
        # Mean Absolute Percentage Deviation
        if np.mean(y_true) != 0:
            mapd = np.mean(np.abs(y_true - y_pred)) / np.mean(y_true) * 100
            metrics['mapd'] = mapd
        
        # Weighted Mean Absolute Percentage Error (if y_true != 0)
        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            wmape = np.sum(np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask])) / np.sum(np.abs(y_true[nonzero_mask])) * 100
            metrics['wmape'] = wmape
        
        return metrics
    
    @staticmethod
    def _calculate_scaled_metrics(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate scaled metrics."""
        metrics = {}
        
        # Mean Absolute Scaled Error (MASE)
        if len(y_true) >= 2:
            # Use naive forecast (previous value) as baseline
            naive_forecast = y_true[:-1]
            actual_next = y_true[1:]
            naive_mae = np.mean(np.abs(actual_next - naive_forecast))
            
            if naive_mae > 0:
                forecast_mae = np.mean(np.abs(y_true - y_pred))
                metrics['mase'] = forecast_mae / naive_mae
        
        # Theil's U statistic
        if len(y_true) >= 2:
            mse_forecast = np.mean((y_true - y_pred) ** 2)
            
            # Naive forecast MSE
            naive_forecast = y_true[:-1]
            actual_next = y_true[1:]
            mse_naive = np.mean((actual_next - naive_forecast) ** 2)
            
            if mse_naive > 0:
                metrics['theil_u'] = np.sqrt(mse_forecast) / np.sqrt(mse_naive)
        
        # Normalized metrics
        y_range = np.max(y_true) - np.min(y_true)
        if y_range > 0:
            metrics['normalized_mae'] = np.mean(np.abs(y_true - y_pred)) / y_range
            metrics['normalized_rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2)) / y_range
        
        return metrics
    
    @staticmethod
    def _calculate_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate directional accuracy metrics."""
        metrics = {}
        
        if len(y_true) >= 2:
            # Directional accuracy
            actual_direction = np.diff(y_true) > 0
            predicted_direction = np.diff(y_pred) > 0
            
            correct_directions = np.sum(actual_direction == predicted_direction)
            total_directions = len(actual_direction)
            
            if total_directions > 0:
                metrics['directional_accuracy'] = (correct_directions / total_directions) * 100
            
            # Hit rate (percentage of predictions within tolerance)
            for tolerance in [0.05, 0.10, 0.20]:  # 5%, 10%, 20% tolerance
                if np.mean(np.abs(y_true)) > 0:
                    relative_errors = np.abs(y_true - y_pred) / np.abs(y_true)
                    hit_rate = np.mean(relative_errors <= tolerance) * 100
                    metrics[f'hit_rate_{int(tolerance*100)}pct'] = hit_rate
        
        return metrics
    
    @staticmethod
    def _calculate_distribution_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate distribution-based metrics."""
        metrics = {}
        
        residuals = y_true - y_pred
        
        # Residual distribution statistics
        if len(residuals) >= 3:
            metrics['residual_skewness'] = stats.skew(residuals)
            
        if len(residuals) >= 4:
            metrics['residual_kurtosis'] = stats.kurtosis(residuals)  # Excess kurtosis
        
        # Normality tests
        if len(residuals) >= 8:  # Minimum for Shapiro-Wilk
            try:
                _, p_value = stats.shapiro(residuals)
                metrics['shapiro_p_value'] = p_value
                metrics['residuals_normal'] = p_value > 0.05  # Null hypothesis: normal distribution
            except:
                pass
        
        # Prediction vs actual distribution comparison
        if len(y_true) >= 5 and len(y_pred) >= 5:
            try:
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.ks_2samp(y_true, y_pred)
                metrics['ks_statistic'] = ks_statistic
                metrics['ks_p_value'] = ks_p_value
                metrics['distributions_similar'] = ks_p_value > 0.05
                
                # Mann-Whitney U test (non-parametric)
                u_statistic, u_p_value = stats.mannwhitneyu(y_true, y_pred, alternative='two-sided')
                metrics['mannwhitney_p_value'] = u_p_value
            except:
                pass
        
        return metrics
    
    @staticmethod
    def _calculate_threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate threshold-based classification metrics."""
        metrics = {}
        
        if len(y_true) >= 2:
            # Convert to binary classification problem based on direction
            actual_up = np.diff(y_true) > 0
            predicted_up = np.diff(y_pred) > 0
            
            if len(actual_up) > 0:
                # Classification metrics
                tp = np.sum(actual_up & predicted_up)
                tn = np.sum(~actual_up & ~predicted_up)
                fp = np.sum(~actual_up & predicted_up)
                fn = np.sum(actual_up & ~predicted_up)
                
                total = tp + tn + fp + fn
                
                if total > 0:
                    accuracy = (tp + tn) / total
                    metrics['classification_accuracy'] = accuracy * 100
                
                if (tp + fp) > 0:
                    precision = tp / (tp + fp)
                    metrics['precision'] = precision * 100
                
                if (tp + fn) > 0:
                    recall = tp / (tp + fn)
                    metrics['recall'] = recall * 100
                
                if 'precision' in metrics and 'recall' in metrics and (metrics['precision'] + metrics['recall']) > 0:
                    f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                    metrics['f1_score'] = f1
        
        return metrics
    
    @staticmethod
    def calculate_confidence_intervals(residuals: np.ndarray, 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence intervals for residuals.
        
        Parameters:
        ----------
        residuals : np.ndarray
            Model residuals
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
            
        Returns:
        -------
        Tuple[float, float]
            Lower and upper confidence bounds
        """
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(residuals, lower_percentile)
        upper_bound = np.percentile(residuals, upper_percentile)
        
        return lower_bound, upper_bound
    
    @staticmethod
    def calculate_prediction_intervals(predictions: np.ndarray,
                                     residuals: np.ndarray,
                                     confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals.
        
        Parameters:
        ----------
        predictions : np.ndarray
            Model predictions
        residuals : np.ndarray
            Historical residuals
        confidence_level : float
            Confidence level
            
        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            Lower and upper prediction intervals
        """
        lower_ci, upper_ci = TimeSeriesMetrics.calculate_confidence_intervals(residuals, confidence_level)
        
        lower_intervals = predictions + lower_ci
        upper_intervals = predictions + upper_ci
        
        return lower_intervals, upper_intervals
    
    @staticmethod
    def rolling_metrics(y_true: np.ndarray, 
                       y_pred: np.ndarray,
                       window_size: int = 30,
                       metrics: List[str] = None) -> pd.DataFrame:
        """
        Calculate rolling metrics over time.
        
        Parameters:
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        window_size : int
            Rolling window size
        metrics : List[str], optional
            List of metrics to calculate
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ['mae', 'rmse', 'mape']
        
        n_points = len(y_true)
        rolling_results = []
        
        for i in range(window_size, n_points + 1):
            window_true = y_true[i-window_size:i]
            window_pred = y_pred[i-window_size:i]
            
            window_metrics = TimeSeriesMetrics.calculate_all_metrics(window_true, window_pred)
            
            # Only keep requested metrics
            filtered_metrics = {k: v for k, v in window_metrics.items() if k in metrics}
            filtered_metrics['window_end'] = i
            
            rolling_results.append(filtered_metrics)
        
        return pd.DataFrame(rolling_results)
    
    @staticmethod
    def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models' performance metrics.
        
        Parameters:
        ----------
        results_dict : Dict[str, Dict[str, float]]
            Dictionary of model results: {model_name: {metric: value}}
            
        Returns:
        -------
        pd.DataFrame
            Comparison table with models as rows and metrics as columns
        """
        comparison_df = pd.DataFrame(results_dict).T
        
        # Add ranking for key metrics (lower is better for error metrics)
        error_metrics = ['mae', 'rmse', 'mse', 'mape', 'smape']
        accuracy_metrics = ['r2', 'directional_accuracy']
        
        for metric in comparison_df.columns:
            if metric in error_metrics:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=True)
            elif metric in accuracy_metrics:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
        
        return comparison_df