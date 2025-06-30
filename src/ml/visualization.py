"""
Consolidated ML visualization module for time series forecasting.

This module contains all visualization functionality including performance plots,
forecasting plots, comparison plots, and enhanced analytical visualizations.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..utils.visualization_utils import get_organized_visualization_path, get_organized_aggregate_path
from .training import ValidationResult, TrainingResult

logger = logging.getLogger(__name__)

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set style defaults
plt.style.use('default')
sns.set_palette("husl")


# =============================================================================
# BASE VISUALIZER CLASS
# =============================================================================

class BaseVisualizer:
    """Base class for visualization functionality."""
    
    def __init__(self, output_dir: str = "data/output/visualizations"):
        """
        Initialize base visualizer.
        
        Parameters:
        ----------
        output_dir : str
            Base output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'light': '#17becf',
            'dark': '#8c564b'
        }
        
        self.model_colors = {
            'linear': '#1f77b4',
            'ridge': '#ff7f0e',
            'lasso': '#2ca02c',
            'random_forest': '#d62728',
            'xgboost': '#9467bd',
            'arima': '#8c564b',
            'sarima': '#e377c2'
        }
    
    def _setup_plot_style(self, figsize: Tuple[int, int] = (12, 8)):
        """Setup consistent plot styling."""
        fig, ax = plt.subplots(figsize=figsize)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        return fig, ax
    
    def _save_plot(self, fig: plt.Figure, 
                   asset_name: str, 
                   model_name: str, 
                   plot_name: str, 
                   is_aggregate: bool = False) -> str:
        """
        Save plot using organized visualization structure.
        
        Parameters:
        ----------
        fig : plt.Figure
            Figure to save
        asset_name : str
            Asset name (ignored if is_aggregate=True)
        model_name : str
            Model name (ignored if is_aggregate=True)
        plot_name : str
            Name of the plot
        is_aggregate : bool
            Whether this is an aggregate plot
            
        Returns:
        -------
        str
            Path where plot was saved
        """
        try:
            if is_aggregate:
                file_path = get_organized_aggregate_path(str(self.output_dir), plot_name)
            else:
                file_path = get_organized_visualization_path(
                    str(self.output_dir), asset_name, model_name, plot_name
                )
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save with high DPI
            fig.savefig(file_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            logger.debug(f"Saved plot to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save plot {plot_name}: {str(e)}")
            return ""
    
    def _get_model_color(self, model_name: str) -> str:
        """Get color for model."""
        return self.model_colors.get(model_name.lower(), self.colors['primary'])


# =============================================================================
# PERFORMANCE VISUALIZER
# =============================================================================

class PerformanceVisualizer(BaseVisualizer):
    """Visualizer for model performance metrics and comparisons."""
    
    def plot_metrics_comparison(self, 
                               results: Dict[str, TrainingResult],
                               asset_name: str,
                               metrics: List[str] = None) -> str:
        """
        Create comprehensive metrics comparison plot.
        
        Parameters:
        ----------
        results : Dict[str, TrainingResult]
            Training results by model
        asset_name : str
            Name of the asset
        metrics : List[str], optional
            Metrics to compare
            
        Returns:
        -------
        str
            Path to saved plot
        """
        metrics = metrics or ['mae', 'rmse', 'r2', 'mape']
        
        # Filter successful results
        successful_results = {name: result for name, result in results.items() 
                            if result.success and result.validation_result}
        
        if not successful_results:
            logger.warning("No successful results for metrics comparison")
            return ""
        
        # Setup subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            ax = axes[i]
            
            model_names = []
            metric_values = []
            colors = []
            
            for model_name, result in successful_results.items():
                if metric in result.validation_result.metrics:
                    model_names.append(model_name)
                    metric_values.append(result.validation_result.metrics[metric])
                    colors.append(self._get_model_color(model_name))
            
            if metric_values:
                bars = ax.bar(model_names, metric_values, color=colors, alpha=0.7)
                ax.set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric.upper())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.annotate(f'{value:.3f}',
                              xy=(bar.get_x() + bar.get_width()/2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=10)
        
        plt.suptitle(f'Model Performance Comparison - {asset_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, asset_name, "general", "metrics_comparison")
    
    def plot_training_progress(self, 
                              results: Dict[str, TrainingResult],
                              asset_name: str) -> str:
        """Plot training times and validation scores."""
        
        successful_results = {name: result for name, result in results.items() 
                            if result.success and result.validation_result}
        
        if not successful_results:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training times
        model_names = list(successful_results.keys())
        training_times = [result.training_time for result in successful_results.values()]
        colors = [self._get_model_color(name) for name in model_names]
        
        bars1 = ax1.bar(model_names, training_times, color=colors, alpha=0.7)
        ax1.set_title('Training Time Comparison', fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars1, training_times):
            height = bar.get_height()
            ax1.annotate(f'{time:.2f}s',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        # Validation scores (RMSE)
        rmse_scores = []
        for result in successful_results.values():
            rmse = result.validation_result.metrics.get('rmse', 0)
            rmse_scores.append(rmse)
        
        bars2 = ax2.bar(model_names, rmse_scores, color=colors, alpha=0.7)
        ax2.set_title('Validation RMSE Comparison', fontweight='bold')
        ax2.set_ylabel('RMSE')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, rmse in zip(bars2, rmse_scores):
            height = bar.get_height()
            ax2.annotate(f'{rmse:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.suptitle(f'Training Progress - {asset_name}', fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, asset_name, "general", "training_progress")
    
    def plot_error_distribution(self, 
                               validation_result: ValidationResult,
                               asset_name: str,
                               model_name: str) -> str:
        """Plot error distribution analysis."""
        
        if validation_result.residuals is None or len(validation_result.residuals) == 0:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        residuals = validation_result.residuals
        
        # Box plot
        ax1.boxplot(residuals, patch_artist=True, 
                   boxprops=dict(facecolor=self._get_model_color(model_name), alpha=0.7))
        ax1.set_title('Residuals Box Plot', fontweight='bold')
        ax1.set_ylabel('Residual Value')
        ax1.grid(True, alpha=0.3)
        
        # Histogram with normal overlay
        ax2.hist(residuals, bins=30, alpha=0.7, density=True, 
                color=self._get_model_color(model_name), edgecolor='black')
        
        # Normal distribution overlay
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = ((1/(sigma * np.sqrt(2 * np.pi))) * 
                       np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        ax2.plot(x, normal_curve, 'r--', linewidth=2, label='Normal Distribution')
        
        ax2.set_title('Residuals Distribution', fontweight='bold')
        ax2.set_xlabel('Residual Value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Error Distribution Analysis - {model_name}', fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, asset_name, model_name, "error_distribution")
    
    def plot_model_complexity_vs_performance(self, 
                                           results: Dict[str, TrainingResult],
                                           asset_name: str) -> str:
        """Plot model complexity vs performance trade-off."""
        
        successful_results = {name: result for name, result in results.items() 
                            if result.success and result.validation_result}
        
        if len(successful_results) < 2:
            return ""
        
        fig, ax = self._setup_plot_style((12, 8))
        
        # Define complexity scores for different models
        complexity_scores = {
            'linear': 1, 'ridge': 2, 'lasso': 2,
            'random_forest': 6, 'xgboost': 8,
            'arima': 4, 'sarima': 7
        }
        
        x_values = []  # complexity
        y_values = []  # performance (1/RMSE for better = higher)
        sizes = []     # training time
        colors = []
        labels = []
        
        for model_name, result in successful_results.items():
            complexity = complexity_scores.get(model_name.lower(), 5)
            rmse = result.validation_result.metrics.get('rmse', float('inf'))
            if rmse > 0:
                performance = 1 / rmse  # Higher is better
                
                x_values.append(complexity)
                y_values.append(performance)
                sizes.append(result.training_time * 50)  # Scale for visibility
                colors.append(self._get_model_color(model_name))
                labels.append(model_name)
        
        if x_values:
            scatter = ax.scatter(x_values, y_values, s=sizes, c=colors, alpha=0.7)
            
            # Add labels
            for i, label in enumerate(labels):
                ax.annotate(label, (x_values[i], y_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Model Complexity Score', fontweight='bold')
            ax.set_ylabel('Performance (1/RMSE)', fontweight='bold')
            ax.set_title(f'Model Complexity vs Performance - {asset_name}', fontweight='bold')
            
            # Add bubble size legend
            sizes_legend = [10, 50, 100]  # seconds
            labels_legend = ['0.2s', '1s', '2s']
            legend_elements = [plt.scatter([], [], s=s*50, c='gray', alpha=0.6) 
                             for s in sizes_legend]
            ax.legend(legend_elements, labels_legend, title="Training Time", 
                     loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return self._save_plot(fig, asset_name, "general", "complexity_vs_performance")
    
    def create_performance_dashboard(self, 
                                   results: Dict[str, TrainingResult],
                                   asset_name: str) -> List[str]:
        """Create comprehensive performance dashboard."""
        saved_plots = []
        
        # Generate all performance plots
        plot_methods = [
            self.plot_metrics_comparison,
            self.plot_training_progress,
            self.plot_model_complexity_vs_performance
        ]
        
        for method in plot_methods:
            try:
                path = method(results, asset_name)
                if path:
                    saved_plots.append(path)
            except Exception as e:
                logger.error(f"Failed to create {method.__name__}: {str(e)}")
        
        # Individual error distribution plots
        for model_name, result in results.items():
            if result.success and result.validation_result:
                try:
                    path = self.plot_error_distribution(
                        result.validation_result, asset_name, model_name
                    )
                    if path:
                        saved_plots.append(path)
                except Exception as e:
                    logger.error(f"Failed to create error distribution for {model_name}: {str(e)}")
        
        return saved_plots


# =============================================================================
# FORECASTING VISUALIZER
# =============================================================================

class ForecastingVisualizer(BaseVisualizer):
    """Visualizer for forecasting-specific plots and analysis."""
    
    def plot_prediction_vs_actual(self, 
                                 validation_result: ValidationResult,
                                 asset_name: str,
                                 model_name: str) -> str:
        """Plot predictions vs actual values."""
        
        if len(validation_result.predictions) == 0 or len(validation_result.actuals) == 0:
            return ""
        
        fig, ax = self._setup_plot_style((12, 8))
        
        predictions = validation_result.predictions
        actuals = validation_result.actuals
        
        # Scatter plot
        ax.scatter(actuals, predictions, alpha=0.6, 
                  color=self._get_model_color(model_name), s=50)
        
        # Perfect prediction line
        min_val = min(min(actuals), min(predictions))
        max_val = max(max(actuals), max(predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Perfect Prediction')
        
        # Metrics in text box
        metrics = validation_result.metrics
        textstr = f"MAE: {metrics.get('mae', 'N/A'):.4f}\n"
        textstr += f"RMSE: {metrics.get('rmse', 'N/A'):.4f}\n"
        textstr += f"R²: {metrics.get('r2', 'N/A'):.4f}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual Values', fontweight='bold')
        ax.set_ylabel('Predicted Values', fontweight='bold')
        ax.set_title(f'Predictions vs Actual - {model_name}', fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        return self._save_plot(fig, asset_name, model_name, "predictions")
    
    def plot_residual_analysis(self, 
                             validation_result: ValidationResult,
                             asset_name: str,
                             model_name: str) -> str:
        """Create comprehensive 4-panel residual analysis."""
        
        if len(validation_result.residuals) == 0:
            return ""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        residuals = validation_result.residuals
        predictions = validation_result.predictions
        
        # 1. Residuals vs Fitted
        ax1.scatter(predictions, residuals, alpha=0.6, 
                   color=self._get_model_color(model_name))
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normal)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram of residuals
        ax3.hist(residuals, bins=30, alpha=0.7, 
                color=self._get_model_color(model_name), edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residuals Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals over time (if dates available)
        if validation_result.prediction_dates is not None:
            ax4.plot(validation_result.prediction_dates, residuals, 
                    color=self._get_model_color(model_name), alpha=0.7)
            ax4.axhline(y=0, color='r', linestyle='--')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residuals Over Time')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.plot(range(len(residuals)), residuals, 
                    color=self._get_model_color(model_name), alpha=0.7)
            ax4.axhline(y=0, color='r', linestyle='--')
            ax4.set_xlabel('Observation Index')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residuals vs Index')
        
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, asset_name, model_name, "residuals")
    
    def plot_train_val_test_split(self, 
                                 data: pd.DataFrame,
                                 split_info: Dict[str, Any],
                                 asset_name: str,
                                 price_column: str = 'price(SGD)') -> str:
        """Visualize train/validation/test split."""
        
        if price_column not in data.columns:
            return ""
        
        fig, ax = self._setup_plot_style((15, 8))
        
        # Assuming split_info contains split indices or dates
        train_end = split_info.get('train_end', len(data) * 0.7)
        val_end = split_info.get('val_end', len(data) * 0.85)
        
        train_data = data.iloc[:int(train_end)]
        val_data = data.iloc[int(train_end):int(val_end)]
        test_data = data.iloc[int(val_end):]
        
        # Plot each split
        ax.plot(train_data.index, train_data[price_column], 
               color=self.colors['primary'], label='Training Data', linewidth=1.5)
        ax.plot(val_data.index, val_data[price_column], 
               color=self.colors['warning'], label='Validation Data', linewidth=1.5)
        ax.plot(test_data.index, test_data[price_column], 
               color=self.colors['success'], label='Test Data', linewidth=1.5)
        
        # Add vertical lines at split points
        if len(train_data) > 0:
            ax.axvline(x=train_data.index[-1], color='gray', linestyle='--', alpha=0.7)
        if len(val_data) > 0:
            ax.axvline(x=val_data.index[-1], color='gray', linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Price (SGD)', fontweight='bold')
        ax.set_title(f'Train/Validation/Test Split - {asset_name}', fontweight='bold')
        ax.legend()
        
        # Format x-axis dates
        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return self._save_plot(fig, asset_name, "general", "data_split")
    
    def plot_feature_importance(self, 
                               feature_importance: Dict[str, float],
                               asset_name: str,
                               model_name: str,
                               top_n: int = 20) -> str:
        """Plot feature importance."""
        
        if not feature_importance:
            return ""
        
        # Sort and get top N features
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:top_n]
        
        features, importances = zip(*sorted_features)
        
        fig, ax = self._setup_plot_style((12, max(8, len(features) * 0.4)))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importances, 
                      color=self._get_model_color(model_name), alpha=0.7)
        
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance', fontweight='bold')
        ax.set_title(f'Feature Importance - {model_name}', fontweight='bold')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        # Invert y-axis to show most important at top
        ax.invert_yaxis()
        
        plt.tight_layout()
        return self._save_plot(fig, asset_name, model_name, "importance")
    
    def plot_forecast_with_uncertainty(self, 
                                     historical_data: pd.DataFrame,
                                     predictions: np.ndarray,
                                     prediction_dates: pd.DatetimeIndex,
                                     confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]],
                                     asset_name: str,
                                     model_name: str,
                                     price_column: str = 'price(SGD)') -> str:
        """Plot forecast with uncertainty bands."""
        
        fig, ax = self._setup_plot_style((15, 8))
        
        # Plot historical data
        recent_data = historical_data.tail(min(200, len(historical_data)))
        ax.plot(recent_data.index, recent_data[price_column], 
               color=self.colors['primary'], label='Historical Data', linewidth=1.5)
        
        # Plot predictions
        ax.plot(prediction_dates, predictions, 
               color=self._get_model_color(model_name), 
               label=f'{model_name} Forecast', linewidth=2, marker='o', markersize=4)
        
        # Add confidence intervals if available
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            ax.fill_between(prediction_dates, lower, upper, 
                           color=self._get_model_color(model_name), 
                           alpha=0.3, label='95% Confidence Interval')
        
        # Add vertical line at forecast start
        if len(recent_data) > 0:
            ax.axvline(x=recent_data.index[-1], color='gray', 
                      linestyle='--', alpha=0.7, label='Forecast Start')
        
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Price (SGD)', fontweight='bold')
        ax.set_title(f'Forecast with Uncertainty - {model_name}', fontweight='bold')
        ax.legend()
        
        # Format x-axis
        if len(prediction_dates) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return self._save_plot(fig, asset_name, model_name, "complete_forecast")
    
    def create_comprehensive_forecast_report(self, 
                                           validation_result: ValidationResult,
                                           feature_importance: Optional[Dict[str, float]],
                                           asset_name: str,
                                           model_name: str) -> List[str]:
        """Create comprehensive forecasting report."""
        saved_plots = []
        
        # Core forecasting plots
        try:
            path = self.plot_prediction_vs_actual(validation_result, asset_name, model_name)
            if path:
                saved_plots.append(path)
        except Exception as e:
            logger.error(f"Failed to create prediction plot: {str(e)}")
        
        try:
            path = self.plot_residual_analysis(validation_result, asset_name, model_name)
            if path:
                saved_plots.append(path)
        except Exception as e:
            logger.error(f"Failed to create residual analysis: {str(e)}")
        
        # Feature importance if available
        if feature_importance:
            try:
                path = self.plot_feature_importance(feature_importance, asset_name, model_name)
                if path:
                    saved_plots.append(path)
            except Exception as e:
                logger.error(f"Failed to create feature importance plot: {str(e)}")
        
        return saved_plots


# =============================================================================
# COMPARISON VISUALIZER
# =============================================================================

class ComparisonVisualizer(BaseVisualizer):
    """Visualizer for comparing multiple models and assets."""
    
    def plot_model_ranking(self, 
                          asset_results: Dict[str, Dict[str, TrainingResult]],
                          metric: str = 'rmse') -> str:
        """Plot model rankings across all assets."""
        
        # Collect data for ranking
        model_scores = {}
        assets = []
        
        for asset_name, results in asset_results.items():
            assets.append(asset_name)
            
            for model_name, result in results.items():
                if result.success and result.validation_result:
                    score = result.validation_result.metrics.get(metric)
                    if score is not None:
                        if model_name not in model_scores:
                            model_scores[model_name] = []
                        model_scores[model_name].append((asset_name, score))
        
        if not model_scores:
            return ""
        
        # Calculate average rankings
        model_avg_rank = {}
        for model_name, scores in model_scores.items():
            # Sort by asset and get ranking position
            asset_scores = dict(scores)
            avg_score = np.mean(list(asset_scores.values()))
            model_avg_rank[model_name] = avg_score
        
        # Sort models by performance
        sorted_models = sorted(model_avg_rank.items(), key=lambda x: x[1])
        
        fig, ax = self._setup_plot_style((12, 8))
        
        models, scores = zip(*sorted_models)
        colors = [self._get_model_color(model) for model in models]
        
        bars = ax.bar(models, scores, color=colors, alpha=0.7)
        ax.set_title(f'Model Rankings - Average {metric.upper()}', fontweight='bold')
        ax.set_ylabel(f'Average {metric.upper()}')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.annotate(f'{score:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        return self._save_plot(fig, "", "", "model_rankings", is_aggregate=True)
    
    def plot_asset_performance_heatmap(self, 
                                     asset_results: Dict[str, Dict[str, TrainingResult]],
                                     metric: str = 'rmse') -> str:
        """Create performance heatmap across assets and models."""
        
        # Create matrix of performance scores
        models = set()
        for results in asset_results.values():
            models.update(results.keys())
        models = sorted(list(models))
        
        assets = sorted(asset_results.keys())
        
        # Build performance matrix
        performance_matrix = np.full((len(assets), len(models)), np.nan)
        
        for i, asset_name in enumerate(assets):
            for j, model_name in enumerate(models):
                result = asset_results[asset_name].get(model_name)
                if result and result.success and result.validation_result:
                    score = result.validation_result.metrics.get(metric)
                    if score is not None:
                        performance_matrix[i, j] = score
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(models) * 0.8), max(8, len(assets) * 0.6)))
        
        # Mask NaN values
        mask = np.isnan(performance_matrix)
        
        # Create heatmap
        im = ax.imshow(performance_matrix, cmap='viridis_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(models)))
        ax.set_yticks(range(len(assets)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(assets)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric.upper()} Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(assets)):
            for j in range(len(models)):
                if not mask[i, j]:
                    text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="white", fontweight='bold')
        
        ax.set_title(f'Model Performance Heatmap - {metric.upper()}', fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, "", "", "performance_heatmap", is_aggregate=True)
    
    def plot_cross_asset_predictions(self, 
                                   asset_results: Dict[str, Dict[str, TrainingResult]],
                                   model_name: str) -> str:
        """Plot predictions across multiple assets for a single model."""
        
        # Collect prediction data for the specified model
        asset_predictions = {}
        
        for asset_name, results in asset_results.items():
            result = results.get(model_name)
            if result and result.success and result.validation_result:
                val_result = result.validation_result
                if len(val_result.predictions) > 0 and len(val_result.actuals) > 0:
                    asset_predictions[asset_name] = {
                        'predictions': val_result.predictions,
                        'actuals': val_result.actuals,
                        'dates': val_result.prediction_dates
                    }
        
        if not asset_predictions:
            return ""
        
        n_assets = len(asset_predictions)
        fig, axes = plt.subplots(n_assets, 1, figsize=(15, 6 * n_assets))
        
        if n_assets == 1:
            axes = [axes]
        
        for i, (asset_name, data) in enumerate(asset_predictions.items()):
            ax = axes[i]
            
            predictions = data['predictions']
            actuals = data['actuals']
            dates = data['dates']
            
            if dates is not None:
                ax.plot(dates, actuals, label='Actual', color=self.colors['primary'], linewidth=2)
                ax.plot(dates, predictions, label='Predicted', 
                       color=self._get_model_color(model_name), linewidth=2, linestyle='--')
            else:
                x_vals = range(len(actuals))
                ax.plot(x_vals, actuals, label='Actual', color=self.colors['primary'], linewidth=2)
                ax.plot(x_vals, predictions, label='Predicted', 
                       color=self._get_model_color(model_name), linewidth=2, linestyle='--')
            
            ax.set_title(f'{asset_name} - {model_name} Predictions', fontweight='bold')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if dates is not None and len(dates) > 0:
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Cross-Asset Predictions - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, "", "", f"cross_asset_{model_name}", is_aggregate=True)
    
    def create_comprehensive_comparison(self, 
                                      asset_results: Dict[str, Dict[str, TrainingResult]]) -> List[str]:
        """Create comprehensive comparison dashboard."""
        saved_plots = []
        
        metrics = ['rmse', 'mae', 'r2']
        
        # Model rankings for each metric
        for metric in metrics:
            try:
                path = self.plot_model_ranking(asset_results, metric)
                if path:
                    saved_plots.append(path)
            except Exception as e:
                logger.error(f"Failed to create model ranking for {metric}: {str(e)}")
        
        # Performance heatmaps
        for metric in metrics:
            try:
                path = self.plot_asset_performance_heatmap(asset_results, metric)
                if path:
                    saved_plots.append(path)
            except Exception as e:
                logger.error(f"Failed to create heatmap for {metric}: {str(e)}")
        
        # Cross-asset predictions for each model
        all_models = set()
        for results in asset_results.values():
            all_models.update(results.keys())
        
        for model_name in all_models:
            try:
                path = self.plot_cross_asset_predictions(asset_results, model_name)
                if path:
                    saved_plots.append(path)
            except Exception as e:
                logger.error(f"Failed to create cross-asset plot for {model_name}: {str(e)}")
        
        return saved_plots


# =============================================================================
# ENHANCED FORECASTING VISUALIZER
# =============================================================================

class EnhancedForecastingVisualizer(BaseVisualizer):
    """Advanced forecasting visualizations with decomposition and ensemble analysis."""
    
    def plot_forecast_decomposition(self, 
                                  data: pd.DataFrame,
                                  asset_name: str,
                                  model_name: str,
                                  price_column: str = 'price(SGD)') -> str:
        """Plot seasonal decomposition of the time series."""
        
        if price_column not in data.columns or len(data) < 24:
            return ""
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            # Perform decomposition
            decomposition = seasonal_decompose(
                data[price_column].dropna(), 
                model='additive', 
                period=min(12, len(data) // 4),
                extrapolate_trend='freq'
            )
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            # Original
            axes[0].plot(decomposition.observed, color=self.colors['primary'])
            axes[0].set_title('Original Time Series', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend, color=self.colors['secondary'])
            axes[1].set_title('Trend Component', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal, color=self.colors['success'])
            axes[2].set_title('Seasonal Component', fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid, color=self.colors['warning'])
            axes[3].set_title('Residual Component', fontweight='bold')
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xlabel('Date')
            
            plt.suptitle(f'Time Series Decomposition - {asset_name}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            return self._save_plot(fig, asset_name, model_name, "decomposition")
            
        except Exception as e:
            logger.error(f"Failed to create decomposition plot: {str(e)}")
            return ""
    
    def plot_ensemble_forecast_analysis(self, 
                                      asset_results: Dict[str, TrainingResult],
                                      asset_name: str) -> str:
        """Create ensemble forecast analysis combining multiple models."""
        
        successful_results = {name: result for name, result in asset_results.items() 
                            if result.success and result.validation_result}
        
        if len(successful_results) < 2:
            return ""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Collect all predictions and actuals
        all_predictions = {}
        all_actuals = {}
        
        for model_name, result in successful_results.items():
            val_result = result.validation_result
            if len(val_result.predictions) > 0:
                all_predictions[model_name] = val_result.predictions
                all_actuals[model_name] = val_result.actuals
        
        if not all_predictions:
            return ""
        
        # Plot 1: Individual model predictions
        dates = None
        actuals = None
        
        for model_name, predictions in all_predictions.items():
            actuals = all_actuals[model_name]  # Should be same for all
            dates = successful_results[model_name].validation_result.prediction_dates
            
            if dates is not None:
                ax1.plot(dates, predictions, label=f'{model_name} Forecast', 
                        color=self._get_model_color(model_name), alpha=0.7, linewidth=1.5)
            else:
                x_vals = range(len(predictions))
                ax1.plot(x_vals, predictions, label=f'{model_name} Forecast', 
                        color=self._get_model_color(model_name), alpha=0.7, linewidth=1.5)
        
        # Plot actual values
        if actuals is not None:
            if dates is not None:
                ax1.plot(dates, actuals, label='Actual', color='black', linewidth=2)
            else:
                ax1.plot(range(len(actuals)), actuals, label='Actual', color='black', linewidth=2)
        
        ax1.set_title('Ensemble Model Predictions', fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Ensemble statistics
        pred_matrix = np.array(list(all_predictions.values()))
        ensemble_mean = np.mean(pred_matrix, axis=0)
        ensemble_std = np.std(pred_matrix, axis=0)
        
        if dates is not None:
            ax2.plot(dates, ensemble_mean, label='Ensemble Mean', 
                    color=self.colors['info'], linewidth=2)
            ax2.fill_between(dates, 
                           ensemble_mean - ensemble_std, 
                           ensemble_mean + ensemble_std,
                           alpha=0.3, color=self.colors['info'], 
                           label='±1 Std Dev')
            if actuals is not None:
                ax2.plot(dates, actuals, label='Actual', color='black', linewidth=2)
        else:
            x_vals = range(len(ensemble_mean))
            ax2.plot(x_vals, ensemble_mean, label='Ensemble Mean', 
                    color=self.colors['info'], linewidth=2)
            ax2.fill_between(x_vals, 
                           ensemble_mean - ensemble_std, 
                           ensemble_mean + ensemble_std,
                           alpha=0.3, color=self.colors['info'], 
                           label='±1 Std Dev')
            if actuals is not None:
                ax2.plot(x_vals, actuals, label='Actual', color='black', linewidth=2)
        
        ax2.set_title('Ensemble Statistics', fontweight='bold')
        ax2.set_ylabel('Price')
        ax2.set_xlabel('Date' if dates is not None else 'Index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Ensemble Forecast Analysis - {asset_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, asset_name, "ensemble", "forecast_analysis")
    
    def create_enhanced_forecast_dashboard(self, 
                                         asset_results: Dict[str, TrainingResult],
                                         data: pd.DataFrame,
                                         asset_name: str) -> List[str]:
        """Create enhanced forecasting dashboard."""
        saved_plots = []
        
        # Ensemble analysis
        try:
            path = self.plot_ensemble_forecast_analysis(asset_results, asset_name)
            if path:
                saved_plots.append(path)
        except Exception as e:
            logger.error(f"Failed to create ensemble analysis: {str(e)}")
        
        # Decomposition analysis
        for model_name in asset_results.keys():
            try:
                path = self.plot_forecast_decomposition(data, asset_name, model_name)
                if path:
                    saved_plots.append(path)
                    break  # Only need one decomposition per asset
            except Exception as e:
                logger.error(f"Failed to create decomposition: {str(e)}")
        
        return saved_plots


# =============================================================================
# MAIN VISUALIZATION CLASSES (EXPORTED)
# =============================================================================

# Export the main visualizers for backward compatibility
__all__ = [
    'PerformanceVisualizer', 
    'ForecastingVisualizer', 
    'ComparisonVisualizer',
    'EnhancedForecastingVisualizer'
]