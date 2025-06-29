"""
Performance visualization components.

This module creates visualizations for model performance metrics,
training progress, and comparative analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

# Import plotting libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Plotting libraries not available. Install matplotlib and seaborn for visualizations.")

logger = logging.getLogger(__name__)


class PerformanceVisualizer:
    """Create performance analysis visualizations."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the performance visualizer.
        
        Parameters:
        ----------
        style : str
            Matplotlib style to use
        figsize : Tuple[int, int]
            Default figure size
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib and seaborn are required for visualizations")
        
        self.style = style
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 8)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_metrics_comparison(self,
                              results_data: Dict[str, Dict[str, float]],
                              metrics: List[str] = ['mae', 'rmse', 'r2', 'mape'],
                              title: str = "Model Performance Comparison",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive metrics comparison plot.
        
        Parameters:
        ----------
        results_data : Dict[str, Dict[str, float]]
            Results data: {model_name: {metric: value}}
        metrics : List[str]
            Metrics to compare
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Filter available metrics
        available_metrics = []
        for metric in metrics:
            if any(metric in model_results for model_results in results_data.values()):
                available_metrics.append(metric)
        
        if not available_metrics:
            raise ValueError("No valid metrics found in results data")
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        models = list(results_data.keys())
        
        for i, metric in enumerate(available_metrics[:4]):  # Limit to 4 metrics
            ax = axes[i]
            
            # Extract metric values
            values = []
            model_names = []
            
            for model_name, model_results in results_data.items():
                if metric in model_results and not np.isnan(model_results[metric]):
                    values.append(model_results[metric])
                    model_names.append(model_name)
            
            if not values:
                ax.text(0.5, 0.5, f'No data for {metric.upper()}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{metric.upper()}')
                continue
            
            # Create bar plot
            bars = ax.bar(model_names, values, color=self.colors[:len(model_names)])
            
            # Customize plot
            ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Value', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Rotate x-axis labels if needed
            if len(max(model_names, key=len)) > 8:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Hide unused subplots
        for j in range(len(available_metrics), 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics comparison to {save_path}")
        
        return fig
    
    def plot_training_progress(self,
                             training_history: Dict[str, List[float]],
                             title: str = "Training Progress",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training progress over epochs/iterations.
        
        Parameters:
        ----------
        training_history : Dict[str, List[float]]
            Training history: {metric: [values_over_time]}
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (metric_name, values) in enumerate(training_history.items()):
            if values and len(values) > 1:
                epochs = range(1, len(values) + 1)
                ax.plot(epochs, values, marker='o', linewidth=2, 
                       label=metric_name, color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Epoch/Iteration', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training progress to {save_path}")
        
        return fig
    
    def plot_error_distribution(self,
                               errors_by_model: Dict[str, np.ndarray],
                               title: str = "Error Distribution by Model",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot error distribution comparison across models.
        
        Parameters:
        ----------
        errors_by_model : Dict[str, np.ndarray]
            Errors by model: {model_name: error_array}
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Box plot
        model_names = list(errors_by_model.keys())
        error_data = [errors_by_model[model] for model in model_names]
        
        box_plot = ax1.boxplot(error_data, labels=model_names, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], self.colors[:len(model_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Error Distribution (Box Plot)', fontsize=14)
        ax1.set_ylabel('Prediction Error (SGD)', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        if len(max(model_names, key=len)) > 8:
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Histogram overlay
        for i, (model_name, errors) in enumerate(errors_by_model.items()):
            ax2.hist(errors, bins=30, alpha=0.6, label=model_name, 
                    color=self.colors[i % len(self.colors)], density=True)
        
        ax2.set_title('Error Distribution (Histogram)', fontsize=14)
        ax2.set_xlabel('Prediction Error (SGD)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved error distribution to {save_path}")
        
        return fig
    
    def plot_accuracy_vs_time(self,
                            time_series_accuracy: Dict[str, pd.DataFrame],
                            metric: str = 'mae',
                            title: str = "Model Accuracy Over Time",
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model accuracy evolution over time.
        
        Parameters:
        ----------
        time_series_accuracy : Dict[str, pd.DataFrame]
            Time series accuracy data: {model_name: df_with_time_and_accuracy}
        metric : str
            Accuracy metric to plot
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (model_name, df) in enumerate(time_series_accuracy.items()):
            if metric in df.columns and 'window_end' in df.columns:
                ax.plot(df['window_end'], df[metric], 
                       marker='o', linewidth=2, label=model_name,
                       color=self.colors[i % len(self.colors)])
        
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved accuracy over time to {save_path}")
        
        return fig
    
    def plot_model_complexity_vs_performance(self,
                                           model_stats: Dict[str, Dict],
                                           complexity_metric: str = 'training_time',
                                           performance_metric: str = 'mae',
                                           title: str = "Model Complexity vs Performance",
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot model complexity vs performance trade-off.
        
        Parameters:
        ----------
        model_stats : Dict[str, Dict]
            Model statistics: {model_name: {complexity_metric: value, performance_metric: value}}
        complexity_metric : str
            Metric representing model complexity
        performance_metric : str
            Metric representing model performance
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract data
        complexity_values = []
        performance_values = []
        model_names = []
        
        for model_name, stats in model_stats.items():
            if complexity_metric in stats and performance_metric in stats:
                complexity_values.append(stats[complexity_metric])
                performance_values.append(stats[performance_metric])
                model_names.append(model_name)
        
        if not complexity_values:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create scatter plot
        scatter = ax.scatter(complexity_values, performance_values, 
                           s=100, alpha=0.7, c=range(len(model_names)), 
                           cmap='viridis')
        
        # Add model name labels
        for i, name in enumerate(model_names):
            ax.annotate(name, (complexity_values[i], performance_values[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel(f'{complexity_metric.replace("_", " ").title()}', fontsize=12)
        ax.set_ylabel(f'{performance_metric.upper()}', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add trend line if enough points
        if len(complexity_values) > 2:
            z = np.polyfit(complexity_values, performance_values, 1)
            p = np.poly1d(z)
            ax.plot(complexity_values, p(complexity_values), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved complexity vs performance to {save_path}")
        
        return fig
    
    def create_performance_dashboard(self,
                                   training_results: Dict[str, Any],
                                   output_dir: str = "data/output/visualizations") -> Dict[str, str]:
        """
        Create a comprehensive performance dashboard.
        
        Parameters:
        ----------
        training_results : Dict[str, Any]
            Complete training results
        output_dir : str
            Output directory for plots
            
        Returns:
        -------
        Dict[str, str]
            Dictionary of plot names and file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        # 1. Extract metrics for all models across all assets
        all_metrics = {}
        model_stats = {}
        
        for asset_name, asset_results in training_results.items():
            for model_name, result in asset_results.items():
                if result.success and result.validation_result:
                    metrics = result.validation_result.metrics
                    
                    if model_name not in all_metrics:
                        all_metrics[model_name] = {metric: [] for metric in metrics.keys()}
                        model_stats[model_name] = {
                            'training_times': [],
                            'feature_counts': []
                        }
                    
                    for metric, value in metrics.items():
                        if not np.isnan(value):
                            all_metrics[model_name][metric].append(value)
                    
                    model_stats[model_name]['training_times'].append(result.training_time)
                    model_stats[model_name]['feature_counts'].append(
                        len(result.feature_importance) if result.feature_importance else 0
                    )
        
        # 2. Calculate average metrics
        avg_metrics = {}
        for model_name, metrics_dict in all_metrics.items():
            avg_metrics[model_name] = {
                metric: np.mean(values) if values else np.nan
                for metric, values in metrics_dict.items()
            }
        
        # 3. Create metrics comparison plot
        if avg_metrics:
            metrics_file = output_path / "model_metrics_comparison.png"
            fig1 = self.plot_metrics_comparison(
                avg_metrics,
                title="Average Model Performance Across All Assets",
                save_path=str(metrics_file)
            )
            plt.close(fig1)
            plot_files['metrics_comparison'] = str(metrics_file)
        
        # 4. Create complexity vs performance plot
        complexity_data = {}
        for model_name, stats in model_stats.items():
            if stats['training_times'] and model_name in avg_metrics:
                complexity_data[model_name] = {
                    'training_time': np.mean(stats['training_times']),
                    'mae': avg_metrics[model_name].get('mae', np.nan)
                }
        
        if complexity_data:
            complexity_file = output_path / "complexity_vs_performance.png"
            fig2 = self.plot_model_complexity_vs_performance(
                complexity_data,
                title="Model Training Time vs Performance",
                save_path=str(complexity_file)
            )
            plt.close(fig2)
            plot_files['complexity_analysis'] = str(complexity_file)
        
        # 5. Create error distribution plot
        all_errors = {}
        for asset_name, asset_results in training_results.items():
            for model_name, result in asset_results.items():
                if result.success and result.validation_result:
                    residuals = result.validation_result.residuals
                    
                    if model_name not in all_errors:
                        all_errors[model_name] = []
                    
                    all_errors[model_name].extend(residuals)
        
        # Convert to numpy arrays
        error_arrays = {model: np.array(errors) for model, errors in all_errors.items() if errors}
        
        if error_arrays:
            error_file = output_path / "error_distributions.png"
            fig3 = self.plot_error_distribution(
                error_arrays,
                title="Prediction Error Distributions Across All Assets",
                save_path=str(error_file)
            )
            plt.close(fig3)
            plot_files['error_distributions'] = str(error_file)
        
        logger.info(f"Created performance dashboard with {len(plot_files)} plots")
        
        return plot_files
    
    def plot_prediction_scatter_matrix(self,
                                     predictions_data: Dict[str, Dict[str, np.ndarray]],
                                     actuals_data: Dict[str, np.ndarray],
                                     title: str = "Prediction vs Actual Scatter Matrix",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create scatter plot matrix showing predictions vs actuals for all models.
        
        Parameters:
        ----------
        predictions_data : Dict[str, Dict[str, np.ndarray]]
            Predictions: {asset_name: {model_name: predictions}}
        actuals_data : Dict[str, np.ndarray]
            Actual values: {asset_name: actuals}
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Collect all model names
        all_models = set()
        for asset_preds in predictions_data.values():
            all_models.update(asset_preds.keys())
        all_models = sorted(list(all_models))
        
        if len(all_models) == 0:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No prediction data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        n_models = len(all_models)
        fig, axes = plt.subplots(1, min(n_models, 4), figsize=(min(n_models * 4, 16), 4))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Collect all predictions and actuals for each model
        for i, model_name in enumerate(all_models[:4]):  # Limit to 4 models
            ax = axes[i]
            
            all_preds = []
            all_actuals = []
            
            for asset_name, asset_preds in predictions_data.items():
                if (model_name in asset_preds and 
                    asset_name in actuals_data):
                    preds = asset_preds[model_name]
                    actuals = actuals_data[asset_name]
                    
                    if len(preds) == len(actuals):
                        all_preds.extend(preds)
                        all_actuals.extend(actuals)
            
            if not all_preds:
                ax.text(0.5, 0.5, f'No data for {model_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create scatter plot
            ax.scatter(all_actuals, all_preds, alpha=0.6, 
                      color=self.colors[i % len(self.colors)], s=20)
            
            # Add perfect prediction line
            min_val = min(min(all_actuals), min(all_preds))
            max_val = max(max(all_actuals), max(all_preds))
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
            
            # Calculate R²
            if len(all_preds) > 1:
                correlation_matrix = np.corrcoef(all_actuals, all_preds)
                r_squared = correlation_matrix[0, 1] ** 2
                ax.text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                       transform=ax.transAxes, fontsize=10, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Actual Price (SGD)', fontsize=10)
            ax.set_ylabel('Predicted Price (SGD)', fontsize=10)
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Hide unused subplots
        for j in range(len(all_models), min(4, len(axes))):
            if j < len(axes):
                axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved prediction scatter matrix to {save_path}")
        
        return fig
    
    def plot_model_stability_analysis(self,
                                     cross_validation_results: Dict[str, List[Dict]],
                                     metric: str = 'mae',
                                     title: str = "Model Stability Analysis",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze model stability across cross-validation folds.
        
        Parameters:
        ----------
        cross_validation_results : Dict[str, List[Dict]]
            CV results: {model_name: [fold_results]}
        metric : str
            Metric to analyze
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        model_names = list(cross_validation_results.keys())
        
        # Extract metric values for each fold
        stability_data = {}
        for model_name, fold_results in cross_validation_results.items():
            metric_values = []
            for fold_result in fold_results:
                if metric in fold_result:
                    metric_values.append(fold_result[metric])
            if metric_values:
                stability_data[model_name] = metric_values
        
        if not stability_data:
            ax1.text(0.5, 0.5, f'No {metric} data available', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, f'No {metric} data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            return fig
        
        # 1. Box plot showing distribution across folds
        model_names = list(stability_data.keys())
        metric_data = [stability_data[model] for model in model_names]
        
        bp = ax1.boxplot(metric_data, labels=model_names, patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], self.colors[:len(model_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title(f'{metric.upper()} Distribution Across CV Folds', fontsize=14)
        ax1.set_ylabel(f'{metric.upper()}', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Stability coefficient (CV / mean)
        stability_coeffs = []
        model_labels = []
        
        for model_name, values in stability_data.items():
            if len(values) > 1:
                cv_coeff = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                stability_coeffs.append(cv_coeff)
                model_labels.append(model_name)
        
        if stability_coeffs:
            bars = ax2.bar(model_labels, stability_coeffs, 
                          color=self.colors[:len(model_labels)], alpha=0.7)
            
            # Add value labels
            for bar, coeff in zip(bars, stability_coeffs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + max(stability_coeffs) * 0.01,
                        f'{coeff:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax2.set_title('Model Stability Coefficient (Lower = More Stable)', fontsize=14)
            ax2.set_ylabel('Coefficient of Variation', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved stability analysis to {save_path}")
        
        return fig
    
    def plot_learning_curves(self,
                           learning_data: Dict[str, Dict[str, List[float]]],
                           title: str = "Model Learning Curves",
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot learning curves showing training and validation performance.
        
        Parameters:
        ----------
        learning_data : Dict[str, Dict[str, List[float]]]
            Learning data: {model_name: {'train_scores': [...], 'val_scores': [...]}}
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
            
        Returns:
        -------
        plt.Figure
            The created figure
        """
        n_models = len(learning_data)
        if n_models == 0:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No learning curve data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create subplots
        cols = min(2, n_models)
        rows = (n_models + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, (model_name, data) in enumerate(learning_data.items()):
            ax = axes[i] if i < len(axes) else axes[0]
            
            train_scores = data.get('train_scores', [])
            val_scores = data.get('val_scores', [])
            
            if train_scores or val_scores:
                if train_scores:
                    epochs = range(1, len(train_scores) + 1)
                    ax.plot(epochs, train_scores, 'o-', 
                           color=self.colors[i % len(self.colors)], 
                           label='Training', linewidth=2, alpha=0.8)
                
                if val_scores:
                    epochs = range(1, len(val_scores) + 1)
                    ax.plot(epochs, val_scores, 's-', 
                           color=self.colors[(i + 1) % len(self.colors)], 
                           label='Validation', linewidth=2, alpha=0.8)
                
                ax.set_title(model_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('Epoch/Iteration', fontsize=10)
                ax.set_ylabel('Score', fontsize=10)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No data for {model_name}', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide unused subplots
        for j in range(n_models, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curves to {save_path}")
        
        return fig