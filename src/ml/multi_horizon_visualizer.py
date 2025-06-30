"""
Multi-Horizon Forecasting Visualizer

Creates comprehensive visualizations for multi-horizon watch price forecasting models.
Addresses the issues with single-point predictions by showing complete time series context,
horizon comparisons, and meaningful forecasting insights.

Usage:
    python -m src.ml.multi_horizon_visualizer
    python -m src.ml.multi_horizon_visualizer --asset "Audemars_Piguet-Steeloyal_Oak_Offshore-22378"
    python -m src.ml.multi_horizon_visualizer --horizons 7 14 30 --models linear ridge
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta

from src.ml.base import BaseTimeSeriesModel
from src.ml.run_training import load_featured_data
from src.utils.logging_config import get_training_logger
from src.utils.visualization_utils import get_organized_aggregate_path, get_organized_visualization_path

logger = get_training_logger()


class MultiHorizonVisualizer:
    """Create comprehensive multi-horizon forecasting visualizations."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (14, 10)):
        self.style = style
        self.figsize = figsize
        self.colors = {
            "actual": "#1f77b4",
            "horizon_7": "#ff7f0e", 
            "horizon_14": "#2ca02c",
            "horizon_30": "#d62728",
            "confidence": "#9467bd",
            "background": "#f8f9fa"
        }
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")
        
        # Set color palette
        sns.set_palette("husl")

    def plot_multi_horizon_comparison(
        self,
        asset_name: str,
        horizon_models: Dict[int, Dict[str, BaseTimeSeriesModel]],
        featured_data: pd.DataFrame,
        horizons: List[int] = [7, 14, 30],
        model_type: str = "linear",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive multi-horizon comparison plot showing:
        1. Historical data with clear context
        2. Predictions for different horizons
        3. Model performance metrics
        4. Meaningful time axis (no more 1970 dates!)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"{asset_name} - Multi-Horizon Forecasting Analysis", fontsize=16, fontweight='bold')

        # Prepare data splits
        n_samples = len(featured_data)
        test_size = 0.2
        val_size = 0.15
        
        test_start = int(n_samples * (1 - test_size))
        val_start = int(n_samples * (1 - test_size - val_size))
        
        train_data = featured_data.iloc[:val_start]
        val_data = featured_data.iloc[val_start:test_start]
        test_data = featured_data.iloc[test_start:]
        
        # Create proper time index
        dates = self._create_time_index(featured_data)
        train_dates = dates[:len(train_data)] if dates is not None else range(len(train_data))
        val_dates = dates[len(train_data):len(train_data)+len(val_data)] if dates is not None else range(len(train_data), len(train_data)+len(val_data))
        test_dates = dates[len(train_data)+len(val_data):] if dates is not None else range(len(train_data)+len(val_data), len(featured_data))

        # Plot 1: Complete Time Series with Predictions
        ax1 = axes[0, 0]
        
        # Plot historical data
        ax1.plot(train_dates, train_data['target'], color=self.colors['actual'], 
                linewidth=2, alpha=0.8, label='Training Data')
        ax1.plot(val_dates, val_data['target'], color=self.colors['actual'], 
                linewidth=2, alpha=0.6, label='Validation Data')
        ax1.plot(test_dates, test_data['target'], color=self.colors['actual'], 
                linewidth=2, alpha=1.0, label='Test Data (Actual)')

        # Generate predictions for each horizon
        feature_cols = [col for col in featured_data.columns if col not in ['target', 'asset_id']]
        
        for horizon in horizons:
            if horizon in horizon_models and model_type in horizon_models[horizon]:
                model = horizon_models[horizon][model_type]
                
                # Make predictions on test set
                if len(test_data) > 0:
                    test_features = test_data[feature_cols]
                    predictions = model.predict(test_features)
                    
                    color_key = f'horizon_{horizon}'
                    ax1.plot(test_dates, predictions, 
                            color=self.colors.get(color_key, '#333333'),
                            linewidth=2, alpha=0.8, linestyle='--',
                            label=f'{horizon}-day Model Predictions')

        ax1.set_title('Multi-Horizon Predictions vs Actual')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (SGD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if dates is not None and isinstance(dates, pd.DatetimeIndex):
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Horizon Performance Comparison
        ax2 = axes[0, 1]
        self._plot_horizon_performance_comparison(ax2, asset_name, horizon_models, 
                                                 featured_data, horizons, model_type)

        # Plot 3: Prediction Accuracy by Horizon
        ax3 = axes[1, 0]
        self._plot_prediction_accuracy_by_horizon(ax3, asset_name, horizon_models, 
                                                 featured_data, horizons, model_type)

        # Plot 4: Forecast Horizon Impact
        ax4 = axes[1, 1]
        self._plot_forecast_horizon_impact(ax4, asset_name, horizon_models, 
                                          featured_data, horizons, model_type)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multi-horizon comparison to {save_path}")

        return fig

    def _create_time_index(self, data: pd.DataFrame) -> Optional[pd.DatetimeIndex]:
        """Create proper time index from data, avoiding 1970 dates."""
        try:
            # Method 1: Use existing datetime index if available
            if isinstance(data.index, pd.DatetimeIndex):
                return data.index
            
            # Method 2: Reconstruct from year/month/day columns
            if all(col in data.columns for col in ['year', 'month']):
                day_col = 'day_of_month' if 'day_of_month' in data.columns else 'day'
                if day_col in data.columns:
                    # Ensure valid dates
                    date_df = data[['year', 'month', day_col]].copy()
                    date_df = date_df.dropna()
                    
                    if len(date_df) > 0:
                        # Filter out obviously invalid dates (like 1970)
                        valid_years = (date_df['year'] >= 2000) & (date_df['year'] <= 2030)
                        if valid_years.sum() > len(date_df) * 0.5:  # If most dates are valid
                            date_df = date_df[valid_years]
                            dates = pd.to_datetime(date_df)
                            
                            # Create full index for all data
                            if len(dates) < len(data):
                                # Extend dates to cover full dataset
                                start_date = dates.min()
                                full_dates = pd.date_range(start=start_date, periods=len(data), freq='D')
                                return full_dates
                            return dates
            
            # Method 3: Create synthetic dates based on data length
            logger.debug("Creating synthetic date index for visualization")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(data)-1)
            return pd.date_range(start=start_date, end=end_date, freq='D')
            
        except Exception as e:
            logger.debug(f"Could not create time index: {e}")
            return None

    def _plot_horizon_performance_comparison(self, ax, asset_name, horizon_models, 
                                           featured_data, horizons, model_type):
        """Plot performance metrics comparison across horizons."""
        metrics_data = []
        
        # Calculate test set performance for each horizon
        n_samples = len(featured_data)
        test_start = int(n_samples * 0.8)
        test_data = featured_data.iloc[test_start:]
        feature_cols = [col for col in featured_data.columns if col not in ['target', 'asset_id']]
        
        for horizon in horizons:
            if horizon in horizon_models and model_type in horizon_models[horizon]:
                model = horizon_models[horizon][model_type]
                
                if len(test_data) > 0:
                    test_features = test_data[feature_cols]
                    y_true = test_data['target'].values
                    y_pred = model.predict(test_features)
                    
                    mae = np.mean(np.abs(y_true - y_pred))
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                    
                    metrics_data.append({
                        'horizon': f'{horizon}d',
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })
        
        if metrics_data:
            df = pd.DataFrame(metrics_data)
            
            # Create grouped bar plot
            x = np.arange(len(df))
            width = 0.25
            
            ax.bar(x - width, df['MAE'], width, label='MAE', alpha=0.8)
            ax.bar(x, df['RMSE'], width, label='RMSE', alpha=0.8)
            ax.bar(x + width, df['MAPE'], width, label='MAPE (%)', alpha=0.8)
            
            ax.set_xlabel('Prediction Horizon')
            ax.set_ylabel('Error Metric')
            ax.set_title('Performance by Horizon')
            ax.set_xticks(x)
            ax.set_xticklabels(df['horizon'])
            ax.legend()
            ax.grid(True, alpha=0.3)

    def _plot_prediction_accuracy_by_horizon(self, ax, asset_name, horizon_models, 
                                           featured_data, horizons, model_type):
        """Plot how prediction accuracy changes with horizon length."""
        
        # Calculate accuracy metrics
        horizon_performance = []
        feature_cols = [col for col in featured_data.columns if col not in ['target', 'asset_id']]
        
        n_samples = len(featured_data)
        test_start = int(n_samples * 0.8)
        test_data = featured_data.iloc[test_start:]
        
        for horizon in sorted(horizons):
            if horizon in horizon_models and model_type in horizon_models[horizon]:
                model = horizon_models[horizon][model_type]
                
                if len(test_data) > 0:
                    test_features = test_data[feature_cols]
                    y_true = test_data['target'].values
                    y_pred = model.predict(test_features)
                    
                    # Calculate R² score
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    horizon_performance.append({
                        'horizon': horizon,
                        'r2_score': max(0, r2),  # Cap at 0 for visualization
                        'mae': np.mean(np.abs(y_true - y_pred))
                    })
        
        if horizon_performance:
            df = pd.DataFrame(horizon_performance)
            
            # Plot R² score trend
            ax.plot(df['horizon'], df['r2_score'], marker='o', linewidth=3, 
                   markersize=8, color=self.colors['horizon_7'])
            ax.fill_between(df['horizon'], 0, df['r2_score'], alpha=0.3, 
                          color=self.colors['horizon_7'])
            
            ax.set_xlabel('Prediction Horizon (Days)')
            ax.set_ylabel('R² Score')
            ax.set_title('Model Accuracy vs Forecast Horizon')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Add trend annotation
            if len(df) > 1:
                trend = "↑" if df['r2_score'].iloc[-1] > df['r2_score'].iloc[0] else "↓"
                ax.text(0.05, 0.95, f'Trend: {trend}', transform=ax.transAxes, 
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def _plot_forecast_horizon_impact(self, ax, asset_name, horizon_models, 
                                     featured_data, horizons, model_type):
        """Plot the impact of different forecast horizons on prediction distribution."""
        
        predictions_by_horizon = {}
        feature_cols = [col for col in featured_data.columns if col not in ['target', 'asset_id']]
        
        n_samples = len(featured_data)
        test_start = int(n_samples * 0.8)
        test_data = featured_data.iloc[test_start:]
        
        if len(test_data) > 0:
            y_true = test_data['target'].values
            test_features = test_data[feature_cols]
            
            for horizon in horizons:
                if horizon in horizon_models and model_type in horizon_models[horizon]:
                    model = horizon_models[horizon][model_type]
                    predictions = model.predict(test_features)
                    residuals = y_true - predictions
                    predictions_by_horizon[f'{horizon}d'] = residuals
        
        if predictions_by_horizon:
            # Create box plot of residuals
            data_for_plot = [residuals for residuals in predictions_by_horizon.values()]
            labels = list(predictions_by_horizon.keys())
            
            box_plot = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = [self.colors.get(f'horizon_{h}', '#333333') for h in horizons]
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Prediction Horizon')
            ax.set_ylabel('Prediction Residuals (SGD)')
            ax.set_title('Prediction Error Distribution by Horizon')
            ax.grid(True, alpha=0.3)

    def create_horizon_performance_summary(
        self,
        multi_horizon_data: Dict,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Create a summary dashboard of multi-horizon performance."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Horizon Forecasting Performance Summary', fontsize=16, fontweight='bold')
        
        # Extract comparison data
        comparison_data = multi_horizon_data.get('comparison', [])
        
        # Plot 1: Success Rate by Horizon
        ax1 = axes[0, 0]
        if comparison_data:
            horizons = [int(item['horizon_days']) for item in comparison_data]
            success_rates = [item['success_rate'] * 100 for item in comparison_data]
            
            bars = ax1.bar(horizons, success_rates, color=[self.colors.get(f'horizon_{h}', '#333333') for h in horizons])
            ax1.set_xlabel('Prediction Horizon (Days)')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_title('Model Training Success Rate by Horizon')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Plot 2: Model Count by Horizon
        ax2 = axes[0, 1]
        if comparison_data:
            total_models = [item['total_models'] for item in comparison_data]
            successful_models = [item['successful_models'] for item in comparison_data]
            
            x = np.arange(len(horizons))
            width = 0.35
            
            ax2.bar(x - width/2, total_models, width, label='Total Models', alpha=0.8)
            ax2.bar(x + width/2, successful_models, width, label='Successful Models', alpha=0.8)
            
            ax2.set_xlabel('Prediction Horizon (Days)')
            ax2.set_ylabel('Number of Models')
            ax2.set_title('Model Training Statistics')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'{h}d' for h in horizons])
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Plot 3: Best Model Performance by Horizon (MAE)
        ax3 = axes[1, 0]
        self._plot_best_model_performance(ax3, multi_horizon_data)

        # Plot 4: Training Time vs Performance Trade-off
        ax4 = axes[1, 1]
        self._plot_training_time_vs_performance(ax4, multi_horizon_data)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved horizon performance summary to {save_path}")

        return fig

    def _plot_best_model_performance(self, ax, multi_horizon_data):
        """Plot best model performance across horizons."""
        detailed_results = multi_horizon_data.get('detailed_results', {})
        
        horizon_rmses = []
        horizons = []
        
        for horizon_key, results in detailed_results.items():
            if results.get('success') and 'summary' in results:
                best_models = results['summary'].get('best_models_by_asset', {})
                if best_models:
                    # Use RMSE as primary metric, fallback to MAE if RMSE not available
                    rmses = []
                    for model_info in best_models.values():
                        rmse = model_info.get('rmse')
                        if rmse is None or rmse == 'N/A':
                            # Fallback to MAE if RMSE not available (for older data)
                            rmse = model_info.get('mae', 0)
                        rmses.append(rmse)
                    
                    avg_rmse = np.mean(rmses)
                    
                    horizon_num = int(horizon_key.replace('_day', ''))
                    horizons.append(horizon_num)
                    horizon_rmses.append(avg_rmse)
        
        if horizons and horizon_rmses:
            # Sort by horizon
            sorted_data = sorted(zip(horizons, horizon_rmses))
            horizons, horizon_rmses = zip(*sorted_data)
            
            ax.plot(horizons, horizon_rmses, marker='o', linewidth=3, markersize=8)
            ax.fill_between(horizons, 0, horizon_rmses, alpha=0.3)
            
            ax.set_xlabel('Prediction Horizon (Days)')
            ax.set_ylabel('Average RMSE (SGD)')
            ax.set_title('Best Model Performance by Horizon (RMSE)')
            ax.grid(True, alpha=0.3)
            
            # Add performance annotations
            for i, (h, rmse) in enumerate(zip(horizons, horizon_rmses)):
                ax.annotate(f'{rmse:.1f}', (h, rmse), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontweight='bold')

    def _plot_training_time_vs_performance(self, ax, multi_horizon_data):
        """Plot training time vs performance trade-off."""
        detailed_results = multi_horizon_data.get('detailed_results', {})
        
        training_times = []
        rmses = []
        horizon_labels = []
        
        for horizon_key, results in detailed_results.items():
            if results.get('success') and 'summary' in results:
                best_models = results['summary'].get('best_models_by_asset', {})
                if best_models:
                    for model_info in best_models.values():
                        training_times.append(model_info.get('training_time', 0))
                        # Use RMSE as primary metric, fallback to MAE if RMSE not available
                        rmse = model_info.get('rmse')
                        if rmse is None or rmse == 'N/A':
                            rmse = model_info.get('mae', 0)
                        rmses.append(rmse)
                        horizon_labels.append(horizon_key.replace('_day', 'd'))
        
        if training_times and rmses:
            # Create scatter plot
            colors = [self.colors.get(f'horizon_{label[:-1]}', '#333333') for label in horizon_labels]
            scatter = ax.scatter(training_times, rmses, c=colors, alpha=0.6, s=50)
            
            ax.set_xlabel('Training Time (seconds)')
            ax.set_ylabel('RMSE (SGD)')
            ax.set_title('Training Time vs Performance Trade-off (RMSE)')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(training_times) > 1:
                z = np.polyfit(training_times, rmses, 1)
                p = np.poly1d(z)
                ax.plot(training_times, p(training_times), "r--", alpha=0.8, linewidth=2)


def load_horizon_models(base_dir: str, horizons: List[int], asset_name: str) -> Dict[int, Dict[str, BaseTimeSeriesModel]]:
    """Load trained models for different horizons."""
    horizon_models = {}
    
    for horizon in horizons:
        models_dir = Path(base_dir) / f"models_{horizon}_day" / asset_name
        if models_dir.exists():
            horizon_models[horizon] = {}
            for model_file in models_dir.glob("*.pkl"):
                model_name = model_file.stem
                try:
                    model = BaseTimeSeriesModel.load_model(model_file)
                    horizon_models[horizon][model_name] = model
                    logger.debug(f"Loaded {horizon}d/{model_name} for {asset_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_file}: {e}")
    
    return horizon_models


def main():
    """Main entry point for multi-horizon visualization."""
    parser = argparse.ArgumentParser(
        description="Create comprehensive multi-horizon forecasting visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ml.multi_horizon_visualizer
  python -m src.ml.multi_horizon_visualizer --asset "Audemars_Piguet-Steeloyal_Oak_Offshore-22378"
  python -m src.ml.multi_horizon_visualizer --horizons 7 14 30 --models linear ridge
  python -m src.ml.multi_horizon_visualizer --summary-only
        """
    )
    
    parser.add_argument(
        "--asset", 
        default="Audemars_Piguet-Steeloyal_Oak_Offshore-22378",
        help="Specific asset to visualize (default: Audemars_Piguet-Steeloyal_Oak_Offshore-22378)"
    )
    
    parser.add_argument(
        "--horizons", 
        nargs="+", 
        type=int, 
        default=[7, 14, 30],
        help="Prediction horizons to compare (default: 7 14 30)"
    )
    
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["linear", "ridge", "xgboost"],
        help="Model types to visualize (default: linear ridge xgboost)"
    )
    
    parser.add_argument(
        "--data-dir", 
        default="data/output",
        help="Data directory (default: data/output)"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="data/output/visualizations",
        help="Output directory (default: data/output/visualizations)"
    )
    
    parser.add_argument(
        "--summary-only", 
        action="store_true",
        help="Only create summary plots, not individual asset plots"
    )

    args = parser.parse_args()

    try:
        logger.info("🎨 MULTI-HORIZON FORECASTING VISUALIZATION")
        logger.info("=" * 60)

        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer
        viz = MultiHorizonVisualizer()

        # Load multi-horizon comparison data
        comparison_file = Path(args.data_dir) / "multi_horizon_comparison.json"
        if comparison_file.exists():
            with open(comparison_file, 'r') as f:
                multi_horizon_data = json.load(f)
            logger.info(f"Loaded multi-horizon comparison data")
        else:
            logger.error(f"Multi-horizon comparison file not found: {comparison_file}")
            sys.exit(1)

        # Create summary visualization
        logger.info("📊 Creating multi-horizon performance summary...")
        summary_path = get_organized_aggregate_path(
            str(output_path), "multi_horizon_performance_summary"
        )
        summary_fig = viz.create_horizon_performance_summary(
            multi_horizon_data, save_path=str(summary_path)
        )
        plt.close(summary_fig)
        logger.info(f"✅ Created performance summary: {summary_path}")

        if not args.summary_only:
            # Load featured data
            logger.info(f"📈 Loading featured data for asset analysis...")
            try:
                featured_data_dict = load_featured_data(args.data_dir)
            except Exception as e:
                logger.error(f"Failed to load featured data: {e}")
                sys.exit(1)

            # Create asset-specific visualizations
            if args.asset in featured_data_dict:
                logger.info(f"🔍 Creating detailed analysis for {args.asset}...")
                
                # Load models for different horizons
                horizon_models = load_horizon_models(args.data_dir, args.horizons, args.asset)
                
                if horizon_models:
                    for model_type in args.models:
                        # Check if model exists for at least one horizon
                        model_exists = any(model_type in models for models in horizon_models.values())
                        
                        if model_exists:
                            logger.info(f"  📊 Creating {model_type} model comparison...")
                            asset_path = get_organized_visualization_path(
                                str(output_path), args.asset, model_type, "multi_horizon_analysis"
                            )
                            
                            asset_fig = viz.plot_multi_horizon_comparison(
                                asset_name=args.asset,
                                horizon_models=horizon_models,
                                featured_data=featured_data_dict[args.asset],
                                horizons=args.horizons,
                                model_type=model_type,
                                save_path=str(asset_path)
                            )
                            plt.close(asset_fig)
                            logger.info(f"    ✅ Created: {asset_path}")
                        else:
                            logger.warning(f"  ⚠️  No {model_type} models found for {args.asset}")
                else:
                    logger.warning(f"⚠️  No models found for {args.asset}")
            else:
                logger.error(f"❌ Asset not found in featured data: {args.asset}")

        logger.info("\n🎉 MULTI-HORIZON VISUALIZATION COMPLETE!")
        logger.info(f"📁 Output directory: {args.output_dir}")
        logger.info(f"📊 Created visualizations show:")
        logger.info("  ✅ Complete time series context (no more single points!)")
        logger.info("  ✅ Multi-horizon model comparisons")
        logger.info("  ✅ Performance metrics across horizons")
        logger.info("  ✅ Proper date handling (no more 1970 dates!)")
        logger.info("  ✅ Forecast accuracy trends")

    except KeyboardInterrupt:
        logger.info("❌ Visualization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Visualization failed: {str(e)}")
        import traceback
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()