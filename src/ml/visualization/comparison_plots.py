"""
Model comparison visualization components.

This module creates comparative visualizations across models, assets,
and time periods.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import plotting libraries with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn(
        "Plotting libraries not available. Install matplotlib and seaborn for visualizations."
    )

logger = logging.getLogger(__name__)


class ModelComparisonVisualizer:
    """Create model comparison visualizations."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the comparison visualizer.

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

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")

        # Set up colors
        self.colors = sns.color_palette("husl", 10)

    def plot_asset_performance_heatmap(
        self,
        results_matrix: pd.DataFrame,
        metric: str = "mae",
        title: str = "Model Performance Across Assets",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a heatmap of model performance across different assets.

        Parameters:
        ----------
        results_matrix : pd.DataFrame
            Matrix with models as columns and assets as rows
        metric : str
            Metric being visualized
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(
            figsize=(
                max(12, len(results_matrix.columns) * 2),
                max(8, len(results_matrix) * 0.5),
            )
        )

        # Create heatmap
        sns.heatmap(
            results_matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            center=results_matrix.median().median(),
            cbar_kws={"label": f"{metric.upper()}"},
            ax=ax,
        )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Assets", fontsize=12)

        # Rotate labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved performance heatmap to {save_path}")

        return fig

    def plot_model_ranking(
        self,
        ranking_data: Dict[str, List[str]],
        title: str = "Model Rankings Across Assets",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot model rankings across different assets.

        Parameters:
        ----------
        ranking_data : Dict[str, List[str]]
            Rankings: {asset_name: [ranked_model_list]}
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Extract all unique models
        all_models = set()
        for rankings in ranking_data.values():
            all_models.update(rankings)
        all_models = sorted(list(all_models))

        # Create ranking matrix
        ranking_matrix = []
        asset_names = []

        for asset_name, rankings in ranking_data.items():
            asset_names.append(asset_name)
            asset_rankings = []

            for model in all_models:
                if model in rankings:
                    rank = rankings.index(model) + 1
                else:
                    rank = len(rankings) + 1  # Worst possible rank
                asset_rankings.append(rank)

            ranking_matrix.append(asset_rankings)

        # Create plot
        fig, ax = plt.subplots(
            figsize=(max(10, len(all_models) * 1.5), max(6, len(asset_names) * 0.4))
        )

        # Create heatmap with reversed colormap (lower rank = better = darker)
        sns.heatmap(
            ranking_matrix,
            annot=True,
            fmt="d",
            cmap="RdYlGn_r",
            xticklabels=all_models,
            yticklabels=asset_names,
            cbar_kws={"label": "Rank (1=Best)"},
            ax=ax,
        )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Models", fontsize=12)
        ax.set_ylabel("Assets", fontsize=12)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved model ranking to {save_path}")

        return fig

    def plot_cross_asset_predictions(
        self,
        predictions_data: Dict[str, Dict[str, np.ndarray]],
        actuals_data: Dict[str, np.ndarray],
        sample_assets: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot predictions vs actuals across multiple assets.

        Parameters:
        ----------
        predictions_data : Dict[str, Dict[str, np.ndarray]]
            Predictions: {asset_name: {model_name: predictions_array}}
        actuals_data : Dict[str, np.ndarray]
            Actual values: {asset_name: actuals_array}
        sample_assets : List[str], optional
            Assets to include (uses first 4 if None)
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Select assets to plot
        if sample_assets is None:
            sample_assets = list(predictions_data.keys())[:4]

        n_assets = len(sample_assets)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Predictions vs Actuals Across Assets", fontsize=16, fontweight="bold"
        )

        axes = axes.flatten()

        for i, asset_name in enumerate(sample_assets):
            if i >= 4:  # Limit to 4 subplots
                break

            ax = axes[i]

            if asset_name not in actuals_data:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {asset_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            actuals = actuals_data[asset_name]
            x_axis = range(len(actuals))

            # Plot actual values
            ax.plot(
                x_axis, actuals, label="Actual", color="black", linewidth=2, alpha=0.8
            )

            # Plot predictions for each model
            if asset_name in predictions_data:
                for j, (model_name, predictions) in enumerate(
                    predictions_data[asset_name].items()
                ):
                    if len(predictions) == len(actuals):
                        ax.plot(
                            x_axis,
                            predictions,
                            label=model_name,
                            color=self.colors[j % len(self.colors)],
                            linewidth=1.5,
                            alpha=0.7,
                        )

            # Clean asset name for title
            clean_name = asset_name.replace("_", " ").replace("-", " ")
            if len(clean_name) > 30:
                clean_name = clean_name[:30] + "..."

            ax.set_title(clean_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Period", fontsize=10)
            ax.set_ylabel("Price (SGD)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(len(sample_assets), 4):
            axes[j].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved cross-asset predictions to {save_path}")

        return fig

    def plot_metric_distribution(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        metric: str = "mae",
        title: str = "Metric Distribution Across Assets",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot distribution of a metric across all assets for each model.

        Parameters:
        ----------
        all_results : Dict[str, Dict[str, Dict]]
            All results: {asset_name: {model_name: {metrics}}}
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
        # Extract metric values for each model
        model_metrics = {}

        for asset_name, asset_results in all_results.items():
            for model_name, results in asset_results.items():
                if metric in results and not np.isnan(results[metric]):
                    if model_name not in model_metrics:
                        model_metrics[model_name] = []
                    model_metrics[model_name].append(results[metric])

        if not model_metrics:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(
                0.5,
                0.5,
                f"No data available for {metric}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Violin plot
        model_names = list(model_metrics.keys())
        metric_data = [model_metrics[model] for model in model_names]

        parts = ax1.violinplot(
            metric_data,
            positions=range(len(model_names)),
            showmeans=True,
            showmedians=True,
        )

        # Color the violin plots
        for pc, color in zip(parts["bodies"], self.colors[: len(model_names)]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha="right")
        ax1.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax1.set_title("Distribution (Violin Plot)", fontsize=14)
        ax1.grid(True, alpha=0.3, axis="y")

        # 2. Box plot with individual points
        bp = ax2.boxplot(metric_data, labels=model_names, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], self.colors[: len(model_names)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add individual points
        for i, (model_name, values) in enumerate(model_metrics.items()):
            y = values
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax2.scatter(x, y, alpha=0.6, s=20, color="black")

        ax2.set_xticklabels(model_names, rotation=45, ha="right")
        ax2.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax2.set_title("Distribution with Data Points", fontsize=14)
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved metric distribution to {save_path}")

        return fig

    def plot_win_rate_matrix(
        self,
        pairwise_comparisons: Dict[str, Dict[str, float]],
        title: str = "Model Win Rate Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot win rate matrix between models.

        Parameters:
        ----------
        pairwise_comparisons : Dict[str, Dict[str, float]]
            Win rates: {model1: {model2: win_rate}}
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Convert to DataFrame
        win_rate_df = pd.DataFrame(pairwise_comparisons).fillna(
            0.5
        )  # 0.5 for self-comparison

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            win_rate_df,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0.5,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Win Rate"},
            ax=ax,
        )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Opponent Model", fontsize=12)
        ax.set_ylabel("Base Model", fontsize=12)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved win rate matrix to {save_path}")

        return fig

    def create_comprehensive_comparison(
        self,
        training_results: Dict[str, Dict[str, Any]],
        output_dir: str = "data/output/visualizations",
    ) -> Dict[str, str]:
        """
        Create comprehensive model comparison visualizations.

        Parameters:
        ----------
        training_results : Dict[str, Dict[str, Any]]
            Complete training results: {asset_name: {model_name: result}}
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

        # 1. Extract successful results
        successful_results = {}
        for asset_name, asset_results in training_results.items():
            successful_results[asset_name] = {}
            for model_name, result in asset_results.items():
                if result.success and result.validation_result:
                    successful_results[asset_name][model_name] = (
                        result.validation_result.metrics
                    )

        if not successful_results:
            logger.warning("No successful results found for comparison")
            return plot_files

        # 2. Create performance heatmap
        metrics_df_data = []
        for asset_name, models in successful_results.items():
            for model_name, metrics in models.items():
                row = {"Asset": asset_name, "Model": model_name}
                row.update(metrics)
                metrics_df_data.append(row)

        if metrics_df_data:
            metrics_df = pd.DataFrame(metrics_df_data)

            for metric in ["mae", "rmse", "r2", "mape"]:
                if metric in metrics_df.columns:
                    pivot_df = metrics_df.pivot(
                        index="Asset", columns="Model", values=metric
                    )

                    if not pivot_df.empty:
                        heatmap_file = output_path / f"performance_heatmap_{metric}.png"
                        fig1 = self.plot_asset_performance_heatmap(
                            pivot_df,
                            metric=metric,
                            title=f"Model Performance ({metric.upper()}) Across Assets",
                            save_path=str(heatmap_file),
                        )
                        plt.close(fig1)
                        plot_files[f"heatmap_{metric}"] = str(heatmap_file)

        # 3. Create metric distribution plots
        for metric in ["mae", "rmse", "r2"]:
            if any(
                metric in models.values()
                for models in successful_results.values()
                for models in [models]
            ):
                dist_file = output_path / f"metric_distribution_{metric}.png"
                fig2 = self.plot_metric_distribution(
                    successful_results,
                    metric=metric,
                    title=f"{metric.upper()} Distribution Across All Assets",
                    save_path=str(dist_file),
                )
                plt.close(fig2)
                plot_files[f"distribution_{metric}"] = str(dist_file)

        # 4. Create model ranking visualization
        rankings = {}
        for asset_name, models in successful_results.items():
            # Sort models by MAE (lower is better)
            mae_scores = {
                model: metrics.get("mae", float("inf"))
                for model, metrics in models.items()
            }
            ranked_models = sorted(mae_scores.keys(), key=lambda x: mae_scores[x])
            rankings[asset_name] = ranked_models

        if rankings:
            ranking_file = output_path / "model_rankings.png"
            fig3 = self.plot_model_ranking(
                rankings,
                title="Model Rankings (by MAE) Across Assets",
                save_path=str(ranking_file),
            )
            plt.close(fig3)
            plot_files["rankings"] = str(ranking_file)

        # 5. Calculate and plot win rates
        all_models = set()
        for models in successful_results.values():
            all_models.update(models.keys())
        all_models = list(all_models)

        if len(all_models) > 1:
            win_rates = {}
            for model1 in all_models:
                win_rates[model1] = {}
                for model2 in all_models:
                    if model1 == model2:
                        win_rates[model1][model2] = 0.5
                    else:
                        wins = 0
                        comparisons = 0

                        for asset_name, models in successful_results.items():
                            if model1 in models and model2 in models:
                                mae1 = models[model1].get("mae", float("inf"))
                                mae2 = models[model2].get("mae", float("inf"))

                                if mae1 < mae2:
                                    wins += 1
                                comparisons += 1

                        win_rates[model1][model2] = (
                            wins / comparisons if comparisons > 0 else 0.5
                        )

            winrate_file = output_path / "win_rate_matrix.png"
            fig4 = self.plot_win_rate_matrix(
                win_rates,
                title="Model Win Rate Matrix (based on MAE)",
                save_path=str(winrate_file),
            )
            plt.close(fig4)
            plot_files["win_rates"] = str(winrate_file)

        logger.info(f"Created comprehensive comparison with {len(plot_files)} plots")

        return plot_files
