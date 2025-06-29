"""
Enhanced forecasting visualization components.

This module provides advanced visualization capabilities specifically designed
for time series forecasting results, including interactive plots, ensemble
analysis, and comprehensive reporting.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class EnhancedForecastingVisualizer:
    """Advanced forecasting visualization capabilities."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (14, 8)):
        """
        Initialize the enhanced forecasting visualizer.

        Parameters:
        ----------
        style : str
            Matplotlib style to use
        figsize : Tuple[int, int]
            Default figure size
        """

        self.style = style
        self.figsize = figsize

        # Enhanced color palette for forecasting
        self.colors = {
            "actual": "#1f77b4",  # Blue
            "predicted": "#ff7f0e",  # Orange
            "ensemble": "#2ca02c",  # Green
            "confidence_95": "#d62728",  # Red
            "confidence_50": "#9467bd",  # Purple
            "train": "#8c564b",  # Brown
            "validation": "#e377c2",  # Pink
            "test": "#7f7f7f",  # Gray
            "trend": "#bcbd22",  # Olive
            "seasonal": "#17becf",  # Cyan
        }

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")

    def plot_forecast_decomposition(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        trend_component: Optional[np.ndarray] = None,
        seasonal_component: Optional[np.ndarray] = None,
        residual_component: Optional[np.ndarray] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Forecast Decomposition Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a comprehensive forecast decomposition plot.

        Parameters:
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        trend_component : np.ndarray, optional
            Trend component of the forecast
        seasonal_component : np.ndarray, optional
            Seasonal component of the forecast
        residual_component : np.ndarray, optional
            Residual component
        dates : pd.DatetimeIndex, optional
            Date index for x-axis
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Determine number of subplots needed
        n_plots = 2  # Always have actual vs predicted and residuals
        if trend_component is not None:
            n_plots += 1
        if seasonal_component is not None:
            n_plots += 1

        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], n_plots * 3))
        if n_plots == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight="bold")

        x_axis = dates if dates is not None else range(len(y_true))
        plot_idx = 0

        # 1. Actual vs Predicted
        ax = axes[plot_idx]
        ax.plot(
            x_axis, y_true, label="Actual", color=self.colors["actual"], linewidth=2
        )
        ax.plot(
            x_axis,
            y_pred,
            label="Predicted",
            color=self.colors["predicted"],
            linewidth=2,
            alpha=0.8,
        )

        # Add error bands
        errors = np.abs(y_true - y_pred)
        ax.fill_between(
            x_axis,
            y_pred - errors,
            y_pred + errors,
            alpha=0.2,
            color=self.colors["confidence_95"],
            label="Error Band",
        )

        ax.set_title("Forecast vs Actual", fontsize=12, fontweight="bold")
        ax.set_ylabel("Price (SGD)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # 2. Trend Component (if available)
        if trend_component is not None:
            ax = axes[plot_idx]
            ax.plot(x_axis, trend_component, color=self.colors["trend"], linewidth=2)
            ax.set_title("Trend Component", fontsize=12, fontweight="bold")
            ax.set_ylabel("Trend Value")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # 3. Seasonal Component (if available)
        if seasonal_component is not None:
            ax = axes[plot_idx]
            ax.plot(
                x_axis, seasonal_component, color=self.colors["seasonal"], linewidth=2
            )
            ax.set_title("Seasonal Component", fontsize=12, fontweight="bold")
            ax.set_ylabel("Seasonal Value")
            ax.grid(True, alpha=0.3)
            plot_idx += 1

        # 4. Residuals
        ax = axes[plot_idx]
        residuals = y_true - y_pred
        ax.plot(x_axis, residuals, color=self.colors["test"], alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.8)
        ax.fill_between(x_axis, residuals, 0, alpha=0.3, color=self.colors["test"])
        ax.set_title("Forecast Residuals", fontsize=12, fontweight="bold")
        ax.set_ylabel("Residual Value")
        ax.set_xlabel("Date" if dates is not None else "Time Period")
        ax.grid(True, alpha=0.3)

        # Format dates if provided
        if dates is not None:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved forecast decomposition to {save_path}")

        return fig

    def plot_ensemble_forecast_analysis(
        self,
        individual_predictions: Dict[str, np.ndarray],
        ensemble_prediction: np.ndarray,
        y_true: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Ensemble Forecast Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Analyze ensemble forecasting results.

        Parameters:
        ----------
        individual_predictions : Dict[str, np.ndarray]
            Individual model predictions: {model_name: predictions}
        ensemble_prediction : np.ndarray
            Combined ensemble prediction
        y_true : np.ndarray
            True values
        weights : Dict[str, float], optional
            Model weights in ensemble
        dates : pd.DatetimeIndex, optional
            Date index
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        x_axis = dates if dates is not None else range(len(y_true))

        # 1. All predictions comparison
        ax1.plot(
            x_axis,
            y_true,
            label="Actual",
            color=self.colors["actual"],
            linewidth=3,
            alpha=0.9,
        )

        # Plot individual model predictions
        colors = plt.cm.Set3(np.linspace(0, 1, len(individual_predictions)))
        for i, (model_name, predictions) in enumerate(individual_predictions.items()):
            ax1.plot(
                x_axis,
                predictions,
                label=model_name,
                color=colors[i],
                linewidth=1.5,
                alpha=0.7,
            )

        # Plot ensemble prediction
        ax1.plot(
            x_axis,
            ensemble_prediction,
            label="Ensemble",
            color=self.colors["ensemble"],
            linewidth=2.5,
            alpha=0.9,
        )

        ax1.set_title("Individual Models vs Ensemble", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Price (SGD)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(True, alpha=0.3)

        # 2. Model weights (if available)
        if weights:
            model_names = list(weights.keys())
            weight_values = list(weights.values())

            bars = ax2.bar(
                model_names, weight_values, color=colors[: len(model_names)], alpha=0.7
            )
            ax2.set_title("Ensemble Model Weights", fontsize=12, fontweight="bold")
            ax2.set_ylabel("Weight")
            ax2.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, weight in zip(bars, weight_values):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + max(weight_values) * 0.01,
                    f"{weight:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        else:
            ax2.text(
                0.5,
                0.5,
                "No weight information available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
        ax2.grid(True, alpha=0.3)

        # 3. Error comparison
        ensemble_errors = np.abs(y_true - ensemble_prediction)
        individual_errors = {
            name: np.abs(y_true - pred) for name, pred in individual_predictions.items()
        }

        error_data = list(individual_errors.values()) + [ensemble_errors]
        error_labels = list(individual_errors.keys()) + ["Ensemble"]

        bp = ax3.boxplot(error_data, labels=error_labels, patch_artist=True)

        # Color the boxes
        all_colors = list(colors[: len(individual_predictions)]) + [
            self.colors["ensemble"]
        ]
        for patch, color in zip(bp["boxes"], all_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax3.set_title("Prediction Error Distribution", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Absolute Error")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3, axis="y")

        # 4. Prediction variance over time
        prediction_matrix = np.array(list(individual_predictions.values()))
        prediction_std = np.std(prediction_matrix, axis=0)

        ax4.plot(
            x_axis, prediction_std, color=self.colors["confidence_95"], linewidth=2
        )
        ax4.fill_between(
            x_axis, 0, prediction_std, alpha=0.3, color=self.colors["confidence_95"]
        )
        ax4.set_title(
            "Prediction Uncertainty (Model Disagreement)",
            fontsize=12,
            fontweight="bold",
        )
        ax4.set_ylabel("Standard Deviation")
        ax4.set_xlabel("Date" if dates is not None else "Time Period")
        ax4.grid(True, alpha=0.3)

        # Format dates if provided
        if dates is not None:
            for ax in [ax1, ax4]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ensemble analysis to {save_path}")

        return fig

    def plot_forecast_confidence_evolution(
        self,
        confidence_data: Dict[str, Dict[str, np.ndarray]],
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Forecast Confidence Evolution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Show how forecast confidence changes over time and horizon.

        Parameters:
        ----------
        confidence_data : Dict[str, Dict[str, np.ndarray]]
            Confidence data: {model_name: {'lower_95': array, 'upper_95': array, 'predictions': array}}
        dates : pd.DatetimeIndex, optional
            Date index
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        n_models = len(confidence_data)
        if n_models == 0:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(
                0.5,
                0.5,
                "No confidence data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        fig, axes = plt.subplots(n_models, 1, figsize=(self.figsize[0], n_models * 4))
        if n_models == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=16, fontweight="bold")

        x_axis = dates if dates is not None else None

        for i, (model_name, data) in enumerate(confidence_data.items()):
            ax = axes[i]

            predictions = data.get("predictions", np.array([]))
            lower_95 = data.get("lower_95", np.array([]))
            upper_95 = data.get("upper_95", np.array([]))
            lower_50 = data.get("lower_50", np.array([]))
            upper_50 = data.get("upper_50", np.array([]))

            if len(predictions) == 0:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {model_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            if x_axis is None:
                x_axis = range(len(predictions))

            # Plot predictions
            ax.plot(
                x_axis,
                predictions,
                label="Prediction",
                color=self.colors["predicted"],
                linewidth=2,
            )

            # Plot confidence intervals
            if len(lower_95) == len(predictions) and len(upper_95) == len(predictions):
                ax.fill_between(
                    x_axis,
                    lower_95,
                    upper_95,
                    alpha=0.2,
                    color=self.colors["confidence_95"],
                    label="95% Confidence",
                )

            if len(lower_50) == len(predictions) and len(upper_50) == len(predictions):
                ax.fill_between(
                    x_axis,
                    lower_50,
                    upper_50,
                    alpha=0.3,
                    color=self.colors["confidence_50"],
                    label="50% Confidence",
                )

            # Calculate and show confidence interval width evolution
            if len(lower_95) == len(upper_95) == len(predictions):
                ci_width = upper_95 - lower_95
                ax2 = ax.twinx()
                ax2.plot(x_axis, ci_width, "--", color="red", alpha=0.7, linewidth=1)
                ax2.set_ylabel("95% CI Width", color="red", fontsize=10)
                ax2.tick_params(axis="y", labelcolor="red")

            ax.set_title(
                f"{model_name} - Confidence Evolution", fontsize=12, fontweight="bold"
            )
            ax.set_ylabel("Price (SGD)")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

            if i == len(confidence_data) - 1:  # Only on last subplot
                ax.set_xlabel("Date" if dates is not None else "Time Period")

        # Format dates if provided
        if dates is not None:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved confidence evolution to {save_path}")

        return fig

    def plot_forecast_error_heatmap(
        self,
        error_matrix: pd.DataFrame,
        title: str = "Forecast Error Heatmap",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create a heatmap of forecast errors across time and models.

        Parameters:
        ----------
        error_matrix : pd.DataFrame
            Error matrix with time as index and models as columns
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
                max(12, len(error_matrix.columns) * 1.5),
                max(8, len(error_matrix) * 0.1),
            )
        )

        # Create heatmap
        sns.heatmap(
            error_matrix.T,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Forecast Error"},
            ax=ax,
        )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Time Period", fontsize=12)
        ax.set_ylabel("Models", fontsize=12)

        # Format x-axis if datetime index
        if isinstance(error_matrix.index, pd.DatetimeIndex):
            # Sample x-axis labels to avoid overcrowding
            n_ticks = min(10, len(error_matrix))
            tick_positions = np.linspace(0, len(error_matrix) - 1, n_ticks, dtype=int)
            tick_labels = [
                error_matrix.index[i].strftime("%Y-%m") for i in tick_positions
            ]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved error heatmap to {save_path}")

        return fig

    def create_enhanced_forecast_dashboard(
        self,
        forecast_results: Dict[str, Any],
        output_dir: str = "data/output/visualizations/enhanced",
    ) -> Dict[str, str]:
        """
        Create a comprehensive enhanced forecasting dashboard.

        Parameters:
        ----------
        forecast_results : Dict[str, Any]
            Complete forecast results with all analysis data
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

        logger.info("Creating enhanced forecasting dashboard...")

        # Extract common data
        y_true = forecast_results.get("actual_values", np.array([]))
        dates = forecast_results.get("dates", None)

        # 1. Forecast decomposition (if components available)
        if "decomposition" in forecast_results:
            decomp = forecast_results["decomposition"]
            decomp_file = output_path / "forecast_decomposition.png"

            fig1 = self.plot_forecast_decomposition(
                y_true=y_true,
                y_pred=forecast_results.get("predictions", np.array([])),
                trend_component=decomp.get("trend"),
                seasonal_component=decomp.get("seasonal"),
                residual_component=decomp.get("residual"),
                dates=dates,
                save_path=str(decomp_file),
            )
            plt.close(fig1)
            plot_files["decomposition"] = str(decomp_file)

        # 2. Ensemble analysis (if multiple models)
        if "ensemble_data" in forecast_results:
            ensemble_data = forecast_results["ensemble_data"]
            ensemble_file = output_path / "ensemble_analysis.png"

            fig2 = self.plot_ensemble_forecast_analysis(
                individual_predictions=ensemble_data.get("individual_predictions", {}),
                ensemble_prediction=ensemble_data.get(
                    "ensemble_prediction", np.array([])
                ),
                y_true=y_true,
                weights=ensemble_data.get("weights"),
                dates=dates,
                save_path=str(ensemble_file),
            )
            plt.close(fig2)
            plot_files["ensemble"] = str(ensemble_file)

        # 3. Confidence evolution
        if "confidence_data" in forecast_results:
            confidence_file = output_path / "confidence_evolution.png"

            fig3 = self.plot_forecast_confidence_evolution(
                confidence_data=forecast_results["confidence_data"],
                dates=dates,
                save_path=str(confidence_file),
            )
            plt.close(fig3)
            plot_files["confidence"] = str(confidence_file)

        # 4. Error heatmap
        if "error_matrix" in forecast_results:
            error_file = output_path / "forecast_error_heatmap.png"

            fig4 = self.plot_forecast_error_heatmap(
                error_matrix=forecast_results["error_matrix"], save_path=str(error_file)
            )
            plt.close(fig4)
            plot_files["error_heatmap"] = str(error_file)

        logger.info(f"Created enhanced dashboard with {len(plot_files)} plots")

        return plot_files
