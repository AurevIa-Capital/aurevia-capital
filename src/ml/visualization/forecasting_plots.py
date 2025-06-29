"""
Forecasting visualization components.

This module creates comprehensive plots for time series forecasting
including predictions, confidence intervals, and residual analysis.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import plotting libraries with fallbacks
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..base import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


class ForecastingVisualizer:
    """Create comprehensive forecasting visualizations."""

    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the forecasting visualizer.

        Parameters:
        ----------
        style : str
            Matplotlib style to use
        figsize : Tuple[int, int]
            Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = {
            "actual": "#2E86AB",
            "predicted": "#A23B72",
            "confidence": "#F18F01",
            "residuals": "#C73E1D",
            "train": "#4CAF50",
            "validation": "#FF9800",
            "test": "#F44336",
        }

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use("default")

    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Predictions vs Actual",
        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create prediction vs actual time series plot.

        Parameters:
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        dates : pd.DatetimeIndex, optional
            Date index for x-axis
        title : str
            Plot title
        confidence_intervals : Tuple[np.ndarray, np.ndarray], optional
            Lower and upper confidence intervals
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Use dates or range index
        x_axis = dates if dates is not None else range(len(y_true))

        # Plot actual vs predicted
        ax.plot(
            x_axis,
            y_true,
            label="Actual",
            color=self.colors["actual"],
            linewidth=2,
            alpha=0.8,
        )
        ax.plot(
            x_axis,
            y_pred,
            label="Predicted",
            color=self.colors["predicted"],
            linewidth=2,
            alpha=0.8,
        )

        # Add confidence intervals if provided
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            ax.fill_between(
                x_axis,
                lower,
                upper,
                alpha=0.3,
                color=self.colors["confidence"],
                label="95% Confidence Interval",
            )

        # Formatting
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Date" if dates is not None else "Time Period", fontsize=12)
        ax.set_ylabel("Price (SGD)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format dates if provided
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved prediction plot to {save_path}")

        return fig

    def plot_residual_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Residual Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive residual analysis plots.

        Parameters:
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
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
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # 1. Residuals over time
        x_axis = dates if dates is not None else range(len(residuals))
        axes[0, 0].plot(x_axis, residuals, color=self.colors["residuals"], alpha=0.7)
        axes[0, 0].axhline(y=0, color="black", linestyle="--", alpha=0.8)
        axes[0, 0].set_title("Residuals Over Time")
        axes[0, 0].set_xlabel("Date" if dates is not None else "Time Period")
        axes[0, 0].set_ylabel("Residuals (SGD)")
        axes[0, 0].grid(True, alpha=0.3)

        if dates is not None:
            axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

        # 2. Residual distribution
        axes[0, 1].hist(
            residuals,
            bins=30,
            alpha=0.7,
            color=self.colors["residuals"],
            edgecolor="black",
        )
        axes[0, 1].axvline(x=0, color="black", linestyle="--", alpha=0.8)
        axes[0, 1].set_title("Residual Distribution")
        axes[0, 1].set_xlabel("Residuals (SGD)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)

        # Add normal distribution overlay
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        # Scale to match histogram
        y = y * len(residuals) * (residuals.max() - residuals.min()) / 30
        axes[0, 1].plot(x, y, "r-", linewidth=2, label="Normal Distribution")
        axes[0, 1].legend()

        # 3. Q-Q plot (approximate)
        from scipy import stats

        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normal Distribution)")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Residuals vs Predicted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6, color=self.colors["residuals"])
        axes[1, 1].axhline(y=0, color="black", linestyle="--", alpha=0.8)
        axes[1, 1].set_title("Residuals vs Predicted Values")
        axes[1, 1].set_xlabel("Predicted Values (SGD)")
        axes[1, 1].set_ylabel("Residuals (SGD)")
        axes[1, 1].grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(y_pred, p(y_pred), "r--", alpha=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved residual analysis to {save_path}")

        return fig

    def plot_train_val_test_split(
        self,
        data: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        price_column: str = "target",
        title: str = "Train/Validation/Test Split",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Visualize the train/validation/test split.

        Parameters:
        ----------
        data : pd.DataFrame
            Time series data
        train_size : float
            Training set proportion
        val_size : float
            Validation set proportion
        price_column : str
            Price column name
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

        n_samples = len(data)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + val_size))

        # Plot price series
        ax.plot(data.index, data[price_column], color="gray", alpha=0.7, linewidth=1)

        # Highlight different splits
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        ax.plot(
            train_data.index,
            train_data[price_column],
            color=self.colors["train"],
            linewidth=2,
            label=f"Training ({len(train_data)} points)",
        )
        ax.plot(
            val_data.index,
            val_data[price_column],
            color=self.colors["validation"],
            linewidth=2,
            label=f"Validation ({len(val_data)} points)",
        )
        ax.plot(
            test_data.index,
            test_data[price_column],
            color=self.colors["test"],
            linewidth=2,
            label=f"Test ({len(test_data)} points)",
        )

        # Add vertical lines to separate splits
        if len(train_data) > 0:
            ax.axvline(x=train_data.index[-1], color="red", linestyle="--", alpha=0.7)
        if len(val_data) > 0:
            ax.axvline(x=val_data.index[-1], color="red", linestyle="--", alpha=0.7)

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (SGD)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format dates
        if isinstance(data.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved split visualization to {save_path}")

        return fig

    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot feature importance.

        Parameters:
        ----------
        feature_importance : Dict[str, float]
            Feature importance scores
        title : str
            Plot title
        top_n : int
            Number of top features to show
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
        )[:top_n]

        features, importance = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.3)))

        # Create horizontal bar plot
        bars = ax.barh(range(len(features)), importance, color=self.colors["predicted"])

        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for bar, value in zip(bars, importance):
            width = bar.get_width()
            ax.text(
                width + max(importance) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {save_path}")

        return fig

    def plot_prediction_horizon_comparison(
        self,
        horizons_data: Dict[str, Dict],
        metric: str = "mae",
        title: str = "Performance by Prediction Horizon",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare model performance across different prediction horizons.

        Parameters:
        ----------
        horizons_data : Dict[str, Dict]
            Data for different horizons {horizon: {model: metrics}}
        metric : str
            Metric to compare
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

        # Extract data for plotting
        horizons = []
        models = set()

        for horizon, model_data in horizons_data.items():
            horizons.append(int(horizon.replace("_day", "")))
            models.update(model_data.keys())

        models = sorted(list(models))
        horizons = sorted(horizons)

        # Create data matrix
        data_matrix = []
        for model in models:
            model_scores = []
            for horizon in horizons:
                horizon_key = f"{horizon}_day"
                if horizon_key in horizons_data and model in horizons_data[horizon_key]:
                    score = horizons_data[horizon_key][model].get(metric, np.nan)
                    model_scores.append(score)
                else:
                    model_scores.append(np.nan)
            data_matrix.append(model_scores)

        # Plot lines for each model
        for i, model in enumerate(models):
            valid_indices = ~np.isnan(data_matrix[i])
            if np.any(valid_indices):
                x_vals = np.array(horizons)[valid_indices]
                y_vals = np.array(data_matrix[i])[valid_indices]
                ax.plot(
                    x_vals, y_vals, marker="o", linewidth=2, label=model, markersize=8
                )

        ax.set_xlabel("Prediction Horizon (Days)", fontsize=12)
        ax.set_ylabel(f"{metric.upper()}", fontsize=12)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(horizons)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved horizon comparison to {save_path}")

        return fig

    def plot_forecast_with_uncertainty(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        prediction_intervals: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        title: str = "Forecast with Uncertainty Bands",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create advanced forecast plot with multiple uncertainty bands.

        Parameters:
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        dates : pd.DatetimeIndex, optional
            Date index for x-axis
        prediction_intervals : Dict[str, Tuple[np.ndarray, np.ndarray]], optional
            Multiple prediction intervals: {'50%': (lower, upper), '95%': (lower, upper)}
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

        # Use dates or range index
        x_axis = dates if dates is not None else range(len(y_true))

        # Plot actual vs predicted
        ax.plot(
            x_axis,
            y_true,
            label="Actual",
            color=self.colors["actual"],
            linewidth=2.5,
            alpha=0.9,
        )
        ax.plot(
            x_axis,
            y_pred,
            label="Predicted",
            color=self.colors["predicted"],
            linewidth=2,
            alpha=0.8,
        )

        # Add multiple prediction intervals if provided
        if prediction_intervals:
            # Sort intervals by confidence level (widest first)
            sorted_intervals = sorted(
                prediction_intervals.items(), key=lambda x: x[0], reverse=True
            )

            alphas = [0.15, 0.25, 0.35]  # Different transparency levels
            colors = ["#FFB74D", "#FF8A65", "#FFAB91"]  # Different colors

            for i, (level, (lower, upper)) in enumerate(sorted_intervals):
                alpha = alphas[i % len(alphas)]
                color = colors[i % len(colors)]
                ax.fill_between(
                    x_axis,
                    lower,
                    upper,
                    alpha=alpha,
                    color=color,
                    label=f"{level} Prediction Interval",
                )

        # Enhanced formatting
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Date" if dates is not None else "Time Period", fontsize=12)
        ax.set_ylabel("Price (SGD)", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        # Format dates if provided
        if dates is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Add statistics box
        if len(y_true) > 0 and len(y_pred) > 0:
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            textstr = f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
            ax.text(
                0.02,
                0.98,
                textstr,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved uncertainty forecast plot to {save_path}")

        return fig

    def plot_multi_horizon_forecast(
        self,
        horizon_results: Dict[int, Dict[str, np.ndarray]],
        actual_values: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Multi-Horizon Forecasts",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot forecasts for multiple prediction horizons.

        Parameters:
        ----------
        horizon_results : Dict[int, Dict[str, np.ndarray]]
            Results by horizon: {horizon_days: {'predictions': array, 'dates': array}}
        actual_values : np.ndarray
            Actual values for comparison
        dates : pd.DatetimeIndex, optional
            Base date index
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot actual values
        base_x = dates if dates is not None else range(len(actual_values))
        ax.plot(
            base_x, actual_values, label="Actual", color="black", linewidth=3, alpha=0.8
        )

        # Color palette for different horizons
        horizon_colors = plt.cm.viridis(np.linspace(0, 1, len(horizon_results)))

        # Plot each horizon forecast
        for i, (horizon, results) in enumerate(sorted(horizon_results.items())):
            predictions = results["predictions"]
            pred_dates = results.get("dates", None)

            if pred_dates is not None:
                x_axis = pred_dates
            else:
                # Offset predictions by horizon days
                if dates is not None and isinstance(dates, pd.DatetimeIndex):
                    start_date = dates[-1] + pd.Timedelta(days=1)
                    x_axis = pd.date_range(
                        start=start_date, periods=len(predictions), freq="D"
                    )
                else:
                    x_axis = range(
                        len(actual_values), len(actual_values) + len(predictions)
                    )

            ax.plot(
                x_axis,
                predictions,
                label=f"{horizon}-day forecast",
                color=horizon_colors[i],
                linewidth=2,
                alpha=0.7,
                marker="o",
                markersize=4,
            )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Date" if dates is not None else "Time Period", fontsize=12)
        ax.set_ylabel("Price (SGD)", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        # Format dates if provided
        if dates is not None or any(
            "dates" in results for results in horizon_results.values()
        ):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved multi-horizon forecast to {save_path}")

        return fig

    def plot_forecast_accuracy_evolution(
        self,
        rolling_metrics: pd.DataFrame,
        metric: str = "mae",
        window_size: int = 30,
        title: str = "Forecast Accuracy Evolution",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot how forecast accuracy evolves over time.

        Parameters:
        ----------
        rolling_metrics : pd.DataFrame
            Rolling metrics with datetime index and model columns
        metric : str
            Metric to plot (mae, rmse, etc.)
        window_size : int
            Rolling window size for display
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

        # Plot rolling accuracy for each model
        for i, model_name in enumerate(rolling_metrics.columns):
            if model_name in rolling_metrics.columns:
                values = rolling_metrics[model_name].dropna()
                if len(values) > 0:
                    ax.plot(
                        values.index,
                        values.values,
                        label=f"{model_name}",
                        color=self.colors[i % len(self.colors)],
                        linewidth=2,
                        alpha=0.8,
                    )

        ax.set_title(
            f"{title} ({window_size}-day rolling {metric.upper()})",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(f"Rolling {metric.upper()}", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        if isinstance(rolling_metrics.index, pd.DatetimeIndex):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved accuracy evolution plot to {save_path}")

        return fig

    def plot_full_forecast_with_context(
        self,
        full_data: pd.DataFrame,
        trained_model,
        target_column: str = "target",
        test_size: float = 0.2,
        validation_size: float = 0.15,
        title: str = "Complete Forecast Analysis",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Create comprehensive forecasting plot showing full time series context.

        Parameters:
        ----------
        full_data : pd.DataFrame
            Complete dataset with all periods
        trained_model : BaseTimeSeriesModel
            Trained model for predictions
        target_column : str
            Target column name
        test_size : float
            Test set proportion
        validation_size : float
            Validation set proportion
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot

        Returns:
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=(16, 8))

        # Calculate split indices (same logic as in training)
        n_samples = len(full_data)
        test_start_idx = int(n_samples * (1 - test_size))
        val_start_idx = int(n_samples * (1 - test_size - validation_size))

        # Split the data
        train_data = full_data.iloc[:val_start_idx]
        val_data = full_data.iloc[val_start_idx:test_start_idx]
        test_data = full_data.iloc[test_start_idx:]

        # Get feature columns (exclude target)
        feature_cols = [
            col for col in full_data.columns if col not in [target_column, "asset_id"]
        ]

        # Try to reconstruct proper dates for x-axis
        use_reconstructed_dates = False
        if (
            not isinstance(full_data.index, pd.DatetimeIndex)
            and "year" in full_data.columns
            and "month" in full_data.columns
        ):
            try:
                day_col = (
                    "day_of_month" if "day_of_month" in full_data.columns else "day"
                )
                if day_col in full_data.columns:
                    reconstructed_dates = pd.to_datetime(
                        full_data[["year", "month", day_col]]
                    )

                    # Update data with proper dates as index
                    train_data = train_data.copy()
                    val_data = val_data.copy()
                    test_data = test_data.copy()

                    if len(train_data) > 0:
                        train_data.index = reconstructed_dates[: len(train_data)]
                    if len(val_data) > 0:
                        val_start_idx_date = len(train_data)
                        val_end_idx_date = val_start_idx_date + len(val_data)
                        val_data.index = reconstructed_dates[
                            val_start_idx_date:val_end_idx_date
                        ]
                    if len(test_data) > 0:
                        test_start_idx_date = len(train_data) + len(val_data)
                        test_data.index = reconstructed_dates[test_start_idx_date:]

                    use_reconstructed_dates = True
                    logger.debug(
                        f"Reconstructed dates from {reconstructed_dates.min()} to {reconstructed_dates.max()}"
                    )

            except Exception as e:
                logger.debug(f"Could not reconstruct dates: {e}")

        # Plot actual values for all periods
        if len(train_data) > 0:
            ax.plot(
                train_data.index,
                train_data[target_column],
                color=self.colors["train"],
                linewidth=2,
                alpha=0.8,
                label=f"Training Data ({len(train_data)} points)",
            )

        if len(val_data) > 0:
            ax.plot(
                val_data.index,
                val_data[target_column],
                color=self.colors["validation"],
                linewidth=2,
                alpha=0.8,
                label=f"Validation Actual ({len(val_data)} points)",
            )

        if len(test_data) > 0:
            ax.plot(
                test_data.index,
                test_data[target_column],
                color=self.colors["test"],
                linewidth=2,
                alpha=0.8,
                label=f"Test Actual ({len(test_data)} points)",
            )

        # Generate and plot predictions for validation and test periods
        try:
            if len(val_data) > 0 and len(feature_cols) > 0:
                val_features = val_data[feature_cols]
                if len(val_features.columns) > 0:
                    val_predictions = trained_model.predict(val_features)
                    ax.plot(
                        val_data.index,
                        val_predictions,
                        color="orange",
                        linewidth=2,
                        alpha=0.9,
                        linestyle="--",
                        label="Validation Predictions",
                    )
        except Exception as e:
            logger.debug(f"Could not generate validation predictions: {e}")

        try:
            if len(test_data) > 0 and len(feature_cols) > 0:
                test_features = test_data[feature_cols]
                if len(test_features.columns) > 0:
                    test_predictions = trained_model.predict(test_features)
                    ax.plot(
                        test_data.index,
                        test_predictions,
                        color="red",
                        linewidth=2,
                        alpha=0.9,
                        linestyle="--",
                        label="Test Predictions",
                    )
        except Exception as e:
            logger.debug(f"Could not generate test predictions: {e}")

        # Add vertical lines to mark split boundaries
        if len(train_data) > 0 and len(val_data) > 0:
            ax.axvline(
                x=val_data.index[0], color="gray", linestyle=":", alpha=0.7, linewidth=2
            )
            ax.text(
                val_data.index[0],
                ax.get_ylim()[1] * 0.95,
                "Val Start",
                rotation=90,
                verticalalignment="top",
                fontsize=10,
            )

        if len(val_data) > 0 and len(test_data) > 0:
            ax.axvline(
                x=test_data.index[0],
                color="gray",
                linestyle=":",
                alpha=0.7,
                linewidth=2,
            )
            ax.text(
                test_data.index[0],
                ax.get_ylim()[1] * 0.95,
                "Test Start",
                rotation=90,
                verticalalignment="top",
                fontsize=10,
            )

        # Formatting
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (SGD)", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        # Format dates if we have datetime data
        if (
            use_reconstructed_dates
            or isinstance(full_data.index, pd.DatetimeIndex)
            or (len(train_data) > 0 and isinstance(train_data.index, pd.DatetimeIndex))
        ):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved full forecast plot to {save_path}")

        return fig

    def create_comprehensive_forecast_report(
        self,
        model: BaseTimeSeriesModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        asset_name: str,
        output_dir: str = "data/output/visualizations",
    ) -> Dict[str, str]:
        """
        Create a comprehensive forecasting report with multiple visualizations.

        Parameters:
        ----------
        model : BaseTimeSeriesModel
            Trained model
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test targets
        asset_name : str
            Name of the asset
        output_dir : str
            Output directory for plots

        Returns:
        -------
        Dict[str, str]
            Dictionary of plot names and file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Make predictions
        predictions = model.predict(X_test)

        # Try to get confidence intervals
        prediction_result = model.predict(X_test, return_confidence=True)
        if (
            hasattr(prediction_result, "confidence_intervals")
            and prediction_result.confidence_intervals
        ):
            confidence_intervals = prediction_result.confidence_intervals
        else:
            confidence_intervals = None

        plot_files = {}

        # 1. Prediction vs Actual
        safe_asset_name = asset_name.replace("/", "_").replace("\\", "_")

        pred_file = (
            output_path / f"{safe_asset_name}_{model.model_name}_predictions.png"
        )
        fig1 = self.plot_prediction_vs_actual(
            y_test.values,
            predictions,
            dates=X_test.index if isinstance(X_test.index, pd.DatetimeIndex) else None,
            title=f"{asset_name} - {model.model_name} Predictions",
            confidence_intervals=confidence_intervals,
            save_path=str(pred_file),
        )
        plt.close(fig1)
        plot_files["predictions"] = str(pred_file)

        # 2. Residual Analysis
        residual_file = (
            output_path / f"{safe_asset_name}_{model.model_name}_residuals.png"
        )
        fig2 = self.plot_residual_analysis(
            y_test.values,
            predictions,
            dates=X_test.index if isinstance(X_test.index, pd.DatetimeIndex) else None,
            title=f"{asset_name} - {model.model_name} Residual Analysis",
            save_path=str(residual_file),
        )
        plt.close(fig2)
        plot_files["residuals"] = str(residual_file)

        # 3. Feature Importance (if available)
        feature_importance = model.get_feature_importance()
        if feature_importance:
            importance_file = (
                output_path / f"{safe_asset_name}_{model.model_name}_importance.png"
            )
            fig3 = self.plot_feature_importance(
                feature_importance,
                title=f"{asset_name} - {model.model_name} Feature Importance",
                save_path=str(importance_file),
            )
            plt.close(fig3)
            plot_files["feature_importance"] = str(importance_file)

        # 4. Advanced forecast with uncertainty (if supported)
        try:
            # Try to get prediction intervals
            if hasattr(model, "predict_interval"):
                intervals_50 = model.predict_interval(X_test, confidence=0.5)
                intervals_95 = model.predict_interval(X_test, confidence=0.95)
                prediction_intervals = {"50%": intervals_50, "95%": intervals_95}

                uncertainty_file = (
                    output_path
                    / f"{safe_asset_name}_{model.model_name}_uncertainty.png"
                )
                fig4 = self.plot_forecast_with_uncertainty(
                    y_test.values,
                    predictions,
                    dates=X_test.index
                    if isinstance(X_test.index, pd.DatetimeIndex)
                    else None,
                    prediction_intervals=prediction_intervals,
                    title=f"{asset_name} - {model.model_name} Forecast with Uncertainty",
                    save_path=str(uncertainty_file),
                )
                plt.close(fig4)
                plot_files["uncertainty_forecast"] = str(uncertainty_file)
        except Exception as e:
            logger.debug(f"Could not create uncertainty plot: {e}")

        logger.info(
            f"Created comprehensive forecast report for {asset_name} - {model.model_name}"
        )

        return plot_files
