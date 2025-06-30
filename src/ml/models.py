"""
Consolidated ML model implementations for time series forecasting.

This module contains all model implementations including linear models,
ensemble models, and time series specific models.
"""

import logging
import warnings
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from ..pipeline.config import PipelineConfig
from .base import SklearnTimeSeriesModel, BaseTimeSeriesModel

logger = logging.getLogger(__name__)

# Suppress statsmodels warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


# =============================================================================
# LINEAR MODELS
# =============================================================================

class LinearRegressionModel(SklearnTimeSeriesModel):
    """Linear regression model for time series forecasting."""
    
    def __init__(self, 
                 config: PipelineConfig, 
                 fit_intercept: bool = True,
                 normalize: bool = False):
        """
        Initialize Linear Regression model.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        fit_intercept : bool
            Whether to calculate intercept
        normalize : bool
            Whether to normalize features
        """
        estimator = LinearRegression(
            fit_intercept=fit_intercept,
            n_jobs=-1
        )
        
        super().__init__(config, "LinearRegression", estimator)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
    
    def get_coefficients(self) -> Optional[Dict[str, float]]:
        """Get model coefficients."""
        if not self.is_fitted:
            return None
            
        if self.feature_names:
            return dict(zip(self.feature_names, self.estimator.coef_))
        return None
    
    def get_intercept(self) -> Optional[float]:
        """Get model intercept."""
        if not self.is_fitted:
            return None
        return float(self.estimator.intercept_)


class RidgeModel(SklearnTimeSeriesModel):
    """Ridge regression model with L2 regularization."""
    
    def __init__(self, 
                 config: PipelineConfig,
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 max_iter: int = 1000):
        """
        Initialize Ridge regression model.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        alpha : float
            Regularization strength
        fit_intercept : bool
            Whether to calculate intercept
        max_iter : int
            Maximum number of iterations
        """
        estimator = Ridge(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            random_state=config.modeling.random_state
        )
        
        super().__init__(config, "Ridge", estimator)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
    
    def get_regularization_path(self, X: pd.DataFrame, y: pd.Series, alphas: np.ndarray) -> Dict[str, Any]:
        """
        Compute regularization path for different alpha values.
        
        Parameters:
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        alphas : np.ndarray
            Array of alpha values to test
            
        Returns:
        -------
        Dict[str, Any]
            Regularization path results
        """
        from sklearn.linear_model import ridge_regression
        
        X_processed, y_processed = self._prepare_training_data(X, y)
        
        coefs = []
        for alpha in alphas:
            coef = ridge_regression(X_processed, y_processed, alpha=alpha)
            coefs.append(coef)
        
        return {
            'alphas': alphas,
            'coefficients': np.array(coefs),
            'feature_names': self.feature_names or list(X.columns)
        }


class LassoModel(SklearnTimeSeriesModel):
    """Lasso regression model with L1 regularization."""
    
    def __init__(self, 
                 config: PipelineConfig,
                 alpha: float = 1.0,
                 fit_intercept: bool = True,
                 max_iter: int = 1000,
                 tol: float = 1e-4):
        """
        Initialize Lasso regression model.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        alpha : float
            Regularization strength
        fit_intercept : bool
            Whether to calculate intercept
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for optimization
        """
        estimator = Lasso(
            alpha=alpha,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            random_state=config.modeling.random_state
        )
        
        super().__init__(config, "Lasso", estimator)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
    
    def get_selected_features(self) -> Optional[Dict[str, float]]:
        """Get features selected by Lasso (non-zero coefficients)."""
        if not self.is_fitted or not self.feature_names:
            return None
            
        coefficients = self.estimator.coef_
        selected_features = {}
        
        for feature, coef in zip(self.feature_names, coefficients):
            if abs(coef) > 1e-10:  # Consider very small coefficients as zero
                selected_features[feature] = coef
                
        return selected_features
    
    def get_sparsity_ratio(self) -> Optional[float]:
        """Get the sparsity ratio (percentage of zero coefficients)."""
        if not self.is_fitted:
            return None
            
        zero_coefs = np.sum(np.abs(self.estimator.coef_) < 1e-10)
        total_coefs = len(self.estimator.coef_)
        
        return zero_coefs / total_coefs


class PolynomialRegressionModel(SklearnTimeSeriesModel):
    """Polynomial regression model for capturing non-linear trends."""
    
    def __init__(self, 
                 config: PipelineConfig,
                 degree: int = 2,
                 interaction_only: bool = False,
                 include_bias: bool = True,
                 alpha: float = 0.0):
        """
        Initialize Polynomial regression model.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        degree : int
            Degree of polynomial features
        interaction_only : bool
            Whether to include only interaction features
        include_bias : bool
            Whether to include bias column
        alpha : float
            Ridge regularization parameter (0 for no regularization)
        """
        # Use Ridge if alpha > 0, otherwise LinearRegression
        if alpha > 0:
            base_estimator = Ridge(alpha=alpha, random_state=config.modeling.random_state)
        else:
            base_estimator = LinearRegression()
            
        super().__init__(config, "PolynomialRegression", base_estimator)
        
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.alpha = alpha
        self.poly_features = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias
        )
        self.poly_feature_names = None
    
    def _prepare_training_data(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare data with polynomial features."""
        
        # First apply base preprocessing
        X_processed, y_processed = super()._prepare_training_data(X, y)
        
        # Generate polynomial features
        X_poly = self.poly_features.fit_transform(X_processed)
        self.poly_feature_names = self.poly_features.get_feature_names_out(X_processed.columns)
        
        # Convert back to DataFrame
        X_poly_df = pd.DataFrame(
            X_poly,
            index=X_processed.index,
            columns=self.poly_feature_names
        )
        
        return X_poly_df, y_processed
    
    def _prepare_prediction_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare prediction data with polynomial features."""
        
        # First apply base preprocessing
        X_processed = super()._prepare_prediction_data(X)
        
        # Generate polynomial features
        X_poly = self.poly_features.transform(X_processed)
        
        # Convert back to DataFrame
        X_poly_df = pd.DataFrame(
            X_poly,
            index=X_processed.index,
            columns=self.poly_feature_names
        )
        
        return X_poly_df
    
    def get_polynomial_feature_names(self) -> Optional[list]:
        """Get the names of polynomial features."""
        return list(self.poly_feature_names) if self.poly_feature_names is not None else None


# =============================================================================
# ENSEMBLE MODELS
# =============================================================================

class RandomForestModel(SklearnTimeSeriesModel):
    """Random Forest model for time series forecasting."""
    
    def __init__(self, 
                 config: PipelineConfig,
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: str = 'sqrt',
                 bootstrap: bool = True,
                 oob_score: bool = True):
        """
        Initialize Random Forest model.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        n_estimators : int
            Number of trees
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split node
        min_samples_leaf : int
            Minimum samples required at leaf node
        max_features : str
            Number of features to consider for best split
        bootstrap : bool
            Whether to use bootstrap sampling
        oob_score : bool
            Whether to use out-of-bag samples for scoring
        """
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=config.modeling.random_state,
            n_jobs=-1
        )
        
        super().__init__(config, "RandomForest", estimator)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
    
    def get_oob_score(self) -> Optional[float]:
        """Get out-of-bag score if available."""
        if not self.is_fitted or not self.oob_score:
            return None
        return float(self.estimator.oob_score_)
    
    def get_tree_feature_importance(self) -> Optional[Dict[str, List[float]]]:
        """Get feature importance from individual trees."""
        if not self.is_fitted or not self.feature_names:
            return None
            
        tree_importances = []
        for tree in self.estimator.estimators_:
            tree_importances.append(tree.feature_importances_)
        
        return {
            'features': self.feature_names,
            'importances_per_tree': tree_importances,
            'mean_importance': np.mean(tree_importances, axis=0).tolist(),
            'std_importance': np.std(tree_importances, axis=0).tolist()
        }
    
    def _calculate_confidence_intervals(self, 
                                     X: pd.DataFrame, 
                                     predictions: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Calculate prediction intervals using individual tree predictions."""
        
        if not self.is_fitted:
            return None
            
        # Get predictions from all trees
        # Convert to numpy array to avoid sklearn feature name warnings
        X_values = X.values if hasattr(X, 'values') else X
        tree_predictions = np.array([
            tree.predict(X_values) for tree in self.estimator.estimators_
        ])
        
        # Calculate percentiles for confidence intervals
        lower = np.percentile(tree_predictions, 2.5, axis=0)
        upper = np.percentile(tree_predictions, 97.5, axis=0)
        
        return lower, upper


class XGBoostModel(BaseTimeSeriesModel):
    """XGBoost model for time series forecasting."""
    
    def __init__(self, 
                 config: PipelineConfig,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 early_stopping_rounds: Optional[int] = 10):
        """
        Initialize XGBoost model.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum depth of trees
        learning_rate : float
            Learning rate
        subsample : float
            Subsample ratio of training instances
        colsample_bytree : float
            Subsample ratio of columns for each tree
        reg_alpha : float
            L1 regularization
        reg_lambda : float
            L2 regularization
        early_stopping_rounds : int, optional
            Early stopping rounds for validation
        """
        super().__init__(config, "XGBoost")
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        
        # Initialize XGBoost with version-compatible parameters
        model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': config.modeling.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # Don't set early_stopping_rounds in constructor, handle in fit method
        self.model = xgb.XGBRegressor(**model_params)
        
        self._training_residuals = None
        self.eval_results = {}
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the XGBoost model."""
        
        # Convert to numpy arrays to avoid sklearn feature name warnings
        X_values = X.values if hasattr(X, 'values') else X
        y_values = y.values if hasattr(y, 'values') else y
        
        # Fit the model (simplified to avoid early stopping parameter issues)
        self.model.fit(X_values, y_values)
        
        # Store training residuals
        train_predictions = self.model.predict(X_values)
        self._training_residuals = y_values - train_predictions
        
        # Store evaluation results
        if hasattr(self.model, 'evals_result_'):
            self.eval_results = self.model.evals_result_
    
    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost model."""
        # Convert to numpy array to avoid sklearn feature name warnings
        X_values = X.values if hasattr(X, 'values') else X
        return self.model.predict(X_values)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from XGBoost model."""
        if not self.is_fitted or not self.feature_names:
            return None
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get XGBoost hyperparameters."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'early_stopping_rounds': self.early_stopping_rounds
        }
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history and evaluation results."""
        return {
            'eval_results': self.eval_results,
            'best_iteration': getattr(self.model, 'best_iteration', None),
            'best_score': getattr(self.model, 'best_score', None)
        }
    
    def plot_importance(self, max_num_features: int = 20) -> Optional[Any]:
        """Plot feature importance."""
        if not self.is_fitted:
            return None
            
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            xgb.plot_importance(
                self.model,
                ax=ax,
                max_num_features=max_num_features,
                importance_type='gain'
            )
            plt.title(f'Feature Importance - {self.model_name}')
            plt.tight_layout()
            return fig
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None
    
    def _calculate_confidence_intervals(self, 
                                     X: pd.DataFrame, 
                                     predictions: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Calculate confidence intervals using quantile regression."""
        
        # For now, use simple approach based on training residuals
        if self._training_residuals is not None:
            residual_std = np.std(self._training_residuals)
            lower = predictions - 1.96 * residual_std
            upper = predictions + 1.96 * residual_std
            return lower, upper
        
        return None


class GradientBoostingModel(SklearnTimeSeriesModel):
    """Gradient Boosting model using sklearn implementation."""
    
    def __init__(self, 
                 config: PipelineConfig,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0,
                 alpha: float = 0.9):
        """
        Initialize Gradient Boosting model.
        
        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        n_estimators : int
            Number of boosting stages
        learning_rate : float
            Learning rate
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split node
        min_samples_leaf : int
            Minimum samples required at leaf node
        subsample : float
            Fraction of samples used for fitting trees
        alpha : float
            Alpha-quantile for Huber loss and quantile regression
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        estimator = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            alpha=alpha,
            random_state=config.modeling.random_state
        )
        
        super().__init__(config, "GradientBoosting", estimator)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.alpha = alpha
    
    def get_training_deviance(self) -> Optional[List[float]]:
        """Get training deviance history."""
        if not self.is_fitted:
            return None
        return self.estimator.train_score_.tolist()
    
    def get_staged_predictions(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get predictions at each boosting stage."""
        if not self.is_fitted:
            return None
            
        X_processed = self._prepare_prediction_data(X)
        return np.array(list(self.estimator.staged_predict(X_processed)))


# =============================================================================
# TIME SERIES MODELS
# =============================================================================

class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA model for time series forecasting."""

    def __init__(
        self,
        config: PipelineConfig,
        order: Tuple[int, int, int] = (1, 1, 1),
        auto_order: bool = True,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        information_criterion: str = "aic",
    ):
        """
        Initialize ARIMA model.

        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        order : Tuple[int, int, int]
            ARIMA order (p, d, q)
        auto_order : bool
            Whether to automatically determine order
        max_p : int
            Maximum p value for auto order selection
        max_d : int
            Maximum d value for auto order selection
        max_q : int
            Maximum q value for auto order selection
        seasonal : bool
            Whether to consider seasonality
        information_criterion : str
            Information criterion for model selection
        """
        super().__init__(config, "ARIMA")

        self.order = order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.information_criterion = information_criterion

        self.model = None
        self.model_fit = None
        self.selected_order = None
        self.aic_values = {}

    def _determine_order(self, y: pd.Series) -> Tuple[int, int, int]:
        """Automatically determine ARIMA order."""

        logger.info("Determining optimal ARIMA order...")

        # Check stationarity
        d = self._determine_differencing(y)

        if self.auto_order:
            # Grid search for best (p, q) combination
            best_aic = float("inf")
            best_order = (0, d, 0)

            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    try:
                        temp_model = ARIMA(y, order=(p, d, q))
                        temp_fit = temp_model.fit()

                        aic = temp_fit.aic
                        self.aic_values[(p, d, q)] = aic

                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)

                    except Exception as e:
                        logger.debug(f"Failed to fit ARIMA({p},{d},{q}): {str(e)}")
                        continue

            logger.info(f"Selected ARIMA order: {best_order} (AIC: {best_aic:.2f})")
            return best_order
        else:
            # Use provided order but adjust differencing
            p, _, q = self.order
            return (p, d, q)

    def _determine_differencing(self, y: pd.Series, max_d: int = None) -> int:
        """Determine the number of differences needed for stationarity."""

        max_d = max_d or self.max_d

        # Test original series
        adf_result = adfuller(y.dropna())

        if adf_result[1] <= 0.05:  # p-value <= 0.05 means stationary
            return 0

        # Test differenced series
        for d in range(1, max_d + 1):
            y_diff = y.diff(d).dropna()

            if len(y_diff) < 10:  # Need minimum observations
                break

            adf_result = adfuller(y_diff)

            if adf_result[1] <= 0.05:
                logger.info(
                    f"Series is stationary after {d} differences (p-value: {adf_result[1]:.4f})"
                )
                return d

        logger.warning(f"Series may not be stationary after {max_d} differences")
        return max_d

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the ARIMA model."""

        # ARIMA uses only the target variable, not external features
        # We'll use the price column from the original data
        price_column = self.config.get_asset_config("watch").price_column

        if price_column in X.columns:
            ts_data = X[price_column]
        else:
            # Fallback to target if available
            ts_data = y

        # Remove missing values
        ts_data = ts_data.dropna()

        if len(ts_data) < 10:
            raise ValueError(
                "Insufficient data for ARIMA model (need at least 10 observations)"
            )

        # Determine order
        if self.auto_order:
            self.selected_order = self._determine_order(ts_data)
        else:
            self.selected_order = self.order

        # Fit the model
        logger.info(f"Fitting ARIMA{self.selected_order} model...")

        try:
            self.model = ARIMA(ts_data, order=self.selected_order)
            self.model_fit = self.model.fit()

            logger.info(
                f"ARIMA model fitted successfully. AIC: {self.model_fit.aic:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to fit ARIMA model: {str(e)}")
            raise

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with ARIMA model."""

        if self.model_fit is None:
            raise ValueError("Model must be fitted before making predictions")

        # Number of steps to forecast
        steps = len(X)

        # Get forecast
        forecast_result = self.model_fit.forecast(steps=steps)

        if isinstance(forecast_result, pd.Series):
            return forecast_result.values
        else:
            return np.array(forecast_result)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """ARIMA doesn't have feature importance in traditional sense."""
        if not self.is_fitted or self.model_fit is None:
            return None

        # Return coefficient information
        params = self.model_fit.params
        return {f"param_{i}": float(val) for i, val in enumerate(params)}

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get ARIMA hyperparameters."""
        return {
            "order": self.selected_order or self.order,
            "auto_order": self.auto_order,
            "max_p": self.max_p,
            "max_d": self.max_d,
            "max_q": self.max_q,
            "information_criterion": self.information_criterion,
        }

    def get_model_summary(self) -> Optional[str]:
        """Get model summary from statsmodels."""
        if self.model_fit is None:
            return None
        return str(self.model_fit.summary())

    def get_residuals(self) -> Optional[pd.Series]:
        """Get model residuals."""
        if self.model_fit is None:
            return None
        return self.model_fit.resid

    def diagnose_residuals(self) -> Dict[str, Any]:
        """Perform residual diagnostics."""
        if self.model_fit is None:
            return {}

        residuals = self.model_fit.resid

        # Ljung-Box test for autocorrelation
        from statsmodels.stats.diagnostic import acorr_ljungbox

        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)

        # Jarque-Bera test for normality
        from statsmodels.stats.stattools import jarque_bera

        jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

        return {
            "ljung_box_test": ljung_box.to_dict(),
            "jarque_bera_test": {
                "statistic": jb_stat,
                "p_value": jb_pvalue,
                "skewness": skew,
                "kurtosis": kurtosis,
            },
            "residual_stats": {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "min": float(residuals.min()),
                "max": float(residuals.max()),
            },
        }


class SARIMAModel(BaseTimeSeriesModel):
    """SARIMA model for seasonal time series forecasting."""

    def __init__(
        self,
        config: PipelineConfig,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
        auto_order: bool = True,
        max_p: int = 3,
        max_d: int = 2,
        max_q: int = 3,
        max_P: int = 2,
        max_D: int = 1,
        max_Q: int = 2,
        seasonal_periods: int = 12,
    ):
        """
        Initialize SARIMA model.

        Parameters:
        ----------
        config : PipelineConfig
            Pipeline configuration
        order : Tuple[int, int, int]
            Non-seasonal ARIMA order (p, d, q)
        seasonal_order : Tuple[int, int, int, int]
            Seasonal order (P, D, Q, s)
        auto_order : bool
            Whether to automatically determine order
        max_p, max_d, max_q : int
            Maximum non-seasonal parameters
        max_P, max_D, max_Q : int
            Maximum seasonal parameters
        seasonal_periods : int
            Number of periods in a season
        """
        super().__init__(config, "SARIMA")

        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self.seasonal_periods = seasonal_periods

        self.model = None
        self.model_fit = None
        self.selected_order = None
        self.selected_seasonal_order = None
        self.aic_values = {}

    def _detect_seasonality(self, y: pd.Series) -> Dict[str, Any]:
        """Detect seasonality in the time series."""

        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                y.dropna(),
                model="additive",
                period=self.seasonal_periods,
                extrapolate_trend="freq",
            )

            # Calculate seasonal strength
            seasonal_var = np.var(decomposition.seasonal)
            residual_var = np.var(decomposition.resid.dropna())
            seasonal_strength = seasonal_var / (seasonal_var + residual_var)

            return {
                "seasonal_strength": seasonal_strength,
                "has_seasonality": seasonal_strength > 0.3,
                "decomposition": decomposition,
            }

        except Exception as e:
            logger.warning(f"Could not perform seasonal decomposition: {str(e)}")
            return {"has_seasonality": False, "seasonal_strength": 0}

    def _determine_seasonal_order(
        self, y: pd.Series
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """Automatically determine SARIMA orders."""

        logger.info("Determining optimal SARIMA order...")

        # Check seasonality
        seasonality_info = self._detect_seasonality(y)

        if not seasonality_info.get("has_seasonality", False):
            logger.info("No significant seasonality detected, using simple ARIMA")
            # Use ARIMA approach
            arima_model = ARIMAModel(self.config, auto_order=True)
            d = arima_model._determine_differencing(y)

            if self.auto_order:
                best_aic = float("inf")
                best_order = (0, d, 0)

                for p in range(self.max_p + 1):
                    for q in range(self.max_q + 1):
                        try:
                            temp_model = SARIMAX(y, order=(p, d, q))
                            temp_fit = temp_model.fit(disp=False)

                            aic = temp_fit.aic
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)

                        except:
                            continue

                return best_order, (0, 0, 0, 0)

        # Grid search for seasonal model
        best_aic = float("inf")
        best_order = (1, 1, 1)
        best_seasonal_order = (0, 0, 0, self.seasonal_periods)

        # Determine differencing
        d = self._determine_differencing(y)
        D = (
            0 if d > 0 else 1
        )  # Typically don't need both non-seasonal and seasonal differencing

        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                for P in range(self.max_P + 1):
                    for Q in range(self.max_Q + 1):
                        try:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, self.seasonal_periods)

                            temp_model = SARIMAX(
                                y, order=order, seasonal_order=seasonal_order
                            )
                            temp_fit = temp_model.fit(disp=False)

                            aic = temp_fit.aic
                            self.aic_values[(order, seasonal_order)] = aic

                            if aic < best_aic:
                                best_aic = aic
                                best_order = order
                                best_seasonal_order = seasonal_order

                        except Exception as e:
                            logger.debug(
                                f"Failed to fit SARIMA{order}x{seasonal_order}: {str(e)}"
                            )
                            continue

        logger.info(
            f"Selected SARIMA order: {best_order}x{best_seasonal_order} (AIC: {best_aic:.2f})"
        )
        return best_order, best_seasonal_order

    def _determine_differencing(self, y: pd.Series) -> int:
        """Determine differencing order (reuse from ARIMA)."""
        arima_model = ARIMAModel(self.config)
        return arima_model._determine_differencing(y, self.max_d)

    def _fit_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the SARIMA model."""

        # Use price column or target
        price_column = self.config.get_asset_config("watch").price_column

        if price_column in X.columns:
            ts_data = X[price_column]
        else:
            ts_data = y

        ts_data = ts_data.dropna()

        if len(ts_data) < max(24, 2 * self.seasonal_periods):
            raise ValueError(
                f"Insufficient data for SARIMA model (need at least {2 * self.seasonal_periods} observations)"
            )

        # Determine orders
        if self.auto_order:
            self.selected_order, self.selected_seasonal_order = (
                self._determine_seasonal_order(ts_data)
            )
        else:
            self.selected_order = self.order
            self.selected_seasonal_order = self.seasonal_order

        # Fit the model
        logger.info(
            f"Fitting SARIMA{self.selected_order}x{self.selected_seasonal_order} model..."
        )

        try:
            self.model = SARIMAX(
                ts_data,
                order=self.selected_order,
                seasonal_order=self.selected_seasonal_order,
            )
            self.model_fit = self.model.fit(disp=False)

            logger.info(
                f"SARIMA model fitted successfully. AIC: {self.model_fit.aic:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to fit SARIMA model: {str(e)}")
            raise

    def _predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with SARIMA model."""

        if self.model_fit is None:
            raise ValueError("Model must be fitted before making predictions")

        steps = len(X)
        forecast_result = self.model_fit.forecast(steps=steps)

        if isinstance(forecast_result, pd.Series):
            return forecast_result.values
        else:
            return np.array(forecast_result)

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """SARIMA doesn't have traditional feature importance."""
        if not self.is_fitted or self.model_fit is None:
            return None

        params = self.model_fit.params
        return {f"param_{param}": float(val) for param, val in params.items()}

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get SARIMA hyperparameters."""
        return {
            "order": self.selected_order or self.order,
            "seasonal_order": self.selected_seasonal_order or self.seasonal_order,
            "auto_order": self.auto_order,
            "seasonal_periods": self.seasonal_periods,
        }

    def get_seasonal_decomposition(self, y: pd.Series) -> Optional[Any]:
        """Get seasonal decomposition of the time series."""
        try:
            return seasonal_decompose(
                y.dropna(),
                model="additive",
                period=self.seasonal_periods,
                extrapolate_trend="freq",
            )
        except Exception as e:
            logger.warning(f"Could not perform seasonal decomposition: {str(e)}")
            return None