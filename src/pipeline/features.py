"""
Comprehensive feature engineering for time series data.

This module consolidates feature engineering and feature selection
from the previous modelling structure.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, RFECV
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV
from scipy.stats import pearsonr

from .config import PipelineConfig, FeatureConfig, ModelConfig
from .base import PipelineComponent

logger = logging.getLogger(__name__)


class FeatureEngineer(PipelineComponent):
    """Advanced feature engineering for time series data."""
    
    def __init__(self, config: PipelineConfig, asset_type: str = "watch"):
        super().__init__(config)
        self.asset_type = asset_type
        self.asset_config = config.get_asset_config(asset_type)
        self.feature_config = config.features
        
    def engineer_features(self, 
                         df: pd.DataFrame, 
                         price_column: Optional[str] = None) -> pd.DataFrame:
        """
        Engineer comprehensive features for time series data.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame with datetime index
        price_column : str, optional
            Name of the price column (uses asset config if None)
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with engineered features
        """
        price_column = price_column or self.asset_config.price_column
        logger.info(f"Engineering features for {len(df)} records")
        
        df_features = df.copy()
        
        # Temporal features
        if self.feature_config.include_temporal_features:
            df_features = self._add_temporal_features(df_features)
            
        # Lag features
        df_features = self._add_lag_features(df_features, price_column)
        
        # Rolling window features
        df_features = self._add_rolling_features(df_features, price_column)
        
        # Momentum features
        if self.feature_config.include_momentum_features:
            df_features = self._add_momentum_features(df_features, price_column)
            
        # Volatility features
        if self.feature_config.include_volatility_features:
            df_features = self._add_volatility_features(df_features, price_column)
            
        # Technical indicators
        if self.feature_config.include_technical_indicators:
            df_features = self._add_technical_indicators(df_features, price_column)
            
        # Target variable
        df_features = self._add_target_variable(df_features, price_column)
        
        # Drop NaN values created by feature engineering
        initial_length = len(df_features)
        df_features.dropna(inplace=True)
        final_length = len(df_features)
        
        logger.info(f"Feature engineering complete: {initial_length} → {final_length} records")
        logger.info(f"Generated {len(df_features.columns)} features")
        
        return df_features
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features based on datetime index."""
        
        df_temp = df.copy()
        
        # Basic temporal features
        df_temp['day_of_week'] = df_temp.index.dayofweek
        df_temp['day_of_month'] = df_temp.index.day
        df_temp['month'] = df_temp.index.month
        df_temp['quarter'] = df_temp.index.quarter
        df_temp['year'] = df_temp.index.year
        df_temp['week_of_year'] = df_temp.index.isocalendar().week
        
        # Cyclical encoding for temporal features
        df_temp['day_of_week_sin'] = np.sin(2 * np.pi * df_temp['day_of_week'] / 7)
        df_temp['day_of_week_cos'] = np.cos(2 * np.pi * df_temp['day_of_week'] / 7)
        df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
        df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
        
        # Boolean features
        df_temp['is_weekend'] = df_temp['day_of_week'] >= 5
        df_temp['is_month_start'] = df_temp.index.is_month_start
        df_temp['is_month_end'] = df_temp.index.is_month_end
        df_temp['is_quarter_start'] = df_temp.index.is_quarter_start
        df_temp['is_quarter_end'] = df_temp.index.is_quarter_end
        df_temp['is_year_start'] = df_temp.index.is_year_start
        df_temp['is_year_end'] = df_temp.index.is_year_end
        
        return df_temp
    
    def _add_lag_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add lagged price features."""
        
        df_lag = df.copy()
        
        for lag in self.feature_config.lag_periods:
            df_lag[f'price_lag_{lag}'] = df_lag[price_column].shift(lag)
            
        return df_lag
    
    def _add_rolling_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add rolling window features."""
        
        df_roll = df.copy()
        
        for window in self.feature_config.rolling_windows:
            # Rolling mean
            df_roll[f'rolling_mean_{window}'] = df_roll[price_column].rolling(window=window).mean()
            
            # Rolling standard deviation
            df_roll[f'rolling_std_{window}'] = df_roll[price_column].rolling(window=window).std()
            
            # Rolling min/max
            df_roll[f'rolling_min_{window}'] = df_roll[price_column].rolling(window=window).min()
            df_roll[f'rolling_max_{window}'] = df_roll[price_column].rolling(window=window).max()
            
            # Rolling median
            df_roll[f'rolling_median_{window}'] = df_roll[price_column].rolling(window=window).median()
            
            # Rolling quantiles
            df_roll[f'rolling_q25_{window}'] = df_roll[price_column].rolling(window=window).quantile(0.25)
            df_roll[f'rolling_q75_{window}'] = df_roll[price_column].rolling(window=window).quantile(0.75)
            
            # Position within rolling window
            rolling_range = df_roll[f'rolling_max_{window}'] - df_roll[f'rolling_min_{window}']
            df_roll[f'price_position_{window}'] = (
                (df_roll[price_column] - df_roll[f'rolling_min_{window}']) / rolling_range
            ).replace([np.inf, -np.inf], np.nan)
            
        return df_roll
    
    def _add_momentum_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add momentum and change features."""
        
        df_momentum = df.copy()
        
        # Price differences
        for period in [1, 3, 7, 14, 30]:
            df_momentum[f'price_diff_{period}'] = df_momentum[price_column].diff(period)
            df_momentum[f'price_pct_change_{period}'] = df_momentum[price_column].pct_change(period)
            
        # Acceleration (second derivative)
        df_momentum['price_acceleration'] = df_momentum[price_column].diff().diff()
        
        # Rate of change ratio
        df_momentum['roc_3_vs_7'] = (
            df_momentum['price_pct_change_3'] / df_momentum['price_pct_change_7']
        ).replace([np.inf, -np.inf], np.nan)
        
        # Momentum oscillator
        for window in [7, 14]:
            df_momentum[f'momentum_{window}'] = (
                df_momentum[price_column] / df_momentum[price_column].shift(window) - 1
            )
            
        return df_momentum
    
    def _add_volatility_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add volatility and risk features."""
        
        df_vol = df.copy()
        
        # Simple volatility measures
        for window in [7, 14, 30]:
            returns = df_vol[price_column].pct_change()
            
            # Rolling volatility (std of returns)
            df_vol[f'volatility_{window}'] = returns.rolling(window=window).std()
            
            # Parkinson volatility (using rolling max/min)
            rolling_max = df_vol[price_column].rolling(window=window).max()
            rolling_min = df_vol[price_column].rolling(window=window).min()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratio = np.log(rolling_max / rolling_min)
                df_vol[f'parkinson_vol_{window}'] = np.sqrt(log_ratio ** 2)
            
            # Average True Range (ATR) approximation
            high_low = rolling_max - rolling_min
            df_vol[f'atr_{window}'] = high_low.rolling(window=window).mean()
            
        # Volatility ratios
        df_vol['vol_ratio_7_14'] = (df_vol['volatility_7'] / df_vol['volatility_14']).replace([np.inf, -np.inf], np.nan)
        df_vol['vol_ratio_14_30'] = (df_vol['volatility_14'] / df_vol['volatility_30']).replace([np.inf, -np.inf], np.nan)
        
        return df_vol
    
    def _add_technical_indicators(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add technical analysis indicators."""
        
        df_tech = df.copy()
        
        # Simple Moving Average convergence/divergence
        df_tech['sma_7'] = df_tech[price_column].rolling(window=7).mean()
        df_tech['sma_21'] = df_tech[price_column].rolling(window=21).mean()
        df_tech['sma_convergence'] = df_tech['sma_7'] - df_tech['sma_21']
        
        # Exponential Moving Average
        df_tech['ema_7'] = df_tech[price_column].ewm(span=7).mean()
        df_tech['ema_21'] = df_tech[price_column].ewm(span=21).mean()
        df_tech['ema_convergence'] = df_tech['ema_7'] - df_tech['ema_21']
        
        # Bollinger Bands
        for window in [20]:
            rolling_mean = df_tech[price_column].rolling(window=window).mean()
            rolling_std = df_tech[price_column].rolling(window=window).std()
            
            df_tech[f'bb_upper_{window}'] = rolling_mean + (2 * rolling_std)
            df_tech[f'bb_lower_{window}'] = rolling_mean - (2 * rolling_std)
            
            bb_range = df_tech[f'bb_upper_{window}'] - df_tech[f'bb_lower_{window}']
            df_tech[f'bb_position_{window}'] = (
                (df_tech[price_column] - df_tech[f'bb_lower_{window}']) / bb_range
            ).replace([np.inf, -np.inf], np.nan)
            
            df_tech[f'bb_width_{window}'] = (bb_range / rolling_mean).replace([np.inf, -np.inf], np.nan)
            
        # RSI (Relative Strength Index) approximation
        for window in [14]:
            delta = df_tech[price_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            rs = (gain / loss).replace([np.inf, -np.inf], np.nan)
            df_tech[f'rsi_{window}'] = 100 - (100 / (1 + rs))
            
        return df_tech
    
    def _add_target_variable(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add target variable for supervised learning."""
        
        df_target = df.copy()
        
        # Future price (shifted by target_shift days)
        df_target['target'] = df_target[price_column].shift(self.feature_config.target_shift)
        
        # Target as percentage change
        df_target['target_pct_change'] = (
            (df_target['target'] - df_target[price_column]) / df_target[price_column]
        )
        
        # Target direction (binary classification)
        df_target['target_direction'] = (df_target['target'] > df_target[price_column]).astype(int)
        
        return df_target
    
    def process_multiple_assets(self, 
                              asset_data: Dict[str, pd.DataFrame],
                              price_column: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Process features for multiple assets.
        
        Parameters:
        ----------
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset data
        price_column : str, optional
            Name of the price column
            
        Returns:
        -------
        Dict[str, pd.DataFrame]
            Dictionary of assets with engineered features
        """
        price_column = price_column or self.asset_config.price_column
        featured_data = {}
        
        for asset_name, df in asset_data.items():
            try:
                df_featured = self.engineer_features(df, price_column)
                featured_data[asset_name] = df_featured
                logger.info(f"Processed features for {asset_name}: {len(df_featured.columns)} features")
            except Exception as e:
                logger.error(f"Failed to process features for {asset_name}: {str(e)}")
                
        return featured_data
    
    def process(self, 
                asset_data: Dict[str, pd.DataFrame],
                price_column: Optional[str] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Main processing method for the pipeline component.
        
        Parameters:
        ----------
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset data
        price_column : str, optional
            Name of the price column
            
        Returns:
        -------
        Tuple[Dict[str, pd.DataFrame], Dict]
            Featured data and processing report
        """
        logger.info(f"Starting feature engineering for {len(asset_data)} assets")
        
        try:
            featured_data = self.process_multiple_assets(asset_data, price_column)
            
            report = self.generate_report(
                assets_processed=len(featured_data),
                total_assets=len(asset_data),
                avg_features=np.mean([len(df.columns) for df in featured_data.values()]) if featured_data else 0
            )
            
            return featured_data, report
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            report = self.generate_report(success=False, error=str(e))
            return {}, report


class FeatureSelector:
    """Advanced feature selection for time series data."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.max_features = config.max_features
        
    def select_features(self, 
                       df: pd.DataFrame, 
                       target_column: str = 'target',
                       method: str = 'correlation') -> Tuple[List[str], Dict]:
        """
        Select the best features using specified method.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame with features and target
        target_column : str
            Name of the target column
        method : str
            Feature selection method
            
        Returns:
        -------
        Tuple[List[str], Dict]
            Selected feature names and selection report
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
            
        # Prepare feature matrix and target
        feature_cols = self._get_feature_columns(df, target_column)
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Remove samples with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Remove features with too many missing values
        X = self._remove_high_missing_features(X)
        
        # Fill remaining missing values
        X = X.fillna(X.median())
        
        logger.info(f"Starting feature selection with {len(X.columns)} features")
        
        # Apply feature selection method
        if method == 'correlation':
            selected_features, report = self._correlation_selection(X, y)
        elif method == 'mutual_info':
            selected_features, report = self._mutual_info_selection(X, y)
        elif method == 'random_forest':
            selected_features, report = self._random_forest_selection(X, y)
        elif method == 'lasso':
            selected_features, report = self._lasso_selection(X, y)
        elif method == 'rfe':
            selected_features, report = self._rfe_selection(X, y)
        elif method == 'hybrid':
            selected_features, report = self._hybrid_selection(X, y)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return selected_features, report
    
    def _get_feature_columns(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Get list of feature columns (excluding target and non-numeric)."""
        
        exclude_cols = {target_column, 'target_pct_change', 'target_direction'}
        
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'bool']:
                feature_cols.append(col)
                
        return feature_cols
    
    def _remove_high_missing_features(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Remove features with high missing value percentage."""
        
        missing_pct = X.isnull().sum() / len(X)
        high_missing_cols = missing_pct[missing_pct > threshold].index
        
        if len(high_missing_cols) > 0:
            logger.info(f"Removing {len(high_missing_cols)} features with >{threshold*100}% missing values")
            X = X.drop(columns=high_missing_cols)
            
        return X
    
    def _correlation_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict]:
        """Select features based on correlation with target."""
        
        correlations = {}
        for col in X.columns:
            try:
                corr, p_value = pearsonr(X[col].fillna(X[col].median()), y)
                if not np.isnan(corr):
                    correlations[col] = {'correlation': abs(corr), 'p_value': p_value}
            except:
                correlations[col] = {'correlation': 0, 'p_value': 1}
                
        # Sort by absolute correlation
        sorted_features = sorted(correlations.items(), 
                               key=lambda x: x[1]['correlation'], 
                               reverse=True)
        
        # Select top features
        n_features = self.max_features or len(sorted_features)
        n_features = min(n_features, len(sorted_features))
        
        selected_features = [feat[0] for feat in sorted_features[:n_features]]
        
        report = {
            'method': 'correlation',
            'total_features': len(X.columns),
            'selected_features': n_features,
            'feature_scores': {feat: score['correlation'] for feat, score in sorted_features[:n_features]}
        }
        
        return selected_features, report
    
    def _mutual_info_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict]:
        """Select features based on mutual information."""
        
        n_features = self.max_features or len(X.columns)
        n_features = min(n_features, len(X.columns))
        
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        # Convert to numpy arrays to avoid sklearn feature name warnings
        X_filled = X.fillna(X.median())
        X_values = X_filled.values
        y_values = y.values if hasattr(y, 'values') else y
        X_selected = selector.fit_transform(X_values, y_values)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        scores = dict(zip(X.columns, selector.scores_))
        
        report = {
            'method': 'mutual_info',
            'total_features': len(X.columns),
            'selected_features': len(selected_features),
            'feature_scores': {feat: scores[feat] for feat in selected_features}
        }
        
        return selected_features, report
    
    def _random_forest_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict]:
        """Select features based on Random Forest importance."""
        
        # Fit Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        X_filled = X.fillna(X.median())
        # Convert to numpy arrays to avoid sklearn feature name warnings
        X_values = X_filled.values
        y_values = y.values if hasattr(y, 'values') else y
        rf.fit(X_values, y_values)
        
        # Get feature importance
        importance_scores = dict(zip(X.columns, rf.feature_importances_))
        
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        # Select top features
        n_features = self.max_features or len(sorted_features)
        n_features = min(n_features, len(sorted_features))
        
        selected_features = [feat[0] for feat in sorted_features[:n_features]]
        
        report = {
            'method': 'random_forest',
            'total_features': len(X.columns),
            'selected_features': n_features,
            'feature_scores': {feat: score for feat, score in sorted_features[:n_features]}
        }
        
        return selected_features, report
    
    def _lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict]:
        """Select features using Lasso regularization."""
        
        # Standardize features
        X_filled = X.fillna(X.median())
        X_scaled = (X_filled - X_filled.mean()) / X_filled.std().replace(0, 1)  # Avoid division by zero
        
        # Use LassoCV for automatic cross-validated alpha selection
        alphas = np.logspace(-4, 1, 20)  # Reduced from 50 to 20 for speed
        lasso_cv = LassoCV(
            alphas=alphas, 
            cv=3,  # 3-fold CV for speed
            random_state=self.config.random_state,
            max_iter=2000  # Increase iterations to help convergence
        )
        
        # Convert to numpy arrays to avoid sklearn feature name warnings
        X_values = X_scaled.values
        y_values = y.values if hasattr(y, 'values') else y
        
        try:
            lasso_cv.fit(X_values, y_values)
            best_alpha = lasso_cv.alpha_
        except:
            # Fallback to simple Lasso with default alpha
            best_alpha = 0.01
            
        # Fit final model
        lasso = Lasso(alpha=best_alpha, random_state=self.config.random_state, max_iter=2000)
        lasso.fit(X_values, y_values)
        
        # Select features with non-zero coefficients
        selected_mask = lasso.coef_ != 0
        selected_features = X.columns[selected_mask].tolist()
        
        # If too many features selected, take top by absolute coefficient
        if self.max_features and len(selected_features) > self.max_features:
            coef_abs = np.abs(lasso.coef_[selected_mask])
            top_indices = np.argsort(coef_abs)[-self.max_features:]
            selected_features = [selected_features[i] for i in top_indices]
            
        feature_scores = {feat: abs(coef) for feat, coef in zip(X.columns, lasso.coef_) if feat in selected_features}
        
        report = {
            'method': 'lasso',
            'total_features': len(X.columns),
            'selected_features': len(selected_features),
            'best_alpha': best_alpha,
            'feature_scores': feature_scores
        }
        
        return selected_features, report
    
    def _rfe_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict]:
        """Select features using Recursive Feature Elimination."""
        
        n_features = self.max_features or len(X.columns) // 2
        n_features = min(n_features, len(X.columns))
        
        estimator = RandomForestRegressor(
            n_estimators=50,
            random_state=self.config.random_state
        )
        
        selector = RFE(estimator=estimator, n_features_to_select=n_features)
        X_filled = X.fillna(X.median())
        # Convert to numpy arrays to avoid sklearn feature name warnings
        X_values = X_filled.values
        y_values = y.values if hasattr(y, 'values') else y
        selector.fit(X_values, y_values)
        
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get feature rankings
        rankings = dict(zip(X.columns, selector.ranking_))
        
        report = {
            'method': 'rfe',
            'total_features': len(X.columns),
            'selected_features': len(selected_features),
            'feature_rankings': {feat: rankings[feat] for feat in selected_features}
        }
        
        return selected_features, report
    
    def _hybrid_selection(self, X: pd.DataFrame, y: pd.Series) -> Tuple[List[str], Dict]:
        """Hybrid feature selection combining multiple methods."""
        
        # Step 1: Correlation filter
        corr_features, _ = self._correlation_selection(X, y)
        top_corr_features = corr_features[:len(X.columns)//2] if len(corr_features) > 20 else corr_features
        
        # Step 2: Random Forest importance
        X_filtered = X[top_corr_features]
        rf_features, _ = self._random_forest_selection(X_filtered, y)
        
        # Step 3: Final selection with Lasso
        if len(rf_features) > 10:
            X_rf = X[rf_features]
            final_features, lasso_report = self._lasso_selection(X_rf, y)
        else:
            final_features = rf_features
            lasso_report = {}
            
        report = {
            'method': 'hybrid',
            'total_features': len(X.columns),
            'correlation_filtered': len(top_corr_features),
            'rf_filtered': len(rf_features),
            'final_selected': len(final_features),
            'lasso_details': lasso_report
        }
        
        return final_features, report