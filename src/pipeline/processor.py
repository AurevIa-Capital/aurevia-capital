"""
Comprehensive data processing pipeline.

This module consolidates data cleaning, validation, outlier detection,
and interpolation from the previous modelling structure.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from scipy import interpolate
from statsmodels.tsa.stattools import adfuller

from .config import PipelineConfig, ProcessingConfig
from .base import PipelineComponent

logger = logging.getLogger(__name__)


class DataValidator:
    """Data validation and quality checking."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def validate_dataframe(self, df: pd.DataFrame, asset_name: str, price_column: str) -> Dict:
        """
        Perform comprehensive validation on a DataFrame.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame
        asset_name : str
            Name of the asset
        price_column : str
            Name of the price column
            
        Returns:
        -------
        Dict
            Validation results
        """
        results = {
            'asset_name': asset_name,
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        # Basic structure validation
        self._validate_structure(df, results, price_column)
        
        # Data quality validation
        self._validate_data_quality(df, results, price_column)
        
        # Time series validation
        self._validate_time_series(df, results, price_column)
        
        # Statistical validation
        self._validate_statistics(df, results, price_column)
        
        # Set overall validity
        results['valid'] = len(results['errors']) == 0
        
        return results
    
    def _validate_structure(self, df: pd.DataFrame, results: Dict, price_column: str) -> None:
        """Validate basic DataFrame structure."""
        
        if df.empty:
            results['errors'].append("DataFrame is empty")
            return
            
        if price_column not in df.columns:
            results['errors'].append(f"Missing required '{price_column}' column")
            
        if not isinstance(df.index, pd.DatetimeIndex):
            results['errors'].append("Index is not DatetimeIndex")
            
        if len(df) < self.config.min_data_points:
            results['warnings'].append(
                f"Only {len(df)} data points, minimum recommended: {self.config.min_data_points}"
            )
            
        results['statistics']['total_rows'] = len(df)
        results['statistics']['total_columns'] = len(df.columns)
    
    def _validate_data_quality(self, df: pd.DataFrame, results: Dict, price_column: str) -> None:
        """Validate data quality issues."""
        
        if price_column not in df.columns:
            return
            
        price_col = df[price_column]
        
        # Missing values
        missing_count = price_col.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        results['statistics']['missing_values'] = missing_count
        results['statistics']['missing_percentage'] = missing_pct
        
        if missing_pct > 50:
            results['errors'].append(f"High missing data percentage: {missing_pct:.1f}%")
        elif missing_pct > 20:
            results['warnings'].append(f"Moderate missing data: {missing_pct:.1f}%")
            
        # Invalid prices
        invalid_prices = (price_col <= 0).sum()
        if invalid_prices > 0:
            results['errors'].append(f"Found {invalid_prices} non-positive prices")
            
        # Duplicate timestamps
        duplicate_dates = df.index.duplicated().sum()
        if duplicate_dates > 0:
            results['warnings'].append(f"Found {duplicate_dates} duplicate timestamps")
            
        results['statistics']['invalid_prices'] = invalid_prices
        results['statistics']['duplicate_dates'] = duplicate_dates
    
    def _validate_time_series(self, df: pd.DataFrame, results: Dict, price_column: str) -> None:
        """Validate time series properties."""
        
        if price_column not in df.columns or len(df) < 2:
            return
            
        # Check frequency
        expected_freq = pd.infer_freq(df.index)
        if expected_freq is None:
            results['warnings'].append("Could not infer regular frequency")
        else:
            results['statistics']['inferred_frequency'] = expected_freq
            
        # Date range
        date_range = df.index.max() - df.index.min()
        results['statistics']['date_range_days'] = date_range.days
        results['statistics']['start_date'] = df.index.min()
        results['statistics']['end_date'] = df.index.max()
        
        # Time gaps
        time_diffs = df.index.to_series().diff().dropna()
        if len(time_diffs) > 0:
            max_gap = time_diffs.max()
            median_gap = time_diffs.median()
            
            if max_gap > median_gap * 5:
                results['warnings'].append(
                    f"Large time gap detected: {max_gap} (median: {median_gap})"
                )
                
            results['statistics']['max_time_gap'] = str(max_gap)
            results['statistics']['median_time_gap'] = str(median_gap)
    
    def _validate_statistics(self, df: pd.DataFrame, results: Dict, price_column: str) -> None:
        """Validate statistical properties."""
        
        if price_column not in df.columns:
            return
            
        price_col = df[price_column].dropna()
        
        if len(price_col) == 0:
            results['errors'].append("No valid price data after removing nulls")
            return
            
        # Basic statistics
        stats_dict = {
            'mean': price_col.mean(),
            'median': price_col.median(),
            'std': price_col.std(),
            'min': price_col.min(),
            'max': price_col.max(),
            'skewness': price_col.skew(),
            'kurtosis': price_col.kurtosis()
        }
        
        results['statistics'].update(stats_dict)
        
        # Check for extreme values
        if abs(stats_dict['skewness']) > 3:
            results['warnings'].append(f"High skewness: {stats_dict['skewness']:.2f}")
            
        if abs(stats_dict['kurtosis']) > 10:
            results['warnings'].append(f"High kurtosis: {stats_dict['kurtosis']:.2f}")
            
        # Coefficient of variation
        cv = stats_dict['std'] / stats_dict['mean'] if stats_dict['mean'] != 0 else float('inf')
        results['statistics']['coefficient_of_variation'] = cv
        
        if cv > 1:
            results['warnings'].append(f"High volatility (CV: {cv:.2f})")


class OutlierDetector:
    """Advanced outlier detection for time series data."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.method = config.outlier_method
        self.threshold = config.outlier_threshold
        
    def detect_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Detect outliers using configured method.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame
        column : str
            Column to analyze
            
        Returns:
        -------
        pd.Series
            Boolean series indicating outliers
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        data = df[column].dropna()
        
        if len(data) == 0:
            return pd.Series(dtype=bool, index=df.index)
            
        if self.method == 'iqr':
            return self._detect_iqr_outliers(df, column)
        elif self.method == 'zscore':
            return self._detect_zscore_outliers(df, column)
        elif self.method == 'isolation_forest':
            return self._detect_isolation_forest_outliers(df, column)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
    
    def _detect_iqr_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Detect outliers using Interquartile Range method."""
        
        data = df[column]
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR
        
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers.fillna(False)
    
    def _detect_zscore_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Detect outliers using Z-score method."""
        
        data = df[column].dropna()
        z_scores = np.abs(stats.zscore(data))
        
        outliers = pd.Series(False, index=df.index)
        outliers.loc[data.index] = z_scores > self.threshold
        
        return outliers
    
    def _detect_isolation_forest_outliers(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        
        data = df[column].dropna()
        
        if len(data) < 10:
            logger.warning("Too few samples for Isolation Forest, using IQR instead")
            return self._detect_iqr_outliers(df, column)
            
        # Reshape for sklearn
        X = data.values.reshape(-1, 1)
        
        contamination = min(0.1, self.threshold / 100)
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(X)
        
        outliers = pd.Series(False, index=df.index)
        outliers.loc[data.index] = outlier_labels == -1
        
        return outliers
    
    def clean_outliers(self, df: pd.DataFrame, outliers: pd.Series, column: str, method: str = 'interpolate') -> pd.DataFrame:
        """Clean outliers from DataFrame."""
        
        df_clean = df.copy()
        
        if method == 'remove':
            df_clean = df_clean[~outliers]
        elif method == 'interpolate':
            df_clean.loc[outliers, column] = np.nan
            df_clean[column] = df_clean[column].interpolate(method='linear')
        elif method == 'cap':
            data = df_clean[column]
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_clean.loc[outliers & (data < lower_bound), column] = lower_bound
            df_clean.loc[outliers & (data > upper_bound), column] = upper_bound
            
        return df_clean


class InterpolationEngine:
    """Advanced interpolation methods for time series data."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.method = config.interpolation_method
        self.fill_limit = config.fill_limit
        
    def interpolate_series(self, df: pd.DataFrame, column: str, method: Optional[str] = None) -> pd.DataFrame:
        """
        Interpolate missing values in a time series.
        
        Parameters:
        ----------
        df : pd.DataFrame
            DataFrame with datetime index
        column : str
            Column to interpolate
        method : str, optional
            Interpolation method
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with interpolated values
        """
        method = method or self.method
        df_result = df.copy()
        
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame")
            return df_result
            
        missing_before = df[column].isnull().sum()
        logger.info(f"Interpolating {missing_before} missing values using {method}")
        
        if method == 'backfill':
            df_result[column] = df_result[column].bfill(limit=self.fill_limit)
        elif method == 'forward':
            df_result[column] = df_result[column].ffill(limit=self.fill_limit)
        elif method == 'linear':
            df_result[column] = df_result[column].interpolate(method='linear', limit=self.fill_limit)
        elif method == 'spline':
            try:
                df_result[column] = df_result[column].interpolate(method='spline', order=3, limit=self.fill_limit)
            except Exception as e:
                logger.warning(f"Spline interpolation failed: {str(e)}, falling back to linear")
                df_result[column] = df_result[column].interpolate(method='linear', limit=self.fill_limit)
        elif method == 'seasonal':
            df_result[column] = self._seasonal_interpolation(df_result[column])
        elif method == 'hybrid':
            df_result[column] = self._hybrid_interpolation(df_result[column])
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
            
        missing_after = df_result[column].isnull().sum()
        logger.info(f"Interpolation complete. Missing values: {missing_before} → {missing_after}")
        
        return df_result
    
    def _seasonal_interpolation(self, series: pd.Series, period: int = 7) -> pd.Series:
        """Seasonal interpolation using similar periods."""
        result = series.copy()
        missing_mask = series.isnull()
        
        if not missing_mask.any():
            return result
            
        for idx in series.index[missing_mask]:
            seasonal_values = []
            
            for lag in [period, period*2, period*3, period*4]:
                past_idx = idx - pd.Timedelta(days=lag)
                future_idx = idx + pd.Timedelta(days=lag)
                
                if past_idx in series.index and not pd.isna(series[past_idx]):
                    seasonal_values.append(series[past_idx])
                if future_idx in series.index and not pd.isna(series[future_idx]):
                    seasonal_values.append(series[future_idx])
                    
            if seasonal_values:
                result[idx] = np.median(seasonal_values)
        
        # Fill remaining gaps with linear interpolation
        if result.isnull().any():
            result = result.interpolate(method='linear', limit=self.fill_limit)
            
        return result
    
    def _hybrid_interpolation(self, series: pd.Series) -> pd.Series:
        """Hybrid interpolation combining multiple methods."""
        result = series.copy()
        
        # Step 1: Forward fill small gaps (1-2 points)
        result = result.ffill(limit=2)
        
        # Step 2: Seasonal for medium gaps
        result = self._seasonal_interpolation(result)
        
        # Step 3: Linear for remaining gaps
        if result.isnull().any():
            result = result.interpolate(method='linear', limit=self.fill_limit)
            
        return result
    
    def resample_to_frequency(self, df: pd.DataFrame, freq: str, column: str) -> pd.DataFrame:
        """Resample time series to specified frequency."""
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")
            
        # Create full date range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        # Reindex and interpolate
        df_resampled = df.reindex(full_range)
        df_resampled = self.interpolate_series(df_resampled, column)
        
        logger.info(f"Resampled from {len(df)} to {len(df_resampled)} records at {freq} frequency")
        
        return df_resampled


class DataProcessor(PipelineComponent):
    """Comprehensive data processing pipeline."""
    
    def __init__(self, config: PipelineConfig, asset_type: str = "watch"):
        super().__init__(config)
        self.asset_type = asset_type
        self.asset_config = config.get_asset_config(asset_type)
        self.validator = DataValidator(config.processing)
        self.outlier_detector = OutlierDetector(config.processing)
        self.interpolation_engine = InterpolationEngine(config.processing)
        
    def process_single_asset(self, df: pd.DataFrame, asset_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Process a single asset's data through the complete pipeline.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame
        asset_name : str
            Name of the asset
            
        Returns:
        -------
        Tuple[pd.DataFrame, Dict]
            Processed DataFrame and processing report
        """
        logger.info(f"Processing {asset_name}")
        
        price_column = self.asset_config.price_column
        
        # Initialize report
        report = {
            'asset_name': asset_name,
            'original_length': len(df),
            'steps': []
        }
        
        df_clean = df.copy()
        
        # Step 1: Validation
        validation_result = self.validator.validate_dataframe(df_clean, asset_name, price_column)
        report['validation'] = validation_result
        
        if not validation_result['valid']:
            logger.error(f"Data validation failed for {asset_name}")
            return df_clean, report
            
        # Step 2: Remove invalid prices
        df_clean, step_report = self._remove_invalid_prices(df_clean, price_column)
        report['steps'].append(step_report)
        
        # Step 3: Handle duplicates
        df_clean, step_report = self._handle_duplicates(df_clean)
        report['steps'].append(step_report)
        
        # Step 4: Detect and handle outliers
        df_clean, step_report = self._handle_outliers(df_clean, price_column)
        report['steps'].append(step_report)
        
        # Step 5: Resample to consistent frequency
        df_clean, step_report = self._resample_data(df_clean, price_column)
        report['steps'].append(step_report)
        
        # Step 6: Interpolate missing values
        df_clean, step_report = self._interpolate_missing_values(df_clean, price_column)
        report['steps'].append(step_report)
        
        # Step 7: Final validation
        df_clean, step_report = self._final_validation(df_clean, price_column)
        report['steps'].append(step_report)
        
        report['final_length'] = len(df_clean)
        report['data_reduction'] = (report['original_length'] - report['final_length']) / report['original_length']
        
        logger.info(f"Processing completed for {asset_name}: {report['original_length']} → {report['final_length']} records")
        
        return df_clean, report
    
    def _remove_invalid_prices(self, df: pd.DataFrame, price_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Remove invalid price values."""
        
        original_length = len(df)
        invalid_mask = (df[price_column] <= 0) | df[price_column].isnull()
        df_clean = df[~invalid_mask].copy()
        removed_count = original_length - len(df_clean)
        
        return df_clean, {
            'step': 'remove_invalid_prices',
            'removed_records': removed_count,
            'remaining_records': len(df_clean),
            'description': f"Removed {removed_count} records with invalid prices"
        }
    
    def _handle_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Handle duplicate timestamps."""
        
        original_length = len(df)
        duplicate_mask = df.index.duplicated(keep='first')
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count > 0:
            df_clean = df[~duplicate_mask].copy()
        else:
            df_clean = df.copy()
            
        return df_clean, {
            'step': 'handle_duplicates',
            'removed_records': original_length - len(df_clean),
            'remaining_records': len(df_clean),
            'description': f"Removed {duplicate_count} duplicate timestamps"
        }
    
    def _handle_outliers(self, df: pd.DataFrame, price_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect and handle outliers."""
        
        outliers = self.outlier_detector.detect_outliers(df, price_column)
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            df_clean = self.outlier_detector.clean_outliers(df, outliers, price_column, method='interpolate')
        else:
            df_clean = df.copy()
            
        return df_clean, {
            'step': 'handle_outliers',
            'outliers_detected': int(outlier_count),
            'outlier_percentage': (outlier_count / len(df)) * 100 if len(df) > 0 else 0,
            'handling_method': 'interpolate',
            'remaining_records': len(df_clean),
            'description': f"Handled {outlier_count} outliers"
        }
    
    def _resample_data(self, df: pd.DataFrame, price_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Resample data to consistent frequency."""
        
        original_length = len(df)
        df_resampled = self.interpolation_engine.resample_to_frequency(
            df, self.config.processing.frequency, price_column
        )
        
        return df_resampled, {
            'step': 'resample_data',
            'original_records': original_length,
            'resampled_records': len(df_resampled),
            'target_frequency': self.config.processing.frequency,
            'description': f"Resampled to {self.config.processing.frequency} frequency"
        }
    
    def _interpolate_missing_values(self, df: pd.DataFrame, price_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Interpolate missing values."""
        
        missing_before = df[price_column].isnull().sum()
        df_interpolated = self.interpolation_engine.interpolate_series(df, price_column)
        missing_after = df_interpolated[price_column].isnull().sum()
        interpolated_count = missing_before - missing_after
        
        return df_interpolated, {
            'step': 'interpolate_missing_values',
            'missing_before': int(missing_before),
            'missing_after': int(missing_after),
            'interpolated_count': int(interpolated_count),
            'interpolation_method': self.config.processing.interpolation_method,
            'remaining_records': len(df_interpolated),
            'description': f"Interpolated {interpolated_count} missing values"
        }
    
    def _final_validation(self, df: pd.DataFrame, price_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Perform final validation and cleanup."""
        
        original_length = len(df)
        
        # Remove any remaining invalid data
        valid_mask = (
            df[price_column].notna() & 
            (df[price_column] > 0) & 
            np.isfinite(df[price_column])
        )
        
        df_final = df[valid_mask].copy()
        df_final.sort_index(inplace=True)
        removed_count = original_length - len(df_final)
        
        return df_final, {
            'step': 'final_validation',
            'removed_records': removed_count,
            'final_records': len(df_final),
            'data_quality_score': len(df_final) / original_length if original_length > 0 else 0,
            'description': f"Final cleanup removed {removed_count} invalid records"
        }
    
    def process_multiple_assets(self, asset_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Process multiple assets.
        
        Parameters:
        ----------
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset data
            
        Returns:
        -------
        Tuple[Dict[str, pd.DataFrame], Dict]
            Processed data and processing reports
        """
        processed_data = {}
        processing_reports = {}
        
        for asset_name, df in asset_data.items():
            try:
                processed_df, report = self.process_single_asset(df, asset_name)
                processed_data[asset_name] = processed_df
                processing_reports[asset_name] = report
            except Exception as e:
                logger.error(f"Failed to process {asset_name}: {str(e)}")
                processing_reports[asset_name] = {
                    'asset_name': asset_name,
                    'error': str(e),
                    'success': False
                }
                
        return processed_data, processing_reports
    
    def process(self, asset_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Main processing method for the pipeline component.
        
        Parameters:
        ----------
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset data to process
            
        Returns:
        -------
        Tuple[Dict[str, pd.DataFrame], Dict]
            Processed data and processing report
        """
        logger.info(f"Starting data processing for {len(asset_data)} assets")
        
        try:
            processed_data, processing_reports = self.process_multiple_assets(asset_data)
            
            report = self.generate_report(
                assets_processed=len(processed_data),
                total_assets=len(asset_data),
                processing_reports=processing_reports
            )
            
            return processed_data, report
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            report = self.generate_report(success=False, error=str(e))
            return {}, report