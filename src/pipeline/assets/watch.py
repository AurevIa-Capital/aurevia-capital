"""
Watch-specific pipeline implementations.

This module consolidates all watch-specific logic from the previous
modelling/assets/watch/ structure into a single comprehensive module.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from ..config import PipelineConfig, WatchConfig
from ..base import AssetProcessor, AssetFeatureEngineer
from ..features import FeatureEngineer

logger = logging.getLogger(__name__)


class WatchProcessor(AssetProcessor):
    """Watch-specific data processing with luxury market domain knowledge."""
    
    def __init__(self, config: PipelineConfig, asset_type: str = "watch"):
        super().__init__(config, asset_type)
        self.watch_config = config.watch
        
    def validate_asset_data(self, df: pd.DataFrame, asset_name: str) -> Dict[str, Any]:
        """
        Perform watch-specific validation.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame
        asset_name : str
            Name of the watch asset
            
        Returns:
        -------
        Dict[str, Any]
            Watch-specific validation results
        """
        validation_results = {
            'asset_name': asset_name,
            'asset_type': 'watch',
            'valid': True,
            'warnings': [],
            'errors': [],
            'watch_specific': {}
        }
        
        price_column = self.asset_config.price_column
        
        # Check if it's a valid watch price range
        if price_column in df.columns:
            prices = df[price_column].dropna()
            if len(prices) > 0:
                min_price = prices.min()
                max_price = prices.max()
                mean_price = prices.mean()
                
                # Watch price validation
                if min_price < 100:  # SGD
                    validation_results['warnings'].append(f"Unusually low watch price: {min_price} SGD")
                if max_price > 1000000:  # 1M SGD
                    validation_results['warnings'].append(f"Exceptionally high watch price: {max_price} SGD")
                    
                # Detect luxury tier
                luxury_tier = self._classify_luxury_tier(mean_price)
                validation_results['watch_specific']['luxury_tier'] = luxury_tier
                validation_results['watch_specific']['price_range'] = {
                    'min': min_price,
                    'max': max_price,
                    'mean': mean_price
                }
                
                # Volatility check for watch market
                returns = prices.pct_change().dropna()
                if len(returns) > 1:
                    daily_vol = returns.std()
                    annual_vol = daily_vol * np.sqrt(252)
                    
                    if annual_vol > 0.5:  # 50% annual volatility
                        validation_results['warnings'].append(f"High volatility for watch: {annual_vol:.2%}")
                    
                    validation_results['watch_specific']['volatility'] = {
                        'daily': daily_vol,
                        'annualized': annual_vol
                    }
        
        # Parse watch metadata
        metadata = self.parse_asset_metadata(asset_name)
        validation_results['watch_specific']['metadata'] = metadata
        
        # Validate brand recognition
        brand = metadata.get('brand', '').lower()
        if brand and not self._is_recognized_brand(brand):
            validation_results['warnings'].append(f"Unrecognized watch brand: {metadata.get('brand')}")
            
        return validation_results
    
    def _classify_luxury_tier(self, mean_price: float) -> str:
        """Classify watch into luxury tier based on price."""
        for tier, config in self.watch_config.luxury_tiers.items():
            if config['min_price'] <= mean_price < config['max_price']:
                return tier
        return 'unknown'
    
    def _is_recognized_brand(self, brand: str) -> bool:
        """Check if brand is in our recognized luxury watch brands."""
        all_brands = []
        for tier_brands in self.watch_config.brand_tiers.values():
            all_brands.extend(tier_brands)
        
        return any(recognized_brand in brand for recognized_brand in all_brands)
    
    def get_domain_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """
        Add watch-specific domain features.
        
        This method delegates to WatchFeatureEngineer for consistency.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame with basic features
        price_column : str
            Name of the price column
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with watch-specific features added
        """
        # This will be called by the main pipeline
        # The actual feature engineering is done by WatchFeatureEngineer
        return df
    
    def parse_asset_metadata(self, asset_name: str) -> Dict[str, str]:
        """
        Parse watch metadata from asset name.
        
        Parameters:
        ----------
        asset_name : str
            Watch asset name (e.g., "Rolex-Submariner-638")
            
        Returns:
        -------
        Dict[str, str]
            Parsed metadata
        """
        try:
            match = re.match(self.asset_config.id_pattern, asset_name)
            if match and len(match.groups()) >= 3:
                brand, model, asset_id = match.groups()[:3]
                return {
                    'brand': brand,
                    'model': model, 
                    'asset_id': asset_id,
                    'full_name': f"{brand} {model}"
                }
        except Exception as e:
            logger.debug(f"Failed to parse watch metadata for {asset_name}: {str(e)}")
            
        # Fallback parsing
        parts = asset_name.split('-')
        if len(parts) >= 3:
            return {
                'brand': parts[0],
                'model': '-'.join(parts[1:-1]),
                'asset_id': parts[-1],
                'full_name': f"{parts[0]} {'-'.join(parts[1:-1])}"
            }
        elif len(parts) == 2:
            return {
                'brand': parts[0],
                'model': parts[1],
                'asset_id': 'unknown',
                'full_name': f"{parts[0]} {parts[1]}"
            }
        else:
            return {
                'brand': asset_name,
                'model': 'unknown',
                'asset_id': 'unknown',
                'full_name': asset_name
            }
    
    def get_brand_summary(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary by watch brands.
        
        Parameters:
        ----------
        processed_data : Dict[str, pd.DataFrame]
            Dictionary of processed watch data
            
        Returns:
        -------
        pd.DataFrame
            Brand-based summary
        """
        brand_data = []
        
        for asset_name, df in processed_data.items():
            metadata = self.parse_asset_metadata(asset_name)
            brand = metadata.get('brand', 'Unknown')
            
            brand_info = {
                'brand': brand,
                'asset_name': asset_name,
                'total_records': len(df),
                'date_range_days': (df.index.max() - df.index.min()).days,
                'avg_price': df[self.asset_config.price_column].mean(),
                'price_volatility': df[self.asset_config.price_column].std(),
                'data_quality': 1 - (df[self.asset_config.price_column].isnull().sum() / len(df)),
                'luxury_tier': self._classify_luxury_tier(df[self.asset_config.price_column].mean())
            }
            
            brand_data.append(brand_info)
            
        brand_df = pd.DataFrame(brand_data)
        
        if len(brand_df) == 0:
            return brand_df
            
        # Aggregate by brand
        brand_summary = brand_df.groupby('brand').agg({
            'asset_name': 'count',
            'total_records': ['sum', 'mean'],
            'date_range_days': 'mean',
            'avg_price': 'mean',
            'price_volatility': 'mean',
            'data_quality': 'mean'
        }).round(2)
        
        # Flatten column names
        brand_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] 
                               for col in brand_summary.columns]
        
        brand_summary.rename(columns={
            'asset_name_count': 'watch_count',
            'total_records_sum': 'total_data_points',
            'total_records_mean': 'avg_data_points_per_watch'
        }, inplace=True)
        
        return brand_summary.reset_index()


class WatchFeatureEngineer(AssetFeatureEngineer):
    """Watch-specific feature engineering with luxury market domain knowledge."""
    
    def __init__(self, config: PipelineConfig, asset_type: str = "watch"):
        super().__init__(config, asset_type)
        self.watch_config = config.watch
        
    def engineer_domain_features(self, 
                                df: pd.DataFrame, 
                                price_column: str,
                                asset_metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Engineer watch-specific features.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame with basic features
        price_column : str
            Name of the price column
        asset_metadata : Dict, optional
            Watch metadata for feature engineering
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with watch-specific features
        """
        logger.info("Engineering watch-specific features")
        
        df_features = df.copy()
        
        # Add luxury market features
        df_features = self._add_luxury_market_features(df_features, price_column)
        
        # Add watch seasonality features
        df_features = self._add_watch_seasonality_features(df_features, price_column)
        
        # Add price tier features
        df_features = self._add_price_tier_features(df_features, price_column)
        
        # Add brand-specific features if metadata available
        if asset_metadata:
            df_features = self._add_brand_features(df_features, asset_metadata, price_column)
            
        return df_features
    
    def _add_luxury_market_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add features specific to luxury watch market dynamics."""
        
        df_luxury = df.copy()
        
        # Luxury market volatility (shorter windows for luxury goods)
        for window in [3, 5, 10]:
            returns = df_luxury[price_column].pct_change()
            df_luxury[f'luxury_vol_{window}'] = returns.rolling(window=window).std() * np.sqrt(252)
            
        # Premium pricing indicators
        rolling_30 = df_luxury[price_column].rolling(window=30)
        df_luxury['price_premium_30d'] = (
            (df_luxury[price_column] - rolling_30.min()) / rolling_30.min()
        ).replace([np.inf, -np.inf], np.nan)
        
        df_luxury['price_discount_30d'] = (
            (rolling_30.max() - df_luxury[price_column]) / rolling_30.max()
        ).replace([np.inf, -np.inf], np.nan)
        
        # Luxury trend strength (momentum with volatility adjustment)
        for period in [7, 14, 21]:
            momentum = df_luxury[price_column].pct_change(period)
            volatility = df_luxury[price_column].pct_change().rolling(period).std()
            df_luxury[f'trend_strength_{period}'] = (
                momentum / volatility
            ).replace([np.inf, -np.inf], np.nan)
            
        # Market stress indicators (rapid price changes)
        rolling_3_std = df_luxury[price_column].rolling(3).std()
        rolling_30_std = df_luxury[price_column].rolling(30).std()
        
        df_luxury['price_stress_3d'] = (
            rolling_3_std / rolling_30_std
        ).replace([np.inf, -np.inf], np.nan)
        
        rolling_7_std = df_luxury[price_column].rolling(7).std()
        df_luxury['price_stress_7d'] = (
            rolling_7_std / rolling_30_std
        ).replace([np.inf, -np.inf], np.nan)
        
        return df_luxury
    
    def _add_watch_seasonality_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add watch market seasonality features."""
        
        df_season = df.copy()
        
        # Holiday seasons (luxury watch sales patterns)
        df_season['is_holiday_season'] = df_season.index.month.isin([11, 12, 1])  # Nov-Jan
        df_season['is_spring_season'] = df_season.index.month.isin([3, 4, 5])     # Mar-May
        df_season['is_summer_season'] = df_season.index.month.isin([6, 7, 8])     # Jun-Aug
        df_season['is_fall_season'] = df_season.index.month.isin([9, 10])         # Sep-Oct
        
        # Year-end effects (fiscal year, bonuses, etc.)
        year_end_dates = pd.to_datetime([f"{year}-12-31" for year in df_season.index.year.unique()])
        days_to_year_end = []
        
        for date in df_season.index:
            year_end = pd.Timestamp(f"{date.year}-12-31")
            days_to_year_end.append((year_end - date).days)
            
        df_season['days_to_year_end'] = days_to_year_end
        df_season['is_year_end_month'] = df_season.index.month == 12
        df_season['is_new_year_month'] = df_season.index.month == 1
        
        # Watch fair seasonality (Basel World / Watches & Wonders - March-April)
        df_season['is_watch_fair_season'] = df_season.index.month.isin([3, 4])
        
        # Seasonal price patterns
        if len(df_season) > 12:  # Need enough data for monthly patterns
            monthly_avg = df_season.groupby(df_season.index.month)[price_column].transform('mean')
            df_season['seasonal_price_index'] = (
                df_season[price_column] / monthly_avg
            ).replace([np.inf, -np.inf], np.nan)
        else:
            df_season['seasonal_price_index'] = 1.0
        
        return df_season
    
    def _add_price_tier_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Add features based on watch price tiers."""
        
        df_tier = df.copy()
        
        # Determine price tier based on mean price
        price_mean = df_tier[price_column].mean()
        
        # Initialize all tier indicators to 0
        for tier in self.watch_config.luxury_tiers.keys():
            df_tier[f'price_tier_{tier}'] = 0
            
        # Set the appropriate tier to 1
        current_tier = 'entry_luxury'  # default
        for tier, config in self.watch_config.luxury_tiers.items():
            if config['min_price'] <= price_mean < config['max_price']:
                current_tier = tier
                df_tier[f'price_tier_{tier}'] = 1
                break
                
        # Tier-specific volatility expectations
        tier_config = self.watch_config.luxury_tiers[current_tier]
        expected_vol = 1.0 / tier_config['volatility_factor']  # Inverse relationship
        
        returns = df_tier[price_column].pct_change()
        actual_vol = returns.rolling(30).std() * np.sqrt(252)
        df_tier['vol_vs_expected'] = (actual_vol / expected_vol).replace([np.inf, -np.inf], np.nan)
        
        # Price positioning within historical range
        if len(df_tier) >= 30:
            rolling_365 = df_tier[price_column].rolling(window=min(365, len(df_tier)), min_periods=30)
            df_tier['price_percentile_365d'] = (
                (df_tier[price_column] - rolling_365.min()) / 
                (rolling_365.max() - rolling_365.min())
            ).replace([np.inf, -np.inf], np.nan)
        else:
            df_tier['price_percentile_365d'] = 0.5  # neutral position
        
        return df_tier
    
    def _add_brand_features(self, 
                           df: pd.DataFrame, 
                           asset_metadata: Dict, 
                           price_column: str) -> pd.DataFrame:
        """Add brand-specific features."""
        
        df_brand = df.copy()
        
        brand = asset_metadata.get('brand', 'Unknown').lower().replace(' ', '_')
        model = asset_metadata.get('model', '').lower()
        
        # Initialize all brand tier indicators to 0
        for tier in self.watch_config.brand_tiers.keys():
            df_brand[f'brand_tier_{tier}'] = 0
            
        # Determine brand tier and set volatility factor
        brand_volatility_factor = 1.5  # default for unknown brands
        
        for tier, tier_brands in self.watch_config.brand_tiers.items():
            if any(tier_brand in brand for tier_brand in tier_brands):
                df_brand[f'brand_tier_{tier}'] = 1
                
                # Set volatility factor based on tier
                if tier == 'ultra_luxury':
                    brand_volatility_factor = 0.8
                elif tier == 'high_luxury':
                    brand_volatility_factor = 1.0
                elif tier == 'mid_luxury':
                    brand_volatility_factor = 1.2
                break
                
        # Brand-adjusted volatility
        returns = df_brand[price_column].pct_change()
        base_vol = returns.rolling(30).std()
        df_brand['brand_adjusted_vol'] = base_vol * brand_volatility_factor
        
        # Sports watch indicators (affects seasonality and volatility)
        sports_keywords = ['speedmaster', 'submariner', 'daytona', 'gmt', 'diver', 'chrono', 'pilot']
        df_brand['is_sports_watch'] = int(any(keyword in model for keyword in sports_keywords))
        
        # Dress watch indicators
        dress_keywords = ['calatrava', 'master', 'patrimony', 'royal_oak', 'nautilus', 'dress', 'classic']
        df_brand['is_dress_watch'] = int(any(keyword in model for keyword in dress_keywords))
        
        # Complication indicators (affects price stability)
        complication_keywords = ['perpetual', 'tourbillon', 'minute_repeater', 'chronograph', 'moonphase']
        df_brand['has_complications'] = int(any(keyword in model for keyword in complication_keywords))
        
        return df_brand
    
    def process_multiple_watches(self, 
                                watch_data: Dict[str, pd.DataFrame],
                                price_column: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Process features for multiple watches with cross-watch insights.
        
        Parameters:
        ----------
        watch_data : Dict[str, pd.DataFrame]
            Dictionary of watch data
        price_column : str, optional
            Name of the price column
            
        Returns:
        -------
        Dict[str, pd.DataFrame]
            Dictionary of watches with engineered features
        """
        price_column = price_column or self.asset_config.price_column
        featured_data = {}
        
        # Calculate market-wide statistics for relative features
        all_prices = []
        for df in watch_data.values():
            if price_column in df.columns:
                all_prices.extend(df[price_column].dropna().tolist())
                
        if all_prices:
            market_stats = {
                'market_mean': np.mean(all_prices),
                'market_std': np.std(all_prices),
                'market_median': np.median(all_prices)
            }
        else:
            market_stats = {'market_mean': 0, 'market_std': 1, 'market_median': 0}
        
        # Process each watch
        for asset_name, df in watch_data.items():
            try:
                # Parse asset metadata
                processor = WatchProcessor(self.config, self.asset_type)
                asset_metadata = processor.parse_asset_metadata(asset_name)
                
                # First apply base feature engineering
                base_engineer = FeatureEngineer(self.config, self.asset_type)
                df_featured = base_engineer.engineer_features(df, price_column)
                
                # Then add watch-specific features
                df_featured = self.engineer_domain_features(df_featured, price_column, asset_metadata)
                
                # Add market-relative features
                df_featured = self.add_market_relative_features(df_featured, price_column, market_stats)
                
                featured_data[asset_name] = df_featured
                
                logger.info(f"Processed features for {asset_name}: {len(df_featured.columns)} features")
                
            except Exception as e:
                logger.error(f"Failed to process features for {asset_name}: {str(e)}")
                
        return featured_data