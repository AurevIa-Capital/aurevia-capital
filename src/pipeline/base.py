"""
Abstract base classes for the pipeline.

This module defines the core abstractions that asset-specific
implementations must follow.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from .config import PipelineConfig, AssetConfig

logger = logging.getLogger(__name__)


class AssetProcessor(ABC):
    """Abstract base class for asset-specific data processing."""
    
    def __init__(self, config: PipelineConfig, asset_type: str):
        self.config = config
        self.asset_type = asset_type
        self.asset_config = config.get_asset_config(asset_type)
        
    @abstractmethod
    def validate_asset_data(self, df: pd.DataFrame, asset_name: str) -> Dict[str, Any]:
        """
        Perform asset-specific validation.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame
        asset_name : str
            Name of the asset
            
        Returns:
        -------
        Dict[str, Any]
            Validation results
        """
        pass
    
    @abstractmethod
    def get_domain_features(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """
        Add asset-specific domain features.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame with basic features
        price_column : str
            Name of the price column
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with domain-specific features added
        """
        pass
    
    @abstractmethod
    def parse_asset_metadata(self, asset_name: str) -> Dict[str, str]:
        """
        Parse metadata from asset name.
        
        Parameters:
        ----------
        asset_name : str
            Name/identifier of the asset
            
        Returns:
        -------
        Dict[str, str]
            Parsed metadata (brand, model, etc.)
        """
        pass
    
    def get_asset_summary(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate summary statistics for processed assets.
        
        This method can be overridden for asset-specific summaries.
        
        Parameters:
        ----------
        processed_data : Dict[str, pd.DataFrame]
            Dictionary of processed asset data
            
        Returns:
        -------
        pd.DataFrame
            Summary statistics
        """
        summary_data = []
        
        for asset_name, df in processed_data.items():
            metadata = self.parse_asset_metadata(asset_name)
            
            summary = {
                'asset_name': asset_name,
                'asset_type': self.asset_type,
                'total_records': len(df),
                'date_range_days': (df.index.max() - df.index.min()).days,
                'avg_price': df[self.asset_config.price_column].mean(),
                'price_volatility': df[self.asset_config.price_column].std(),
                'data_quality': 1 - (df[self.asset_config.price_column].isnull().sum() / len(df))
            }
            
            # Add parsed metadata
            summary.update(metadata)
            summary_data.append(summary)
            
        return pd.DataFrame(summary_data)


class AssetFeatureEngineer(ABC):
    """Abstract base class for asset-specific feature engineering."""
    
    def __init__(self, config: PipelineConfig, asset_type: str):
        self.config = config
        self.asset_type = asset_type
        self.asset_config = config.get_asset_config(asset_type)
        self.feature_config = config.features
        
    @abstractmethod
    def engineer_domain_features(self, 
                                df: pd.DataFrame, 
                                price_column: str,
                                asset_metadata: Optional[Dict] = None) -> pd.DataFrame:
        """
        Engineer asset-specific features.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame with basic features
        price_column : str
            Name of the price column
        asset_metadata : Dict, optional
            Asset metadata for feature engineering
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with domain-specific features
        """
        pass
    
    def add_market_relative_features(self, 
                                   df: pd.DataFrame, 
                                   price_column: str,
                                   market_stats: Dict) -> pd.DataFrame:
        """
        Add features relative to overall market.
        
        This is a common pattern that can be overridden for asset-specific logic.
        
        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame
        price_column : str
            Name of the price column
        market_stats : Dict
            Market-wide statistics
            
        Returns:
        -------
        pd.DataFrame
            DataFrame with market-relative features
        """
        df_market = df.copy()
        
        # Price relative to market
        df_market['price_vs_market_mean'] = df_market[price_column] / market_stats['market_mean']
        df_market['price_vs_market_median'] = df_market[price_column] / market_stats['market_median']
        
        # Z-score relative to market
        df_market['market_z_score'] = (
            (df_market[price_column] - market_stats['market_mean']) / market_stats['market_std']
        )
        
        return df_market


class AssetAnalyzer(ABC):
    """Abstract base class for asset-specific analysis."""
    
    def __init__(self, config: PipelineConfig, asset_type: str):
        self.config = config
        self.asset_type = asset_type
        self.asset_config = config.get_asset_config(asset_type)
        
    @abstractmethod
    def generate_asset_insights(self, 
                              processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate asset-specific insights and analysis.
        
        Parameters:
        ----------
        processed_data : Dict[str, pd.DataFrame]
            Dictionary of processed asset data
            
        Returns:
        -------
        Dict[str, Any]
            Asset-specific insights and analysis results
        """
        pass
    
    def get_correlation_analysis(self, 
                               asset_data: Dict[str, pd.DataFrame],
                               price_column: str) -> Dict[str, Any]:
        """
        Analyze correlations between assets of the same type.
        
        Parameters:
        ----------
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset data
        price_column : str
            Name of the price column
            
        Returns:
        -------
        Dict[str, Any]
            Correlation analysis results
        """
        if len(asset_data) < 2:
            return {'error': 'Need at least 2 assets for correlation analysis'}
            
        # Create combined DataFrame
        price_series = {}
        for asset_name, df in asset_data.items():
            if price_column in df.columns:
                price_series[asset_name] = df[price_column]
                
        combined_df = pd.DataFrame(price_series)
        correlation_matrix = combined_df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'asset1': correlation_matrix.index[i],
                        'asset2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
                    
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'average_correlation': correlation_matrix.values[
                pd.np.triu_indices_from(correlation_matrix.values, k=1)
            ].mean() if len(correlation_matrix) > 1 else 0
        }


class PipelineComponent(ABC):
    """Base class for pipeline components."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    @abstractmethod
    def process(self, *args, **kwargs) -> Tuple[Any, Dict]:
        """
        Process data and return results with report.
        
        Returns:
        -------
        Tuple[Any, Dict]
            Processing results and processing report
        """
        pass
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """Validate input parameters."""
        return True
    
    def generate_report(self, **kwargs) -> Dict:
        """Generate processing report."""
        return {
            'component': self.__class__.__name__,
            'timestamp': pd.Timestamp.now(),
            'success': True
        }


def create_asset_processor(asset_type: str, config: PipelineConfig) -> AssetProcessor:
    """
    Factory function to create asset-specific processor.
    
    Parameters:
    ----------
    asset_type : str
        Type of asset to process
    config : PipelineConfig
        Pipeline configuration
        
    Returns:
    -------
    AssetProcessor
        Asset-specific processor instance
    """
    if asset_type == "watch":
        from .assets.watch import WatchProcessor
        return WatchProcessor(config, asset_type)
    elif asset_type == "stock":
        from .assets.stock import StockProcessor
        return StockProcessor(config, asset_type)
    elif asset_type == "crypto":
        from .assets.crypto import CryptoProcessor
        return CryptoProcessor(config, asset_type)
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")


def create_feature_engineer(asset_type: str, config: PipelineConfig) -> AssetFeatureEngineer:
    """
    Factory function to create asset-specific feature engineer.
    
    Parameters:
    ----------
    asset_type : str
        Type of asset
    config : PipelineConfig
        Pipeline configuration
        
    Returns:
    -------
    AssetFeatureEngineer
        Asset-specific feature engineer instance
    """
    if asset_type == "watch":
        from .assets.watch import WatchFeatureEngineer
        return WatchFeatureEngineer(config, asset_type)
    elif asset_type == "stock":
        from .assets.stock import StockFeatureEngineer
        return StockFeatureEngineer(config, asset_type)
    elif asset_type == "crypto":
        from .assets.crypto import CryptoFeatureEngineer
        return CryptoFeatureEngineer(config, asset_type)
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")