"""
Enhanced data loading with support for multiple asset types.

This module consolidates data loading functionality from the previous
modelling/preprocessing/data_loader.py structure.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import PipelineConfig, AssetConfig
from .base import PipelineComponent

logger = logging.getLogger(__name__)


class DataLoader(PipelineComponent):
    """Enhanced data loader with asset-type awareness."""
    
    def __init__(self, config: PipelineConfig, asset_type: str = "watch"):
        super().__init__(config)
        self.asset_type = asset_type
        self.asset_config = config.get_asset_config(asset_type)
        
    def discover_asset_files(self, 
                            pattern: Optional[str] = None,
                            max_files: Optional[int] = None) -> List[Path]:
        """
        Discover asset files in the data directory.
        
        Parameters:
        ----------
        pattern : str, optional
            File pattern to match (default: *.csv)
        max_files : int, optional
            Maximum number of files to return
            
        Returns:
        -------
        List[Path]
            List of discovered file paths
        """
        scrape_path = self.config.data_paths.scrape_path
        
        if not scrape_path.exists():
            logger.warning(f"Scrape directory does not exist: {scrape_path}")
            return []
            
        # Use provided pattern or default CSV pattern
        file_pattern = pattern or "*.csv"
        
        # Get all CSV files
        csv_files = list(scrape_path.glob(file_pattern))
        
        if max_files:
            csv_files = csv_files[:max_files]
            
        logger.info(f"Discovered {len(csv_files)} files in {scrape_path}")
        
        return sorted(csv_files)
    
    def load_single_asset(self, file_path: Path) -> Tuple[str, pd.DataFrame]:
        """
        Load a single asset file.
        
        Parameters:
        ----------
        file_path : Path
            Path to the asset file
            
        Returns:
        -------
        Tuple[str, pd.DataFrame]
            Asset name and loaded DataFrame
        """
        try:
            # Extract asset name from filename
            asset_name = self._extract_asset_name(file_path)
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Validate required columns
            if self.asset_config.date_column not in df.columns:
                raise ValueError(f"Missing date column: {self.asset_config.date_column}")
            if self.asset_config.price_column not in df.columns:
                raise ValueError(f"Missing price column: {self.asset_config.price_column}")
                
            # Convert date column to datetime
            df[self.asset_config.date_column] = pd.to_datetime(df[self.asset_config.date_column])
            
            # Set date as index
            df.set_index(self.asset_config.date_column, inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            logger.info(f"Loaded {asset_name}: {len(df)} records from {df.index.min()} to {df.index.max()}")
            
            return asset_name, df
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            raise
    
    def load_multiple_assets(self, 
                           file_paths: Optional[List[Path]] = None,
                           max_files: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load multiple asset files.
        
        Parameters:
        ----------
        file_paths : List[Path], optional
            Specific file paths to load. If None, discovers all files.
        max_files : int, optional
            Maximum number of files to load
            
        Returns:
        -------
        Dict[str, pd.DataFrame]
            Dictionary mapping asset names to DataFrames
        """
        if file_paths is None:
            file_paths = self.discover_asset_files(max_files=max_files)
        elif max_files:
            file_paths = file_paths[:max_files]
            
        asset_data = {}
        failed_files = []
        
        for file_path in file_paths:
            try:
                asset_name, df = self.load_single_asset(file_path)
                asset_data[asset_name] = df
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                failed_files.append(file_path)
                
        logger.info(f"Successfully loaded {len(asset_data)} assets")
        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files")
            
        return asset_data
    
    def _extract_asset_name(self, file_path: Path) -> str:
        """
        Extract asset name from file path based on configured pattern.
        
        Parameters:
        ----------
        file_path : Path
            Path to the asset file
            
        Returns:
        -------
        str
            Extracted asset name
        """
        filename = file_path.stem  # Filename without extension
        
        # Try to parse using the configured pattern
        try:
            match = re.match(self.asset_config.id_pattern, filename)
            if match:
                # For watch files: Brand-Model-ID
                if len(match.groups()) >= 3:
                    brand, model, asset_id = match.groups()[:3]
                    return f"{brand}-{model}-{asset_id}"
                elif len(match.groups()) >= 2:
                    brand, model = match.groups()[:2]
                    return f"{brand}-{model}"
                elif len(match.groups()) >= 1:
                    return match.groups()[0]
        except Exception as e:
            logger.debug(f"Pattern matching failed for {filename}: {str(e)}")
            
        # Fallback to filename
        return filename
    
    def get_asset_metadata(self, asset_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate metadata summary for loaded assets.
        
        Parameters:
        ----------
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of loaded asset data
            
        Returns:
        -------
        pd.DataFrame
            Metadata summary
        """
        metadata_list = []
        
        for asset_name, df in asset_data.items():
            
            # Parse asset name for metadata
            brand, model, asset_id = self._parse_asset_name(asset_name)
            
            metadata = {
                'asset_name': asset_name,
                'asset_type': self.asset_type,
                'brand': brand,
                'model': model,
                'asset_id': asset_id,
                'total_records': len(df),
                'start_date': df.index.min(),
                'end_date': df.index.max(),
                'date_range_days': (df.index.max() - df.index.min()).days,
                'missing_values': df[self.asset_config.price_column].isnull().sum(),
                'min_price': df[self.asset_config.price_column].min(),
                'max_price': df[self.asset_config.price_column].max(),
                'mean_price': df[self.asset_config.price_column].mean(),
                'price_std': df[self.asset_config.price_column].std()
            }
            
            metadata_list.append(metadata)
            
        return pd.DataFrame(metadata_list)
    
    def _parse_asset_name(self, asset_name: str) -> Tuple[str, str, str]:
        """Parse asset name into components."""
        try:
            match = re.match(self.asset_config.id_pattern, asset_name)
            if match and len(match.groups()) >= 3:
                return match.groups()[:3]
        except Exception:
            pass
            
        # Fallback parsing based on asset type
        if self.asset_type == "watch":
            parts = asset_name.split('-')
            if len(parts) >= 3:
                return parts[0], '-'.join(parts[1:-1]), parts[-1]
            elif len(parts) == 2:
                return parts[0], parts[1], 'unknown'
            else:
                return asset_name, 'unknown', 'unknown'
        elif self.asset_type == "stock":
            return asset_name, 'stock', 'unknown'
        elif self.asset_type == "crypto":
            parts = asset_name.split('-')
            if len(parts) >= 2:
                return parts[0], parts[1], 'unknown'
            else:
                return asset_name, 'crypto', 'unknown'
        else:
            return asset_name, 'unknown', 'unknown'
    
    def process(self, 
                file_paths: Optional[List[Path]] = None,
                max_files: Optional[int] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """
        Main processing method for the pipeline component.
        
        Parameters:
        ----------
        file_paths : List[Path], optional
            Specific file paths to load
        max_files : int, optional
            Maximum number of files to load
            
        Returns:
        -------
        Tuple[Dict[str, pd.DataFrame], Dict]
            Loaded asset data and processing report
        """
        logger.info(f"Starting data loading for asset type: {self.asset_type}")
        
        try:
            asset_data = self.load_multiple_assets(file_paths, max_files)
            metadata = self.get_asset_metadata(asset_data)
            
            report = self.generate_report(
                assets_loaded=len(asset_data),
                total_records=sum(len(df) for df in asset_data.values()),
                metadata_summary=metadata.describe().to_dict()
            )
            
            return asset_data, report
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            report = self.generate_report(success=False, error=str(e))
            return {}, report


class MultiAssetLoader:
    """Loader for multiple asset types simultaneously."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def load_all_assets(self, 
                       asset_types: List[str],
                       max_files_per_type: Optional[int] = None) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load multiple asset types.
        
        Parameters:
        ----------
        asset_types : List[str]
            List of asset types to load
        max_files_per_type : int, optional
            Maximum files per asset type
            
        Returns:
        -------
        Dict[str, Dict[str, pd.DataFrame]]
            Nested dictionary: {asset_type: {asset_name: df}}
        """
        all_data = {}
        
        for asset_type in asset_types:
            try:
                loader = DataLoader(self.config, asset_type)
                asset_data, _ = loader.process(max_files=max_files_per_type)
                all_data[asset_type] = asset_data
                logger.info(f"Loaded {len(asset_data)} {asset_type} assets")
            except Exception as e:
                logger.error(f"Failed to load {asset_type} assets: {str(e)}")
                all_data[asset_type] = {}
                
        return all_data