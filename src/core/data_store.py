"""
Centralized Data Store for the AurevIa Pipeline.

This module implements a centralized data storage abstraction that reduces
file system coupling between pipeline components and provides a unified
interface for data operations.
"""

import logging
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Iterator
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle

import pandas as pd
import numpy as np

from ..pipeline.config import PipelineConfig, DataPaths

logger = logging.getLogger(__name__)


@dataclass
class DataInfo:
    """Metadata about stored data."""
    
    key: str
    data_type: str  # 'scraped', 'processed', 'features', 'models', 'predictions'
    format: str  # 'csv', 'pickle', 'json'
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    description: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DataStore(ABC):
    """Abstract base class for data storage implementations."""
    
    @abstractmethod
    def save(self, key: str, data: Any, format: str = 'auto', **kwargs) -> str:
        """Save data with the given key."""
        pass
    
    @abstractmethod
    def load(self, key: str, **kwargs) -> Any:
        """Load data with the given key."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if data exists for the given key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete data with the given key."""
        pass
    
    @abstractmethod
    def list_keys(self, pattern: str = None) -> List[str]:
        """List all available keys, optionally filtered by pattern."""
        pass
    
    @abstractmethod
    def get_info(self, key: str) -> Optional[DataInfo]:
        """Get metadata about stored data."""
        pass


class FileSystemDataStore(DataStore):
    """File system-based data store implementation."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data_paths = config.data_paths
        self.base_path = Path(self.data_paths.base_data_dir)
        self.metadata_file = self.base_path / ".data_store_metadata.json"
        
        # Create directories
        self._create_directories()
        
        # Load metadata
        self._metadata = self._load_metadata()
    
    def _create_directories(self):
        """Create necessary directories."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "scrape" / "prices").mkdir(parents=True, exist_ok=True)
        (self.base_path / "output").mkdir(parents=True, exist_ok=True)
        (self.base_path / "models").mkdir(parents=True, exist_ok=True)
        (self.base_path / "cache").mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict[str, DataInfo]:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            metadata = {}
            for key, info_dict in metadata_dict.items():
                # Convert datetime strings back to datetime objects
                info_dict['created_at'] = datetime.fromisoformat(info_dict['created_at'])
                info_dict['modified_at'] = datetime.fromisoformat(info_dict['modified_at'])
                metadata[key] = DataInfo(**info_dict)
            
            return metadata
        
        except Exception as e:
            logger.warning(f"Could not load metadata: {str(e)}")
            return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            metadata_dict = {}
            for key, info in self._metadata.items():
                info_dict = asdict(info)
                # Convert datetime objects to strings
                info_dict['created_at'] = info.created_at.isoformat()
                info_dict['modified_at'] = info.modified_at.isoformat()
                metadata_dict[key] = info_dict
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        
        except Exception as e:
            logger.error(f"Could not save metadata: {str(e)}")
    
    def _determine_format(self, data: Any, format: str = 'auto') -> str:
        """Determine the appropriate format for data."""
        if format != 'auto':
            return format
        
        if isinstance(data, pd.DataFrame):
            return 'csv'
        elif isinstance(data, (dict, list)):
            return 'json'
        else:
            return 'pickle'
    
    def _get_file_path(self, key: str, format: str) -> Path:
        """Get the file path for a given key and format."""
        # Determine subdirectory based on data type
        if key.startswith('scraped_'):
            subdir = "scrape/prices"
        elif key.startswith('processed_') or key.startswith('features_'):
            subdir = "output"
        elif key.startswith('model_'):
            subdir = "models"
        else:
            subdir = "cache"
        
        # File extension based on format
        extensions = {
            'csv': '.csv',
            'json': '.json',
            'pickle': '.pkl'
        }
        
        filename = key + extensions.get(format, '.pkl')
        return self.base_path / subdir / filename
    
    def save(self, key: str, data: Any, format: str = 'auto', **kwargs) -> str:
        """Save data with the given key."""
        format = self._determine_format(data, format)
        file_path = self._get_file_path(key, format)
        
        # Create parent directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, **kwargs)
                else:
                    raise ValueError(f"Cannot save {type(data)} as CSV")
            
            elif format == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str, **kwargs)
            
            elif format == 'pickle':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, **kwargs)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Update metadata
            now = datetime.now()
            data_type = key.split('_')[0] if '_' in key else 'unknown'
            
            self._metadata[key] = DataInfo(
                key=key,
                data_type=data_type,
                format=format,
                size_bytes=file_path.stat().st_size,
                created_at=self._metadata.get(key, DataInfo(key, data_type, format, 0, now, now)).created_at,
                modified_at=now,
                description=kwargs.get('description'),
                tags=kwargs.get('tags', [])
            )
            
            self._save_metadata()
            
            logger.info(f"Saved {key} as {format} to {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Failed to save {key}: {str(e)}")
            raise
    
    def load(self, key: str, **kwargs) -> Any:
        """Load data with the given key."""
        if not self.exists(key):
            raise KeyError(f"Key not found: {key}")
        
        info = self._metadata[key]
        file_path = self._get_file_path(key, info.format)
        
        try:
            if info.format == 'csv':
                return pd.read_csv(file_path, **kwargs)
            
            elif info.format == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f, **kwargs)
            
            elif info.format == 'pickle':
                with open(file_path, 'rb') as f:
                    return pickle.load(f, **kwargs)
            
            else:
                raise ValueError(f"Unsupported format: {info.format}")
        
        except Exception as e:
            logger.error(f"Failed to load {key}: {str(e)}")
            raise
    
    def exists(self, key: str) -> bool:
        """Check if data exists for the given key."""
        return key in self._metadata
    
    def delete(self, key: str) -> bool:
        """Delete data with the given key."""
        if not self.exists(key):
            return False
        
        try:
            info = self._metadata[key]
            file_path = self._get_file_path(key, info.format)
            
            if file_path.exists():
                file_path.unlink()
            
            del self._metadata[key]
            self._save_metadata()
            
            logger.info(f"Deleted {key}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete {key}: {str(e)}")
            return False
    
    def list_keys(self, pattern: str = None) -> List[str]:
        """List all available keys, optionally filtered by pattern."""
        keys = list(self._metadata.keys())
        
        if pattern:
            import fnmatch
            keys = [key for key in keys if fnmatch.fnmatch(key, pattern)]
        
        return sorted(keys)
    
    def get_info(self, key: str) -> Optional[DataInfo]:
        """Get metadata about stored data."""
        return self._metadata.get(key)
    
    def list_by_type(self, data_type: str) -> List[str]:
        """List keys by data type."""
        return [key for key, info in self._metadata.items() if info.data_type == data_type]
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get summary of storage usage."""
        total_size = sum(info.size_bytes for info in self._metadata.values())
        
        type_counts = {}
        type_sizes = {}
        
        for info in self._metadata.values():
            type_counts[info.data_type] = type_counts.get(info.data_type, 0) + 1
            type_sizes[info.data_type] = type_sizes.get(info.data_type, 0) + info.size_bytes
        
        return {
            'total_items': len(self._metadata),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'by_type': {
                'counts': type_counts,
                'sizes_mb': {k: round(v / (1024 * 1024), 2) for k, v in type_sizes.items()}
            }
        }


class ScrapedDataManager:
    """Manager for scraped data operations."""
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
    
    def save_watch_prices(self, brand: str, model: str, watch_id: str, data: pd.DataFrame) -> str:
        """Save scraped watch price data."""
        key = f"scraped_{brand}_{model}_{watch_id}"
        return self.data_store.save(key, data, description=f"Price data for {brand} {model}")
    
    def load_watch_prices(self, brand: str, model: str, watch_id: str) -> pd.DataFrame:
        """Load scraped watch price data."""
        key = f"scraped_{brand}_{model}_{watch_id}"
        return self.data_store.load(key)
    
    def list_scraped_watches(self) -> List[Dict[str, str]]:
        """List all scraped watches."""
        keys = self.data_store.list_by_type('scraped')
        watches = []
        
        for key in keys:
            parts = key.replace('scraped_', '').split('_')
            if len(parts) >= 3:
                watches.append({
                    'brand': parts[0],
                    'model': parts[1],
                    'id': parts[2],
                    'key': key
                })
        
        return watches
    
    def save_scraping_progress(self, progress_data: Dict[str, Any]) -> str:
        """Save scraping progress."""
        return self.data_store.save('scraping_progress', progress_data, format='json')
    
    def load_scraping_progress(self) -> Dict[str, Any]:
        """Load scraping progress."""
        if self.data_store.exists('scraping_progress'):
            return self.data_store.load('scraping_progress')
        return {}


class ProcessedDataManager:
    """Manager for processed data operations."""
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
    
    def save_processed_data(self, data: pd.DataFrame, stage: str = 'final') -> str:
        """Save processed data."""
        key = f"processed_{stage}"
        return self.data_store.save(key, data, description=f"Processed data - {stage} stage")
    
    def load_processed_data(self, stage: str = 'final') -> pd.DataFrame:
        """Load processed data."""
        key = f"processed_{stage}"
        return self.data_store.load(key)
    
    def save_feature_data(self, data: pd.DataFrame, feature_set: str = 'default') -> str:
        """Save feature engineered data."""
        key = f"features_{feature_set}"
        return self.data_store.save(key, data, description=f"Feature data - {feature_set} set")
    
    def load_feature_data(self, feature_set: str = 'default') -> pd.DataFrame:
        """Load feature engineered data."""
        key = f"features_{feature_set}"
        return self.data_store.load(key)
    
    def save_pipeline_summary(self, summary: Dict[str, Any]) -> str:
        """Save pipeline processing summary."""
        return self.data_store.save('pipeline_summary', summary, format='json')


class ModelDataManager:
    """Manager for model data operations."""
    
    def __init__(self, data_store: DataStore):
        self.data_store = data_store
    
    def save_trained_model(self, model: Any, model_name: str, horizon: int, asset_id: str = None) -> str:
        """Save a trained model."""
        if asset_id:
            key = f"model_{model_name}_{horizon}d_{asset_id}"
        else:
            key = f"model_{model_name}_{horizon}d"
        
        return self.data_store.save(key, model, format='pickle', 
                                  description=f"{model_name} model for {horizon}-day horizon")
    
    def load_trained_model(self, model_name: str, horizon: int, asset_id: str = None) -> Any:
        """Load a trained model."""
        if asset_id:
            key = f"model_{model_name}_{horizon}d_{asset_id}"
        else:
            key = f"model_{model_name}_{horizon}d"
        
        return self.data_store.load(key)
    
    def save_model_comparison(self, comparison_data: Dict[str, Any], horizon: int) -> str:
        """Save model comparison results."""
        key = f"model_comparison_{horizon}d"
        return self.data_store.save(key, comparison_data, format='json')
    
    def save_predictions(self, predictions: pd.DataFrame, model_name: str, horizon: int) -> str:
        """Save model predictions."""
        key = f"predictions_{model_name}_{horizon}d"
        return self.data_store.save(key, predictions, description=f"Predictions from {model_name}")
    
    def list_trained_models(self) -> List[Dict[str, Any]]:
        """List all trained models."""
        keys = self.data_store.list_by_type('model')
        models = []
        
        for key in keys:
            if key.startswith('model_') and not key.startswith('model_comparison'):
                parts = key.replace('model_', '').split('_')
                if len(parts) >= 2:
                    models.append({
                        'model_name': parts[0],
                        'horizon': parts[1],
                        'asset_id': parts[2] if len(parts) > 2 else None,
                        'key': key
                    })
        
        return models


def create_data_store(config: PipelineConfig) -> DataStore:
    """Factory function to create a data store instance."""
    # For now, only FileSystemDataStore is implemented
    # In the future, could add database or cloud storage implementations
    return FileSystemDataStore(config)


def create_data_managers(data_store: DataStore) -> Dict[str, Any]:
    """Create all data managers."""
    return {
        'scraped': ScrapedDataManager(data_store),
        'processed': ProcessedDataManager(data_store),
        'models': ModelDataManager(data_store)
    }