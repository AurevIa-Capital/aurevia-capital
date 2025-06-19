"""
Core library for multi-asset forecasting platform.

This package provides shared abstractions and utilities for asset forecasting
across different asset types (watches, gold, crypto, etc.).
"""

__version__ = "0.1.0"
__author__ = "Asset Forecasting Platform"

from .assets.base import BaseAsset, BaseDataCollector
from .schemas.timeseries import PricePoint, Forecast

__all__ = [
    "BaseAsset",
    "BaseDataCollector", 
    "PricePoint",
    "Forecast"
]