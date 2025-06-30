"""Time series data schemas for standardized data exchange."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, List, Any


class PricePoint(BaseModel):
    """Standardized price point across all assets."""
    timestamp: datetime
    price: float
    currency: str = "USD"
    volume: Optional[float] = None
    source: str
    asset_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Forecast(BaseModel):
    """Standardized forecast output."""
    asset_id: str
    forecast_date: datetime
    predictions: List[PricePoint]
    confidence_intervals: Dict[str, List[float]]
    model_name: str
    metrics: Dict[str, float]
    horizon_days: int = 30
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ModelMetrics(BaseModel):
    """Standardized model performance metrics."""
    model_name: str
    asset_id: str
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2_score: Optional[float] = None  # R-squared score
    training_date: datetime
    validation_period_days: int
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }