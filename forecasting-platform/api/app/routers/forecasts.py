"""Forecasting endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.dependencies import get_container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{asset_type}/{asset_id}")
async def get_forecast(
    asset_type: str,
    asset_id: str,
    days: int = Query(default=30, le=365, ge=1),
    model: Optional[str] = Query(default=None),
    container=Depends(get_container)
):
    """Get forecast for specific asset."""
    try:
        available_models = ["linear_regression", "xgboost", "arima", "sarima", "random_forest"]
        
        if model and model not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model}. Available: {available_models}"
            )
        
        # Use best performing model if none specified
        if not model:
            model = "xgboost"
        
        # Mock forecast data
        start_date = datetime.utcnow()
        predictions = []
        base_price = 15000.0
        
        for i in range(days):
            date = start_date + timedelta(days=i+1)
            # Mock prediction with some trend and noise
            trend = i * 2.5  # Small upward trend
            noise = (i % 7) * 25 - 75  # Weekly variation
            predicted_price = base_price + trend + noise
            
            predictions.append({
                "date": date.strftime("%Y-%m-%d"),
                "predicted_price": round(predicted_price, 2),
                "lower_bound": round(predicted_price * 0.95, 2),
                "upper_bound": round(predicted_price * 1.05, 2),
                "confidence": 0.85
            })
        
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "model": model,
            "forecast_date": datetime.utcnow().isoformat(),
            "horizon_days": days,
            "currency": "SGD",
            "predictions": predictions,
            "model_metrics": {
                "mae": 125.50,
                "mse": 25000.25,
                "rmse": 158.11,
                "mape": 0.85,
                "r2_score": 0.92
            },
            "confidence_level": 0.85
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate forecast")


@router.get("/{asset_type}/{asset_id}/models")
async def list_available_models(
    asset_type: str,
    asset_id: str,
    container=Depends(get_container)
):
    """List available forecasting models for an asset."""
    try:
        models = [
            {
                "name": "linear_regression",
                "display_name": "Linear Regression",
                "description": "Baseline linear trend analysis",
                "accuracy": 0.78,
                "training_date": "2024-06-10",
                "suitable_for": ["short_term", "trend_analysis"]
            },
            {
                "name": "xgboost",
                "display_name": "XGBoost",
                "description": "Gradient boosting for high accuracy",
                "accuracy": 0.92,
                "training_date": "2024-06-10",
                "suitable_for": ["short_term", "long_term", "complex_patterns"]
            },
            {
                "name": "arima",
                "display_name": "ARIMA",
                "description": "Classical time series forecasting",
                "accuracy": 0.85,
                "training_date": "2024-06-10",
                "suitable_for": ["time_series", "seasonal_patterns"]
            },
            {
                "name": "sarima",
                "display_name": "SARIMA",
                "description": "Seasonal time series with trend decomposition",
                "accuracy": 0.88,
                "training_date": "2024-06-10",
                "suitable_for": ["seasonal_data", "long_term_trends"]
            },
            {
                "name": "random_forest",
                "display_name": "Random Forest",
                "description": "Ensemble method for complex patterns",
                "accuracy": 0.89,
                "training_date": "2024-06-10",
                "suitable_for": ["ensemble", "feature_rich"]
            }
        ]
        
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "available_models": models,
            "recommended_model": "xgboost",
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/{asset_type}/{asset_id}/retrain")
async def retrain_models(
    asset_type: str,
    asset_id: str,
    models: Optional[List[str]] = None,
    container=Depends(get_container)
):
    """Trigger model retraining for an asset."""
    try:
        # This would trigger actual model retraining
        available_models = ["linear_regression", "xgboost", "arima", "sarima", "random_forest"]
        
        if not models:
            models = available_models
        
        invalid_models = [m for m in models if m not in available_models]
        if invalid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid models: {invalid_models}. Available: {available_models}"
            )
        
        # Mock response
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "retraining_triggered": True,
            "models": models,
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=30)).isoformat(),
            "job_id": f"retrain_{asset_id}_{int(datetime.utcnow().timestamp())}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")