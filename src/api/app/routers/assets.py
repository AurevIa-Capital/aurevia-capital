"""Asset management endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.dependencies import get_container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{asset_type}")
async def list_assets(
    asset_type: str,
    limit: int = Query(default=100, le=1000),
    offset: int = Query(default=0, ge=0),
    container=Depends(get_container)
):
    """List all available assets of given type."""
    try:
        # This would be implemented with actual collectors
        supported_types = ["watch", "gold", "crypto"]
        
        if asset_type not in supported_types:
            raise HTTPException(
                status_code=404, 
                detail=f"Unknown asset type: {asset_type}. Supported: {supported_types}"
            )
        
        # Mock response for now
        assets = []
        if asset_type == "watch":
            assets = [
                {
                    "asset_id": "21813-rolex-submariner-124060",
                    "name": "Rolex Submariner 124060",
                    "brand": "Rolex",
                    "model": "Submariner",
                    "reference": "124060",
                    "last_price": 15000.0,
                    "currency": "SGD",
                    "last_updated": datetime.utcnow().isoformat()
                },
                {
                    "asset_id": "326-tudor-black-bay-58-79030n",
                    "name": "Tudor Black Bay 58",
                    "brand": "Tudor", 
                    "model": "Black Bay 58",
                    "reference": "79030N",
                    "last_price": 4500.0,
                    "currency": "SGD",
                    "last_updated": datetime.utcnow().isoformat()
                }
            ]
        
        return {
            "asset_type": asset_type,
            "total": len(assets),
            "limit": limit,
            "offset": offset,
            "assets": assets[offset:offset+limit]
        }
        
    except Exception as e:
        logger.error(f"Error listing assets: {e}")
        raise HTTPException(status_code=500, detail="Failed to list assets")


@router.get("/{asset_type}/{asset_id}")
async def get_asset(
    asset_type: str,
    asset_id: str,
    container=Depends(get_container)
):
    """Get detailed information about a specific asset."""
    try:
        # Mock response for now
        if asset_type == "watch" and asset_id == "21813-rolex-submariner-124060":
            return {
                "asset_id": asset_id,
                "asset_type": asset_type,
                "name": "Rolex Submariner 124060",
                "brand": "Rolex",
                "model": "Submariner",
                "reference": "124060",
                "description": "Rolex Submariner Date 124060",
                "specifications": {
                    "case_size": "41mm",
                    "movement": "Automatic",
                    "water_resistance": "300m"
                },
                "current_price": 15000.0,
                "currency": "SGD",
                "price_range_30d": {
                    "min": 14800.0,
                    "max": 15200.0,
                    "avg": 15000.0
                },
                "last_updated": datetime.utcnow().isoformat(),
                "data_sources": ["watchcharts"],
                "available_forecasts": ["30_day", "90_day"]
            }
        
        raise HTTPException(status_code=404, detail="Asset not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting asset: {e}")
        raise HTTPException(status_code=500, detail="Failed to get asset")


@router.get("/{asset_type}/{asset_id}/history")
async def get_asset_history(
    asset_type: str,
    asset_id: str,
    days: int = Query(default=30, le=365, ge=1),
    container=Depends(get_container)
):
    """Get historical price data for an asset."""
    try:
        # This would fetch actual historical data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Mock historical data
        history = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            price = 15000 + (i * 10) + (i % 7 * 50)  # Mock price variation
            history.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": price,
                "currency": "SGD"
            })
        
        return {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "period": {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "days": days
            },
            "data_points": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error getting asset history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get asset history")