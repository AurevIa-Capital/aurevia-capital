"""Data collection management endpoints."""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from app.dependencies import get_container

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
async def list_collectors(container=Depends(get_container)):
    """List all available data collectors."""
    try:
        collectors = [
            {
                "name": "watchcharts",
                "asset_type": "watch",
                "status": "active",
                "last_run": datetime.utcnow().isoformat(),
                "success_rate": 0.95,
                "rate_limit": 10,
                "sources": ["watchcharts.com"]
            },
            {
                "name": "chrono24",
                "asset_type": "watch", 
                "status": "inactive",
                "last_run": None,
                "success_rate": 0.88,
                "rate_limit": 5,
                "sources": ["chrono24.com"]
            },
            {
                "name": "kitco",
                "asset_type": "gold",
                "status": "planned",
                "last_run": None,
                "success_rate": None,
                "rate_limit": 60,
                "sources": ["kitco.com"]
            }
        ]
        
        return {
            "collectors": collectors,
            "total": len(collectors),
            "active": len([c for c in collectors if c["status"] == "active"]),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing collectors: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collectors")


@router.get("/{collector_name}")
async def get_collector_status(
    collector_name: str,
    container=Depends(get_container)
):
    """Get detailed status of a specific collector."""
    try:
        # Mock collector status
        if collector_name == "watchcharts":
            return {
                "name": collector_name,
                "asset_type": "watch",
                "status": "active",
                "configuration": {
                    "rate_limit": 10,
                    "timeout": 30,
                    "retry_attempts": 3,
                    "user_agent": "ForecastingBot/1.0"
                },
                "statistics": {
                    "total_requests": 1250,
                    "successful_requests": 1188,
                    "failed_requests": 62,
                    "success_rate": 0.95,
                    "avg_response_time": 2.5
                },
                "last_run": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "duration": 45.2,
                    "assets_collected": 12,
                    "status": "completed"
                },
                "next_run": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "health": "healthy"
            }
        
        raise HTTPException(status_code=404, detail=f"Collector '{collector_name}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting collector status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collector status")


@router.post("/{asset_type}/collect")
async def trigger_collection(
    asset_type: str,
    asset_ids: Optional[List[str]] = None,
    force: bool = Query(default=False),
    container=Depends(get_container)
):
    """Trigger data collection for asset type."""
    try:
        supported_types = ["watch", "gold", "crypto"]
        
        if asset_type not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported asset type: {asset_type}. Supported: {supported_types}"
            )
        
        # Mock collection trigger
        return {
            "asset_type": asset_type,
            "collection_triggered": True,
            "asset_ids": asset_ids or "all",
            "force_refresh": force,
            "estimated_duration": "15-30 minutes",
            "job_id": f"collect_{asset_type}_{int(datetime.utcnow().timestamp())}",
            "status_endpoint": f"/api/v1/collectors/jobs/{asset_type}_{int(datetime.utcnow().timestamp())}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering collection: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger collection")


@router.get("/jobs/{job_id}")
async def get_collection_job_status(
    job_id: str,
    container=Depends(get_container)
):
    """Get status of a collection job."""
    try:
        # Mock job status
        return {
            "job_id": job_id,
            "status": "running",
            "progress": {
                "total_assets": 100,
                "completed": 45,
                "failed": 2,
                "percentage": 45.0
            },
            "started_at": (datetime.utcnow() - timedelta(minutes=10)).isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "current_asset": "rolex-submariner-124060",
            "logs": [
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": "info",
                    "message": "Processing asset: rolex-submariner-124060"
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job status")