"""
Asset Forecasting API Gateway

Unified API for all asset types including watches, gold, crypto, etc.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import asyncio
import logging
from datetime import datetime

from app.routers import assets, forecasts, collectors
from app.dependencies import get_container
from app.middleware.logging import LoggingMiddleware

# Configure centralized logging
from src.utils.logging_config import setup_logging
logger = setup_logging("api_server")

# Create FastAPI app
app = FastAPI(
    title="Asset Forecasting API",
    description="Unified API for multi-asset price forecasting",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
app.include_router(forecasts.router, prefix="/api/v1/forecasts", tags=["forecasts"])
app.include_router(collectors.router, prefix="/api/v1/collectors", tags=["collectors"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Asset Forecasting API",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "assets": "/api/v1/assets",
            "forecasts": "/api/v1/forecasts", 
            "collectors": "/api/v1/collectors",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",  # TODO: Add actual health checks
            "collectors": "available"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )