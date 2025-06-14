#!/usr/bin/env python3
"""
Entry point for the API gateway.

This script launches the FastAPI-based API gateway for multi-asset forecasting.
"""

import os
import sys
from pathlib import Path

# Add the forecasting platform to the Python path
platform_dir = Path(__file__).parent.parent
sys.path.insert(0, str(platform_dir))

# Change to API directory
api_dir = platform_dir / "api"
os.chdir(api_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )