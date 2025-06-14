#!/usr/bin/env python3
"""
Entry point for the streamlined dashboard.

This script launches the multi-asset dashboard with plugin architecture.
"""

import os
import sys
from pathlib import Path

# Add the forecasting platform to the Python path
platform_dir = Path(__file__).parent.parent
sys.path.insert(0, str(platform_dir))

# Change to dashboard directory for proper asset loading
dashboard_dir = platform_dir / "dashboard"
os.chdir(dashboard_dir)

def main():
    """Main entry point for the dashboard."""
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py"])

if __name__ == "__main__":
    main()