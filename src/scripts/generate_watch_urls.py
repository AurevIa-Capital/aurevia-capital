#!/usr/bin/env python3
"""
Script to scrape watch URLs from WatchCharts brand pages and generate JSON file.
Run this script to generate watch_targets_100.json with 10 watches per brand.
Uses Selenium with Cloudflare bypass capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from src.collectors.watch.watch_discovery import WatchDiscovery

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



def main():
    """Main function to scrape all brands and generate JSON file."""
    # Create discovery scraper instance
    discovery_scraper = WatchDiscovery()
    
    # Use the discovery scraper's built-in functionality
    all_watches = discovery_scraper.discover_all_watches()

    # Save to JSON file in data/scrape/url/ directory
    output_dir = project_root / "data" / "scrape" / "url"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    output_file = str(output_dir / "watch_targets_100.json")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_watches, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully saved {len(all_watches)} watches to {output_file}")

        # Print summary
        print("\nSummary:")
        print(f"Total watches extracted: {len(all_watches)}")
        
        # Count by brand
        brand_counts = {}
        for watch in all_watches:
            brand = watch["brand"]
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        for brand, count in brand_counts.items():
            print(f"{brand}: {count} watches")

        # Print first few entries as sample
        print("\nFirst 5 entries:")
        for i, watch in enumerate(all_watches[:5]):
            print(f"{i + 1}. {watch['brand']} - {watch['model_name']}")

    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()
