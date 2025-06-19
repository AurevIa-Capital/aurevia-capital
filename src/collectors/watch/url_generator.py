#!/usr/bin/env python3
"""
Watch URL Generator - Discover and save watch URLs from WatchCharts brand pages.

This module provides a streamlined interface for generating watch target URLs
for the scraping pipeline. Uses the WatchDiscovery class for unified functionality.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from .watch_discovery import WatchDiscovery

logger = logging.getLogger(__name__)


class WatchURLGenerator:
    """Streamlined watch URL generator with configurable output."""
    
    def __init__(self, output_dir: str = "data/scrape/url"):
        """Initialize with configurable output directory."""
        self.discovery = WatchDiscovery()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_and_save(self, filename: str = "watch_targets_100.json") -> Dict[str, int]:
        """
        Generate watch URLs and save to JSON file.
        
        Returns:
            Dict with generation statistics
        """
        logger.info("🚀 Starting watch URL generation")
        
        # Generate URLs using discovery scraper
        all_watches = self.discovery.discover_all_watches()
        
        if not all_watches:
            logger.error("No watches discovered!")
            return {"total": 0, "brands": {}}
        
        # Save to JSON file
        output_file = self.output_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_watches, f, indent=2, ensure_ascii=False)
        
        # Generate statistics
        brand_counts = {}
        for watch in all_watches:
            brand = watch["brand"]
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        stats = {
            "total": len(all_watches),
            "brands": brand_counts,
            "output_file": str(output_file)
        }
        
        logger.info(f"✅ Generated {stats['total']} watch URLs")
        logger.info(f"📁 Saved to: {output_file}")
        
        return stats
    
    def print_summary(self, stats: Dict) -> None:
        """Print generation summary."""
        print("\n" + "="*50)
        print("🎯 WATCH URL GENERATION COMPLETE")
        print("="*50)
        print(f"Total watches discovered: {stats['total']}")
        print(f"Target: 100 watches (10 per brand)")
        print(f"Success rate: {stats['total']/100*100:.1f}%")
        
        print("\nBrand Breakdown:")
        for brand, count in stats['brands'].items():
            print(f"  {brand}: {count}/10")
        
        print(f"\nResults saved to: {stats['output_file']}")


def main():
    """CLI entry point for watch URL generation."""
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        generator = WatchURLGenerator()
        stats = generator.generate_and_save()
        generator.print_summary(stats)
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
