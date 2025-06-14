#!/usr/bin/env python3
"""
Convenient runner script for mass watch scraping
"""

import os
import sys
from pathlib import Path

# Add src/dev to Python path
src_dev_path = Path(__file__).parent / "src" / "dev"
sys.path.insert(0, str(src_dev_path))

try:
    from mass_scraper import MassWatchScraper
    
    def main():
        print("🏭 LUXURY WATCH MASS SCRAPER")
        print("=" * 40)
        print("This will scrape 100 luxury watches from WatchCharts")
        print("10 brands × 10 watches each")
        print()
        print("Features:")
        print("✅ Cloudflare bypass")
        print("✅ Parallel processing")
        print("✅ Progress tracking & resume")
        print("✅ Error handling & retries")
        print("✅ Rate limiting")
        print()
        
        confirm = input("Continue? (y/n): ").lower().strip()
        if confirm != 'y':
            print("Cancelled.")
            return
        
        # Run the scraper
        scraper = MassWatchScraper()
        scraper.run()
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print("❌ Import error:", e)
    print()
    print("Please install required dependencies:")
    print("pip install -r requirements_scraping.txt")
    print()
    print("Make sure you're in the project root directory.")
    sys.exit(1)