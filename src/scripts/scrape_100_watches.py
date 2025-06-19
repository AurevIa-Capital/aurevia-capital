#!/usr/bin/env python3
"""
Convenient runner script for mass watch scraping using JSON input
"""

from src.collectors.watch.mass_scraper import MassWatchScraper


def main():
    print("🏭 LUXURY WATCH MASS SCRAPER")
    print("=" * 40)
    print("This will scrape 100 luxury watches from WatchCharts")
    print("Using JSON input from data/scrape/url/watch_targets_100.json")
    print()
    print("Features:")
    print("✅ Cloudflare bypass")
    print("✅ Parallel processing")
    print("✅ Progress tracking & resume")
    print("✅ Error handling & retries")
    print("✅ Rate limiting")
    print("✅ Custom output directory: data/scrape/prices/")
    print()

    confirm = input("Continue? (y/n): ").lower().strip()
    if confirm != "y":
        print("Cancelled.")
        return

    # Run the scraper with new configuration
    scraper = MassWatchScraper(
        output_dir="data/scrape/prices",
        targets_file="data/scrape/url/watch_targets_100.json",
        progress_file="data/scrape/scraping_progress.json"
    )
    scraper.run()


if __name__ == "__main__":
    main()
