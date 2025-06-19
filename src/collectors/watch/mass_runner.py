#!/usr/bin/env python3
"""
Mass Watch Scraper Runner - Streamlined interface for batch watch price scraping.

This module provides a clean interface for running the mass watch scraping pipeline
with configurable options and comprehensive progress tracking.
"""

import logging
from pathlib import Path

from .mass_scraper import MassWatchScraper

logger = logging.getLogger(__name__)


class WatchScrapingRunner:
    """Streamlined runner for mass watch scraping operations."""

    def __init__(
        self,
        targets_file: str = "data/scrape/url/watch_targets_100.json",
        output_dir: str = "data/scrape/prices",
        progress_file: str = "data/scrape/scraping_progress.json",
    ):
        """Initialize with configurable file paths."""
        self.targets_file = Path(targets_file)
        self.output_dir = Path(output_dir)
        self.progress_file = Path(progress_file)

        # Validate inputs
        if not self.targets_file.exists():
            raise FileNotFoundError(f"Targets file not found: {self.targets_file}")

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def display_info(self) -> None:
        """Display scraping configuration and features."""
        print("🏭 LUXURY WATCH MASS SCRAPER")
        print("=" * 50)
        print(f"📂 Input:  {self.targets_file}")
        print(f"📁 Output: {self.output_dir}")
        print(f"📊 Progress: {self.progress_file}")
        print()
        print("Features:")
        print("✅ Cloudflare bypass with stealth mode")
        print("✅ Progress tracking & resume capability")
        print("✅ Error handling & automatic retries")
        print("✅ Rate limiting & brand-based delays")
        print("✅ Incremental data collection")
        print("✅ Unique filename format: {Brand}-{Model}-{ID}.csv")
        print()

    def confirm_execution(self) -> bool:
        """Get user confirmation before starting."""
        try:
            confirm = input("Continue with scraping? (y/n): ").lower().strip()
            return confirm == "y"
        except KeyboardInterrupt:
            print("\n⚠️  Operation cancelled by user")
            return False

    def run_scraping(self, interactive: bool = True) -> bool:
        """
        Execute the mass scraping operation.

        Args:
            interactive: Whether to show info and ask for confirmation

        Returns:
            bool: True if scraping completed successfully
        """
        if interactive:
            self.display_info()
            if not self.confirm_execution():
                print("Operation cancelled.")
                return False

        try:
            # Create and configure scraper
            scraper = MassWatchScraper(
                output_dir=str(self.output_dir),
                targets_file=str(self.targets_file),
                progress_file=str(self.progress_file),
            )

            # Execute scraping
            logger.info("🚀 Starting mass watch scraping")
            scraper.run()

            logger.info("✅ Mass scraping completed")
            return True

        except KeyboardInterrupt:
            print("\n⚠️  Scraping interrupted by user")
            print("Progress has been saved. Run again to resume.")
            return False
        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}")
            raise


def main():
    """CLI entry point for mass watch scraping."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        runner = WatchScrapingRunner()
        runner.run_scraping(interactive=True)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print(
            "💡 Run URL generation first: python -m src.collectors.watch.url_generator"
        )
    except Exception as e:
        logger.error(f"❌ Runner failed: {e}")
        raise


if __name__ == "__main__":
    main()
