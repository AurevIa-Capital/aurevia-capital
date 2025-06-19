"""
Modern watch price scraper with enhanced data collection capabilities.

This module provides a robust scraping system for collecting watch price data
from various sources with proper error handling and rate limiting.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.collectors.watch.base_scraper import BaseScraper
from src.utils.selenium_utils import safe_quit_driver

logger = logging.getLogger(__name__)


@dataclass
class WatchTarget:
    """Data class for watch scraping targets."""

    brand: str
    model_name: str
    url: str
    watch_id: str


class CloudflareBypassScraper(BaseScraper):
    """Enhanced scraper with Cloudflare bypass capabilities."""

    def __init__(self, max_workers: int = 3, delay_range: Tuple[int, int] = (5, 15)):
        super().__init__(delay_range)
        self.max_workers = max_workers
        self.session = requests.Session()
        self.setup_session()

    def setup_session(self):
        """Setup requests session with realistic headers."""
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "max-age=0",
            }
        )

    def process_target(self, target: Dict, **kwargs) -> bool:
        """Process a single watch scraping target."""
        watch = WatchTarget(**target)
        output_dir = kwargs.get('output_dir', 'data/watches')
        return self.scrape_single_watch(watch, output_dir)

    def extract_price_data(self, driver: webdriver.Chrome) -> Optional[pd.DataFrame]:
        """Extract price data using multiple fallback methods."""

        # Method 1: Chart.js extraction (original method)
        chart_script = """
        function extractFromCharts() {
            if (!window.Chart || !window.Chart.instances) {
                return null;
            }
            
            const chartInstances = Object.values(window.Chart.instances);
            
            for (let chart of chartInstances) {
                if (!chart.data || !chart.data.datasets) continue;
                
                for (let dataset of chart.data.datasets) {
                    if (!dataset.data || dataset.data.length === 0) continue;
                    
                    const samplePoint = dataset.data[0];
                    if (samplePoint && 
                        typeof samplePoint === 'object' && 
                        samplePoint.y !== undefined && 
                        samplePoint.x instanceof Date) {
                        
                        return dataset.data.map(point => ({
                            date: point.x.toISOString().split('T')[0],
                            price: point.y
                        }));
                    }
                }
            }
            return null;
        }
        
        return JSON.stringify(extractFromCharts());
        """

        try:
            result = driver.execute_script(chart_script)
            if result and result != "null":
                data = json.loads(result)
                if data:
                    df = pd.DataFrame(data)
                    df.rename(columns={"price": "price(SGD)"}, inplace=True)
                    return df
        except Exception as e:
            print(f"Chart.js extraction failed: {e}")

        # Method 2: DOM scraping fallback
        try:
            price_elements = driver.find_elements(
                By.CSS_SELECTOR, "[data-price], .price-value, .chart-data"
            )
            if price_elements:
                # Extract data from DOM elements
                # This would need to be customized based on WatchCharts' actual DOM structure
                print("Found price elements in DOM, but extraction not implemented")
        except Exception as e:
            print(f"DOM extraction failed: {e}")

        # Method 3: Network monitoring (would require additional setup)
        # Could intercept XHR requests for chart data

        return None

    def load_existing_data(self, output_file: str) -> Optional[pd.DataFrame]:
        """Load existing CSV data and return DataFrame with latest date info."""
        try:
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                if len(df) > 0 and "date" in df.columns:
                    # Convert date column to datetime for proper comparison
                    df["date"] = pd.to_datetime(df["date"])
                    return df
        except Exception as e:
            logger.warning(f"Error loading existing data from {output_file}: {e}")
        return None

    def merge_with_existing_data(
        self, new_df: pd.DataFrame, existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge new data with existing data, keeping only newer data points."""
        if existing_df is None or len(existing_df) == 0:
            return new_df

        # Convert date columns to datetime
        new_df["date"] = pd.to_datetime(new_df["date"])
        existing_df["date"] = pd.to_datetime(existing_df["date"])

        # Get the latest date from existing data
        latest_existing_date = existing_df["date"].max()

        # Filter new data to only include dates after the latest existing date
        newer_data = new_df[new_df["date"] > latest_existing_date]

        if len(newer_data) > 0:
            # Combine existing data with newer data
            combined_df = pd.concat([existing_df, newer_data], ignore_index=True)
            # Sort by date and remove duplicates
            combined_df = combined_df.sort_values("date").drop_duplicates(
                subset=["date"], keep="last"
            )
            logger.info(
                f"Added {len(newer_data)} new data points after {latest_existing_date.date()}"
            )
            return combined_df
        else:
            logger.info(f"No new data points found after {latest_existing_date.date()}")
            return existing_df

    def scrape_single_watch(
        self, watch: WatchTarget, output_dir: str = "data/watches"
    ) -> bool:
        """Scrape a single watch with full error handling and incremental updates."""
        # Create filename in format {brand}-{model}-{id}.csv to ensure uniqueness
        brand_safe = self.make_filename_safe(watch.brand)
        
        # Extract clean model name without the watch ID prefix
        clean_model = watch.model_name
        if " - " in watch.model_name and watch.model_name.split(" - ")[0].isdigit():
            clean_model = watch.model_name.split(" - ", 1)[1]
        
        model_safe = self.make_filename_safe(clean_model)
        filename = f"{brand_safe}-{model_safe}-{watch.watch_id}.csv"
        output_file = os.path.join(output_dir, filename)

        # Load existing data to check for latest date
        existing_data = self.load_existing_data(output_file)
        if existing_data is not None:
            latest_date = existing_data["date"].max()
            logger.info(
                f"🔍 {watch.brand} {watch.model_name} - Checking for data after {latest_date.date()}"
            )
        else:
            logger.info(f"🔄 {watch.brand} {watch.model_name} - Starting fresh scrape")

        driver = None
        try:
            # Create driver and navigate with retries
            driver = self.create_browser_session()
            
            # Define chart elements to wait for
            chart_elements = [
                "canvas",
                "[data-chart]", 
                ".chart"
            ]
            
            if not self.safe_navigate_with_retries(driver, watch.url, wait_for_elements=chart_elements):
                logger.error(f"Failed to navigate to {watch.brand} {watch.model_name}")
                return False

            # Extract data
            new_df = self.extract_price_data(driver)

            if new_df is not None and len(new_df) > 0:
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)

                # Merge with existing data (if any)
                final_df = self.merge_with_existing_data(new_df, existing_data)

                # Convert dates back to string format for CSV storage
                # Handle both datetime and string date columns to avoid "dt accessor" errors
                if pd.api.types.is_datetime64_any_dtype(final_df["date"]):
                    # Date column is already datetime - use .dt accessor directly
                    final_df["date"] = final_df["date"].dt.strftime("%Y-%m-%d")
                elif final_df["date"].dtype == "object":
                    # Date column is string format - convert via datetime to ensure consistent YYYY-MM-DD format
                    final_df["date"] = pd.to_datetime(final_df["date"]).dt.strftime(
                        "%Y-%m-%d"
                    )

                # Save merged data
                final_df.to_csv(output_file, index=False)

                if existing_data is not None:
                    new_points = len(final_df) - len(existing_data)
                    if new_points > 0:
                        logger.info(
                            f"✅ {watch.brand} {watch.model_name} - Updated with {new_points} new data points"
                        )
                    else:
                        logger.info(
                            f"✅ {watch.brand} {watch.model_name} - No new data points found"
                        )
                else:
                    logger.info(
                        f"✅ {watch.brand} {watch.model_name} - Saved {len(final_df)} data points (fresh)"
                    )

                return True
            else:
                logger.warning(
                    f"❌ {watch.brand} {watch.model_name} - No data extracted"
                )
                return False

        except Exception as e:
            logger.error(f"{watch.brand} {watch.model_name} - Error: {str(e)}")

            # Save error screenshot
            try:
                if driver:
                    # Create error directory if it doesn't exist
                    error_dir = os.path.join(output_dir, "error")
                    os.makedirs(error_dir, exist_ok=True)

                    # Save screenshot in error subdirectory
                    screenshot_filename = f"{watch.watch_id}_error.png"
                    screenshot_path = os.path.join(error_dir, screenshot_filename)
                    driver.save_screenshot(screenshot_path)
                    logger.info(f"Error screenshot saved: {screenshot_path}")
            except Exception as screenshot_error:
                logger.warning(f"Failed to save error screenshot: {screenshot_error}")

            return False

        finally:
            safe_quit_driver(driver)

            # Random delay between requests
            self.random_delay()

    def scrape_watches_parallel(
        self, watches: List[WatchTarget], output_dir: str = "data/watches"
    ) -> Dict[str, bool]:
        """Scrape multiple watches in parallel with rate limiting."""
        results = {}
        os.makedirs(output_dir, exist_ok=True)

        # Convert WatchTarget objects to dictionaries for base class compatibility
        targets = [
            {
                'brand': watch.brand,
                'model_name': watch.model_name,
                'url': watch.url,
                'watch_id': watch.watch_id
            }
            for watch in watches
        ]

        # Use base class method for processing with brand-based delays
        return self.process_multiple_targets(targets, brand_delay=60, output_dir=output_dir)


def discover_watch_urls() -> List[WatchTarget]:
    """Discover watch URLs for 10 brands with 10 watches each."""

    # Pre-defined targets (you would need to expand this list)
    # These would ideally be discovered dynamically from WatchCharts
    watch_targets = [
        # Rolex (10 watches)
        WatchTarget(
            "Rolex",
            "Submariner 124060",
            "https://watchcharts.com/watch_model/21813-rolex-submariner-124060/overview",
            "21813-rolex-submariner-124060",
        ),
        WatchTarget(
            "Rolex",
            "GMT-Master II 126710BLNR",
            "https://watchcharts.com/watch_model/1234-rolex-gmt-master-ii-126710blnr/overview",
            "1234-rolex-gmt-master-ii-126710blnr",
        ),
        # Add 8 more Rolex watches...
        # Omega (10 watches)
        WatchTarget(
            "Omega",
            "Speedmaster Professional",
            "https://watchcharts.com/watch_model/30921-omega-speedmaster-professional-moonwatch-310-30-42-50-01-002/overview",
            "30921-omega-speedmaster-professional-moonwatch-310-30-42-50-01-002",
        ),
        # Add 9 more Omega watches...
        # Tudor (10 watches)
        WatchTarget(
            "Tudor",
            "Black Bay 58",
            "https://watchcharts.com/watch_model/326-tudor-black-bay-58-79030n/overview",
            "326-tudor-black-bay-58-79030n",
        ),
        # Add 9 more Tudor watches...
        # Add 7 more brands with 10 watches each...
    ]

    return watch_targets


def main():
    """Main function to scrape 100 watches."""
    print("🚀 Starting enhanced watch scraper for 100 watches")

    # Discover targets
    watches = discover_watch_urls()
    print(f"📋 Found {len(watches)} watch targets")

    # Create scraper with conservative settings
    scraper = CloudflareBypassScraper(
        max_workers=2,  # Conservative to avoid rate limiting
        delay_range=(10, 20),  # Longer delays for Cloudflare
    )

    # Group by brand for organized scraping
    brands = {}
    for watch in watches:
        if watch.brand not in brands:
            brands[watch.brand] = []
        brands[watch.brand].append(watch)

    # Scrape brand by brand
    all_results = {}
    for brand, brand_watches in brands.items():
        print(f"\n📦 Processing {brand} ({len(brand_watches)} watches)")

        brand_results = scraper.scrape_watches_parallel(brand_watches)
        all_results.update(brand_results)

        # Longer pause between brands
        if brand != list(brands.keys())[-1]:  # Not the last brand
            print("⏸️  Pausing 60 seconds before next brand...")
            time.sleep(60)

    # Summary
    print("\n📊 SCRAPING SUMMARY")
    print("=" * 50)

    successful = sum(1 for success in all_results.values() if success)
    total = len(all_results)

    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    print(f"📈 Success rate: {successful / total * 100:.1f}%")

    # Group by brand for detailed report
    for brand in brands.keys():
        brand_results = {k: v for k, v in all_results.items() if k.startswith(brand)}
        brand_success = sum(1 for success in brand_results.values() if success)
        print(f"  {brand}: {brand_success}/{len(brand_results)}")


if __name__ == "__main__":
    main()
