"""
Modern watch price scraper with enhanced data collection capabilities.

This module provides a robust scraping system for collecting watch price data
from various sources with proper error handling and rate limiting.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from selenium import webdriver

from src.collectors.watch.base_scraper import BaseScraper
from src.collectors.selenium_utils import safe_quit_driver

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

    def __init__(self, delay_range: Tuple[int, int] = (5, 15)):
        super().__init__(delay_range)

    def process_target(self, target: Dict, **kwargs) -> bool:
        """Process a single watch scraping target."""
        watch = WatchTarget(**target)
        output_dir = kwargs.get("output_dir", "data/watches")
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

        # No fallback methods currently implemented

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
            chart_elements = ["canvas", "[data-chart]", ".chart"]

            if not self.safe_navigate_with_retries(
                driver, watch.url, wait_for_elements=chart_elements
            ):
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
        _results = {}
        os.makedirs(output_dir, exist_ok=True)

        # Convert WatchTarget objects to dictionaries for base class compatibility
        targets = [
            {
                "brand": watch.brand,
                "model_name": watch.model_name,
                "url": watch.url,
                "watch_id": watch.watch_id,
            }
            for watch in watches
        ]

        # Use base class method for processing with brand-based delays
        return self.process_multiple_targets(
            targets, brand_delay=60, output_dir=output_dir
        )
