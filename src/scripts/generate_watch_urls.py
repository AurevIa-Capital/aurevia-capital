#!/usr/bin/env python3
"""
Script to scrape watch URLs from WatchCharts brand pages and generate JSON file.
Run this script to generate watch_targets_100.json with 10 watches per brand.
Uses Selenium with Cloudflare bypass capabilities.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

from src.utils.selenium_utils import (
    SeleniumDriverFactory,
    check_website_loaded_successfully,
    safe_quit_driver,
)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Brand configuration
BRANDS = {
    "Top Tier": {
        "Patek Philippe": "https://watchcharts.com/watches/brand/patek+philippe",
        "Rolex": "https://watchcharts.com/watches/brand/rolex",
        "Audemars Piguet": "https://watchcharts.com/watches/brand/audemars+piguet",
        "Vacheron Constantin": "https://watchcharts.com/watches/brand/vacheron+constantin",
    },
    "Mid Tier": {
        "Omega": "https://watchcharts.com/watches/brand/omega",
        "Tudor": "https://watchcharts.com/watches/brand/tudor",
        "Hublot": "https://watchcharts.com/watches/brand/hublot",
    },
    "Designer / Entry-Level": {
        "Tissot": "https://watchcharts.com/watches/brand/tissot",
        "Longines": "https://watchcharts.com/watches/brand/longines",
        "Seiko": "https://watchcharts.com/watches/brand/seiko",
    },
}


def get_watch_urls_from_brand_page(
    driver, brand_name: str, brand_url: str, limit: int = 10
) -> List[Dict]:
    """
    Scrape watch URLs from a brand page using Selenium.

    Args:
        driver: Selenium WebDriver instance
        brand_name: Name of the brand
        brand_url: URL of the brand page
        limit: Number of watches to extract (default 10)

    Returns:
        List of dictionaries with watch data
    """
    logger.info(f"Scraping {brand_name} from {brand_url}")

    try:
        # Navigate to the brand page
        driver.get(brand_url)

        # Wait for page to load and check if successful
        time.sleep(5)

        if not check_website_loaded_successfully(driver):
            logger.error(f"Failed to load {brand_name} page properly")
            return []

        # Get page source and parse with BeautifulSoup
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, "html.parser")

        watches = []

        # Look for watch model links - they contain '/watch_model/' in the href
        watch_links = soup.find_all("a", href=True)

        seen_urls = set()  # To avoid duplicates

        for link in watch_links:
            href = link.get("href", "")

            # Filter for watch model URLs
            if "/watch_model/" in href and len(watches) < limit:
                # Build full URL if relative
                full_url = (
                    href
                    if href.startswith("http")
                    else f"https://watchcharts.com{href}"
                )

                # Skip duplicates
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)

                # Get the watch title/model name from the link text or nearby elements
                model_name = link.get_text(strip=True)

                # If link text is empty or too short, try to find model name from parent elements
                if not model_name or len(model_name) < 3:
                    # Try to find model name in parent elements
                    parent = link.parent
                    if parent:
                        model_name = parent.get_text(strip=True)
                        # Take first part if too long
                        if len(model_name) > 100:
                            model_name = model_name[:100] + "..."

                # Extract watch ID from URL
                watch_id = ""
                if "/watch_model/" in href:
                    # Extract ID from URL like: /watch_model/22557-patek-philippe-aquanaut-5167-stainless-steel-5167a/overview
                    url_parts = href.split("/")
                    for part in url_parts:
                        if part and part.startswith(tuple("0123456789")) and "-" in part:
                            watch_id = part.split("-")[0]  # Get just the numeric ID
                            break

                # Clean up model name
                if model_name:
                    # Remove brand name from model name if present
                    model_name = model_name.replace(brand_name, "").strip()
                    if model_name.startswith("-"):
                        model_name = model_name[1:].strip()
                    
                    # Clean up common unwanted text patterns
                    cleanup_patterns = [
                        r"In Production",
                        r"Retail Price[^A-Za-z]*",
                        r"Market Price[^A-Za-z]*",
                        r"S\$[\d,]+",
                        r"~S\$[\d,]+", 
                        r"\d+mm",
                        r"\d+M ",
                        r"Steel\d+mm\d+M",
                        r"Steel\d+mm",
                        r"White gold\d+mm\d+M",
                        r"Rose gold\d+mm\d+M", 
                        r"Yellow gold\d+mm\d+M",
                        r"Gold/steel\d+mm\d+M",
                        r"Titanium\d+mm\d+M",
                        r"Ceramic\d+mm\d+M"
                    ]
                    
                    for pattern in cleanup_patterns:
                        model_name = re.sub(pattern, "", model_name, flags=re.IGNORECASE)
                    
                    # Clean up specific watch model references
                    model_name = re.sub(r"[A-Z0-9]+/[A-Z0-9.-]+", "", model_name)  # Remove reference numbers like 5167A, 126300, etc.
                    model_name = re.sub(r"[A-Z0-9]{4,}", "", model_name)  # Remove long alphanumeric codes
                    
                    # Remove extra whitespace and clean up
                    model_name = " ".join(model_name.split())
                    
                    # If still empty, extract from URL
                    if not model_name or len(model_name) < 3:
                        # Extract model name from URL
                        url_parts = href.split("/")
                        for part in url_parts:
                            if part and "-" in part and len(part) > 10:
                                model_name = part.replace("-", " ").title()
                                break
                
                # Combine watch ID with model name
                if watch_id and model_name:
                    model_name = f"{watch_id} - {model_name}"
                elif watch_id:
                    model_name = watch_id

                # Skip if we still don't have a good model name
                if not model_name or len(model_name) < 3:
                    continue

                watch_data = {
                    "brand": brand_name,
                    "model_name": model_name,
                    "url": full_url,
                    "source": "generated",
                }

                watches.append(watch_data)
                logger.info(f"Found watch: {model_name}")

        logger.info(f"Extracted {len(watches)} watches for {brand_name}")
        return watches[:limit]  # Ensure we don't exceed the limit

    except Exception as e:
        logger.error(f"Error scraping {brand_name}: {e}")
        return []


def main():
    """Main function to scrape all brands and generate JSON file."""
    all_watches = []

    for tier, brands in BRANDS.items():
        logger.info(f"Processing {tier} brands...")

        for brand_name, brand_url in brands.items():
            driver = None
            try:
                # Create fresh browser for each brand
                logger.info(f"Creating new browser for {brand_name}...")
                driver = SeleniumDriverFactory.create_stealth_driver(
                    headless=False
                )  # Set to True for headless

                watches = get_watch_urls_from_brand_page(
                    driver, brand_name, brand_url, limit=10
                )
                all_watches.extend(watches)

            except Exception as e:
                logger.error(f"Error processing {brand_name}: {e}")

            finally:
                # Always clean up the driver after each brand
                if driver:
                    logger.info(f"Closing browser for {brand_name}...")
                    safe_quit_driver(driver)

            # Wait between brands
            logger.info("Waiting 10 seconds before next brand...")
            time.sleep(10)

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
        for tier, brands in BRANDS.items():
            tier_count = len([w for w in all_watches if w["brand"] in brands.keys()])
            print(f"{tier}: {tier_count} watches")

        # Print first few entries as sample
        print("\nFirst 5 entries:")
        for i, watch in enumerate(all_watches[:5]):
            print(f"{i + 1}. {watch['brand']} - {watch['model_name']}")

    except Exception as e:
        logger.error(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()
