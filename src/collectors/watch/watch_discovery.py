"""
Watch discovery script to scrape watchcharts.com brand pages for watch URLs.

This script discovers watch URLs from brand pages and creates a JSON file
with watch targets for scraping.
"""

import json
import logging
import random
import re
import time
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.utils.selenium_utils import (
    SeleniumDriverFactory,
    check_cloudflare_challenge,
    check_website_loaded_successfully,
    safe_quit_driver,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WatchDiscovery:
    """Discover watch URLs from WatchCharts brand pages."""
    
    def __init__(self, delay_range: tuple = (3, 8)):
        """Initialize the discovery engine."""
        self.delay_range = delay_range
        self.base_url = "https://watchcharts.com"
        
        # Brand URLs to scrape (exactly as provided by user)
        self.brand_urls = {
            # Top Tier
            "Patek Philippe": "https://watchcharts.com/watches/brand/patek+philippe",
            "Rolex": "https://watchcharts.com/watches/brand/rolex", 
            "Audemars Piguet": "https://watchcharts.com/watches/brand/audemars+piguet",
            "Vacheron Constantin": "https://watchcharts.com/watches/brand/vacheron+constantin",
            
            # Mid Tier
            "Omega": "https://watchcharts.com/watches/brand/omega",
            "Tudor": "https://watchcharts.com/watches/brand/tudor",
            "Hublot": "https://watchcharts.com/watches/brand/hublot",
            
            # Designer / Entry-Level
            "Tissot": "https://watchcharts.com/watches/brand/tissot",
            "Longines": "https://watchcharts.com/watches/brand/longines",
            "Seiko": "https://watchcharts.com/watches/brand/seiko"
        }
    
    def random_delay(self):
        """Add random delay between requests."""
        delay = random.uniform(*self.delay_range)
        logger.info(f"Waiting {delay:.1f} seconds...")
        time.sleep(delay)
    
    def extract_model_name_from_title(self, title: str, brand: str) -> str:
        """Extract clean model name from watch title."""
        # Remove brand name from title
        model_name = title.replace(brand, "").strip()
        
        # Remove common prefixes/suffixes
        prefixes_to_remove = ["Watch", "Collection", "-", "–", "•"]
        for prefix in prefixes_to_remove:
            if model_name.startswith(prefix):
                model_name = model_name[len(prefix):].strip()
        
        # Clean up extra whitespace
        model_name = re.sub(r'\s+', ' ', model_name).strip()
        
        return model_name if model_name else title
    
    def extract_watch_id_from_url(self, url: str) -> str:
        """Extract watch ID from WatchCharts URL."""
        # Expected format: https://watchcharts.com/watch_model/{ID}-{slug}/overview
        match = re.search(r'/watch_model/(\d+)-([^/]+)', url)
        if match:
            watch_id = match.group(1)
            slug = match.group(2)
            return f"{watch_id}-{slug}"
        
        # Fallback - use the entire path segment
        path_parts = urlparse(url).path.split('/')
        for part in path_parts:
            if part and '-' in part and any(c.isdigit() for c in part):
                return part
        
        return "unknown-watch"
    
    def discover_watches_from_brand_page(self, brand: str, brand_url: str, target_count: int = 10) -> List[Dict]:
        """Discover watches from a brand page."""
        logger.info(f"🔍 Discovering watches for {brand}")
        logger.info(f"URL: {brand_url}")
        
        driver = None
        watches = []
        
        try:
            # Create driver
            driver = SeleniumDriverFactory.create_discovery_driver(headless=True)
            
            # Navigate to brand page
            driver.get(brand_url)
            
            # Wait for page load
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(3)
            
            # Check if page loaded successfully
            if not check_website_loaded_successfully(driver):
                if check_cloudflare_challenge(driver):
                    logger.warning("Cloudflare challenge detected, waiting...")
                    time.sleep(random.uniform(20, 40))
                    
                    if check_cloudflare_challenge(driver):
                        raise Exception("Cloudflare challenge not resolved")
                else:
                    logger.warning("Page may not have loaded properly")
                    # Continue anyway, might still be able to extract data
            
            # Wait for watch listings to load
            try:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/watch_model/']"))
                )
            except:
                logger.warning("No watch model links found with standard selector")
            
            # Find all watch links - try multiple selectors
            selectors_to_try = [
                "a[href*='/watch_model/']",  # Direct watch model links
                "a[href*='/watch/']",        # Alternative watch links
                ".watch-card a",             # Watch card links
                ".watch-item a",             # Watch item links
                ".product-link",             # Product links
                "a[title]",                  # Links with titles (might be watches)
            ]
            
            watch_links = []
            for selector in selectors_to_try:
                try:
                    links = driver.find_elements(By.CSS_SELECTOR, selector)
                    if links:
                        logger.info(f"Found {len(links)} potential links with selector: {selector}")
                        watch_links.extend(links)
                        break  # Use first successful selector
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not watch_links:
                logger.error(f"No watch links found for {brand}")
                return watches
            
            logger.info(f"Processing {len(watch_links)} potential watch links")
            
            # Extract watch information
            processed_urls = set()  # Avoid duplicates
            
            for link in watch_links:
                if len(watches) >= target_count:
                    break
                
                try:
                    url = link.get_attribute('href')
                    title = link.get_attribute('title') or link.text
                    
                    # Skip if no URL or not a watch model URL
                    if not url or 'watch_model' not in url:
                        continue
                    
                    # Skip duplicates
                    if url in processed_urls:
                        continue
                    processed_urls.add(url)
                    
                    # Extract model name
                    model_name = self.extract_model_name_from_title(title, brand)
                    
                    # Skip if no meaningful model name
                    if not model_name or len(model_name.strip()) < 3:
                        continue
                    
                    # Extract watch ID
                    watch_id = self.extract_watch_id_from_url(url)
                    
                    # Ensure URL is absolute
                    if url.startswith('/'):
                        url = urljoin(self.base_url, url)
                    
                    # Add /overview if not present
                    if not url.endswith('/overview'):
                        url = url.rstrip('/') + '/overview'
                    
                    watch_data = {
                        "brand": brand,
                        "model_name": model_name,
                        "url": url,
                        "source": "generated"
                    }
                    
                    watches.append(watch_data)
                    logger.info(f"✅ Found: {brand} - {model_name}")
                    
                except Exception as e:
                    logger.debug(f"Error processing link: {e}")
                    continue
            
            logger.info(f"🎯 Discovered {len(watches)} watches for {brand}")
            return watches
            
        except Exception as e:
            logger.error(f"Error discovering watches for {brand}: {str(e)}")
            return watches
            
        finally:
            safe_quit_driver(driver)
            self.random_delay()
    
    def discover_all_watches(self) -> List[Dict]:
        """Discover watches from all brand pages."""
        logger.info("🚀 Starting watch discovery for all brands")
        
        all_watches = []
        
        for brand, brand_url in self.brand_urls.items():
            try:
                brand_watches = self.discover_watches_from_brand_page(brand, brand_url, 10)
                all_watches.extend(brand_watches)
                
                logger.info(f"📊 {brand}: {len(brand_watches)}/10 watches discovered")
                
                # Longer delay between brands
                if brand != list(self.brand_urls.keys())[-1]:  # Not the last brand
                    logger.info("⏸️  Pausing 30 seconds before next brand...")
                    time.sleep(30)
                    
            except Exception as e:
                logger.error(f"Failed to process {brand}: {e}")
                continue
        
        logger.info(f"🎯 Total watches discovered: {len(all_watches)}")
        
        # Summary by brand
        brand_counts = {}
        for watch in all_watches:
            brand = watch['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        logger.info("📊 Discovery Summary:")
        for brand, count in brand_counts.items():
            logger.info(f"  {brand}: {count}/10 watches")
        
        return all_watches
    
    def save_watch_targets(self, watches: List[Dict], filename: str = "watch_targets_100.json"):
        """Save discovered watches to JSON file."""
        logger.info(f"💾 Saving {len(watches)} watches to {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(watches, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Watch targets saved to {filename}")


def main():
    """Main function to discover and save watch targets."""
    discovery = WatchDiscovery()
    
    # Discover all watches
    watches = discovery.discover_all_watches()
    
    if watches:
        # Save to JSON file in project root
        discovery.save_watch_targets(watches, "watch_targets_100.json")
        
        print("\n" + "="*60)
        print("🎯 WATCH DISCOVERY COMPLETE")
        print("="*60)
        print(f"Total watches discovered: {len(watches)}")
        print(f"Target: 100 watches (10 per brand)")
        print(f"Success rate: {len(watches)/100*100:.1f}%")
        
        # Brand breakdown
        brand_counts = {}
        for watch in watches:
            brand = watch['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
        
        print("\nBrand Breakdown:")
        for brand, count in brand_counts.items():
            print(f"  {brand}: {count}/10")
        
        print(f"\nResults saved to: watch_targets_100.json")
    else:
        print("❌ No watches discovered!")


if __name__ == "__main__":
    main()