"""
Watch discovery script to scrape watchcharts.com brand pages for watch URLs.

This script discovers watch URLs from brand pages and creates a JSON file
with watch targets for scraping.
"""

import json
import logging
from typing import Dict, List

from bs4 import BeautifulSoup

from src.collectors.watch.base_scraper import BaseScraper
from src.utils.selenium_utils import safe_quit_driver

logger = logging.getLogger(__name__)


class WatchDiscovery(BaseScraper):
    """Discover watch URLs from WatchCharts brand pages."""
    
    def __init__(self, delay_range: tuple = (3, 8)):
        """Initialize the discovery engine."""
        super().__init__(delay_range)
        
        # Brand URLs to scrape
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
    
    def process_target(self, target: Dict, **kwargs) -> bool:
        """Process a single brand discovery target."""
        brand = target['brand']
        brand_url = target['url']
        target_count = kwargs.get('target_count', 10)
        
        return len(self.discover_watches_from_brand_page(brand, brand_url, target_count)) > 0
    
    def discover_watches_from_brand_page(self, brand: str, brand_url: str, target_count: int = 10) -> List[Dict]:
        """Discover watches from a brand page."""
        logger.info(f"🔍 Discovering watches for {brand}")
        
        driver = None
        watches = []
        
        try:
            # Create driver and navigate with retries
            driver = self.create_browser_session(headless=True)
            
            # Wait for watch listings to load
            wait_elements = [
                "a[href*='/watch_model/']",  # Direct watch model links
                "a[href*='/watch/']",        # Alternative watch links
                ".watch-card a",             # Watch card links
                ".watch-item a",             # Watch item links
            ]
            
            if not self.safe_navigate_with_retries(driver, brand_url, wait_for_elements=wait_elements):
                logger.error(f"Failed to navigate to {brand} page")
                return watches
            
            # Parse page with BeautifulSoup for efficiency
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")
            
            # Find watch model links
            watch_links = soup.find_all("a", href=True)
            processed_urls = set()
            
            for link in watch_links:
                if len(watches) >= target_count:
                    break
                
                href = link.get("href", "")
                if "/watch_model/" not in href:
                    continue
                
                # Build full URL and avoid duplicates
                full_url = self.ensure_absolute_url(href)
                if full_url in processed_urls:
                    continue
                processed_urls.add(full_url)
                
                # Extract and clean model name
                title = link.get_text(strip=True)
                if not title and link.parent:
                    title = link.parent.get_text(strip=True)[:100]
                
                model_name = self.clean_model_name(title, brand)
                if len(model_name) < 3:
                    continue
                
                # Extract watch ID and combine with model name
                watch_id = self.extract_watch_id_from_url(href)
                if watch_id and watch_id != "unknown":
                    model_name = f"{watch_id} - {model_name}"
                
                watch_data = {
                    "brand": brand,
                    "model_name": model_name,
                    "url": full_url,
                    "source": "generated"
                }
                
                watches.append(watch_data)
                logger.info(f"✅ Found: {brand} - {model_name}")
            
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