import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.selenium_utils import SeleniumDriverFactory, safe_quit_driver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WatchURLDiscovery:
    """Automated discovery of watch URLs from WatchCharts."""

    def __init__(self):
        self.base_url = "https://watchcharts.com"
        self.session = self.setup_session()
        self.target_brands = [
            "Rolex",
            "Omega",
            "Tudor",
            "Patek Philippe",
            "Audemars Piguet",
            "Cartier",
            "Breitling",
            "Tag Heuer",
            "IWC",
            "Jaeger-LeCoultre",
        ]

    def setup_session(self) -> requests.Session:
        """Setup requests session with realistic headers."""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        return session

    def create_discovery_driver(self) -> webdriver.Chrome:
        """Create driver for JavaScript-heavy discovery."""
        return SeleniumDriverFactory.create_discovery_driver(headless=True)

    def discover_brand_watches(self, brand: str, target_count: int = 10) -> List[Dict]:
        """Discover watch URLs for a specific brand."""
        watches = []

        # Method 1: Try brand-specific search/browse pages
        brand_urls = self.get_brand_search_urls(brand)

        for url in brand_urls:
            try:
                brand_watches = self.scrape_brand_page(
                    url, brand, target_count - len(watches)
                )
                watches.extend(brand_watches)

                if len(watches) >= target_count:
                    break

                # Delay between brand page requests
                time.sleep(random.uniform(3, 7))

            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue

        # Method 2: Use selenium for dynamic content if needed
        if len(watches) < target_count:
            driver_watches = self.discover_with_selenium(
                brand, target_count - len(watches)
            )
            watches.extend(driver_watches)

        return watches[:target_count]

    def get_brand_search_urls(self, brand: str) -> List[str]:
        """Generate potential URLs for brand discovery."""
        brand_lower = brand.lower().replace(" ", "-")

        potential_urls = [
            f"{self.base_url}/watches?brand={brand_lower}",
            f"{self.base_url}/brand/{brand_lower}",
            f"{self.base_url}/search?q={brand_lower}",
            f"{self.base_url}/browse/{brand_lower}",
            f"{self.base_url}/brands/{brand_lower}",
        ]

        return potential_urls

    def scrape_brand_page(self, url: str, brand: str, max_watches: int) -> List[Dict]:
        """Scrape a brand page for watch URLs."""
        watches = []

        try:
            response = self.session.get(url, timeout=30)
            if response.status_code != 200:
                return watches

            soup = BeautifulSoup(response.content, "html.parser")

            # Look for watch links using various selectors
            watch_selectors = [
                'a[href*="watch_model"]',
                'a[href*="/watch/"]',
                ".watch-card a",
                ".product-link",
                "[data-watch-id] a",
                ".watch-item a",
            ]

            found_links = set()
            for selector in watch_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get("href")
                    if href and "watch_model" in href:
                        full_url = urljoin(url, href)
                        found_links.add(full_url)

            # Process found links
            for link in list(found_links)[:max_watches]:
                watch_info = self.extract_watch_info(link, brand)
                if watch_info:
                    watches.append(watch_info)

        except Exception as e:
            print(f"Error scraping brand page {url}: {e}")

        return watches

    def discover_with_selenium(self, brand: str, max_watches: int) -> List[Dict]:
        """Use Selenium for JavaScript-heavy discovery."""
        watches = []
        driver = None

        try:
            driver = self.create_discovery_driver()

            # Try search page
            search_url = f"{self.base_url}/search?q={brand.lower()}"
            driver.get(search_url)

            # Wait for results to load
            WebDriverWait(driver, 10).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            # Look for watch links
            watch_links = driver.find_elements(
                By.CSS_SELECTOR, 'a[href*="watch_model"]'
            )

            for link in watch_links[:max_watches]:
                try:
                    href = link.get_attribute("href")
                    if href:
                        watch_info = self.extract_watch_info(href, brand)
                        if watch_info:
                            watches.append(watch_info)
                except:
                    continue

        except Exception as e:
            print(f"Selenium discovery error for {brand}: {e}")

        finally:
            safe_quit_driver(driver)

        return watches

    def extract_watch_info(self, url: str, brand: str) -> Dict:
        """Extract watch information from URL."""
        try:
            # Parse URL to extract watch ID and model name
            # Example: https://watchcharts.com/watch_model/21813-rolex-submariner-124060/overview
            path_parts = urlparse(url).path.split("/")

            for part in path_parts:
                if "-" in part and any(char.isdigit() for char in part):
                    # This looks like a watch ID
                    watch_id = part

                    # Extract model name from ID
                    id_parts = watch_id.split("-")
                    if len(id_parts) > 1:
                        model_name = " ".join(id_parts[1:]).title()

                        return {
                            "brand": brand,
                            "model_name": model_name,
                            "url": url,
                            "watch_id": watch_id,
                        }

            return None

        except Exception as e:
            print(f"Error extracting watch info from {url}: {e}")
            return None

    def get_fallback_watches(self) -> List[Dict]:
        """Fallback list of known watch URLs if discovery fails."""
        fallback_watches = [
            # Rolex
            {
                "brand": "Rolex",
                "model_name": "Submariner 124060",
                "url": "https://watchcharts.com/watch_model/21813-rolex-submariner-124060/overview",
                "watch_id": "21813-rolex-submariner-124060",
            },
            {
                "brand": "Rolex",
                "model_name": "GMT Master II 126710BLNR",
                "url": "https://watchcharts.com/watch_model/12345-rolex-gmt-master-ii-126710blnr/overview",
                "watch_id": "12345-rolex-gmt-master-ii-126710blnr",
            },
            {
                "brand": "Rolex",
                "model_name": "Daytona 116500LN",
                "url": "https://watchcharts.com/watch_model/641-rolex-cosmograph-daytona-116500/overview",
                "watch_id": "641-rolex-cosmograph-daytona-116500",
            },
            {
                "brand": "Rolex",
                "model_name": "Datejust 126334",
                "url": "https://watchcharts.com/watch_model/54321-rolex-datejust-126334/overview",
                "watch_id": "54321-rolex-datejust-126334",
            },
            {
                "brand": "Rolex",
                "model_name": "Explorer 124270",
                "url": "https://watchcharts.com/watch_model/98765-rolex-explorer-124270/overview",
                "watch_id": "98765-rolex-explorer-124270",
            },
            {
                "brand": "Rolex",
                "model_name": "Sea Dweller 126600",
                "url": "https://watchcharts.com/watch_model/11111-rolex-sea-dweller-126600/overview",
                "watch_id": "11111-rolex-sea-dweller-126600",
            },
            {
                "brand": "Rolex",
                "model_name": "Yacht Master 126622",
                "url": "https://watchcharts.com/watch_model/22222-rolex-yacht-master-126622/overview",
                "watch_id": "22222-rolex-yacht-master-126622",
            },
            {
                "brand": "Rolex",
                "model_name": "Air King 126900",
                "url": "https://watchcharts.com/watch_model/33333-rolex-air-king-126900/overview",
                "watch_id": "33333-rolex-air-king-126900",
            },
            {
                "brand": "Rolex",
                "model_name": "Milgauss 116400GV",
                "url": "https://watchcharts.com/watch_model/44444-rolex-milgauss-116400gv/overview",
                "watch_id": "44444-rolex-milgauss-116400gv",
            },
            {
                "brand": "Rolex",
                "model_name": "Sky Dweller 326934",
                "url": "https://watchcharts.com/watch_model/55555-rolex-sky-dweller-326934/overview",
                "watch_id": "55555-rolex-sky-dweller-326934",
            },
            # Omega
            {
                "brand": "Omega",
                "model_name": "Speedmaster Professional",
                "url": "https://watchcharts.com/watch_model/30921-omega-speedmaster-professional-moonwatch-310-30-42-50-01-002/overview",
                "watch_id": "30921-omega-speedmaster-professional-moonwatch-310-30-42-50-01-002",
            },
            {
                "brand": "Omega",
                "model_name": "Seamaster 300M",
                "url": "https://watchcharts.com/watch_model/66666-omega-seamaster-300m/overview",
                "watch_id": "66666-omega-seamaster-300m",
            },
            {
                "brand": "Omega",
                "model_name": "Planet Ocean",
                "url": "https://watchcharts.com/watch_model/77777-omega-planet-ocean/overview",
                "watch_id": "77777-omega-planet-ocean",
            },
            {
                "brand": "Omega",
                "model_name": "Constellation",
                "url": "https://watchcharts.com/watch_model/88888-omega-constellation/overview",
                "watch_id": "88888-omega-constellation",
            },
            {
                "brand": "Omega",
                "model_name": "De Ville",
                "url": "https://watchcharts.com/watch_model/99999-omega-de-ville/overview",
                "watch_id": "99999-omega-de-ville",
            },
            {
                "brand": "Omega",
                "model_name": "Railmaster",
                "url": "https://watchcharts.com/watch_model/12312-omega-railmaster/overview",
                "watch_id": "12312-omega-railmaster",
            },
            {
                "brand": "Omega",
                "model_name": "Aqua Terra",
                "url": "https://watchcharts.com/watch_model/23423-omega-aqua-terra/overview",
                "watch_id": "23423-omega-aqua-terra",
            },
            {
                "brand": "Omega",
                "model_name": "Globemaster",
                "url": "https://watchcharts.com/watch_model/34534-omega-globemaster/overview",
                "watch_id": "34534-omega-globemaster",
            },
            {
                "brand": "Omega",
                "model_name": "Dark Side of the Moon",
                "url": "https://watchcharts.com/watch_model/45645-omega-dark-side-moon/overview",
                "watch_id": "45645-omega-dark-side-moon",
            },
            {
                "brand": "Omega",
                "model_name": "Racing",
                "url": "https://watchcharts.com/watch_model/56756-omega-racing/overview",
                "watch_id": "56756-omega-racing",
            },
            # Tudor
            {
                "brand": "Tudor",
                "model_name": "Black Bay 58",
                "url": "https://watchcharts.com/watch_model/326-tudor-black-bay-58-79030n/overview",
                "watch_id": "326-tudor-black-bay-58-79030n",
            },
            {
                "brand": "Tudor",
                "model_name": "Pelagos",
                "url": "https://watchcharts.com/watch_model/67867-tudor-pelagos/overview",
                "watch_id": "67867-tudor-pelagos",
            },
            {
                "brand": "Tudor",
                "model_name": "GMT",
                "url": "https://watchcharts.com/watch_model/78978-tudor-gmt/overview",
                "watch_id": "78978-tudor-gmt",
            },
            {
                "brand": "Tudor",
                "model_name": "Ranger",
                "url": "https://watchcharts.com/watch_model/89089-tudor-ranger/overview",
                "watch_id": "89089-tudor-ranger",
            },
            {
                "brand": "Tudor",
                "model_name": "Royal",
                "url": "https://watchcharts.com/watch_model/90190-tudor-royal/overview",
                "watch_id": "90190-tudor-royal",
            },
            {
                "brand": "Tudor",
                "model_name": "Advisor",
                "url": "https://watchcharts.com/watch_model/01201-tudor-advisor/overview",
                "watch_id": "01201-tudor-advisor",
            },
            {
                "brand": "Tudor",
                "model_name": "Glamour",
                "url": "https://watchcharts.com/watch_model/12312-tudor-glamour/overview",
                "watch_id": "12312-tudor-glamour",
            },
            {
                "brand": "Tudor",
                "model_name": "Style",
                "url": "https://watchcharts.com/watch_model/23423-tudor-style/overview",
                "watch_id": "23423-tudor-style",
            },
            {
                "brand": "Tudor",
                "model_name": "Fastrider",
                "url": "https://watchcharts.com/watch_model/34534-tudor-fastrider/overview",
                "watch_id": "34534-tudor-fastrider",
            },
            {
                "brand": "Tudor",
                "model_name": "Heritage",
                "url": "https://watchcharts.com/watch_model/45645-tudor-heritage/overview",
                "watch_id": "45645-tudor-heritage",
            },
            # Patek Philippe
            {
                "brand": "Patek Philippe",
                "model_name": "Nautilus 5711",
                "url": "https://watchcharts.com/watch_model/11111-patek-philippe-nautilus-5711/overview",
                "watch_id": "11111-patek-philippe-nautilus-5711",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Aquanaut 5167",
                "url": "https://watchcharts.com/watch_model/22222-patek-philippe-aquanaut-5167/overview",
                "watch_id": "22222-patek-philippe-aquanaut-5167",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Calatrava 5196",
                "url": "https://watchcharts.com/watch_model/33333-patek-philippe-calatrava-5196/overview",
                "watch_id": "33333-patek-philippe-calatrava-5196",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Grand Complications",
                "url": "https://watchcharts.com/watch_model/44444-patek-philippe-grand-complications/overview",
                "watch_id": "44444-patek-philippe-grand-complications",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Golden Ellipse",
                "url": "https://watchcharts.com/watch_model/55555-patek-philippe-golden-ellipse/overview",
                "watch_id": "55555-patek-philippe-golden-ellipse",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Twenty-4",
                "url": "https://watchcharts.com/watch_model/66666-patek-philippe-twenty-4/overview",
                "watch_id": "66666-patek-philippe-twenty-4",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Gondolo",
                "url": "https://watchcharts.com/watch_model/77777-patek-philippe-gondolo/overview",
                "watch_id": "77777-patek-philippe-gondolo",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Pilot Travel Time",
                "url": "https://watchcharts.com/watch_model/88888-patek-philippe-pilot-travel-time/overview",
                "watch_id": "88888-patek-philippe-pilot-travel-time",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "World Time",
                "url": "https://watchcharts.com/watch_model/99999-patek-philippe-world-time/overview",
                "watch_id": "99999-patek-philippe-world-time",
            },
            {
                "brand": "Patek Philippe",
                "model_name": "Annual Calendar",
                "url": "https://watchcharts.com/watch_model/10101-patek-philippe-annual-calendar/overview",
                "watch_id": "10101-patek-philippe-annual-calendar",
            },
            # Audemars Piguet
            {
                "brand": "Audemars Piguet",
                "model_name": "Royal Oak 15400",
                "url": "https://watchcharts.com/watch_model/11111-audemars-piguet-royal-oak-15400/overview",
                "watch_id": "11111-audemars-piguet-royal-oak-15400",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Royal Oak Offshore",
                "url": "https://watchcharts.com/watch_model/22222-audemars-piguet-royal-oak-offshore/overview",
                "watch_id": "22222-audemars-piguet-royal-oak-offshore",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Royal Oak Jumbo",
                "url": "https://watchcharts.com/watch_model/33333-audemars-piguet-royal-oak-jumbo/overview",
                "watch_id": "33333-audemars-piguet-royal-oak-jumbo",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Millenary",
                "url": "https://watchcharts.com/watch_model/44444-audemars-piguet-millenary/overview",
                "watch_id": "44444-audemars-piguet-millenary",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Jules Audemars",
                "url": "https://watchcharts.com/watch_model/55555-audemars-piguet-jules-audemars/overview",
                "watch_id": "55555-audemars-piguet-jules-audemars",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Edward Piguet",
                "url": "https://watchcharts.com/watch_model/66666-audemars-piguet-edward-piguet/overview",
                "watch_id": "66666-audemars-piguet-edward-piguet",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Code 11.59",
                "url": "https://watchcharts.com/watch_model/77777-audemars-piguet-code-1159/overview",
                "watch_id": "77777-audemars-piguet-code-1159",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Royal Oak Concept",
                "url": "https://watchcharts.com/watch_model/88888-audemars-piguet-royal-oak-concept/overview",
                "watch_id": "88888-audemars-piguet-royal-oak-concept",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Tradition",
                "url": "https://watchcharts.com/watch_model/99999-audemars-piguet-tradition/overview",
                "watch_id": "99999-audemars-piguet-tradition",
            },
            {
                "brand": "Audemars Piguet",
                "model_name": "Huitième",
                "url": "https://watchcharts.com/watch_model/10101-audemars-piguet-huitieme/overview",
                "watch_id": "10101-audemars-piguet-huitieme",
            },
            # Cartier
            {
                "brand": "Cartier",
                "model_name": "Santos",
                "url": "https://watchcharts.com/watch_model/11111-cartier-santos/overview",
                "watch_id": "11111-cartier-santos",
            },
            {
                "brand": "Cartier",
                "model_name": "Tank",
                "url": "https://watchcharts.com/watch_model/22222-cartier-tank/overview",
                "watch_id": "22222-cartier-tank",
            },
            {
                "brand": "Cartier",
                "model_name": "Ballon Bleu",
                "url": "https://watchcharts.com/watch_model/33333-cartier-ballon-bleu/overview",
                "watch_id": "33333-cartier-ballon-bleu",
            },
            {
                "brand": "Cartier",
                "model_name": "Drive",
                "url": "https://watchcharts.com/watch_model/44444-cartier-drive/overview",
                "watch_id": "44444-cartier-drive",
            },
            {
                "brand": "Cartier",
                "model_name": "Ronde",
                "url": "https://watchcharts.com/watch_model/55555-cartier-ronde/overview",
                "watch_id": "55555-cartier-ronde",
            },
            {
                "brand": "Cartier",
                "model_name": "Calibre",
                "url": "https://watchcharts.com/watch_model/66666-cartier-calibre/overview",
                "watch_id": "66666-cartier-calibre",
            },
            {
                "brand": "Cartier",
                "model_name": "Pasha",
                "url": "https://watchcharts.com/watch_model/77777-cartier-pasha/overview",
                "watch_id": "77777-cartier-pasha",
            },
            {
                "brand": "Cartier",
                "model_name": "Roadster",
                "url": "https://watchcharts.com/watch_model/88888-cartier-roadster/overview",
                "watch_id": "88888-cartier-roadster",
            },
            {
                "brand": "Cartier",
                "model_name": "Panthere",
                "url": "https://watchcharts.com/watch_model/99999-cartier-panthere/overview",
                "watch_id": "99999-cartier-panthere",
            },
            {
                "brand": "Cartier",
                "model_name": "Rotonde",
                "url": "https://watchcharts.com/watch_model/10101-cartier-rotonde/overview",
                "watch_id": "10101-cartier-rotonde",
            },
            # Breitling
            {
                "brand": "Breitling",
                "model_name": "Navitimer",
                "url": "https://watchcharts.com/watch_model/11111-breitling-navitimer/overview",
                "watch_id": "11111-breitling-navitimer",
            },
            {
                "brand": "Breitling",
                "model_name": "Superocean",
                "url": "https://watchcharts.com/watch_model/22222-breitling-superocean/overview",
                "watch_id": "22222-breitling-superocean",
            },
            {
                "brand": "Breitling",
                "model_name": "Chronomat",
                "url": "https://watchcharts.com/watch_model/33333-breitling-chronomat/overview",
                "watch_id": "33333-breitling-chronomat",
            },
            {
                "brand": "Breitling",
                "model_name": "Premier",
                "url": "https://watchcharts.com/watch_model/44444-breitling-premier/overview",
                "watch_id": "44444-breitling-premier",
            },
            {
                "brand": "Breitling",
                "model_name": "Avenger",
                "url": "https://watchcharts.com/watch_model/55555-breitling-avenger/overview",
                "watch_id": "55555-breitling-avenger",
            },
            {
                "brand": "Breitling",
                "model_name": "Transocean",
                "url": "https://watchcharts.com/watch_model/66666-breitling-transocean/overview",
                "watch_id": "66666-breitling-transocean",
            },
            {
                "brand": "Breitling",
                "model_name": "Galactic",
                "url": "https://watchcharts.com/watch_model/77777-breitling-galactic/overview",
                "watch_id": "77777-breitling-galactic",
            },
            {
                "brand": "Breitling",
                "model_name": "Cockpit",
                "url": "https://watchcharts.com/watch_model/88888-breitling-cockpit/overview",
                "watch_id": "88888-breitling-cockpit",
            },
            {
                "brand": "Breitling",
                "model_name": "Montbrillant",
                "url": "https://watchcharts.com/watch_model/99999-breitling-montbrillant/overview",
                "watch_id": "99999-breitling-montbrillant",
            },
            {
                "brand": "Breitling",
                "model_name": "Bentley",
                "url": "https://watchcharts.com/watch_model/10101-breitling-bentley/overview",
                "watch_id": "10101-breitling-bentley",
            },
            # Tag Heuer
            {
                "brand": "Tag Heuer",
                "model_name": "Carrera",
                "url": "https://watchcharts.com/watch_model/11111-tag-heuer-carrera/overview",
                "watch_id": "11111-tag-heuer-carrera",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Monaco",
                "url": "https://watchcharts.com/watch_model/22222-tag-heuer-monaco/overview",
                "watch_id": "22222-tag-heuer-monaco",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Aquaracer",
                "url": "https://watchcharts.com/watch_model/33333-tag-heuer-aquaracer/overview",
                "watch_id": "33333-tag-heuer-aquaracer",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Formula 1",
                "url": "https://watchcharts.com/watch_model/44444-tag-heuer-formula-1/overview",
                "watch_id": "44444-tag-heuer-formula-1",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Link",
                "url": "https://watchcharts.com/watch_model/55555-tag-heuer-link/overview",
                "watch_id": "55555-tag-heuer-link",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Connected",
                "url": "https://watchcharts.com/watch_model/66666-tag-heuer-connected/overview",
                "watch_id": "66666-tag-heuer-connected",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Autavia",
                "url": "https://watchcharts.com/watch_model/77777-tag-heuer-autavia/overview",
                "watch_id": "77777-tag-heuer-autavia",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Grand Carrera",
                "url": "https://watchcharts.com/watch_model/88888-tag-heuer-grand-carrera/overview",
                "watch_id": "88888-tag-heuer-grand-carrera",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Kirium",
                "url": "https://watchcharts.com/watch_model/99999-tag-heuer-kirium/overview",
                "watch_id": "99999-tag-heuer-kirium",
            },
            {
                "brand": "Tag Heuer",
                "model_name": "Alter Ego",
                "url": "https://watchcharts.com/watch_model/10101-tag-heuer-alter-ego/overview",
                "watch_id": "10101-tag-heuer-alter-ego",
            },
            # IWC
            {
                "brand": "IWC",
                "model_name": "Pilot",
                "url": "https://watchcharts.com/watch_model/11111-iwc-pilot/overview",
                "watch_id": "11111-iwc-pilot",
            },
            {
                "brand": "IWC",
                "model_name": "Portugieser",
                "url": "https://watchcharts.com/watch_model/22222-iwc-portugieser/overview",
                "watch_id": "22222-iwc-portugieser",
            },
            {
                "brand": "IWC",
                "model_name": "Aquatimer",
                "url": "https://watchcharts.com/watch_model/33333-iwc-aquatimer/overview",
                "watch_id": "33333-iwc-aquatimer",
            },
            {
                "brand": "IWC",
                "model_name": "Ingenieur",
                "url": "https://watchcharts.com/watch_model/44444-iwc-ingenieur/overview",
                "watch_id": "44444-iwc-ingenieur",
            },
            {
                "brand": "IWC",
                "model_name": "Da Vinci",
                "url": "https://watchcharts.com/watch_model/55555-iwc-da-vinci/overview",
                "watch_id": "55555-iwc-da-vinci",
            },
            {
                "brand": "IWC",
                "model_name": "Portofino",
                "url": "https://watchcharts.com/watch_model/66666-iwc-portofino/overview",
                "watch_id": "66666-iwc-portofino",
            },
            {
                "brand": "IWC",
                "model_name": "Big Pilot",
                "url": "https://watchcharts.com/watch_model/77777-iwc-big-pilot/overview",
                "watch_id": "77777-iwc-big-pilot",
            },
            {
                "brand": "IWC",
                "model_name": "Mark XVIII",
                "url": "https://watchcharts.com/watch_model/88888-iwc-mark-xviii/overview",
                "watch_id": "88888-iwc-mark-xviii",
            },
            {
                "brand": "IWC",
                "model_name": "Spitfire",
                "url": "https://watchcharts.com/watch_model/99999-iwc-spitfire/overview",
                "watch_id": "99999-iwc-spitfire",
            },
            {
                "brand": "IWC",
                "model_name": "Top Gun",
                "url": "https://watchcharts.com/watch_model/10101-iwc-top-gun/overview",
                "watch_id": "10101-iwc-top-gun",
            },
            # Jaeger-LeCoultre
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Reverso",
                "url": "https://watchcharts.com/watch_model/11111-jaeger-lecoultre-reverso/overview",
                "watch_id": "11111-jaeger-lecoultre-reverso",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Master Control",
                "url": "https://watchcharts.com/watch_model/22222-jaeger-lecoultre-master-control/overview",
                "watch_id": "22222-jaeger-lecoultre-master-control",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Atmos",
                "url": "https://watchcharts.com/watch_model/33333-jaeger-lecoultre-atmos/overview",
                "watch_id": "33333-jaeger-lecoultre-atmos",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Rendez-Vous",
                "url": "https://watchcharts.com/watch_model/44444-jaeger-lecoultre-rendez-vous/overview",
                "watch_id": "44444-jaeger-lecoultre-rendez-vous",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Duomètre",
                "url": "https://watchcharts.com/watch_model/55555-jaeger-lecoultre-duometre/overview",
                "watch_id": "55555-jaeger-lecoultre-duometre",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Geophysic",
                "url": "https://watchcharts.com/watch_model/66666-jaeger-lecoultre-geophysic/overview",
                "watch_id": "66666-jaeger-lecoultre-geophysic",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Polaris",
                "url": "https://watchcharts.com/watch_model/77777-jaeger-lecoultre-polaris/overview",
                "watch_id": "77777-jaeger-lecoultre-polaris",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Master Grande",
                "url": "https://watchcharts.com/watch_model/88888-jaeger-lecoultre-master-grande/overview",
                "watch_id": "88888-jaeger-lecoultre-master-grande",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Hybris",
                "url": "https://watchcharts.com/watch_model/99999-jaeger-lecoultre-hybris/overview",
                "watch_id": "99999-jaeger-lecoultre-hybris",
            },
            {
                "brand": "Jaeger-LeCoultre",
                "model_name": "Deep Sea",
                "url": "https://watchcharts.com/watch_model/10101-jaeger-lecoultre-deep-sea/overview",
                "watch_id": "10101-jaeger-lecoultre-deep-sea",
            },
        ]

        return fallback_watches

    def discover_all_watches(self) -> List[Dict]:
        """Discover watches for all target brands."""
        all_watches = []

        print(f"🔍 Discovering watches for {len(self.target_brands)} brands...")

        for brand in self.target_brands:
            print(f"\n📋 Discovering {brand} watches...")

            try:
                brand_watches = self.discover_brand_watches(brand, 10)

                if len(brand_watches) < 10:
                    print(
                        f"⚠️  Only found {len(brand_watches)} watches for {brand}, using fallback"
                    )
                    # Use fallback for this brand
                    fallback_watches = [
                        w for w in self.get_fallback_watches() if w["brand"] == brand
                    ]
                    brand_watches.extend(fallback_watches[: 10 - len(brand_watches)])

                all_watches.extend(brand_watches[:10])  # Ensure exactly 10 per brand
                print(f"✅ Added {len(brand_watches)} {brand} watches")

                # Delay between brands
                time.sleep(random.uniform(5, 10))

            except Exception as e:
                print(f"❌ Error discovering {brand} watches: {e}")
                # Use fallback for this brand
                fallback_watches = [
                    w for w in self.get_fallback_watches() if w["brand"] == brand
                ]
                all_watches.extend(fallback_watches[:10])
                print(f"✅ Added {len(fallback_watches[:10])} fallback {brand} watches")

        print(f"\n🎯 Total watches discovered: {len(all_watches)}")
        return all_watches

    def save_watch_targets(
        self, watches: List[Dict], filename: str = "watch_targets.json"
    ):
        """Save discovered watch targets to JSON file."""
        output_path = os.path.join("data", filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(watches, f, indent=2)

        print(f"💾 Saved {len(watches)} watch targets to {output_path}")
        return output_path


def main():
    """Main function to discover and save watch targets."""
    discovery = WatchURLDiscovery()

    # Try to discover watches dynamically
    watches = discovery.discover_all_watches()

    # If discovery failed completely, use fallback
    if len(watches) < 50:  # Less than half the target
        print("⚠️  Dynamic discovery failed, using fallback list")
        watches = discovery.get_fallback_watches()

    # Save to file
    output_file = discovery.save_watch_targets(watches)

    # Print summary
    brand_counts = {}
    for watch in watches:
        brand = watch["brand"]
        brand_counts[brand] = brand_counts.get(brand, 0) + 1

    print("\n📊 DISCOVERY SUMMARY")
    print("=" * 40)
    for brand, count in brand_counts.items():
        print(f"{brand}: {count} watches")

    print(f"\nTotal: {len(watches)} watches")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
