"""
Base scraper functionality with shared utilities for watch data collection.

This module provides common functionality used across different watch scraping
components to reduce code duplication and improve maintainability.
"""

import logging
import random
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.utils.selenium_utils import (
    SeleniumDriverFactory,
    check_cloudflare_challenge,
    check_website_loaded_successfully,
)

logger = logging.getLogger(__name__)


class WatchScrapingMixin:
    """Mixin class providing common scraping utilities."""

    def __init__(self, delay_range: Tuple[int, int] = (3, 8)):
        self.delay_range = delay_range
        self.base_url = "https://watchcharts.com"

    def random_delay(self) -> None:
        """Add random delay between requests."""
        delay = random.uniform(*self.delay_range)
        logger.info(f"Waiting {delay:.1f} seconds...")
        time.sleep(delay)

    def create_browser_session(self, headless: bool = True) -> webdriver.Chrome:
        """Create a fresh browser session with stealth capabilities."""
        return SeleniumDriverFactory.create_stealth_driver(headless=headless)

    def safe_navigate_with_retries(
        self,
        driver: webdriver.Chrome,
        url: str,
        max_retries: int = 3,
        wait_for_elements: List[str] = None,
    ) -> bool:
        """
        Navigate to URL with retries and comprehensive error handling.

        Args:
            driver: Selenium WebDriver instance
            url: URL to navigate to
            max_retries: Maximum number of retry attempts
            wait_for_elements: List of CSS selectors to wait for after page load

        Returns:
            bool: True if navigation successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Navigating to {url} (attempt {attempt + 1}/{max_retries})"
                )

                # Navigate to URL
                driver.get(url)

                # Wait for basic page load
                WebDriverWait(driver, 30).until(
                    lambda d: d.execute_script("return document.readyState")
                    == "complete"
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
                        raise Exception("Website failed to load properly")

                # Wait for specific elements if provided
                if wait_for_elements:
                    for selector in wait_for_elements:
                        try:
                            WebDriverWait(driver, 20).until(
                                EC.presence_of_element_located(
                                    (By.CSS_SELECTOR, selector)
                                )
                            )
                            logger.info(f"Found required element: {selector}")
                            break
                        except Exception:
                            logger.debug(f"Element not found: {selector}")
                            continue

                logger.info("Navigation successful")
                return True

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to navigate after {max_retries} attempts: {e}"
                    )
                    return False

                logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying...")
                self.random_delay()

        return False

    def extract_watch_id_from_url(self, url: str) -> str:
        """Extract watch ID from WatchCharts URL."""
        # Expected format: https://watchcharts.com/watch_model/{ID}-{slug}/overview
        match = re.search(r"/watch_model/(\d+)-([^/]+)", url)
        if match:
            return match.group(1)

        # Fallback - look for numeric ID in path
        path_parts = urlparse(url).path.split("/")
        for part in path_parts:
            if part and part.split("-")[0].isdigit():
                return part.split("-")[0]

        return "unknown"

    def clean_model_name(self, model_name: str, brand: str) -> str:
        """
        Clean and standardize model names by removing unwanted patterns.

        Args:
            model_name: Raw model name from scraping
            brand: Brand name to remove from model name

        Returns:
            str: Cleaned model name
        """
        if not model_name:
            return ""

        # Remove brand name from model name if present
        cleaned = model_name.replace(brand, "").strip()
        if cleaned.startswith("-"):
            cleaned = cleaned[1:].strip()

        # Comprehensive cleanup patterns
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
            r"Ceramic\d+mm\d+M",
            r"[A-Z0-9]+/[A-Z0-9.-]+",  # Reference numbers like 5167A, 126300
            r"[A-Z0-9]{4,}",  # Long alphanumeric codes
        ]

        for pattern in cleanup_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Clean up whitespace and normalize
        cleaned = " ".join(cleaned.split())

        return cleaned if len(cleaned) >= 3 else model_name

    def make_filename_safe(self, text: str) -> str:
        """Convert text to filesystem-safe filename."""
        return (
            text.replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
        )

    def ensure_absolute_url(self, url: str, add_overview: bool = True) -> str:
        """Ensure URL is absolute and optionally add /overview suffix."""
        # Make absolute if relative
        if url.startswith("/"):
            url = urljoin(self.base_url, url)

        # Add /overview if requested and not present
        if add_overview and not url.endswith("/overview"):
            url = url.rstrip("/") + "/overview"

        return url


class BaseScraper(WatchScrapingMixin, ABC):
    """Abstract base class for watch scrapers with common functionality."""

    def __init__(self, delay_range: Tuple[int, int] = (3, 8)):
        super().__init__(delay_range)
        self.setup_logging()

    def setup_logging(self) -> None:
        """Setup logging configuration."""
        # Logging now handled by centralized logging_config
        pass

    @abstractmethod
    def process_target(self, target: Dict, **kwargs) -> bool:
        """Process a single scraping target. Must be implemented by subclasses."""
        pass

    def process_multiple_targets(
        self, targets: List[Dict], brand_delay: int = 30, **kwargs
    ) -> Dict[str, bool]:
        """
        Process multiple targets with brand-based batching and delays.

        Args:
            targets: List of target dictionaries
            brand_delay: Delay between processing different brands
            **kwargs: Additional arguments passed to process_target

        Returns:
            Dict mapping target IDs to success status
        """
        results = {}

        # Group targets by brand
        brand_groups = {}
        for target in targets:
            brand = target.get("brand", "unknown")
            if brand not in brand_groups:
                brand_groups[brand] = []
            brand_groups[brand].append(target)

        # Process each brand group
        for i, (brand, brand_targets) in enumerate(brand_groups.items()):
            logger.info(f"Processing {brand} ({len(brand_targets)} targets)")

            # Process targets in this brand
            for target in brand_targets:
                target_id = self._get_target_id(target)
                try:
                    success = self.process_target(target, **kwargs)
                    results[target_id] = success

                    if success:
                        logger.info(f"✅ Successfully processed {target_id}")
                    else:
                        logger.warning(f"❌ Failed to process {target_id}")

                except Exception as e:
                    logger.error(f"❌ Error processing {target_id}: {e}")
                    results[target_id] = False

                # Small delay between targets within same brand
                time.sleep(random.uniform(2, 5))

            # Longer delay between brands (except for last brand)
            if i < len(brand_groups) - 1:
                logger.info(f"Waiting {brand_delay} seconds before next brand...")
                time.sleep(brand_delay)

        return results

    def _get_target_id(self, target: Dict) -> str:
        """Extract a unique identifier from target dictionary."""
        if "watch_id" in target:
            return target["watch_id"]
        elif "model_name" in target and "brand" in target:
            return f"{target['brand']}_{target['model_name']}"
        else:
            return str(hash(str(target)))
