"""
Selenium utilities for web scraping with Cloudflare bypass capabilities.
"""

import logging
from typing import Optional

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class SeleniumDriverFactory:
    """Factory for creating configured Selenium drivers."""
    
    @staticmethod
    def create_stealth_driver(headless: bool = False) -> webdriver.Chrome:
        """Create a stealthy Chrome driver with Cloudflare bypass techniques."""
        options = Options()
        
        if headless:
            options.add_argument('--headless')
        
        # Anti-detection options
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Realistic browser profile
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--disable-extensions')
        
        # Performance optimizations
        options.add_argument('--no-first-run')
        options.add_argument('--disable-default-apps')
        options.add_argument('--disable-background-timer-throttling')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--disable-backgrounding-occluded-windows')
        
        try:
            # Create driver
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            
            # Execute stealth script
            stealth_script = """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            window.chrome = {runtime: {}};
            """
            driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {'source': stealth_script})
            
            logger.info("Successfully created stealth Chrome driver")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to create Chrome driver: {e}")
            raise
    
    @staticmethod
    def create_discovery_driver(headless: bool = True) -> webdriver.Chrome:
        """Create driver optimized for URL discovery tasks."""
        options = Options()
        
        if headless:
            options.add_argument('--headless')
        
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            logger.info("Successfully created discovery Chrome driver")
            return driver
            
        except Exception as e:
            logger.error(f"Failed to create discovery driver: {e}")
            raise


def safe_quit_driver(driver: Optional[webdriver.Chrome]) -> None:
    """Safely quit a Selenium driver with error handling."""
    if driver:
        try:
            driver.quit()
            logger.debug("Driver quit successfully")
        except Exception as e:
            logger.warning(f"Error quitting driver: {e}")


def check_cloudflare_challenge(driver: webdriver.Chrome) -> bool:
    """Check if the current page contains a Cloudflare challenge."""
    try:
        page_source = driver.page_source.lower()
        cloudflare_indicators = [
            "cloudflare",
            "checking your browser",
            "please wait while we check your browser",
            "cf-browser-verification"
        ]
        
        for indicator in cloudflare_indicators:
            if indicator in page_source:
                logger.warning("Cloudflare challenge detected")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking for Cloudflare challenge: {e}")
        return False