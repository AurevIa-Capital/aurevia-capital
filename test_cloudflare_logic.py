#!/usr/bin/env python3
"""
Test script to validate the improved Cloudflare detection logic.
This script tests the website loading sequence without requiring pandas.
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from utils.selenium_utils import (
        SeleniumDriverFactory,
        check_cloudflare_challenge,
        check_website_loaded_successfully,
        safe_quit_driver,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run this from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_website_loading():
    """Test the improved website loading and Cloudflare detection logic."""
    
    test_url = "https://watchcharts.com/watch_model/21813-rolex-submariner-124060/overview"
    
    print(f"🧪 Testing website loading logic for: {test_url}")
    print("=" * 60)
    
    driver = None
    try:
        # Create driver
        print("🚀 Creating stealth Chrome driver...")
        driver = SeleniumDriverFactory.create_stealth_driver()
        
        # Navigate to URL
        print(f"🌐 Navigating to {test_url}")
        driver.get(test_url)
        
        # Wait for page load
        print("⏳ Waiting for page to load...")
        from selenium.webdriver.support.ui import WebDriverWait
        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        
        # Wait for dynamic content
        print("⏳ Waiting for dynamic content (3 seconds)...")
        time.sleep(3)
        
        # Check website loading status
        print("🔍 Checking if website loaded successfully...")
        website_loaded = check_website_loaded_successfully(driver)
        
        if website_loaded:
            print("✅ Website loaded successfully!")
            print("📊 Page info:")
            print(f"   Title: {driver.title}")
            print(f"   URL: {driver.current_url}")
            
            # Look for chart elements
            try:
                from selenium.webdriver.common.by import By
                canvas_elements = driver.find_elements(By.TAG_NAME, "canvas")
                chart_elements = driver.find_elements(By.CSS_SELECTOR, ".chart")
                
                print(f"   Canvas elements found: {len(canvas_elements)}")
                print(f"   Chart elements found: {len(chart_elements)}")
                
                if canvas_elements or chart_elements:
                    print("✅ Chart elements detected - ready for data extraction!")
                else:
                    print("⚠️  No chart elements found")
                    
            except Exception as e:
                print(f"❌ Error checking for chart elements: {e}")
        else:
            print("❌ Website failed to load properly")
            
            # Check for Cloudflare
            print("🔍 Checking for Cloudflare challenge...")
            cloudflare_detected = check_cloudflare_challenge(driver)
            
            if cloudflare_detected:
                print("🛡️  Cloudflare challenge detected")
                print("📄 Page source snippet:")
                page_source = driver.page_source[:500] + "..." if len(driver.page_source) > 500 else driver.page_source
                print(page_source)
            else:
                print("❓ No Cloudflare challenge detected - unknown issue")
                
    except Exception as e:
        print(f"❌ Error during test: {e}")
        
    finally:
        if driver:
            print("🔄 Cleaning up driver...")
            safe_quit_driver(driver)
            
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_website_loading()