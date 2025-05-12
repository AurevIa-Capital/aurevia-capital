import json
import os
import time

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


def scrape_watchcharts(urls, base_dir="data/watches"):
    """
    Scrape watch price data from multiple WatchCharts URLs

    Args:
        urls (list): List of WatchCharts URLs to scrape
        base_dir (str): Base directory to save output files

    Returns:
        dict: Dictionary mapping URLs to DataFrames (or None for failed scrapes)
    """
    results = {}

    # Setup Chrome options - only do this once
    options = Options()
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    # Initialize WebDriver once for all scrapes
    print("Setting up Chrome browser...")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    try:
        for i, url in enumerate(urls):
            print(f"\n{'=' * 60}")
            print(f"Processing watch {i + 1}/{len(urls)}: {url}")
            print(f"{'=' * 60}")

            # Generate output file path
            output_file = get_output_path_from_url(url, base_dir)
            print(f"Will save to: {output_file}")

            # Create output directory if needed
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            try:
                # Navigate to page
                print(f"Opening {url}...")
                driver.get(url)

                # Wait for user input for EVERY watch - as requested
                print("\n============================================================")
                print(f"Watch {i + 1}/{len(urls)}: {url}")
                print("MANUAL ACTION REQUIRED: Please solve CAPTCHA if needed")
                print(
                    "After the page fully loads with the watch data, press Enter to continue"
                )
                print("============================================================\n")
                input("Press Enter when the page is loaded...")

                # Sleep for 10 seconds as requested
                print("Waiting 10 seconds to ensure page is fully loaded...")
                time.sleep(10)

                # Try to locate a chart element to confirm page loaded properly
                try:
                    WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.TAG_NAME, "canvas"))
                    )
                    print("Chart elements detected on page")
                except Exception:
                    print(
                        "Warning: Could not detect chart elements - page may not be fully loaded"
                    )

                # JavaScript to extract price data
                extract_script = """
                function extractPriceData() {
                    // Check if Chart.js is available
                    if (!window.Chart || !window.Chart.instances) {
                        return { error: "Chart.js not found or not initialized" };
                    }
                    
                    // Get all chart instances
                    const chartInstances = Object.values(window.Chart.instances);
                    console.log(`Found ${chartInstances.length} chart instances`);
                    
                    // Find price history data
                    for (let i = 0; i < chartInstances.length; i++) {
                        const chart = chartInstances[i];
                        
                        if (!chart.data || !chart.data.datasets) continue;
                        
                        for (let j = 0; j < chart.data.datasets.length; j++) {
                            const dataset = chart.data.datasets[j];
                            
                            if (!dataset.data || dataset.data.length === 0) continue;
                            
                            // Check if this looks like price history data
                            const samplePoint = dataset.data[0];
                            if (samplePoint && 
                                typeof samplePoint === 'object' && 
                                samplePoint.y !== undefined && 
                                samplePoint.x instanceof Date) {
                                
                                console.log(`Found price data in chart ${i}, dataset ${j}`);
                                
                                // Format data for return
                                return {
                                    success: true,
                                    watchName: document.title.split('-')[0]?.trim() || 'Watch',
                                    data: dataset.data.map(point => ({
                                        date: point.x.toISOString().split('T')[0],  // Format as YYYY-MM-DD
                                        price: point.y
                                    }))
                                };
                            }
                        }
                    }
                    
                    return { error: "No price history data found in any chart" };
                }
                
                return JSON.stringify(extractPriceData());
                """

                # Execute JavaScript to extract data
                print("Extracting price data from charts...")
                result = driver.execute_script(extract_script)
                data = json.loads(result)

                if "error" in data:
                    print(f"Error extracting data: {data['error']}")

                    # Additional debugging
                    print("Checking available charts...")
                    chart_info = driver.execute_script("""
                        if (window.Chart && window.Chart.instances) {
                            return Object.keys(window.Chart.instances).length + " charts found";
                        }
                        return "No Chart.js instances found";
                    """)
                    print(chart_info)

                    results[url] = None
                    continue  # Skip to next URL

                if not data.get("success"):
                    print("Unknown error during data extraction")
                    results[url] = None
                    continue  # Skip to next URL

                # Create DataFrame with the data
                print(f"Successfully extracted {len(data['data'])} price points")
                df = pd.DataFrame(data["data"])

                # Rename price column to include currency
                df.rename(columns={"price": "price(SGD)"}, inplace=True)

                # Save to CSV with proper headers
                df.to_csv(output_file, index=False)
                print(f"Data saved to {output_file}")

                # Show preview
                print("\nData preview:")
                print(df.head())

                # Store result
                results[url] = df

            except Exception as e:
                print(f"Error during scraping: {str(e)}")

                # Try to take screenshot for debugging
                try:
                    screenshot_path = output_file.replace(".csv", "_error.png")
                    driver.save_screenshot(screenshot_path)
                    print(f"Error screenshot saved to: {screenshot_path}")
                except Exception:
                    pass

                results[url] = None

        return results

    finally:
        # Always close the browser at the end
        driver.quit()
        print("Browser closed")


def get_output_path_from_url(url, base_dir="data/watches"):
    """
    Generate an output file path from a WatchCharts URL

    Args:
        url (str): URL of the WatchCharts watch page
        base_dir (str): Base directory to save output files

    Returns:
        str: Path to save the CSV file
    """
    # Extract watch identifier from URL
    # Example URL: https://watchcharts.com/watch_model/641-rolex-cosmograph-daytona-116500/overview
    url_parts = url.strip("/").split("/")

    # Find the part containing the watch identifier (typically after "watch_model")
    watch_id = None
    for i, part in enumerate(url_parts):
        if part == "watch_model" and i + 1 < len(url_parts):
            watch_id = url_parts[i + 1]
            break

    if not watch_id:
        # Fallback if we couldn't extract a proper ID
        import hashlib

        watch_id = hashlib.md5(url.encode()).hexdigest()[:10]

    # Create filename from watch ID
    filename = f"{watch_id}.csv"

    # Create full path
    os.makedirs(base_dir, exist_ok=True)
    output_path = os.path.join(base_dir, filename)

    return output_path


# Usage
urls = [
    # "https://watchcharts.com/watch_model/21813-rolex-submariner-124060/overview",  # Rolex Submariner (Rolex 124060)
    "https://watchcharts.com/watch_model/30921-omega-speedmaster-professional-moonwatch-310-30-42-50-01-002/overview",  # Omega Speedmaster
    # "https://watchcharts.com/watch_model/326-tudor-black-bay-58-79030n/overview",  # Tudor Black Bay 58
]

# Process all URLs at once
results = scrape_watchcharts(urls)

# Summary of results
print("\nSummary of scraping results:")
for url, df in results.items():
    watch_id = url.split("/")[-2]  # Extract watch ID from URL
    status = "SUCCESS" if df is not None else "FAILED"
    print(f"{watch_id}: {status}")

print("\nAll watches processed!")
