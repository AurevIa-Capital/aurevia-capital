#!/usr/bin/env python3
"""
Mass Watch Scraper - Scrape 100 luxury watches efficiently with Cloudflare bypass
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper import CloudflareBypassScraper, WatchTarget
from watch_discovery import WatchURLDiscovery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MassWatchScraper:
    """Orchestrates mass scraping of 100 watches with progress tracking."""
    
    def __init__(self, output_dir: str = "data/watches"):
        self.output_dir = output_dir
        self.progress_file = os.path.join("data", "scraping_progress.json")
        self.targets_file = os.path.join("data", "watch_targets.json")
        self.scraper = CloudflareBypassScraper(max_workers=2, delay_range=(10, 20))
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
    
    def load_or_discover_targets(self) -> list:
        """Load existing targets or discover new ones."""
        if os.path.exists(self.targets_file):
            logger.info(f"Loading existing targets from {self.targets_file}")
            with open(self.targets_file, 'r') as f:
                targets_data = json.load(f)
                return [WatchTarget(**target) for target in targets_data]
        
        logger.info("No existing targets found, discovering new ones...")
        discovery = WatchURLDiscovery()
        watches_data = discovery.discover_all_watches()
        
        # Save for future use
        discovery.save_watch_targets(watches_data)
        
        return [WatchTarget(**watch) for watch in watches_data]
    
    def load_progress(self) -> dict:
        """Load scraping progress from file."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        
        return {
            'completed': [],
            'failed': [],
            'started_at': None,
            'last_updated': None,
            'total_targets': 0
        }
    
    def save_progress(self, progress: dict):
        """Save scraping progress to file."""
        progress['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def get_pending_targets(self, all_targets: list, progress: dict) -> list:
        """Get list of targets that haven't been completed."""
        completed_ids = set(progress.get('completed', []))
        failed_ids = set(progress.get('failed', []))
        
        # Filter out completed and failed
        pending = []
        for target in all_targets:
            if target.watch_id not in completed_ids and target.watch_id not in failed_ids:
                pending.append(target)
        
        return pending
    
    def scrape_with_progress(self, targets: list) -> dict:
        """Scrape all targets with progress tracking."""
        progress = self.load_progress()
        
        if progress['started_at'] is None:
            progress['started_at'] = datetime.now().isoformat()
            progress['total_targets'] = len(targets)
        
        pending_targets = self.get_pending_targets(targets, progress)
        
        logger.info("SCRAPING STATUS")
        logger.info(f"Total targets: {len(targets)}")
        logger.info(f"Completed: {len(progress.get('completed', []))}")
        logger.info(f"Failed: {len(progress.get('failed', []))}")
        logger.info(f"Pending: {len(pending_targets)}")
        
        if not pending_targets:
            logger.info("All targets already processed!")
            return progress
        
        logger.info(f"Starting scraper for {len(pending_targets)} pending targets...")
        
        # Group by brand for organized processing
        brand_groups = {}
        for target in pending_targets:
            if target.brand not in brand_groups:
                brand_groups[target.brand] = []
            brand_groups[target.brand].append(target)
        
        # Process each brand
        for brand_name, brand_targets in brand_groups.items():
            logger.info(f"Processing {brand_name} ({len(brand_targets)} watches)")
            
            # Process watches in this brand
            for i, target in enumerate(brand_targets, 1):
                logger.info(f"[{len(progress['completed']) + len(progress['failed']) + 1}/{len(targets)}] "
                      f"{brand_name} {i}/{len(brand_targets)}: {target.model_name}")
                
                try:
                    success = self.scraper.scrape_single_watch(target, self.output_dir)
                    
                    if success:
                        progress['completed'].append(target.watch_id)
                        logger.info(f"Successfully scraped {target.brand} {target.model_name}")
                    else:
                        progress['failed'].append(target.watch_id)
                        logger.warning(f"Failed to scrape {target.brand} {target.model_name}")
                
                except Exception as e:
                    progress['failed'].append(target.watch_id)
                    logger.error(f"Error scraping {target.brand} {target.model_name}: {e}")
                
                # Save progress after each watch
                self.save_progress(progress)
                
                # Print current stats
                completed = len(progress['completed'])
                failed = len(progress['failed'])
                total = len(targets)
                success_rate = (completed / (completed + failed) * 100) if (completed + failed) > 0 else 0
                
                logger.info(f"Progress: {completed + failed}/{total} "
                      f"(✅{completed} ❌{failed}) "
                      f"Success: {success_rate:.1f}%")
            
            # Longer pause between brands
            if brand_name != list(brand_groups.keys())[-1]:
                logger.info("Waiting 60 seconds before next brand...")
                import time
                time.sleep(60)
        
        return progress
    
    def print_final_report(self, progress: dict, targets: list):
        """Print comprehensive final report."""
        completed = len(progress.get('completed', []))
        failed = len(progress.get('failed', []))
        total = len(targets)
        success_rate = (completed / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("🎯 FINAL SCRAPING REPORT")
        print("="*60)
        
        print(f"📊 Overall Statistics:")
        print(f"  Total targets: {total}")
        print(f"  ✅ Successful: {completed} ({completed/total*100:.1f}%)")
        print(f"  ❌ Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"  📈 Success rate: {success_rate:.1f}%")
        
        # Brand breakdown
        brand_stats = {}
        for target in targets:
            brand = target.brand
            if brand not in brand_stats:
                brand_stats[brand] = {'total': 0, 'completed': 0, 'failed': 0}
            
            brand_stats[brand]['total'] += 1
            
            if target.watch_id in progress.get('completed', []):
                brand_stats[brand]['completed'] += 1
            elif target.watch_id in progress.get('failed', []):
                brand_stats[brand]['failed'] += 1
        
        print(f"\n📦 Brand Breakdown:")
        for brand, stats in brand_stats.items():
            brand_success_rate = (stats['completed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {brand}: {stats['completed']}/{stats['total']} "
                  f"({brand_success_rate:.1f}%) "
                  f"[✅{stats['completed']} ❌{stats['failed']}]")
        
        # Time information
        if progress.get('started_at'):
            started = datetime.fromisoformat(progress['started_at'])
            if progress.get('last_updated'):
                ended = datetime.fromisoformat(progress['last_updated'])
                duration = ended - started
                print(f"\n⏱️  Duration: {duration}")
        
        # File locations
        print(f"\n📁 Output:")
        print(f"  Watch data: {self.output_dir}/")
        print(f"  Progress: {self.progress_file}")
        print(f"  Targets: {self.targets_file}")
        
        print("\n" + "="*60)
    
    def run(self):
        """Main execution method."""
        print("🚀 MASS WATCH SCRAPER")
        print("Targeting 100 luxury watches across 10 brands")
        print("=" * 50)
        
        try:
            # Load or discover targets
            targets = self.load_or_discover_targets()
            print(f"🎯 Loaded {len(targets)} watch targets")
            
            # Execute scraping with progress tracking
            progress = self.scrape_with_progress(targets)
            
            # Print final report
            self.print_final_report(progress, targets)
            
            # Check if we should retry failed ones
            failed_count = len(progress.get('failed', []))
            if failed_count > 0:
                print(f"\n⚠️  {failed_count} watches failed to scrape.")
                retry = input("Would you like to retry failed watches? (y/n): ").lower().strip()
                
                if retry == 'y':
                    logger.info("Retrying failed watches...")
                    
                    # Reset failed watches to pending (clear failed list)
                    failed_count = len(progress.get('failed', []))
                    progress['failed'] = []
                    self.save_progress(progress)
                    
                    logger.info(f"Reset {failed_count} failed watches to retry")
                    
                    # Retry scraping - scrape_with_progress will automatically pick up the reset watches
                    progress = self.scrape_with_progress(targets)
                    self.print_final_report(progress, targets)
        
        except KeyboardInterrupt:
            print("\n⚠️  Scraping interrupted by user")
            print("Progress has been saved. Run the script again to resume.")
        
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            print("Check the progress file to see what was completed.")


def main():
    """Entry point for mass scraping."""
    scraper = MassWatchScraper()
    scraper.run()


if __name__ == "__main__":
    main()