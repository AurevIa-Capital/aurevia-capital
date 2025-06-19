#!/usr/bin/env python3
"""
CSV Data Validator - Check scraped CSV files for insufficient data points.

This script validates CSV files in the data/scrape/prices directory and identifies
files with fewer than the minimum required rows for reliable forecasting.
"""

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class CSVDataValidator:
    """Validates CSV files for sufficient data points."""

    def __init__(
        self,
        data_dir: str = "data/scrape/prices",
        min_rows: int = 90,
        log_dir: str = "logs",
    ):
        """
        Initialize validator with configurable parameters.
        
        Args:
            data_dir: Directory containing CSV files to validate
            min_rows: Minimum number of rows required
            log_dir: Directory for log files
        """
        self.data_dir = Path(data_dir)
        self.min_rows = min_rows
        self.log_dir = Path(log_dir)
        
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"csv_validation_{self.timestamp}.log"
        
        # Ensure data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.setup_file_logging()

    def setup_file_logging(self) -> None:
        """Setup logging to both console and file."""
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    def find_csv_files(self) -> List[Path]:
        """Find all CSV files in the data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        return csv_files

    def validate_csv_file(self, csv_file: Path) -> Tuple[bool, int, str, bool]:
        """
        Validate a single CSV file.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            Tuple of (is_valid, row_count, error_message, is_date_ascending)
        """
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if len(rows) < 2:  # Less than header + 1 data row
                    return False, 0, "Empty or header-only file", True
                
                row_count = len(rows) - 1  # Subtract 1 for header
                
                # Check date ordering if we have enough rows
                is_date_ascending = True
                date_error = ""
                
                if row_count >= 2:
                    try:
                        # Assume first column is date
                        first_date = rows[1][0]  # First data row
                        last_date = rows[-1][0]  # Last data row
                        
                        # Try to parse dates in common formats
                        from datetime import datetime
                        
                        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
                        first_parsed = None
                        last_parsed = None
                        
                        for fmt in date_formats:
                            try:
                                first_parsed = datetime.strptime(first_date, fmt)
                                last_parsed = datetime.strptime(last_date, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if first_parsed and last_parsed:
                            is_date_ascending = first_parsed <= last_parsed
                            if not is_date_ascending:
                                date_error = " (DESCENDING ORDER)"
                        
                    except Exception:
                        # If date parsing fails, assume it's okay
                        pass
            
            # Combine validation checks
            has_sufficient_rows = row_count >= self.min_rows
            
            if not has_sufficient_rows and not is_date_ascending:
                error_msg = f"Insufficient data: {row_count} < {self.min_rows}{date_error}"
            elif not has_sufficient_rows:
                error_msg = f"Insufficient data: {row_count} < {self.min_rows}"
            elif not is_date_ascending:
                error_msg = f"Date ordering issue{date_error}"
            else:
                error_msg = ""
            
            is_valid = has_sufficient_rows and is_date_ascending
            
            return is_valid, row_count, error_msg, is_date_ascending
            
        except Exception as e:
            return False, 0, f"Error reading file: {str(e)}", True

    def extract_watch_info(self, filename: str) -> Dict[str, str]:
        """
        Extract watch information from filename.
        
        Expected format: {Brand}-{Model}-{ID}.csv
        """
        try:
            # Remove .csv extension
            name = filename.replace('.csv', '')
            
            # Split by hyphens to extract components
            parts = name.split('-')
            
            if len(parts) >= 3:
                brand = parts[0]
                # ID is typically the last part
                watch_id = parts[-1]
                # Model is everything in between
                model = '-'.join(parts[1:-1])
                
                return {
                    'brand': brand,
                    'model': model,
                    'watch_id': watch_id,
                    'filename': filename
                }
            else:
                return {
                    'brand': 'Unknown',
                    'model': name,
                    'watch_id': 'Unknown',
                    'filename': filename
                }
                
        except Exception:
            return {
                'brand': 'Unknown',
                'model': filename,
                'watch_id': 'Unknown',
                'filename': filename
            }

    def validate_all_files(self) -> Dict:
        """
        Validate all CSV files and return comprehensive results.
        
        Returns:
            Dictionary with validation results and statistics
        """
        csv_files = self.find_csv_files()
        
        if not csv_files:
            logger.warning("No CSV files found to validate!")
            return {
                'total_files': 0,
                'valid_files': [],
                'invalid_files': [],
                'error_files': [],
                'date_order_issues': []
            }
        
        valid_files = []
        invalid_files = []
        error_files = []
        date_order_issues = []
        
        logger.info(f"Validating CSV files (minimum {self.min_rows} rows required, ascending date order)...")
        logger.info("=" * 80)
        
        for csv_file in sorted(csv_files):
            is_valid, row_count, error_msg, is_date_ascending = self.validate_csv_file(csv_file)
            watch_info = self.extract_watch_info(csv_file.name)
            
            file_info = {
                'file': csv_file.name,
                'path': str(csv_file),
                'row_count': row_count,
                'brand': watch_info['brand'],
                'model': watch_info['model'],
                'watch_id': watch_info['watch_id'],
                'error': error_msg,
                'is_date_ascending': is_date_ascending
            }
            
            if error_msg and "Error reading file" in error_msg:
                error_files.append(file_info)
                logger.error(f"❌ ERROR: {csv_file.name} - {error_msg}")
            elif is_valid:
                valid_files.append(file_info)
                logger.info(f"✅ VALID: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']}")
            else:
                invalid_files.append(file_info)
                if not is_date_ascending:
                    date_order_issues.append(file_info)
                    logger.warning(f"📅 DATE ORDER: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']} - {error_msg}")
                else:
                    logger.warning(f"⚠️  INSUFFICIENT: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']}")
        
        return {
            'total_files': len(csv_files),
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'error_files': error_files,
            'date_order_issues': date_order_issues
        }

    def print_summary_report(self, results: Dict) -> None:
        """Print comprehensive validation summary."""
        total = results['total_files']
        valid_count = len(results['valid_files'])
        invalid_count = len(results['invalid_files'])
        error_count = len(results['error_files'])
        date_order_count = len(results['date_order_issues'])
        
        logger.info("=" * 80)
        logger.info("📊 CSV VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total CSV files: {total}")
        logger.info(f"✅ Valid files (≥{self.min_rows} rows, ascending dates): {valid_count}")
        logger.info(f"⚠️  Invalid files: {invalid_count}")
        logger.info(f"   📅 Date order issues (descending): {date_order_count}")
        logger.info(f"   📊 Insufficient data (<{self.min_rows} rows): {invalid_count - date_order_count}")
        logger.info(f"❌ Error files: {error_count}")
        
        if total > 0:
            success_rate = valid_count / total * 100
            logger.info(f"📈 Success rate: {success_rate:.1f}%")
        
        # Date order issues
        if date_order_count > 0:
            logger.info(f"\n📅 FILES WITH DATE ORDER ISSUES (DESCENDING ORDER):")
            logger.info("-" * 80)
            
            brand_groups = {}
            for file_info in results['date_order_issues']:
                brand = file_info['brand']
                if brand not in brand_groups:
                    brand_groups[brand] = []
                brand_groups[brand].append(file_info)
            
            for brand, files in sorted(brand_groups.items()):
                logger.info(f"\n{brand}:")
                for file_info in sorted(files, key=lambda x: x['row_count']):
                    logger.info(f"  • {file_info['model']} (ID: {file_info['watch_id']}) - {file_info['row_count']} rows - DESCENDING ORDER")
        
        # Insufficient data files (excluding date order issues)
        insufficient_files = [f for f in results['invalid_files'] if f['is_date_ascending']]
        if insufficient_files:
            logger.info(f"\n⚠️  FILES WITH INSUFFICIENT DATA (<{self.min_rows} rows):")
            logger.info("-" * 80)
            
            brand_groups = {}
            for file_info in insufficient_files:
                brand = file_info['brand']
                if brand not in brand_groups:
                    brand_groups[brand] = []
                brand_groups[brand].append(file_info)
            
            for brand, files in sorted(brand_groups.items()):
                logger.info(f"\n{brand}:")
                for file_info in sorted(files, key=lambda x: x['row_count']):
                    logger.info(f"  • {file_info['model']} (ID: {file_info['watch_id']}) - {file_info['row_count']} rows")
        
        # Error files details
        if error_count > 0:
            logger.info(f"\n❌ FILES WITH ERRORS:")
            logger.info("-" * 80)
            for file_info in results['error_files']:
                logger.info(f"  • {file_info['file']} - {file_info['error']}")
        
        logger.info(f"\n📁 Validation log saved to: {self.log_file}")

    def get_files_needing_rescrape(self, results: Dict) -> List[Dict[str, str]]:
        """
        Get detailed list of watches that need to be rescraped.
        
        Returns:
            List of dictionaries with full watch details
        """
        rescrape_list = []
        for file_info in results['invalid_files']:
            watch_detail = {
                'brand': file_info['brand'],
                'model': file_info['model'], 
                'watch_id': file_info['watch_id'],
                'filename': file_info['file'],
                'issue': file_info['error'],
                'row_count': file_info['row_count']
            }
            rescrape_list.append(watch_detail)
        
        return rescrape_list

    def run_validation(self) -> Dict:
        """
        Run complete validation process.
        
        Returns:
            Validation results dictionary
        """
        logger.info("🔍 CSV DATA VALIDATION STARTED")
        logger.info(f"Directory: {self.data_dir}")
        logger.info(f"Minimum rows required: {self.min_rows}")
        logger.info(f"Log file: {self.log_file}")
        
        try:
            results = self.validate_all_files()
            self.print_summary_report(results)
            
            # Save detailed watch info needing rescrape
            rescrape_details = self.get_files_needing_rescrape(results)
            if rescrape_details:
                rescrape_file = self.data_dir.parent / f"rescrape_needed_{self.timestamp}.txt"
                with open(rescrape_file, 'w', encoding='utf-8') as f:
                    f.write("# Watches needing rescrape\n")
                    f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("# Format: Brand | Model | Watch ID | Issue | Current Rows\n")
                    f.write("# " + "="*70 + "\n\n")
                    
                    for watch in rescrape_details:
                        f.write(f"{watch['brand']} | {watch['model']} | {watch['watch_id']} | {watch['issue']} | {watch['row_count']} rows\n")
                
                # Also save JSON format for programmatic use
                import json
                rescrape_json = self.data_dir.parent / f"rescrape_needed_{self.timestamp}.json"
                with open(rescrape_json, 'w', encoding='utf-8') as f:
                    json.dump(rescrape_details, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📝 Detailed rescrape info saved to: {rescrape_file}")
                logger.info(f"📝 JSON rescrape data saved to: {rescrape_json}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise


def main():
    """CLI entry point for CSV validation."""
    try:
        validator = CSVDataValidator()
        results = validator.run_validation()
        
        # Exit with appropriate code
        invalid_count = len(results['invalid_files'])
        error_count = len(results['error_files'])
        
        if error_count > 0:
            exit(2)  # Critical errors
        elif invalid_count > 0:
            exit(1)  # Warnings (insufficient data)
        else:
            exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        exit(130)
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()