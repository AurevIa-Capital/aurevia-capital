#!/usr/bin/env python3
"""
CSV Data Validator - Check scraped CSV files for insufficient data points.

This script validates CSV files in the data/scrape/prices directory and identifies
files with fewer than the minimum required rows for reliable forecasting.
"""

import csv
import logging
import shutil
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
        move_invalid: bool = False,
    ):
        """
        Initialize validator with configurable parameters.

        Args:
            data_dir: Directory containing CSV files to validate
            min_rows: Minimum number of rows required
            log_dir: Directory for log files
            move_invalid: Whether to move invalid files to kiv directory
        """
        self.data_dir = Path(data_dir)
        self.min_rows = min_rows
        self.log_dir = Path(log_dir)
        self.move_invalid = move_invalid

        # Set up kiv directory for invalid files
        self.kiv_dir = self.data_dir / "kiv"

        # Generate timestamp for unique session folder
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"csv_validation_{self.timestamp}"
        self.log_file = self.session_dir / "validation.log"

        # Ensure data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Create kiv directory if move_invalid is enabled
        if self.move_invalid:
            self.kiv_dir.mkdir(parents=True, exist_ok=True)

        # Ensure session directory exists
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        self.setup_file_logging()

    def setup_file_logging(self) -> None:
        """Setup logging to both console and file."""
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Create console handler with UTF-8 encoding for Windows compatibility
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Set console encoding to UTF-8 if possible (Windows compatibility)
        import sys

        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8")
                sys.stderr.reconfigure(encoding="utf-8")
            except Exception:
                pass  # Fallback to default encoding

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Configure logger
        logger.setLevel(logging.INFO)
        # Clear any existing handlers
        logger.handlers.clear()
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
            with open(csv_file, "r", encoding="utf-8") as f:
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

                        date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
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
                error_msg = (
                    f"Insufficient data: {row_count} < {self.min_rows}{date_error}"
                )
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
            name = filename.replace(".csv", "")

            # Split by hyphens to extract components
            parts = name.split("-")

            if len(parts) >= 3:
                brand = parts[0]
                # ID is typically the last part
                watch_id = parts[-1]
                # Model is everything in between
                model = "-".join(parts[1:-1])

                return {
                    "brand": brand,
                    "model": model,
                    "watch_id": watch_id,
                    "filename": filename,
                }
            else:
                return {
                    "brand": "Unknown",
                    "model": name,
                    "watch_id": "Unknown",
                    "filename": filename,
                }

        except Exception:
            return {
                "brand": "Unknown",
                "model": filename,
                "watch_id": "Unknown",
                "filename": filename,
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
                "total_files": 0,
                "valid_files": [],
                "invalid_files": [],
                "error_files": [],
                "date_order_issues": [],
            }

        valid_files = []
        invalid_files = []
        error_files = []
        date_order_issues = []

        logger.info(
            f"Validating CSV files (minimum {self.min_rows} rows required, ascending date order)..."
        )
        logger.info("=" * 80)

        for csv_file in sorted(csv_files):
            is_valid, row_count, error_msg, is_date_ascending = self.validate_csv_file(
                csv_file
            )
            watch_info = self.extract_watch_info(csv_file.name)

            file_info = {
                "file": csv_file.name,
                "path": str(csv_file),
                "row_count": row_count,
                "brand": watch_info["brand"],
                "model": watch_info["model"],
                "watch_id": watch_info["watch_id"],
                "error": error_msg,
                "is_date_ascending": is_date_ascending,
            }

            if error_msg and "Error reading file" in error_msg:
                error_files.append(file_info)
                logger.error(f"[ERROR] ERROR: {csv_file.name} - {error_msg}")
            elif is_valid:
                valid_files.append(file_info)
                logger.info(
                    f"[OK] VALID: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']}"
                )
            else:
                invalid_files.append(file_info)
                if not is_date_ascending:
                    date_order_issues.append(file_info)
                    logger.warning(
                        f"[DATE] DATE ORDER: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']} - {error_msg}"
                    )
                else:
                    logger.warning(
                        f"[WARN] INSUFFICIENT: {csv_file.name} ({row_count} rows) - {watch_info['brand']} {watch_info['model']}"
                    )

        return {
            "total_files": len(csv_files),
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "error_files": error_files,
            "date_order_issues": date_order_issues,
        }

    def print_summary_report(self, results: Dict) -> None:
        """Print comprehensive validation summary."""
        total = results["total_files"]
        valid_count = len(results["valid_files"])
        invalid_count = len(results["invalid_files"])
        error_count = len(results["error_files"])
        date_order_count = len(results["date_order_issues"])
        move_stats = results.get("move_stats", {"moved": 0, "failed": 0})

        logger.info("=" * 80)
        logger.info("[STATS] CSV VALIDATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total CSV files: {total}")
        logger.info(
            f"[OK] Valid files (>={self.min_rows} rows, ascending dates): {valid_count}"
        )
        logger.info(f"[WARN] Invalid files: {invalid_count}")
        logger.info(f"   [DATE] Date order issues (descending): {date_order_count}")
        logger.info(
            f"   [STATS] Insufficient data (<{self.min_rows} rows): {invalid_count - date_order_count}"
        )
        logger.info(f"[ERROR] Error files: {error_count}")

        if total > 0:
            success_rate = valid_count / total * 100
            logger.info(f"[CHART] Success rate: {success_rate:.1f}%")

        # File movement statistics
        if self.move_invalid and (move_stats["moved"] > 0 or move_stats["failed"] > 0):
            logger.info("\n[FOLDER] FILE MOVEMENT SUMMARY:")
            logger.info("-" * 80)
            logger.info(f"[FOLDER] Files moved to kiv/: {move_stats['moved']}")
            if move_stats["failed"] > 0:
                logger.info(f"[ERROR] Failed to move: {move_stats['failed']}")

        # Only show detailed file lists if files weren't moved
        if not self.move_invalid:
            # Date order issues
            if date_order_count > 0:
                logger.info("\n[DATE] FILES WITH DATE ORDER ISSUES (DESCENDING ORDER):")
                logger.info("-" * 80)

                brand_groups = {}
                for file_info in results["date_order_issues"]:
                    brand = file_info["brand"]
                    if brand not in brand_groups:
                        brand_groups[brand] = []
                    brand_groups[brand].append(file_info)

                for brand, files in sorted(brand_groups.items()):
                    logger.info(f"\n{brand}:")
                    for file_info in sorted(files, key=lambda x: x["row_count"]):
                        logger.info(
                            f"  * {file_info['model']} (ID: {file_info['watch_id']}) - {file_info['row_count']} rows - DESCENDING ORDER"
                        )

            # Insufficient data files (excluding date order issues)
            insufficient_files = [
                f for f in results["invalid_files"] if f["is_date_ascending"]
            ]
            if insufficient_files:
                logger.info(
                    f"\n[WARN] FILES WITH INSUFFICIENT DATA (<{self.min_rows} rows):"
                )
                logger.info("-" * 80)

                brand_groups = {}
                for file_info in insufficient_files:
                    brand = file_info["brand"]
                    if brand not in brand_groups:
                        brand_groups[brand] = []
                    brand_groups[brand].append(file_info)

                for brand, files in sorted(brand_groups.items()):
                    logger.info(f"\n{brand}:")
                    for file_info in sorted(files, key=lambda x: x["row_count"]):
                        logger.info(
                            f"  * {file_info['model']} (ID: {file_info['watch_id']}) - {file_info['row_count']} rows"
                        )

            # Error files details
            if error_count > 0:
                logger.info("\n[ERROR] FILES WITH ERRORS:")
                logger.info("-" * 80)
                for file_info in results["error_files"]:
                    logger.info(f"  * {file_info['file']} - {file_info['error']}")
        else:
            # If files were moved, just show they were moved to kiv/
            if invalid_count > 0 or error_count > 0:
                logger.info(
                    f"\n[FOLDER] Invalid files have been moved to: {self.kiv_dir}"
                )

        logger.info(f"\n[FOLDER] Session files saved to: {self.session_dir}")

    def get_files_needing_rescrape(self, results: Dict) -> List[Dict[str, str]]:
        """
        Get detailed list of watches that need to be rescraped.

        Returns:
            List of dictionaries with full watch details
        """
        rescrape_list = []
        for file_info in results["invalid_files"]:
            watch_detail = {
                "brand": file_info["brand"],
                "model": file_info["model"],
                "watch_id": file_info["watch_id"],
                "filename": file_info["file"],
                "issue": file_info["error"],
                "row_count": file_info["row_count"],
            }
            rescrape_list.append(watch_detail)

        return rescrape_list

    def move_invalid_file(self, file_path: Path, reason: str) -> bool:
        """
        Move an invalid file to the kiv directory.

        Args:
            file_path: Path to the file to move
            reason: Reason for moving the file

        Returns:
            True if moved successfully, False otherwise
        """
        if not self.move_invalid:
            return False

        try:
            # Create destination path
            dest_path = self.kiv_dir / file_path.name

            # Handle filename conflicts by adding timestamp
            if dest_path.exists():
                stem = dest_path.stem
                suffix = dest_path.suffix
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = self.kiv_dir / f"{stem}_{timestamp}{suffix}"

            # Move the file
            shutil.move(str(file_path), str(dest_path))
            logger.info(f"[FOLDER] MOVED: {file_path.name} -> kiv/ ({reason})")

            # Create a metadata file alongside it
            metadata_path = dest_path.with_suffix(".info")
            with open(metadata_path, "w", encoding="utf-8") as f:
                f.write(f"Original file: {file_path.name}\n")
                f.write(f"Moved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Reason: {reason}\n")
                f.write(f"Validation session: {self.timestamp}\n")

            return True

        except Exception as e:
            logger.error(f"[ERROR] Failed to move {file_path.name} to kiv: {e}")
            return False

    def move_invalid_files(self, results: Dict) -> Dict[str, int]:
        """
        Move all invalid files to kiv directory.

        Args:
            results: Validation results dictionary

        Returns:
            Dictionary with move statistics
        """
        if not self.move_invalid:
            return {"moved": 0, "failed": 0}

        moved_count = 0
        failed_count = 0

        logger.info("\n[FOLDER] MOVING INVALID FILES TO KIV DIRECTORY")
        logger.info("-" * 80)

        # Move invalid files (insufficient data or date order issues)
        for file_info in results["invalid_files"]:
            file_path = Path(file_info["path"])
            reason = file_info["error"]

            if self.move_invalid_file(file_path, reason):
                moved_count += 1
            else:
                failed_count += 1

        # Move error files (corrupted/unreadable)
        for file_info in results["error_files"]:
            file_path = Path(file_info["path"])
            reason = file_info["error"]

            if self.move_invalid_file(file_path, reason):
                moved_count += 1
            else:
                failed_count += 1

        if moved_count > 0:
            logger.info(
                f"[OK] Successfully moved {moved_count} files to {self.kiv_dir}"
            )
        if failed_count > 0:
            logger.warning(f"[WARN] Failed to move {failed_count} files")

        return {"moved": moved_count, "failed": failed_count}

    def run_validation(self) -> Dict:
        """
        Run complete validation process.

        Returns:
            Validation results dictionary
        """
        logger.info("[SEARCH] CSV DATA VALIDATION STARTED")
        logger.info(f"Directory: {self.data_dir}")
        logger.info(f"Minimum rows required: {self.min_rows}")
        logger.info(f"Move invalid files: {self.move_invalid}")
        if self.move_invalid:
            logger.info(f"KIV directory: {self.kiv_dir}")
        logger.info(f"Log file: {self.log_file}")

        try:
            results = self.validate_all_files()

            # Move invalid files if enabled
            move_stats = {"moved": 0, "failed": 0}
            if self.move_invalid:
                move_stats = self.move_invalid_files(results)
                # Update results to reflect moved files
                results["move_stats"] = move_stats

            self.print_summary_report(results)

            # Save detailed watch info needing rescrape (only if files weren't moved)
            rescrape_details = self.get_files_needing_rescrape(results)
            if rescrape_details and not self.move_invalid:
                rescrape_file = self.session_dir / "rescrape_needed.txt"
                with open(rescrape_file, "w", encoding="utf-8") as f:
                    f.write("# Watches needing rescrape\n")
                    f.write(
                        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(
                        "# Format: Brand | Model | Watch ID | Issue | Current Rows\n"
                    )
                    f.write("# " + "=" * 70 + "\n\n")

                    for watch in rescrape_details:
                        f.write(
                            f"{watch['brand']} | {watch['model']} | {watch['watch_id']} | {watch['issue']} | {watch['row_count']} rows\n"
                        )

                # Also save JSON format for programmatic use
                import json

                rescrape_json = self.session_dir / "rescrape_needed.json"
                with open(rescrape_json, "w", encoding="utf-8") as f:
                    json.dump(rescrape_details, f, indent=2, ensure_ascii=False)

                logger.info("[NOTE] Rescrape files saved to session directory")
                logger.info("   * Text format: rescrape_needed.txt")
                logger.info("   * JSON format: rescrape_needed.json")
            elif self.move_invalid and rescrape_details:
                logger.info(
                    "[NOTE] Invalid files moved to kiv/ - no rescrape list generated"
                )

            return results

        except Exception as e:
            logger.error(f"[ERROR] Validation failed: {e}")
            raise


def main():
    """CLI entry point for CSV validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate CSV files and optionally move invalid ones to kiv directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.collectors.watch.csv_validator                    # Validate only
  python -m src.collectors.watch.csv_validator --move             # Validate and move invalid files
  python -m src.collectors.watch.csv_validator --min-rows 50      # Custom minimum rows
  python -m src.collectors.watch.csv_validator --data-dir data/test --move  # Custom directory
        """,
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Move invalid files to prices/kiv/ directory",
    )

    parser.add_argument(
        "--min-rows",
        type=int,
        default=90,
        help="Minimum number of rows required (default: 90)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/scrape/prices",
        help="Directory containing CSV files to validate (default: data/scrape/prices)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files (default: logs)",
    )

    args = parser.parse_args()

    try:
        validator = CSVDataValidator(
            data_dir=args.data_dir,
            min_rows=args.min_rows,
            log_dir=args.log_dir,
            move_invalid=args.move,
        )
        results = validator.run_validation()

        # Exit with appropriate code
        invalid_count = len(results["invalid_files"])
        error_count = len(results["error_files"])

        if error_count > 0:
            exit(2)  # Critical errors
        elif invalid_count > 0 and not args.move:
            exit(1)  # Warnings (insufficient data) - only if files weren't moved
        else:
            exit(0)  # Success

    except KeyboardInterrupt:
        print("\n[WARN] Validation interrupted by user")
        exit(130)
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
