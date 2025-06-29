"""
Multi-Horizon Training Script

This script trains models for different prediction horizons
(1-day, 3-day, 7-day, 14-day, 30-day ahead).

Usage:
    python -m src.ml.multi_horizon_training
    python -m src.ml.multi_horizon_training --horizons 1 7 30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from src.ml.run_training import run_training
from src.pipeline import PipelineConfig, run_pipeline
from src.utils.logging_config import get_training_logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent

# Setup centralized logging
logger = get_training_logger()


def get_csv_file_count(data_dir: str = "data/scrape/prices") -> int:
    """Count the number of CSV files in the data directory."""
    csv_files = list(Path(data_dir).glob("*.csv"))
    return len(csv_files)


def train_multiple_horizons(
    horizons: List[int] = [1, 3, 7, 14, 30],
    models: List[str] = ["linear", "ridge", "xgboost"],
    max_assets: int = None,
    base_output_dir: str = "data/output",
) -> Dict:
    """
    Train models for multiple prediction horizons.

    Parameters:
    ----------
    horizons : List[int]
        List of prediction horizons in days
    models : List[str]
        Models to train
    max_assets : int, optional
        Maximum assets to train on. If None, trains on all available assets.
    base_output_dir : str
        Base output directory

    Returns:
    -------
    Dict
        Results for all horizons
    """

    # Auto-detect number of assets if not specified
    if max_assets is None:
        max_assets = get_csv_file_count()
        logger.info(f"Auto-detected {max_assets} CSV files for training")
    
    results_by_horizon = {}

    for horizon in horizons:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TRAINING MODELS FOR {horizon}-DAY HORIZON")
        logger.info(f"{'=' * 60}")

        # Step 1: Prepare data with custom target shift
        config = PipelineConfig()
        config.features.target_shift = -horizon  # Negative for future prediction

        logger.info(f"Preparing data for {horizon}-day ahead prediction...")

        pipeline_results = run_pipeline(
            asset_type="watch",
            config=config,
            max_files=None,  # Process all files
            enable_feature_selection=True,
        )

        if not pipeline_results.get("success", False):
            logger.error(f"Data preparation failed for horizon {horizon}")
            results_by_horizon[f"{horizon}_day"] = {
                "success": False,
                "error": "Data preparation failed",
            }
            continue

        # Step 2: Train models
        output_path = f"{base_output_dir}/models_{horizon}_day"

        logger.info(f"Training models for {horizon}-day horizon...")

        training_results = run_training(
            models=models,
            max_assets=max_assets,
            tune_hyperparams=False,  # Skip tuning for speed
            data_path=base_output_dir,
            output_path=output_path,
            target_column="target",
        )

        results_by_horizon[f"{horizon}_day"] = training_results

        # Log summary for this horizon
        if training_results.get("success"):
            summary = training_results.get("summary", {})
            success_rate = summary.get("success_rate", 0)
            logger.info(
                f"✅ {horizon}-day horizon: {success_rate:.1%} models successful"
            )
        else:
            logger.error(f"❌ {horizon}-day horizon: Training failed")

    # Step 3: Compare results across horizons
    logger.info(f"\n{'=' * 60}")
    logger.info("MULTI-HORIZON COMPARISON")
    logger.info(f"{'=' * 60}")

    comparison_data = []

    for horizon_key, results in results_by_horizon.items():
        if results.get("success"):
            summary = results.get("summary", {})
            horizon_days = horizon_key.replace("_day", "")

            comparison_data.append(
                {
                    "horizon_days": horizon_days,
                    "success_rate": summary.get("success_rate", 0),
                    "total_models": summary.get("total_models", 0),
                    "successful_models": summary.get("successful_models", 0),
                }
            )

    # Save comparison
    comparison_file = Path(base_output_dir) / "multi_horizon_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(
            {
                "horizons_trained": horizons,
                "models_used": models,
                "comparison": comparison_data,
                "detailed_results": results_by_horizon,
            },
            f,
            indent=2,
            default=str,
        )

    logger.info(f"Multi-horizon comparison saved to: {comparison_file}")

    # Display summary
    logger.info("\nHorizon Performance Summary:")
    logger.info("Horizon | Success Rate | Models")
    logger.info("--------|--------------|-------")

    for item in comparison_data:
        horizon = item["horizon_days"]
        success_rate = item["success_rate"]
        successful = item["successful_models"]
        total = item["total_models"]
        logger.info(f"{horizon:>7} | {success_rate:>11.1%} | {successful}/{total}")

    return results_by_horizon


def analyze_horizon_performance(
    results_file: str = "data/output/multi_horizon_comparison.json",
):
    """
    Analyze and display performance across different horizons.

    Parameters:
    ----------
    results_file : str
        Path to results file
    """

    results_path = Path(results_file)

    if not results_path.exists():
        logger.error(f"Results file not found: {results_file}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    logger.info("HORIZON ANALYSIS")
    logger.info("=" * 50)

    comparison = data.get("comparison", [])

    if not comparison:
        logger.warning("No comparison data found")
        return

    # Find best and worst performing horizons
    best_horizon = max(comparison, key=lambda x: x["success_rate"])
    worst_horizon = min(comparison, key=lambda x: x["success_rate"])

    logger.info(
        f"Best performing horizon: {best_horizon['horizon_days']} days ({best_horizon['success_rate']:.1%})"
    )
    logger.info(
        f"Worst performing horizon: {worst_horizon['horizon_days']} days ({worst_horizon['success_rate']:.1%})"
    )

    # Show trend
    logger.info("\nPerformance by horizon:")
    for item in sorted(comparison, key=lambda x: int(x["horizon_days"])):
        horizon = item["horizon_days"]
        success_rate = item["success_rate"]
        logger.info(f"  {horizon} days: {success_rate:.1%}")


def main():
    """Main entry point."""
    
    # Use global logger (already configured)
    global logger

    parser = argparse.ArgumentParser(
        description="Train models for multiple prediction horizons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ml.multi_horizon_training
  python -m src.ml.multi_horizon_training --horizons 1 7 30
  python -m src.ml.multi_horizon_training --models linear xgboost --max-assets 3
  python -m src.ml.multi_horizon_training --analyze-only
        """,
    )

    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 3, 7, 14],
        help="Prediction horizons in days (default: 1 3 7 14)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "ridge", "xgboost"],
        help="Models to train (default: linear ridge xgboost)",
    )

    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Maximum assets to train on (default: all available files)",
    )

    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Base output directory (default: data/output)",
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results without training",
    )

    args = parser.parse_args()

    if args.analyze_only:
        analyze_horizon_performance()
        return

    try:
        logger.info("MULTI-HORIZON MODEL TRAINING")
        logger.info(f"Horizons: {args.horizons} days")
        logger.info(f"Models: {args.models}")
        logger.info(f"Max assets: {args.max_assets}")

        results = train_multiple_horizons(
            horizons=args.horizons,
            models=args.models,
            max_assets=args.max_assets,
            base_output_dir=args.output_dir,
        )

        logger.info("Multi-horizon training completed!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
