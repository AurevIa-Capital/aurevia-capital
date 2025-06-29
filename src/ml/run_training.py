"""
ML Model Training Runner

This script runs the complete model training pipeline using data
prepared by the data pipeline. Run after data preparation is complete.

Usage:
    python -m src.ml.run_training
    python -m src.ml.run_training --models linear ridge xgboost
    python -m src.ml.run_training --max-assets 5 --tune-hyperparams
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.ml.base import create_model_factory
from src.ml.training.trainer import ModelTrainer
from src.ml.training.tuner import HyperparameterTuner
from src.pipeline.config import PipelineConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent


from src.utils.logging_config import get_training_logger

logger = get_training_logger()


def load_featured_data(data_path: str = "data/output") -> Dict:
    """
    Load featured data from the data pipeline output.

    Parameters:
    ----------
    data_path : str
        Path to the data directory

    Returns:
    -------
    Dict
        Dictionary containing featured data by asset
    """
    data_dir = Path(data_path)
    featured_data_file = data_dir / "featured_data.csv"

    if not featured_data_file.exists():
        raise FileNotFoundError(
            f"Featured data file not found: {featured_data_file}\n"
            "Please run the data pipeline first: python -m src.pipeline.run_pipeline"
        )

    logger.info(f"Loading featured data from {featured_data_file}")

    # Load the combined featured data
    df = pd.read_csv(featured_data_file)

    if "asset_id" not in df.columns:
        raise ValueError("Featured data must contain 'asset_id' column")

    # Split back into individual assets
    featured_data = {}
    for asset_id in df["asset_id"].unique():
        asset_df = df[df["asset_id"] == asset_id].copy()
        asset_df = asset_df.drop("asset_id", axis=1)

        # Set index if it exists
        if "date" in asset_df.columns:
            asset_df["date"] = pd.to_datetime(asset_df["date"])
            asset_df.set_index("date", inplace=True)
        elif asset_df.index.name != "date":
            # Try to infer datetime index, handling epoch times
            try:
                # Check if index looks like epoch time (very large numbers)
                if asset_df.index.dtype in ["int64", "float64"]:
                    # Convert from nanoseconds to datetime if values are very large
                    if asset_df.index.max() > 1e10:  # Likely nanoseconds since epoch
                        asset_df.index = pd.to_datetime(asset_df.index, unit="ns")
                    elif asset_df.index.max() > 1e6:  # Likely seconds since epoch
                        asset_df.index = pd.to_datetime(asset_df.index, unit="s")
                    else:
                        # Try direct conversion
                        asset_df.index = pd.to_datetime(asset_df.index)
                else:
                    asset_df.index = pd.to_datetime(asset_df.index)
            except:
                # If datetime conversion fails, use range index
                logger.warning(
                    f"Could not convert index to datetime for {asset_id}, using range index"
                )
                asset_df.reset_index(drop=True, inplace=True)

        featured_data[asset_id] = asset_df

    logger.info(f"Loaded {len(featured_data)} assets with featured data")

    # Log data info
    for asset_id, df in list(featured_data.items())[:3]:  # Show first 3
        logger.info(f"  {asset_id}: {len(df)} records, {len(df.columns)} features")

    if len(featured_data) > 3:
        logger.info(f"  ... and {len(featured_data) - 3} more assets")

    return featured_data


def get_available_models() -> List[str]:
    """Get list of available model names."""
    config = PipelineConfig()
    model_factory = create_model_factory(config)
    return list(model_factory.keys())


def run_training(
    models: List[str] = None,
    max_assets: Optional[int] = None,
    tune_hyperparams: bool = False,
    data_path: str = "data/output",
    output_path: str = "data/output/models",
    target_column: str = "target",
) -> Dict:
    """
    Run the complete model training pipeline.

    Parameters:
    ----------
    models : List[str], optional
        List of models to train
    max_assets : int, optional
        Maximum number of assets to train on
    tune_hyperparams : bool
        Whether to perform hyperparameter tuning
    data_path : str
        Path to featured data
    output_path : str
        Path to save trained models
    target_column : str
        Name of target column

    Returns:
    -------
    Dict
        Training results
    """

    # Step 1: Load configuration and data
    logger.info("=" * 60)
    logger.info("ML MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    config = PipelineConfig()

    try:
        featured_data = load_featured_data(data_path)
    except Exception as e:
        logger.error(f"Failed to load featured data: {str(e)}")
        return {"success": False, "error": str(e)}

    if not featured_data:
        logger.error("No featured data available for training")
        return {"success": False, "error": "No data available"}

    # Step 2: Validate models
    available_models = get_available_models()

    if models is None:
        # Default model selection - exclude slow models by default
        models = ["linear", "ridge", "random_forest", "xgboost"]
        logger.info(f"Using default models: {models}")
    else:
        # Validate requested models
        invalid_models = [m for m in models if m not in available_models]
        if invalid_models:
            logger.error(f"Invalid models: {invalid_models}")
            logger.info(f"Available models: {available_models}")
            return {"success": False, "error": f"Invalid models: {invalid_models}"}

    logger.info(f"Training models: {models}")

    # Step 3: Filter data and check target column
    valid_assets = {}
    for asset_name, df in featured_data.items():
        if target_column not in df.columns:
            logger.warning(
                f"Skipping {asset_name}: missing target column '{target_column}'"
            )
            continue

        if len(df) < 50:  # Minimum samples for meaningful training
            logger.warning(
                f"Skipping {asset_name}: insufficient data ({len(df)} samples)"
            )
            continue

        valid_assets[asset_name] = df

    if not valid_assets:
        logger.error("No valid assets found for training")
        return {"success": False, "error": "No valid assets"}

    logger.info(f"Training on {len(valid_assets)} valid assets")

    # Limit assets if requested
    if max_assets and len(valid_assets) > max_assets:
        asset_names = list(valid_assets.keys())[:max_assets]
        valid_assets = {name: valid_assets[name] for name in asset_names}
        logger.info(f"Limited to {max_assets} assets for training")

    # Step 4: Initialize trainer
    trainer = ModelTrainer(config)

    # Step 5: Hyperparameter tuning (optional)
    tuning_results = {}
    if tune_hyperparams:
        logger.info("Performing hyperparameter tuning...")

        # Use first asset for tuning
        first_asset = list(valid_assets.values())[0]
        X_train, X_val, X_test, y_train, y_val, y_test = (
            trainer.prepare_time_series_split(first_asset, target_column)
        )

        tuner = HyperparameterTuner(config)
        model_factory = create_model_factory(config)

        for model_name in models:
            if model_name in model_factory:
                logger.info(f"Tuning {model_name}...")

                tuning_result = tuner.tune_model(
                    model_name=model_name,
                    model_class=model_factory[model_name],
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    method="grid_search",
                )

                tuning_results[model_name] = tuning_result

                if tuning_result.success:
                    logger.info(
                        f"  Best {model_name} score: {tuning_result.best_score:.4f}"
                    )
                    logger.info(f"  Best params: {tuning_result.best_params}")
                else:
                    logger.warning(f"  Tuning failed for {model_name}")

    # Step 6: Train models on all assets
    logger.info("Training models on all assets...")

    # Extract best parameters from tuning if available
    model_params = {}
    for model_name, tuning_result in tuning_results.items():
        if tuning_result.success:
            model_params[model_name] = tuning_result.best_params

    training_results = trainer.train_asset_models(
        featured_data=valid_assets,
        model_names=models,
        target_column=target_column,
        model_params=model_params if model_params else None,
        max_assets=None,  # Already filtered above
    )

    # Step 7: Analyze results
    logger.info("Analyzing training results...")

    total_models = 0
    successful_models = 0
    best_models_by_asset = {}

    for asset_name, asset_results in training_results.items():
        logger.info(f"\nAsset: {asset_name}")

        asset_successful = {}
        for model_name, result in asset_results.items():
            total_models += 1

            if result.success:
                successful_models += 1
                val_metrics = (
                    result.validation_result.metrics if result.validation_result else {}
                )
                mae = val_metrics.get("mae", float("inf"))
                rmse = val_metrics.get("rmse", float("inf"))
                r2 = val_metrics.get("r2", 0)

                logger.info(
                    f"  ✓ {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
                )
                asset_successful[model_name] = result
            else:
                logger.warning(f"  ✗ {model_name}: FAILED - {result.error_message}")

        # Find best model for this asset
        if asset_successful:
            best_result = trainer.get_best_model(asset_successful, metric="mae")
            if best_result:
                best_models_by_asset[asset_name] = {
                    "model_name": best_result.model_name,
                    "mae": best_result.validation_result.metrics.get("mae", "N/A"),
                    "training_time": best_result.training_time,
                }

    # Step 8: Save results
    logger.info(f"\nSaving models and results to {output_path}...")

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trained models
    trainer.save_training_results(training_results, output_path)

    # Save training summary
    from datetime import datetime

    summary = {
        "training_date": datetime.now().isoformat(),
        "models_trained": models,
        "total_assets": len(training_results),
        "total_models": total_models,
        "successful_models": successful_models,
        "success_rate": successful_models / total_models if total_models > 0 else 0,
        "best_models_by_asset": best_models_by_asset,
        "hyperparameter_tuning": tune_hyperparams,
        "tuning_results": {
            k: {"success": v.success, "best_score": v.best_score}
            for k, v in tuning_results.items()
        }
        if tuning_results
        else {},
    }

    summary_file = output_dir / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save model comparison
    if successful_models > 0:
        comparison_data = []
        for asset_name, asset_results in training_results.items():
            for model_name, result in asset_results.items():
                if result.success and result.validation_result:
                    row = {
                        "asset_name": asset_name,
                        "model_name": model_name,
                        "training_time": result.training_time,
                        **result.validation_result.metrics,
                    }
                    comparison_data.append(row)

        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
            logger.info(f"✅ Saved model comparison ({len(comparison_data)} results)")

    # Step 9: Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Models trained: {models}")
    logger.info(f"Assets processed: {len(training_results)}")
    logger.info(
        f"Success rate: {successful_models}/{total_models} ({summary['success_rate']:.1%})"
    )

    if best_models_by_asset:
        logger.info("\nBest models by asset:")
        for asset_name, best_info in best_models_by_asset.items():
            logger.info(
                f"  {asset_name}: {best_info['model_name']} (MAE: {best_info['mae']})"
            )

    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Summary saved to: {summary_file}")

    return {
        "success": True,
        "summary": summary,
        "training_results": training_results,
        "output_path": str(output_path),
    }


def main():
    """Main entry point for the training script."""

    parser = argparse.ArgumentParser(
        description="Train ML models on watch price data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.ml.run_training
  python -m src.ml.run_training --models linear ridge xgboost
  python -m src.ml.run_training --max-assets 5 --tune-hyperparams
  python -m src.ml.run_training --data-path custom/path --target price_next_day
        """,
    )

    # Available models
    available_models = [
        "linear",
        "ridge",
        "lasso",
        "random_forest",
        "xgboost",
        "arima",
        "sarima",
    ]

    parser.add_argument(
        "--models",
        nargs="+",
        choices=available_models,
        help=f"Models to train. Available: {available_models}",
    )

    parser.add_argument(
        "--max-assets", type=int, help="Maximum number of assets to train on"
    )

    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Perform hyperparameter tuning (slower but better results)",
    )

    parser.add_argument(
        "--data-path",
        default="data/output",
        help="Path to featured data directory (default: data/output)",
    )

    parser.add_argument(
        "--output-path",
        default="data/output/models",
        help="Path to save trained models (default: data/output/models)",
    )

    parser.add_argument(
        "--target", default="target", help="Name of target column (default: target)"
    )

    parser.add_argument(
        "--list-models", action="store_true", help="List available models and exit"
    )

    args = parser.parse_args()

    # List models and exit
    if args.list_models:
        print("Available models:")
        for model in available_models:
            print(f"  - {model}")
        return

    # Run training
    try:
        result = run_training(
            models=args.models,
            max_assets=args.max_assets,
            tune_hyperparams=args.tune_hyperparams,
            data_path=args.data_path,
            output_path=args.output_path,
            target_column=args.target,
        )

        if result["success"]:
            logger.info("Training completed successfully!")
            sys.exit(0)
        else:
            logger.error(f"Training failed: {result['error']}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
