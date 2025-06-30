"""
Example script demonstrating model training with the ML pipeline.

This script shows how to use the complete ML training system
with the data pipeline output.
"""

from pathlib import Path

from src.ml import ModelTrainer, train_models
from src.pipeline import PipelineConfig, run_pipeline
from src.utils.logging_config import get_training_logger

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent

logger = get_training_logger()


def example_training_workflow():
    """
    Complete example of training models with the data pipeline.
    """

    # Step 1: Configure pipeline
    config = PipelineConfig()

    # Customize configuration for training
    config.modeling.test_size = 0.2
    config.modeling.validation_size = 0.1
    config.modeling.cross_validation_folds = 3
    config.features.target_shift = -1  # Predict next day

    logger.info("Starting model training example...")

    try:
        # Step 2: Run data pipeline to get featured data
        logger.info("Running data pipeline...")
        pipeline_results = run_pipeline(
            asset_type="watch",
            config=config,
            max_files=5,  # Limit for demonstration
        )

        # Extract featured data from pipeline results
        if not pipeline_results.get("success", False):
            logger.error(
                f"Pipeline failed: {pipeline_results.get('error', 'Unknown error')}"
            )
            return

        # Get featured data from the feature engineering step
        featured_data = (
            pipeline_results.get("steps", {})
            .get("feature_engineering", {})
            .get("data", {})
        )

        if not featured_data:
            logger.error("No featured data available for training")
            return

        logger.info(f"Pipeline complete. {len(featured_data)} assets processed")

        # Step 3: Define models to train
        model_names = [
            "linear",
            "ridge",
            "random_forest",
            "xgboost",
            # Note: ARIMA/SARIMA might take longer, uncomment if desired
            # 'arima'
        ]

        # Step 4: Train models
        logger.info(
            f"Training {len(model_names)} models on {len(featured_data)} assets..."
        )

        training_results = train_models(
            config=config,
            featured_data=featured_data,
            model_names=model_names,
            max_assets=3,  # Limit assets for demonstration
        )

        # Step 5: Analyze results
        logger.info("Training complete. Analyzing results...")

        total_models_trained = 0
        successful_models = 0

        for asset_name, asset_results in training_results.items():
            logger.info(f"\nAsset: {asset_name}")

            for model_name, result in asset_results.items():
                total_models_trained += 1

                if result.success:
                    successful_models += 1
                    val_metrics = (
                        result.validation_result.metrics
                        if result.validation_result
                        else {}
                    )
                    mae = val_metrics.get("mae", "N/A")
                    rmse = val_metrics.get("rmse", "N/A")
                    r2 = val_metrics.get("r2", "N/A")

                    logger.info(
                        f"  {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}"
                    )
                else:
                    logger.warning(f"  {model_name}: FAILED - {result.error_message}")

        logger.info("\nTraining Summary:")
        logger.info(f"  Total models: {total_models_trained}")
        logger.info(f"  Successful: {successful_models}")
        logger.info(f"  Failed: {total_models_trained - successful_models}")

        # Step 6: Find best model for each asset
        logger.info("\nBest models by asset:")

        trainer = ModelTrainer(config)

        for asset_name, asset_results in training_results.items():
            best_result = trainer.get_best_model(asset_results, metric="mae")

            if best_result:
                mae = best_result.validation_result.metrics.get("mae", "N/A")
                logger.info(
                    f"  {asset_name}: {best_result.model_name} (MAE: {mae:.4f})"
                )
            else:
                logger.info(f"  {asset_name}: No successful models")

        # Step 7: Save models (optional)
        output_dir = "data/output/trained_models"
        logger.info(f"Saving models to {output_dir}...")

        trainer.save_training_results(training_results, output_dir)

        logger.info("Example training workflow complete!")

        return training_results

    except Exception as e:
        logger.error(f"Training workflow failed: {str(e)}")
        raise


def example_single_asset_training():
    """
    Example of training models for a single asset with custom configuration.
    """

    logger.info("Single asset training example...")

    config = PipelineConfig()

    # Step 1: Get featured data for one asset
    pipeline_results = run_pipeline(asset_type="watch", config=config, max_files=1)

    if not pipeline_results.get("success", False):
        logger.error(
            f"Pipeline failed: {pipeline_results.get('error', 'Unknown error')}"
        )
        return

    # Extract featured data
    featured_data = (
        pipeline_results.get("steps", {}).get("feature_engineering", {}).get("data", {})
    )

    if not featured_data:
        logger.error("No data available")
        return

    # Get first asset
    asset_name = list(featured_data.keys())[0]
    df = featured_data[asset_name]

    logger.info(f"Training models for {asset_name} ({len(df)} samples)")

    # Step 2: Create trainer and prepare data
    trainer = ModelTrainer(config)

    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_time_series_split(
        df, target_column="target"
    )

    # Step 3: Train individual models
    models_to_train = ["linear", "ridge", "random_forest"]

    results = {}
    for model_name in models_to_train:
        logger.info(f"Training {model_name}...")

        result = trainer.train_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            validate=True,
        )

        results[model_name] = result

        if result.success:
            mae = result.validation_result.metrics.get("mae", "N/A")
            logger.info(f"  {model_name} completed - MAE: {mae:.4f}")
        else:
            logger.error(f"  {model_name} failed: {result.error_message}")

    # Step 4: Test best model on test set
    best_result = trainer.get_best_model(results, "mae")

    if best_result and best_result.model:
        logger.info(f"Testing best model ({best_result.model_name}) on test set...")

        test_predictions = best_result.model.predict(X_test)

        from src.ml.training import TimeSeriesValidator

        validator = TimeSeriesValidator(config)
        test_metrics = validator.calculate_metrics(y_test.values, test_predictions)

        logger.info("Test set performance:")
        logger.info(f"  MAE: {test_metrics.get('mae', 'N/A'):.4f}")
        logger.info(f"  RMSE: {test_metrics.get('rmse', 'N/A'):.4f}")
        logger.info(f"  R²: {test_metrics.get('r2', 'N/A'):.4f}")

    return results


if __name__ == "__main__":
    # Run the example
    example_training_workflow()

    # Uncomment to run single asset example
    # example_single_asset_training()
