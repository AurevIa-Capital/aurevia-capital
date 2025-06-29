"""
Consolidated data pipeline for multi-asset time series processing.

This module provides a unified interface for data loading, processing,
feature engineering, and analysis across different asset types.
"""

from typing import Dict, Optional

from .assets.watch import WatchFeatureEngineer, WatchProcessor
from .base import create_asset_processor, create_feature_engineer
from .config import AssetConfig, CryptoConfig, PipelineConfig, StockConfig, WatchConfig
from .features import FeatureEngineer, FeatureSelector
from .loader import DataLoader, MultiAssetLoader
from .processor import DataProcessor

__version__ = "1.0.0"

__all__ = [
    # Core components
    "PipelineConfig",
    "DataLoader",
    "DataProcessor",
    "FeatureEngineer",
    "FeatureSelector",
    # Multi-asset support
    "MultiAssetLoader",
    # Asset configurations
    "AssetConfig",
    "WatchConfig",
    "StockConfig",
    "CryptoConfig",
    # Asset-specific implementations
    "WatchProcessor",
    "WatchFeatureEngineer",
    # Factory functions
    "create_asset_processor",
    "create_feature_engineer",
    # Main pipeline interface
    "run_pipeline",
    "create_pipeline",
]


def run_pipeline(
    asset_type: str = "watch",
    config: Optional[PipelineConfig] = None,
    max_files: Optional[int] = None,
    enable_feature_selection: bool = True,
) -> Dict:
    """
    Run the complete pipeline for a specific asset type.

    This is the main entry point for processing assets through the entire pipeline.

    Parameters:
    ----------
    asset_type : str
        Type of asset to process ('watch', 'stock', 'crypto')
    config : PipelineConfig, optional
        Pipeline configuration (uses default if None)
    max_files : int, optional
        Maximum number of files to process
    enable_feature_selection : bool
        Whether to perform feature selection

    Returns:
    -------
    Dict
        Pipeline results including processed data and reports
    """
    import logging
    from typing import Dict, Optional

    logger = logging.getLogger(__name__)
    logger.info(f"Starting pipeline for asset type: {asset_type}")

    # Use default config if none provided
    if config is None:
        config = PipelineConfig()

    # Create pipeline components
    loader = DataLoader(config, asset_type)
    processor = DataProcessor(config, asset_type)
    feature_engineer = FeatureEngineer(config, asset_type)

    # Initialize results
    pipeline_results = {"asset_type": asset_type, "config": config, "steps": {}}

    try:
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        raw_data, load_report = loader.process(max_files=max_files)
        pipeline_results["steps"]["data_loading"] = {
            "data": raw_data,
            "report": load_report,
            "assets_count": len(raw_data),
        }

        if not raw_data:
            logger.error("No data loaded successfully")
            return pipeline_results

        # Step 2: Process data (clean, validate, interpolate)
        logger.info("Step 2: Processing data")
        processed_data, process_report = processor.process(raw_data)
        pipeline_results["steps"]["data_processing"] = {
            "data": processed_data,
            "report": process_report,
            "assets_count": len(processed_data),
        }

        if not processed_data:
            logger.error("No data processed successfully")
            return pipeline_results

        # Step 3: Engineer features
        logger.info("Step 3: Engineering features")
        featured_data, feature_report = feature_engineer.process(processed_data)
        pipeline_results["steps"]["feature_engineering"] = {
            "data": featured_data,
            "report": feature_report,
            "assets_count": len(featured_data),
        }

        # Step 4: Feature selection (optional)
        if enable_feature_selection and featured_data:
            logger.info("Step 4: Performing feature selection")
            feature_selector = FeatureSelector(config.modeling)

            selection_results = {}
            for asset_name, df in featured_data.items():
                if "target" in df.columns and len(df) > 20:
                    try:
                        selected_features, selection_report = (
                            feature_selector.select_features(
                                df, "target", method="random_forest"
                            )
                        )
                        selection_results[asset_name] = {
                            "selected_features": selected_features,
                            "selection_report": selection_report,
                        }
                    except Exception as e:
                        logger.warning(
                            f"Feature selection failed for {asset_name}: {str(e)}"
                        )

            pipeline_results["steps"]["feature_selection"] = {
                "data": selection_results,
                "assets_count": len(selection_results)
            }

        # Step 5: Asset-specific processing if available
        if asset_type == "watch":
            logger.info("Step 5: Watch-specific processing")
            watch_processor = WatchProcessor(config, asset_type)
            watch_engineer = WatchFeatureEngineer(config, asset_type)

            # Generate brand summary
            brand_summary = watch_processor.get_brand_summary(processed_data)
            pipeline_results["steps"]["brand_analysis"] = {
                "brand_summary": brand_summary,
                "assets_count": len(brand_summary) if brand_summary is not None else 0
            }

            # Apply watch-specific features
            watch_featured_data = watch_engineer.process_multiple_watches(
                processed_data
            )
            pipeline_results["steps"]["watch_features"] = {
                "data": watch_featured_data,
                "assets_count": len(watch_featured_data),
            }

        # Step 6: Save output files
        logger.info("Step 6: Saving output files")
        _save_pipeline_outputs(pipeline_results, config)
        
        pipeline_results["success"] = True
        logger.info(f"Pipeline completed successfully for {asset_type}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        pipeline_results["success"] = False
        pipeline_results["error"] = str(e)

    return pipeline_results


def _save_pipeline_outputs(pipeline_results: dict, config: PipelineConfig):
    """Save pipeline outputs to CSV files."""
    import pandas as pd
    from pathlib import Path
    
    output_path = Path(config.data_paths.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    steps = pipeline_results.get("steps", {})
    
    # Save main featured data
    if "feature_engineering" in steps:
        featured_data = steps["feature_engineering"].get("data", {})
        if featured_data:
            # Combine all featured data into one DataFrame
            combined_data = []
            for asset_name, df in featured_data.items():
                df_copy = df.copy()
                df_copy['asset_id'] = asset_name
                combined_data.append(df_copy)
            
            if combined_data:
                final_df = pd.concat(combined_data, ignore_index=True)
                final_df.to_csv(output_path / "featured_data.csv", index=False)
                print(f"✅ Saved featured_data.csv ({len(final_df)} records, {len(final_df.columns)} columns)")
    
    # Save brand summary if available
    if "brand_analysis" in steps:
        brand_summary = steps["brand_analysis"].get("brand_summary")
        if brand_summary is not None and len(brand_summary) > 0:
            brand_df = pd.DataFrame(brand_summary)
            brand_df.to_csv(output_path / "brand_summary.csv", index=False)
            print(f"✅ Saved brand_summary.csv ({len(brand_df)} brands)")
    
    # Save feature selection summary
    if "feature_selection" in steps:
        selection_data = steps["feature_selection"].get("data", {})
        if selection_data:
            selection_records = []
            for asset_name, selection_info in selection_data.items():
                record = {
                    'asset_id': asset_name,
                    'total_features': selection_info.get("selection_report", {}).get("total_features", 0),
                    'selected_features': selection_info.get("selection_report", {}).get("selected_features", 0),
                    'method': selection_info.get("selection_report", {}).get("method", "unknown")
                }
                selection_records.append(record)
            
            if selection_records:
                selection_df = pd.DataFrame(selection_records)
                selection_df.to_csv(output_path / "feature_selection_summary.csv", index=False)
                print(f"✅ Saved feature_selection_summary.csv ({len(selection_df)} assets)")
    
    # Save processing steps summary
    processing_steps = []
    for step_name, step_data in steps.items():
        # Handle both old format (direct dict) and new format (with 'data' key)
        if isinstance(step_data, dict):
            if 'assets_count' in step_data:
                assets_count = step_data['assets_count']
            elif 'data' in step_data and isinstance(step_data['data'], dict):
                assets_count = len(step_data['data'])
            else:
                assets_count = 0
        else:
            assets_count = 0
            
        record = {
            'step': step_name,
            'assets_count': assets_count,
            'status': 'completed'
        }
        processing_steps.append(record)
    
    if processing_steps:
        steps_df = pd.DataFrame(processing_steps)
        steps_df.to_csv(output_path / "processing_steps.csv", index=False)
        print(f"✅ Saved processing_steps.csv ({len(steps_df)} steps)")


def create_pipeline(asset_type: str, config: Optional[PipelineConfig] = None):
    """
    Create a pipeline instance for a specific asset type.

    Parameters:
    ----------
    asset_type : str
        Type of asset ('watch', 'stock', 'crypto')
    config : PipelineConfig, optional
        Pipeline configuration

    Returns:
    -------
    Dict
        Dictionary containing pipeline components
    """
    from typing import Optional

    if config is None:
        config = PipelineConfig()

    return {
        "config": config,
        "loader": DataLoader(config, asset_type),
        "processor": DataProcessor(config, asset_type),
        "feature_engineer": FeatureEngineer(config, asset_type),
        "feature_selector": FeatureSelector(config.modeling),
        "asset_processor": create_asset_processor(asset_type, config),
        "asset_feature_engineer": create_feature_engineer(asset_type, config),
    }
