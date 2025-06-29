"""
Data Pipeline Runner

This script runs the complete data preparation pipeline for watch price data.
It handles data loading, processing, feature engineering, and outputs prepared
data for ML model training.

Usage:
    python -m src.pipeline.run_pipeline
    python -m src.pipeline.run_pipeline --max-files 10
    python -m src.pipeline.run_pipeline --asset-type watch --no-feature-selection
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import run_pipeline, PipelineConfig
from src.utils.logging_config import get_pipeline_logger

# Setup centralized logging
logger = get_pipeline_logger()


def main():
    """Main entry point for the data pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Run the data preparation pipeline for watch price forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.pipeline.run_pipeline
  python -m src.pipeline.run_pipeline --max-files 10
  python -m src.pipeline.run_pipeline --asset-type watch --no-feature-selection
  python -m src.pipeline.run_pipeline --output-dir custom/output
        """
    )
    
    parser.add_argument(
        '--asset-type',
        choices=['watch', 'stock', 'crypto'],
        default='watch',
        help='Type of asset to process (default: watch)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of files to process'
    )
    
    parser.add_argument(
        '--no-feature-selection',
        action='store_true',
        help='Skip feature selection step'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/output',
        help='Output directory for processed data (default: data/output)'
    )
    
    parser.add_argument(
        '--interpolation-method',
        choices=['backfill', 'forward', 'linear', 'spline', 'seasonal', 'hybrid'],
        default='backfill',
        help='Interpolation method for missing values (default: backfill)'
    )
    
    parser.add_argument(
        '--outlier-method',
        choices=['iqr', 'zscore', 'isolation_forest'],
        default='iqr',
        help='Outlier detection method (default: iqr)'
    )
    
    parser.add_argument(
        '--min-data-points',
        type=int,
        default=30,
        help='Minimum data points required per asset (default: 30)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig()
    
    # Customize configuration based on arguments
    if args.output_dir != 'data/output':
        config.data_paths.output_dir = args.output_dir
    
    config.processing.interpolation_method = args.interpolation_method
    config.processing.outlier_method = args.outlier_method
    config.processing.min_data_points = args.min_data_points
    
    # Run pipeline
    logger.info("=" * 60)
    logger.info("DATA PREPARATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Asset type: {args.asset_type}")
    logger.info(f"Max files: {args.max_files or 'unlimited'}")
    logger.info(f"Feature selection: {'disabled' if args.no_feature_selection else 'enabled'}")
    logger.info(f"Output directory: {config.data_paths.output_dir}")
    logger.info(f"Interpolation method: {args.interpolation_method}")
    logger.info(f"Outlier detection: {args.outlier_method}")
    logger.info("=" * 60)
    
    try:
        # Run the pipeline
        pipeline_results = run_pipeline(
            asset_type=args.asset_type,
            config=config,
            max_files=args.max_files,
            enable_feature_selection=not args.no_feature_selection
        )
        
        # Check results
        if pipeline_results.get("success", False):
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            # Display summary
            steps = pipeline_results.get("steps", {})
            
            logger.info("Pipeline steps completed:")
            for step_name, step_data in steps.items():
                if isinstance(step_data, dict) and 'assets_count' in step_data:
                    count = step_data['assets_count']
                    logger.info(f"  ✓ {step_name.replace('_', ' ').title()}: {count} assets")
            
            # Show output files
            output_path = Path(config.data_paths.output_dir)
            if output_path.exists():
                output_files = list(output_path.glob("*.csv"))
                if output_files:
                    logger.info(f"\nOutput files saved to {output_path}:")
                    for file in output_files:
                        logger.info(f"  📄 {file.name}")
            
            logger.info(f"\nNext step: Run model training")
            logger.info(f"Command: python -m src.ml.run_training")
            
            sys.exit(0)
            
        else:
            error_msg = pipeline_results.get("error", "Unknown error")
            logger.error(f"Pipeline failed: {error_msg}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        import traceback
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()