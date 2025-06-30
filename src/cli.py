"""
Unified CLI entry point for AurevIa Timepiece.

This module provides a single command-line interface for all operations
including data collection, pipeline processing, model training, and serving.
Uses the new Factory and Builder patterns for improved architecture.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Use absolute imports when running as script
try:
    from .core.config_builder import PipelineConfigBuilder, create_training_config_from_args
    from .core.model_factory import ModelFactory
    from .core.data_store import create_data_store
    from .core.command_pattern import CommandFactory, CommandInvoker
    from .core.pipeline_orchestrator import create_orchestrator
    from .core.event_system import create_default_event_system
except ImportError:
    # Fallback to absolute imports for direct script execution
    from src.core.config_builder import PipelineConfigBuilder, create_training_config_from_args
    from src.core.model_factory import ModelFactory
    from src.core.data_store import create_data_store
    from src.core.command_pattern import CommandFactory, CommandInvoker
    from src.core.pipeline_orchestrator import create_orchestrator
    from src.core.event_system import create_default_event_system

# Configure logging for CLI
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_scrape(args):
    """Execute scraping commands using Command Pattern."""
    event_bus = create_default_event_system()
    command_invoker = CommandInvoker(event_bus)
    
    command = CommandFactory.create_scrape_command(args.scrape_command)
    result = command_invoker.execute_command(command)
    
    if not result.success:
        logger.error(result.message)
        sys.exit(1)
    else:
        logger.info(result.message)


def cmd_pipeline(args):
    """Execute pipeline processing using Command Pattern and new architecture."""
    logger.info(f"Running data pipeline with max-files={args.max_files}")
    
    # Option 1: Use new Pipeline Orchestrator (recommended)
    if getattr(args, 'use_new_architecture', True):
        config = (PipelineConfigBuilder()
                  .from_args(args)
                  .with_processing_options(
                      interpolation_method=getattr(args, 'interpolation_method', 'backfill'),
                      outlier_method=getattr(args, 'outlier_method', 'iqr'),
                      min_data_points=getattr(args, 'min_data_points', 30)
                  )
                  .build())
        
        orchestrator = create_orchestrator(config)
        
        # Execute using new orchestrator (data loading + processing + feature engineering only)
        try:
            from .core.pipeline_strategy import PipelineStage
            
            # Execute data loading
            data_result = orchestrator.execute_single_stage(
                PipelineStage.DATA_LOADING,
                max_files=args.max_files
            )
            
            if not data_result.success:
                logger.error(f"Data loading failed: {data_result.error}")
                sys.exit(1)
            
            # Execute data processing
            processing_result = orchestrator.execute_single_stage(
                PipelineStage.DATA_PROCESSING,
                input_data=data_result.data
            )
            
            if not processing_result.success:
                logger.error(f"Data processing failed: {processing_result.error}")
                sys.exit(1)
            
            # Execute feature engineering
            feature_result = orchestrator.execute_single_stage(
                PipelineStage.FEATURE_ENGINEERING,
                input_data=processing_result.data
            )
            
            if not feature_result.success:
                logger.error(f"Feature engineering failed: {feature_result.error}")
                sys.exit(1)
            
            logger.info("Pipeline processing completed successfully using new architecture")
            
        except Exception as e:
            logger.error(f"New architecture pipeline failed: {str(e)}")
            logger.info("Falling back to legacy pipeline...")
            _execute_legacy_pipeline(args)
    
    else:
        # Option 2: Use Command Pattern with legacy pipeline
        event_bus = create_default_event_system()
        command_invoker = CommandInvoker(event_bus)
        
        command = CommandFactory.create_pipeline_command(args)
        result = command_invoker.execute_command(command)
        
        if not result.success:
            logger.error(result.message)
            sys.exit(1)
        else:
            logger.info(result.message)


def _execute_legacy_pipeline(args):
    """Fallback to legacy pipeline execution."""
    from src.pipeline.run_pipeline import main
    
    pipeline_args = ['--max-files', str(args.max_files)]
    
    if args.interpolation_method:
        pipeline_args.extend(['--interpolation-method', args.interpolation_method])
    
    if args.outlier_method:
        pipeline_args.extend(['--outlier-method', args.outlier_method])
    
    original_argv = sys.argv
    try:
        sys.argv = ['run_pipeline.py'] + pipeline_args
        main()
    finally:
        sys.argv = original_argv


def cmd_train(args):
    """Execute model training using Command Pattern."""
    logger.info(f"Training models with horizons: {args.horizons}")
    
    event_bus = create_default_event_system()
    command_invoker = CommandInvoker(event_bus)
    
    command = CommandFactory.create_training_command(args)
    result = command_invoker.execute_command(command, analyze_only=getattr(args, 'analyze_only', False))
    
    if not result.success:
        logger.error(result.message)
        sys.exit(1)
    else:
        logger.info(result.message)


def cmd_serve(args):
    """Execute serving commands using Command Pattern."""
    event_bus = create_default_event_system()
    command_invoker = CommandInvoker(event_bus)
    
    command = CommandFactory.create_serve_command(args.serve_type)
    result = command_invoker.execute_command(command)
    
    if not result.success:
        logger.error(result.message)
        sys.exit(1)
    else:
        logger.info(result.message)


def cmd_visualize(args):
    """Execute visualization commands using Command Pattern."""
    logger.info("Creating visualizations...")
    
    event_bus = create_default_event_system()
    command_invoker = CommandInvoker(event_bus)
    
    command = CommandFactory.create_visualize_command(args)
    result = command_invoker.execute_command(command)
    
    if not result.success:
        logger.error(result.message)
        sys.exit(1)
    else:
        logger.info(result.message)


def cmd_full_pipeline(args):
    """Execute complete pipeline using new Phase 2 architecture."""
    logger.info("Executing full pipeline with new architecture...")
    logger.info(f"Horizons: {args.horizons}, Models: {args.models}")
    
    # Build configuration
    config = (PipelineConfigBuilder()
              .from_args(args)
              .with_processing_options()
              .with_feature_engineering()
              .with_modeling_options()
              .build())
    
    # Create orchestrator with event monitoring
    orchestrator = create_orchestrator(config)
    
    try:
        # Execute full pipeline using new orchestrator
        result = orchestrator.execute_full_pipeline(
            horizons=args.horizons,
            models=args.models,
            max_files=args.max_files,
            max_assets=args.max_assets
        )
        
        if result['success']:
            logger.info("🎉 Full pipeline completed successfully!")
            logger.info(f"Pipeline ID: {result['pipeline_id']}")
            logger.info(f"Results: {result['results']}")
            
            # Display progress and metrics
            if 'progress' in result:
                logger.info(f"Progress: {result['progress'].get('status', 'unknown')}")
            
            if 'metrics' in result:
                metrics = result['metrics']
                if 'total_execution_time' in metrics:
                    logger.info(f"Total execution time: {metrics['total_execution_time']:.2f}s")
        else:
            logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Full pipeline execution failed: {str(e)}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog='aurevia',
        description='AurevIa Timepiece - Luxury Watch Price Forecasting Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aurevia scrape urls                     # Generate watch URLs
  aurevia scrape prices                   # Scrape price data
  aurevia pipeline --max-files 20         # Run data pipeline
  aurevia train --horizons 7 14 30        # Train models
  aurevia serve dashboard                  # Start dashboard
  aurevia serve api                       # Start API server
  aurevia visualize --max-assets 5        # Create visualizations
  
  # NEW: Full pipeline with Phase 2 architecture
  aurevia full --horizons 7 14 30 --models linear xgboost --max-assets 10
        """)
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Data scraping operations')
    scrape_parser.add_argument('scrape_command', choices=['urls', 'prices'],
                              help='Type of scraping to perform')
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run data pipeline')
    pipeline_parser.add_argument('--max-files', type=int, default=20,
                                help='Maximum number of files to process')
    pipeline_parser.add_argument('--interpolation-method', 
                                choices=['backfill', 'linear', 'spline', 'seasonal', 'hybrid'],
                                help='Interpolation method for missing data')
    pipeline_parser.add_argument('--outlier-method',
                                choices=['iqr', 'zscore', 'isolation_forest'],
                                help='Outlier detection method')
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Model training operations')
    train_parser.add_argument('--horizons', type=int, nargs='+',
                             help='Prediction horizons in days (e.g., 7 14 30)')
    
    # Get available models from factory
    available_models = ModelFactory.get_available_models()
    train_parser.add_argument('--models', nargs='+',
                             choices=available_models,
                             help='Models to train')
    train_parser.add_argument('--max-assets', type=int,
                             help='Maximum number of assets to process')
    train_parser.add_argument('--analyze-only', action='store_true',
                             help='Only analyze existing results')
    train_parser.set_defaults(func=cmd_train)
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start web services')
    serve_parser.add_argument('serve_type', choices=['dashboard', 'api'],
                             help='Type of service to start')
    serve_parser.set_defaults(func=cmd_serve)
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--max-assets', type=int,
                           help='Maximum number of assets to visualize')
    viz_parser.add_argument('--skip-aggregate', action='store_true',
                           help='Skip aggregate visualizations')
    viz_parser.add_argument('--verbose', action='store_true',
                           help='Enable verbose output')
    viz_parser.add_argument('--specific-assets', nargs='+',
                           help='Specific assets to visualize')
    viz_parser.set_defaults(func=cmd_visualize)
    
    # NEW: Full pipeline command using new architecture
    full_parser = subparsers.add_parser('full', help='Execute complete pipeline with new architecture')
    full_parser.add_argument('--horizons', type=int, nargs='+', default=[7, 14, 30],
                            help='Prediction horizons in days')
    full_parser.add_argument('--models', nargs='+', 
                            choices=ModelFactory.get_available_models(),
                            default=['linear', 'xgboost'],
                            help='Models to train')
    full_parser.add_argument('--max-files', type=int, default=20,
                            help='Maximum number of files to process')
    full_parser.add_argument('--max-assets', type=int,
                            help='Maximum number of assets to process')
    full_parser.add_argument('--include-scraping', action='store_true',
                            help='Include data scraping in pipeline')
    full_parser.add_argument('--include-visualization', action='store_true',
                            help='Include visualization generation')
    full_parser.set_defaults(func=cmd_full_pipeline)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle no command
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        # Execute the appropriate command
        args.func(args)
        logger.info("Command completed successfully")
    except KeyboardInterrupt:
        logger.info("Command interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()