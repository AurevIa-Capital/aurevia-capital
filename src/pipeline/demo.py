"""
Demonstration script for the new pipeline structure.

This script shows how to use the consolidated pipeline for watch data processing.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import run_pipeline, create_pipeline, PipelineConfig
from utils.logging_config import get_pipeline_logger

# Setup centralized logging
logger = get_pipeline_logger()


def demo_simple_pipeline():
    """Demonstrate the simplest way to run the pipeline."""
    
    print("\n=== Simple Pipeline Demo ===")
    
    # Run complete pipeline with defaults
    results = run_pipeline(asset_type="watch", max_files=5)
    
    if results.get('success'):
        print(f"✓ Pipeline completed successfully!")
        
        # Show results summary
        for step_name, step_data in results['steps'].items():
            if 'assets_count' in step_data:
                print(f"  {step_name}: {step_data['assets_count']} assets processed")
                
        # Show featured data info
        if 'feature_engineering' in results['steps']:
            featured_data = results['steps']['feature_engineering']['data']
            if featured_data:
                sample_asset = list(featured_data.keys())[0]
                sample_df = featured_data[sample_asset]
                print(f"  Sample asset {sample_asset}: {len(sample_df.columns)} features, {len(sample_df)} records")
                
    else:
        print(f"✗ Pipeline failed: {results.get('error', 'Unknown error')}")


def demo_custom_configuration():
    """Demonstrate custom configuration usage."""
    
    print("\n=== Custom Configuration Demo ===")
    
    # Create custom configuration
    config = PipelineConfig()
    
    # Customize processing settings
    config.processing.interpolation_method = "spline"
    config.processing.outlier_method = "isolation_forest"
    config.features.lag_periods = [1, 2, 3, 7]
    config.features.include_technical_indicators = True
    
    # Run with custom configuration
    results = run_pipeline(
        asset_type="watch", 
        config=config, 
        max_files=3,
        enable_feature_selection=True
    )
    
    if results.get('success'):
        print(f"✓ Custom pipeline completed!")
        print(f"  Used {config.processing.interpolation_method} interpolation")
        print(f"  Used {config.processing.outlier_method} outlier detection")
    else:
        print(f"✗ Custom pipeline failed: {results.get('error', 'Unknown error')}")


def demo_component_level_usage():
    """Demonstrate using individual pipeline components."""
    
    print("\n=== Component-Level Usage Demo ===")
    
    try:
        # Create pipeline components
        pipeline = create_pipeline("watch")
        
        # Use individual components
        loader = pipeline['loader']
        processor = pipeline['processor']
        feature_engineer = pipeline['feature_engineer']
        
        print("✓ Created pipeline components:")
        print(f"  Loader: {type(loader).__name__}")
        print(f"  Processor: {type(processor).__name__}")
        print(f"  Feature Engineer: {type(feature_engineer).__name__}")
        
        # Load data
        raw_data, load_report = loader.process(max_files=2)
        print(f"✓ Loaded {len(raw_data)} assets")
        
        if raw_data:
            # Process data
            processed_data, process_report = processor.process(raw_data)
            print(f"✓ Processed {len(processed_data)} assets")
            
            # Engineer features
            if processed_data:
                featured_data, feature_report = feature_engineer.process(processed_data)
                print(f"✓ Engineered features for {len(featured_data)} assets")
                
                if featured_data:
                    sample_asset = list(featured_data.keys())[0]
                    sample_df = featured_data[sample_asset]
                    print(f"  Sample: {sample_asset} has {len(sample_df.columns)} features")
        
    except Exception as e:
        print(f"✗ Component demo failed: {str(e)}")


def demo_watch_specific_features():
    """Demonstrate watch-specific processing."""
    
    print("\n=== Watch-Specific Features Demo ===")
    
    try:
        from pipeline.assets.watch import WatchProcessor, WatchFeatureEngineer
        
        config = PipelineConfig()
        watch_processor = WatchProcessor(config, "watch")
        watch_engineer = WatchFeatureEngineer(config, "watch")
        
        print("✓ Created watch-specific components")
        
        # Parse watch metadata
        sample_name = "Rolex-Submariner-638"
        metadata = watch_processor.parse_asset_metadata(sample_name)
        print(f"✓ Parsed metadata for {sample_name}:")
        print(f"  Brand: {metadata.get('brand')}")
        print(f"  Model: {metadata.get('model')}")
        print(f"  ID: {metadata.get('asset_id')}")
        
        print("✓ Watch-specific processing available")
        
    except Exception as e:
        print(f"✗ Watch-specific demo failed: {str(e)}")


if __name__ == "__main__":
    print("Pipeline Structure Demo")
    print("=" * 50)
    
    # Run all demos
    demo_simple_pipeline()
    demo_custom_configuration() 
    demo_component_level_usage()
    demo_watch_specific_features()
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("\nTo use the new pipeline in your code:")
    print("  from src.pipeline import run_pipeline, PipelineConfig")
    print("  results = run_pipeline('watch', max_files=10)")