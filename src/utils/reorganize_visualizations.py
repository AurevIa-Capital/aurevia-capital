#!/usr/bin/env python3
"""
Script to reorganize existing visualization files into the new organized structure.

This script moves files from the flat structure:
  data/output/visualizations/{asset}_{model}_{plot_type}.png

To the organized structure:
  data/output/visualizations/{asset}/{model}/{plot_type}.png
"""

import re
from pathlib import Path
from typing import Tuple, Optional


def parse_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse visualization filename to extract asset name, model name, and plot type.
    
    Expected patterns:
    - {asset}_{model}_predictions.png
    - {asset}_{model}_residuals.png
    - {asset}_{model}_importance.png
    - {asset}_{model}_complete_forecast.png
    - {asset}_{model}_decomposition.png
    - {asset}_data_split.png (special case)
    
    Returns:
    -------
    Optional[Tuple[str, str, str]]
        (asset_name, model_name, plot_type) or None if cannot parse
    """
    # Remove .png extension
    name = filename.replace('.png', '')
    
    # Special case for data_split (no model)
    if name.endswith('_data_split'):
        asset_name = name.replace('_data_split', '')
        return asset_name, 'general', 'data_split'
    
    # Special case for aggregate files
    aggregate_files = {
        'complexity_vs_performance': 'complexity_vs_performance',
        'overall_performance_comparison': 'overall_performance_comparison',
        'multi_horizon_performance_summary': 'multi_horizon_performance_summary'
    }
    
    if name in aggregate_files:
        return 'aggregate', 'aggregate', aggregate_files[name]
    
    # Regular pattern matching for model-specific files
    # Try different patterns based on known model types
    model_patterns = [
        r'(.+)_(LinearRegression|RandomForest|Ridge|XGBoost)_(.+)$',
        r'(.+)_(linear|random_forest|ridge|xgboost)_(.+)$'
    ]
    
    for pattern in model_patterns:
        match = re.match(pattern, name)
        if match:
            asset_name, model_name, plot_type = match.groups()
            
            # Normalize model names
            model_mapping = {
                'LinearRegression': 'linear',
                'RandomForest': 'random_forest', 
                'Ridge': 'ridge',
                'XGBoost': 'xgboost'
            }
            
            model_name = model_mapping.get(model_name, model_name.lower())
            return asset_name, model_name, plot_type
            
    return None


def reorganize_visualizations(base_dir: str = "data/output/visualizations", dry_run: bool = True):
    """
    Reorganize existing visualization files into organized directory structure.
    
    Parameters:
    ----------
    base_dir : str
        Base visualization directory
    dry_run : bool
        If True, only print what would be done without moving files
    """
    viz_dir = Path(base_dir)
    
    if not viz_dir.exists():
        print(f"Directory {base_dir} does not exist")
        return
    
    print(f"Reorganizing visualizations in: {viz_dir}")
    print(f"Dry run mode: {dry_run}")
    print("=" * 60)
    
    moved_count = 0
    failed_count = 0
    
    # Get all PNG files in the root directory
    png_files = list(viz_dir.glob("*.png"))
    
    if not png_files:
        print("No PNG files found in root directory to reorganize")
        return
    
    for png_file in png_files:
        filename = png_file.name
        
        # Parse the filename
        parsed = parse_filename(filename)
        
        if parsed is None:
            print(f"❌ Could not parse: {filename}")
            failed_count += 1
            continue
            
        asset_name, model_name, plot_type = parsed
        
        # Create organized path
        if asset_name == 'aggregate':
            new_dir = viz_dir / "aggregate"
            new_path = new_dir / f"{plot_type}.png"
        else:
            new_dir = viz_dir / asset_name / model_name
            new_path = new_dir / f"{plot_type}.png"
        
        print(f"📊 {filename}")
        print(f"   Asset: {asset_name}")
        print(f"   Model: {model_name}")
        print(f"   Type:  {plot_type}")
        print(f"   New:   {new_path.relative_to(viz_dir)}")
        
        if not dry_run:
            # Create directory if it doesn't exist
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            try:
                png_file.rename(new_path)
                print(f"   ✅ Moved successfully")
                moved_count += 1
            except Exception as e:
                print(f"   ❌ Failed to move: {e}")
                failed_count += 1
        else:
            print(f"   🔍 Would move to: {new_path}")
            moved_count += 1
            
        print()
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Files processed: {len(png_files)}")
    print(f"  Successfully {'would be moved' if dry_run else 'moved'}: {moved_count}")
    print(f"  Failed to parse: {failed_count}")
    
    if dry_run:
        print(f"\nTo actually move the files, run with dry_run=False")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize visualization files")
    parser.add_argument("--base-dir", default="data/output/visualizations", 
                       help="Base visualization directory")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually move files (default is dry run)")
    
    args = parser.parse_args()
    
    reorganize_visualizations(
        base_dir=args.base_dir,
        dry_run=not args.execute
    )