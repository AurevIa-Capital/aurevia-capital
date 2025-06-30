"""
Utility functions for organizing visualization outputs.
"""

from pathlib import Path


def get_organized_visualization_path(
    base_output_dir: str,
    asset_name: str,
    model_name: str,
    plot_type: str,
) -> Path:
    """
    Create organized path for visualization files.
    
    Organizes files as: {base_output_dir}/{watch_model}/{prediction_model}/{plot_type}.png
    
    Parameters:
    ----------
    base_output_dir : str
        Base output directory (e.g., "data/output/visualizations")
    asset_name : str
        Name of the asset/watch (e.g., "Rolex-Submariner-638")
    model_name : str
        Name of the ML model (e.g., "xgboost", "linear")
    plot_type : str
        Type of plot (e.g., "predictions", "residuals", "importance")
        
    Returns:
    -------
    Path
        Organized path for the visualization file
    """
    # Clean asset name for directory use
    safe_asset_name = asset_name.replace("/", "_").replace("\\", "_")
    
    # Create the organized directory structure
    organized_dir = Path(base_output_dir) / safe_asset_name / model_name
    organized_dir.mkdir(parents=True, exist_ok=True)
    
    # Return the full file path
    return organized_dir / f"{plot_type}.png"


def get_organized_aggregate_path(
    base_output_dir: str,
    plot_type: str,
) -> Path:
    """
    Create organized path for aggregate visualization files.
    
    Places aggregate files in: {base_output_dir}/aggregate/{plot_type}.png
    
    Parameters:
    ----------
    base_output_dir : str
        Base output directory (e.g., "data/output/visualizations")  
    plot_type : str
        Type of aggregate plot (e.g., "performance_comparison", "complexity_analysis")
        
    Returns:
    -------
    Path
        Organized path for the aggregate visualization file
    """
    # Create aggregate directory
    aggregate_dir = Path(base_output_dir) / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    
    # Return the full file path
    return aggregate_dir / f"{plot_type}.png"