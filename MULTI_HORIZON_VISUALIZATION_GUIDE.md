# Multi-Horizon Visualization Guide

## Overview

The new `multi_horizon_visualizer.py` script creates comprehensive, meaningful visualizations for your 7/14/30-day forecasting models. It addresses the issues with the previous single-point predictions by providing complete time series context and multi-horizon comparisons.

## Key Improvements

### ✅ **Fixed Issues from Original Visualization**
- **No more single-point predictions**: Shows complete time series with training/validation/test context
- **Proper date handling**: No more 1970 dates - creates meaningful time axes
- **Multi-horizon comparison**: Compare all horizons (7d, 14d, 30d) in one comprehensive view
- **Performance analysis**: Clear metrics showing how accuracy changes with forecast horizon
- **Professional presentation**: Publication-ready charts with proper legends and annotations

### ✅ **New Visualization Types**

1. **Multi-Horizon Performance Summary** (`multi_horizon_performance_summary.png`)
   - Success rates across all horizons
   - Model training statistics
   - Performance trends as horizon increases
   - Training time vs accuracy trade-offs

2. **Asset-Specific Analysis** (`{asset}_linear_multi_horizon_analysis.png`)
   - Complete time series with predictions for all horizons
   - Performance comparison across horizons (MAE, RMSE, MAPE)
   - Model accuracy trends (R² scores)
   - Prediction error distributions

## Usage

### Environment Setup
```bash
# IMPORTANT: Always use the timepiece environment
source activate timepiece
```

### Basic Commands

#### 1. Create Summary Dashboard (Recommended first step)
```bash
source activate timepiece && python -m src.ml.multi_horizon_visualizer --summary-only
```
**Output**: `data/output/visualizations/multi_horizon_performance_summary.png`

#### 2. Detailed Asset Analysis
```bash
source activate timepiece && python -m src.ml.multi_horizon_visualizer \
    --asset "Audemars_Piguet-Steeloyal_Oak_Offshore-22378" \
    --horizons 7 14 30 \
    --models linear
```
**Output**: `data/output/visualizations/Audemars_Piguet-Steeloyal_Oak_Offshore-22378_linear_multi_horizon_analysis.png`

#### 3. Compare Multiple Models
```bash
source activate timepiece && python -m src.ml.multi_horizon_visualizer \
    --asset "Audemars_Piguet-Steeloyal_Oak_Offshore-22378" \
    --horizons 7 14 30 \
    --models linear ridge xgboost
```

### Advanced Usage

#### Custom Horizons and Models
```bash
source activate timepiece && python -m src.ml.multi_horizon_visualizer \
    --asset "Rolex-SteelHulk_ubmariner-639" \
    --horizons 7 14 \
    --models linear ridge \
    --output-dir custom/viz
```

#### Available Assets
Your trained models include assets from these brands:
- **Audemars Piguet**: Royal Oak, Offshore models
- **Rolex**: Submariner, GMT-Master, Datejust
- **Patek Philippe**: Nautilus, Aquanaut, Calatrava
- **Omega**: Speedmaster, Seamaster
- **Hublot**: Big Bang, Classic Fusion
- **Longines**: Spirit, HydroConquest, Master Collection

## What the Visualizations Show

### Summary Dashboard (4 panels):
1. **Success Rate by Horizon**: How well models trained for each horizon
2. **Model Statistics**: Total vs successful models per horizon
3. **Performance Trends**: Average MAE progression with longer horizons
4. **Training Efficiency**: Time vs accuracy trade-offs

### Asset-Specific Analysis (4 panels):
1. **Complete Time Series**: Full historical data + predictions for all horizons
2. **Performance Metrics**: MAE, RMSE, MAPE comparison across horizons
3. **Accuracy Trends**: R² scores showing how accuracy degrades with longer horizons
4. **Error Distributions**: Box plots showing prediction error spread by horizon

## Key Insights from Your Models

Based on the generated visualizations:

### Performance Summary:
- **100% success rate** across all horizons (7d, 14d, 30d)
- **261 models** trained per horizon (87 assets × 3 models each)
- **Performance degradation**: Average MAE increases from 18.0 (7d) to 37.0 (30d)
- **Training efficiency**: Most models train in <0.1 seconds

### For Audemars Piguet Royal Oak Offshore:
- **7-day forecasts**: Best accuracy with lowest error spread
- **14-day forecasts**: Moderate accuracy degradation
- **30-day forecasts**: Highest uncertainty but still reasonable performance
- **Model behavior**: Linear regression performs consistently across all horizons

## Integration with Existing Workflow

### After Training New Models:
```bash
# 1. Train multi-horizon models
source activate timepiece && python -m src.ml.multi_horizon_training --horizons 7 14 30

# 2. Create visualizations
source activate timepiece && python -m src.ml.multi_horizon_visualizer --summary-only

# 3. Analyze specific assets
source activate timepiece && python -m src.ml.multi_horizon_visualizer \
    --asset "Your_Asset_Name" --horizons 7 14 30 --models linear ridge xgboost
```

### Files Created:
- **Summary**: `data/output/visualizations/multi_horizon_performance_summary.png`
- **Asset Analysis**: `data/output/visualizations/{asset}_{model}_multi_horizon_analysis.png`
- **Logs**: `./logs/ml_training.log` (detailed execution logs)

## Troubleshooting

### Common Issues:

1. **Environment Error**: Always use `source activate timepiece`
2. **Asset Not Found**: Check available assets in the training logs
3. **Missing Models**: Ensure multi-horizon training completed successfully
4. **Matplotlib Warning**: The `labels` deprecation warning is cosmetic and doesn't affect output

### Debug Commands:
```bash
# Check available assets
ls data/output/models_7_day/

# Check training summary
cat data/output/multi_horizon_comparison.json | head -50

# View logs
tail -100 logs/ml_training.log
```

## Next Steps

1. **Run the summary visualization** to get an overview of all horizon performance
2. **Pick specific assets** for detailed analysis based on your interests
3. **Compare different models** (linear vs ridge vs xgboost) for the same asset
4. **Use insights** to optimize your trading strategies based on forecast horizon accuracy

The new visualizations provide the complete context you need to understand how your multi-horizon models perform across different prediction windows!