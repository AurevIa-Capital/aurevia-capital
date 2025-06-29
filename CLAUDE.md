# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modern luxury watch price forecasting platform that analyzes historical price data and generates predictions using multiple machine learning models. The system features a modular architecture with FastAPI backend, Streamlit dashboard, and comprehensive data collection pipeline with Cloudflare bypass capabilities.

## Quick Start

```bash
# Stage 1: Generate watch URLs (100 watches, 10 per brand)
python -m src.collectors.watch.watch_urls

# Stage 2: Scrape price data 
python -m src.collectors.watch.scrape_runner

# Stage 3: Run data pipeline
python -m src.pipeline.run_pipeline --max-files 20

# Stage 4: Train ML models (trains on all 87 CSV files by default)
python -m src.ml.multi_horizon_training --horizons 7 14 30

# Run the dashboard
python src/scripts/run_dashboard.py

# Run the API (alternative interface)
python src/scripts/run_api.py
```

## Current Architecture

### Core Application Structure
- **Dashboard Entry**: `src/scripts/run_dashboard.py` → launches Streamlit dashboard  
- **API Entry**: `src/scripts/run_api.py` → launches FastAPI backend
- **Data Collection**: `src/collectors/watch/` → watch price scraping pipeline
- **Data Pipeline**: `src/pipeline/` → unified data processing and feature engineering
- **Machine Learning**: `src/ml/` → model training, evaluation, and prediction
- **API Layer**: `src/api/` → FastAPI backend with routers and middleware
- **Dashboard**: `src/dashboard/` → Streamlit interface with plugin architecture
- **Core Services**: `src/core/` → shared domain logic and schemas
- **Analysis**: `src/analysis/` → EDA and visualization
- **Utilities**: `src/utils/` → shared utilities

### Watch Data Collection Pipeline

#### Stage 1: URL Discovery
- **Script**: `python -m src.collectors.watch.watch_urls`
- **Purpose**: Discovers 100 watch URLs (10 per brand) from WatchCharts
- **Output**: `data/scrape/url/watch_targets_100.json`
- **Brands**: Rolex, Patek Philippe, Omega, Tudor, Audemars Piguet, etc.

#### Stage 2: Price Scraping  
- **Script**: `python -m src.collectors.watch.scrape_runner`
- **Purpose**: Scrapes historical price data for each watch
- **Output**: `data/scrape/prices/{Brand}-{Model}-{ID}.csv`
- **Features**: Progress tracking, resume capability, Cloudflare bypass

### Watch Collector Components

#### `base_scraper.py` - Foundation
- **BaseScraper**: Abstract base class with shared scraping utilities
- **WatchScrapingMixin**: Common browser management, navigation, text processing
- **Key Methods**: `safe_navigate_with_retries()`, `make_filename_safe()`, `extract_watch_id_from_url()`

#### `watch_urls.py` - URL Discovery
- **WatchDiscovery**: Discovers watch URLs from brand pages
- **Brands Covered**: 10 luxury watch brands with 10 watches each
- **Output Format**: JSON with brand, model_name, url, source fields

#### `price_scraper.py` - Price Extraction
- **CloudflareBypassScraper**: Extracts price data using Chart.js injection
- **Features**: Incremental updates, data merging, error screenshots
- **Output Format**: CSV with date and price(SGD) columns

#### `batch_scraper.py` - Orchestration
- **MassWatchScraper**: Manages batch processing of multiple watches
- **Features**: Progress tracking, brand-based delays, resume capability
- **Progress File**: `data/scrape/scraping_progress.json`

#### `scrape_runner.py` - CLI Interface
- **WatchScrapingRunner**: User-friendly interface for mass scraping
- **Features**: Configuration validation, interactive prompts, error handling

### Data Directory Structure
```
data/
├── scrape/
│   ├── url/
│   │   └── watch_targets_100.json    # Stage 1: Watch URLs and metadata
│   ├── prices/
│   │   ├── Rolex-Submariner-638.csv      # Stage 2: Price data per watch
│   │   ├── Omega-Speedmaster-30921.csv   # Unique files per watch
│   │   └── ...
│   └── scraping_progress.json        # Progress tracking
├── output/
│   ├── featured_data.csv             # Enhanced dataset with 80+ features
│   ├── brand_summary.csv             # Performance analysis by watch brand
│   ├── processing_steps.csv          # Detailed pipeline execution log
│   ├── feature_selection_summary.csv # Feature importance and selection results
│   ├── multi_horizon_comparison.json # Multi-horizon model comparison
│   ├── models/                       # Trained models (1, 3, 7, 14 day horizons)
│   │   ├── models_1_day/
│   │   ├── models_3_day/
│   │   ├── models_7_day/
│   │   ├── models_14_day/
│   │   └── model_comparison.csv
│   ├── visualizations/               # Model performance and forecasting plots
└── images/                           # Watch reference images
```

### File Naming Standards
- **Watch Targets**: `watch_targets_100.json`
- **Price Data**: `{Brand}-{Model}-{watch_id}.csv` (e.g., `Rolex-Submariner-638.csv`)
- **Progress Files**: `scraping_progress.json`

### Modular Application Architecture

The system features a modular architecture with multiple interface layers:

#### API Layer (`src/api/`)
```
src/api/
├── main.py           # FastAPI application entry point
└── app/
    ├── dependencies.py    # Dependency injection
    ├── middleware/
    │   └── logging.py    # Request logging middleware
    └── routers/
        ├── assets.py     # Asset management endpoints
        ├── collectors.py # Data collection endpoints
        └── forecasts.py  # Forecasting endpoints
```

#### Dashboard Layer (`src/dashboard/`)
```
src/dashboard/
├── main.py           # Streamlit application entry point
├── components/       # Reusable dashboard components
├── config/          # Dashboard configuration
├── core/            # Dashboard core logic
└── plugins/         # Plugin architecture for asset types
```

#### Data Pipeline (`src/pipeline/`)
```
src/pipeline/
├── run_pipeline.py   # CLI entry point with argument parsing
├── config.py         # All configuration (paths, processing, features, models)
├── base.py           # Abstract base classes and factory functions
├── loader.py         # Data loading and discovery (multi-asset support)
├── processor.py      # Data processing (cleaning, validation, interpolation, outliers)
├── features.py       # Feature engineering and selection (80+ features)
├── assets/           # Asset-specific implementations
│   └── watch.py      # Complete watch processing and feature engineering
└── demo.py           # Demonstration script
```

#### Key Pipeline Features

**Configuration-Driven Design**:
- **Centralized settings** in `PipelineConfig`
- **Asset-agnostic architecture** (easily extensible to stocks, crypto)
- **Flexible parameters** for all processing steps

**Advanced Data Processing**:
- **Multiple interpolation methods**: backfill, linear, spline, seasonal, hybrid
- **Sophisticated outlier detection**: IQR, Z-score, Isolation Forest
- **Comprehensive validation**: 15+ quality checks per asset

**Enhanced Feature Engineering**:
- **80+ features** (vs original 15): temporal, technical, momentum, volatility
- **Watch-specific features**: luxury market dynamics, brand tiers, seasonality
- **Cross-asset features**: market-relative positioning

**Intelligent Feature Selection**:
- **Multiple algorithms**: correlation, mutual info, Random Forest, Lasso, RFE
- **Hybrid approach** combining best methods
- **Feature importance analysis** and correlation detection

**Watch Domain Expertise**:
- **Brand classification**: Ultra-luxury (Patek), High-luxury (Rolex), Mid-luxury (Tudor)
- **Watch type detection**: Sports vs Dress watches
- **Luxury market seasonality**: Holiday seasons, watch fairs, year-end effects
- **Price tier analysis**: Entry/Mid/High/Ultra luxury segments

### Machine Learning Models
The system employs multiple forecasting models with multi-horizon training:
- **Linear Regression**: Baseline linear trend analysis
- **Ridge Regression**: Regularized linear model for stability
- **Random Forest**: Ensemble method for complex patterns
- **XGBoost**: Gradient boosting for high accuracy

#### Multi-Horizon Training
Models are trained for multiple prediction horizons with automatic file detection:
- **1-day**: Short-term price movements
- **3-day**: Medium-term trends
- **7-day**: Weekly forecasts
- **14-day**: Bi-weekly projections
- **30-day**: Monthly forecasts

**Key Features**:
- **Automatic File Detection**: Trains on all CSV files in `data/scrape/prices/` by default (87 files)
- **Dual Logging**: Outputs to both terminal and timestamped log files in `./logs/`
- **Configurable Horizons**: Specify custom prediction horizons
- **Resume Capability**: Can restart interrupted training sessions
- **Comprehensive Metrics**: Multi-horizon performance comparison

**Training Commands**:
```bash
# Train all horizons on all files (default)
python -m src.ml.multi_horizon_training

# Train specific horizons on all files
python -m src.ml.multi_horizon_training --horizons 7 14 30

# Train with custom parameters
python -m src.ml.multi_horizon_training --horizons 1 7 --models linear xgboost --max-assets 50

# Analyze existing results only
python -m src.ml.multi_horizon_training --analyze-only
```

#### Model Outputs
- **Trained Models**: Saved to `data/output/models_{horizon}/`
- **Performance Metrics**: Model comparison in `model_comparison.csv`
- **Multi-Horizon Comparison**: Cross-horizon analysis in `multi_horizon_comparison.json`
- **Training Logs**: Detailed logs in `./logs/multi_horizon_training_[timestamp].log`
- **Visualizations**: Comprehensive plots in `data/output/visualizations/`

## Development Commands

### Watch Data Collection
```bash
# Generate watch URLs (Stage 1)
python -m src.collectors.watch.watch_urls

# Scrape price data (Stage 2)
python -m src.collectors.watch.scrape_runner

# Debug individual components
python -m src.collectors.watch.batch_scraper    # Direct batch processing
python -m src.collectors.watch.price_scraper    # Individual watch scraping
```

### Data Pipeline Commands

#### Simple Usage (Recommended)
```bash
# Run complete pipeline with defaults
python -m src.pipeline.run_pipeline --max-files 20

# Run complete pipeline with custom settings
python -m src.pipeline.run_pipeline --max-files 20 --interpolation-method spline --outlier-method isolation_forest

# Run dashboard  
python src/scripts/run_dashboard.py

# Run API server
python src/scripts/run_api.py

# Test pipeline with demo
cd src/pipeline && python demo.py
```

#### Advanced Usage (Custom Configuration)
```python
# Custom configuration example
from src.pipeline import run_pipeline, PipelineConfig

# Create custom configuration
config = PipelineConfig()
config.processing.interpolation_method = "spline"
config.processing.outlier_method = "isolation_forest"  
config.features.lag_periods = [1, 2, 3, 7, 14, 30]
config.modeling.feature_selection = True

# Run with custom settings
results = run_pipeline(asset_type="watch", config=config, max_files=20)
```

#### Component-Level Usage
```python
# Use individual components
from src.pipeline import DataLoader, DataProcessor, FeatureEngineer, PipelineConfig
from src.pipeline.assets.watch import WatchProcessor, WatchFeatureEngineer

config = PipelineConfig()

# Core components
loader = DataLoader(config, "watch")
processor = DataProcessor(config, "watch")  
engineer = FeatureEngineer(config, "watch")

# Asset-specific components
watch_processor = WatchProcessor(config, "watch")
watch_engineer = WatchFeatureEngineer(config, "watch")

# Process step by step
raw_data, load_report = loader.process(max_files=10)
clean_data, process_report = processor.process(raw_data)
featured_data, feature_report = engineer.process(clean_data)
```

## Development Standards

### Code Quality Requirements
- **Type Hints**: All functions must use proper type annotations
- **Logging**: Use centralized Loguru-based logging (see Technical Infrastructure)
- **Error Handling**: Implement specific exception handling with context
- **Documentation**: Add docstrings for all modules and functions

### Git Workflow
**CRITICAL REQUIREMENT**: Always commit after major changes and update CLAUDE.md

#### Commit Strategy
```bash
# After any significant change
git add .
git commit -m "descriptive message"
git push origin main
```

#### Commit Triggers (MUST commit after):
1. **File refactoring or renaming**
2. **Adding new features or components** 
3. **Fixing bugs or issues**
4. **Updating dependencies or configurations**
5. **Updating this CLAUDE.md file**

#### Commit Message Guidelines
- Use clear, descriptive commit messages
- **NEVER include**: "🤖 Generated with Claude Code" or "Co-Authored-By: Claude <noreply@anthropic.com>"
- Use present tense ("Add feature" not "Added feature")
- Focus on what was changed, not implementation details

#### Data Files Policy
- **NEVER commit files in `data/` directory** unless explicitly requested
- Data files are large and change frequently
- Use `.gitignore` to exclude `data/` by default

### CLAUDE.md Maintenance
**CRITICAL REQUIREMENT**: Update this file whenever making major changes

#### Update Guidelines
1. **Focus on current state** - Explain what exists now, not what changed
2. **Remove outdated sections** - Delete historical change logs and completed task lists
3. **Keep it concise** - Focus on essential information for development
4. **Update after refactoring** - Always sync with code changes
5. **Commit immediately** - Always commit CLAUDE.md updates

#### What to Update
- File names and paths when refactoring
- Command examples when scripts change
- Architecture descriptions when restructuring
- Development workflows when processes change

## Technical Infrastructure

### Centralized Logging System
The system uses a **Loguru-based centralized logging** configuration that eliminates manual logging setup across components:

**Key Features**:
- **Zero Configuration**: Components get pre-configured loggers with one import
- **Component-Specific Files**: Each component writes to its own log file (e.g., `ml_training.log`, `data_pipeline.log`)
- **Automatic Rotation**: Log files rotate at 10MB with 30-day retention and compression
- **Dual Output**: Simultaneous terminal and file logging with color-coded console output
- **Structured Logging**: Optional JSON format for log analysis and monitoring
- **Thread-Safe**: Concurrent logging across pipeline components

**Usage Examples**:
```python
# Multi-horizon training
from src.utils.logging_config import get_training_logger
logger = get_training_logger()
logger.info("Training {model} on {files} files", model="xgboost", files=87)

# Data validation
from src.utils.logging_config import get_validation_logger
logger = get_validation_logger()
logger.info("Validated {count} CSV files", count=87)

# Web scraping
from src.utils.logging_config import get_scraping_logger
logger = get_scraping_logger()
logger.info("Scraped {watches} watches from {brand}", watches=10, brand="Rolex")

# Custom component
from src.utils.logging_config import setup_logging
logger = setup_logging("custom_component", json_format=True)
logger.info("Custom operation completed", success=True, duration=45.2)
```

**Log File Locations**:
- `./logs/ml_training.log` → Multi-horizon training and model operations
- `./logs/data_pipeline.log` → Data processing and feature engineering
- `./logs/data_validation.log` → CSV validation and data quality checks  
- `./logs/web_scraping.log` → Watch price scraping operations
- `./logs/aurevia_pipeline.log` → General pipeline operations

### Web Scraping Capabilities
- **Cloudflare Bypass**: Stealth browser configuration with anti-detection
- **Rate Limiting**: Built-in delays (10-20 seconds) between requests
- **Progress Persistence**: All scraping progress saved continuously
- **Resume Capability**: Can restart from interruption points
- **Incremental Updates**: Only scrapes newer data points
- **Error Handling**: Comprehensive retry logic and error screenshots

### Dependencies
- **Core**: streamlit, pandas, numpy, matplotlib, seaborn, plotly
- **ML**: scikit-learn, xgboost, statsmodels
- **API**: fastapi, uvicorn, pydantic
- **Logging**: loguru (centralized logging system)
- **Scraping**: selenium, beautifulsoup4, webdriver-manager (optional)
- **Development**: pytest, black, isort, mypy

### Pipeline Outputs
The enhanced ML pipeline generates comprehensive outputs:

#### Core Data Files
- `data/output/featured_data.csv` → Enhanced dataset with 80+ engineered features
- `data/scrape/prices/*.csv` → Raw historical price data (input)

#### Analysis Reports  
- `data/output/brand_summary.csv` → Performance analysis by watch brand
- `data/output/processing_steps.csv` → Detailed pipeline execution log
- `data/output/feature_selection_summary.csv` → Feature importance and selection results
- `data/output/model_summary.csv` → ML model performance metrics

#### Visualization Outputs
- `data/output/visualizations/` → Comprehensive visualization directory
  - `*_predictions.png` → Model prediction plots
  - `*_residuals.png` → Residual analysis plots
  - `*_importance.png` → Feature importance plots
  - `*_decomposition.png` → Seasonal decomposition plots
  - `*_complete_forecast.png` → Complete forecasting plots
  - `complexity_vs_performance.png` → Model complexity analysis

#### Configuration and Logs
- **Centralized Logging System**: All components use Loguru-based unified logging
- **Component-Specific Logs**: Separate log files for each pipeline component in `./logs/`
- **Automatic Log Rotation**: 10MB file size rotation with 30-day retention
- **Structured Logging**: JSON format support for advanced log analysis
- **Dual Output**: Both terminal and file logging with consistent formatting
- Pipeline execution logs with structured logging
- Data quality scores and validation reports
- Feature correlation matrices and selection rationale

## Troubleshooting

### Common Issues
- **Missing Data Files**: Run the complete data pipeline (URL generation → scraping → ML processing)
- **Scraping Failures**: Check ChromeDriver installation and Cloudflare status
- **Import Errors**: Ensure all requirements are installed (`pip install -r requirements.txt`)
- **Path Issues**: Run commands from project root directory
- **Feature Engineering Errors**: Check data quality and minimum data requirements (30+ points recommended)
- **Pipeline Configuration**: Verify `PipelineConfig` settings match your data structure
- **API Issues**: Check if FastAPI server is running on correct port (8000)
- **Dashboard Issues**: Ensure Streamlit dependencies are installed

### Debug Information
- **Structured Logging**: All components use Python logging with detailed execution traces
- **Dual Logging System**: Multi-horizon training outputs to both terminal and timestamped log files
- **Training Logs**: Complete training history saved in `./logs/multi_horizon_training_[timestamp].log`
- **Error Screenshots**: Saved to `data/scrape/prices/error/` directory
- **Progress Tracking**: Real-time progress in `data/scrape/scraping_progress.json`
- **Pipeline Reports**: Detailed execution logs in `data/output/processing_steps.csv`
- **Data Quality Metrics**: Validation scores and outlier detection results

## Extending the Pipeline

### Adding New Asset Types
The modular architecture supports easy extension to new asset classes:

```bash
# Create new asset module
mkdir -p src/pipeline/assets/stocks
touch src/pipeline/assets/stocks/__init__.py
touch src/pipeline/assets/stocks/stock.py
```

### Custom Feature Engineering
```python
# Extend base feature engineer
from src.pipeline.features import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def add_custom_features(self, df, price_column):
        # Add your domain-specific features
        df['custom_indicator'] = df[price_column].rolling(10).mean()
        return df
```

### Configuration Customization
```python
# Create asset-specific configuration
from src.pipeline.config import PipelineConfig

config = PipelineConfig()
config.assets.asset_type = "stocks"
config.assets.price_column = "close_price"
config.features.lag_periods = [1, 5, 22]  # Daily, weekly, monthly
```

## Repository Information
- **Remote Origin**: https://github.com/AurevIa-Capital/aurevia-capital.git
- **Main Branch**: main
- **Development Practice**: Commit regularly and push frequently for backup and collaboration