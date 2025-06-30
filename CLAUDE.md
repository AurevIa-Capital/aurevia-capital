# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modern luxury watch price forecasting platform that analyzes historical price data and generates predictions using multiple machine learning models. The system features a modular architecture with design patterns, event-driven monitoring, FastAPI backend, Streamlit dashboard, and comprehensive data collection pipeline with Cloudflare bypass capabilities.

## Quick Start

### Unified CLI Interface
```bash
# Use the unified CLI for all operations
python src/cli.py --help

# Data collection
python src/cli.py scrape urls                     # Generate watch URLs
python src/cli.py scrape prices                   # Scrape price data

# Data processing and training
python src/cli.py pipeline --max-files 20         # Run data pipeline
python src/cli.py train --horizons 7 14 30        # Train ML models

# Full pipeline with new architecture
python src/cli.py full --horizons 7 14 30 --models linear xgboost --max-assets 10

# Serving
python src/cli.py serve dashboard                  # Start Streamlit dashboard
python src/cli.py serve api                       # Start FastAPI server

# Visualization
python src/cli.py visualize --max-assets 5        # Create visualizations
```

### Legacy Direct Commands (still supported)
```bash
# Direct module execution (legacy)
python -m src.collectors.watch.watch_urls
python -m src.collectors.watch.scrape_runner
python -m src.pipeline.run_pipeline --max-files 20
python -m src.ml.multi_horizon_training --horizons 7 14 30
python src/scripts/run_dashboard.py
python src/scripts/run_api.py
```

## Current Architecture

### Repository Structure
The codebase is organized with modern design patterns and consolidated modules:

```
src/
├── cli.py                          # Unified CLI entry point
├── core/                          # Design patterns and architecture
│   ├── model_factory.py           # Factory pattern for ML models
│   ├── config_builder.py          # Builder pattern for configuration
│   ├── data_store.py              # Centralized data management
│   ├── pipeline_strategy.py       # Strategy pattern for pipeline components
│   ├── event_system.py            # Observer pattern for monitoring
│   ├── command_pattern.py         # Command pattern for CLI operations
│   └── pipeline_orchestrator.py   # Main orchestrator combining all patterns
├── collectors/
│   ├── selenium_utils.py          # Selenium utilities
│   └── watch/                     # Watch data collection pipeline
├── pipeline/                      # Data processing and feature engineering
│   ├── config.py                  # Configuration classes
│   ├── loader.py                  # Data loading
│   ├── processor.py               # Data processing
│   ├── features.py                # Feature engineering
│   └── assets/                    # Asset-specific processing
├── ml/                           # Machine learning components
│   ├── base.py                   # Base classes for ML models
│   ├── models.py                 # All ML model implementations
│   ├── training.py               # Training logic and validation
│   ├── visualization.py          # Visualization classes
│   └── multi_horizon_training.py # Multi-horizon training orchestrator
├── api/                          # FastAPI backend
├── dashboard/                    # Streamlit interface
├── scripts/                      # Standalone scripts
└── utils/                        # Shared utilities
```

### Core Design Patterns

#### Model Factory Pattern (`src/core/model_factory.py`)
Centralized model creation and management:
- `ModelFactory`: Creates model instances with validation
- `ModelConfigBuilder`: Fluent API for model configuration
- Eliminates code duplication in model instantiation
- Supports all model types: linear, ensemble, time series

#### Configuration Builder Pattern (`src/core/config_builder.py`)
Fluent API for building pipeline configurations:
- `PipelineConfigBuilder`: Main configuration builder
- `TrainingConfigBuilder`: Specialized for training workflows
- Validation and error handling built-in
- CLI argument integration and JSON loading/saving

#### Centralized Data Store (`src/core/data_store.py`)
Unified data management interface:
- `DataStore`: Abstract interface for data operations
- `FileSystemDataStore`: File-based implementation
- Data managers for different types: scraped, processed, models
- Metadata tracking and organized storage

#### Strategy Pattern (`src/core/pipeline_strategy.py`)
Modular and interchangeable pipeline components:
- `PipelineStrategy`: Abstract base for pipeline stages
- `StrategyFactory`: Creates strategies for different stages
- Stage-specific strategies: Data Loading, Processing, Feature Engineering, Training, Visualization
- Enables swapping pipeline components without code changes

#### Observer Pattern (`src/core/event_system.py`)
Event-driven monitoring and logging:
- `EventBus`: Central event dispatcher
- Built-in observers: `LoggingObserver`, `ProgressObserver`, `MetricsObserver`, `FileLogObserver`
- Real-time pipeline visibility and monitoring
- Comprehensive event types for all pipeline operations

#### Command Pattern (`src/core/command_pattern.py`)
Structured CLI operations with event support:
- `Command`: Abstract base for all operations
- `CommandInvoker`: Executes commands with event notifications
- `CommandFactory`: Creates commands from CLI arguments
- Support for undo/redo and composite operations

#### Pipeline Orchestrator (`src/core/pipeline_orchestrator.py`)
Unified orchestrator combining all design patterns:
- Coordinates Strategy, Observer, and Command patterns
- Event-driven pipeline execution
- Comprehensive monitoring and metrics collection
- Integration with Factory, Builder, and Data Store patterns

## Machine Learning Architecture

### Consolidated ML Structure
All ML functionality is organized in focused modules:

#### `src/ml/models.py` - All Model Implementations
Contains all model classes:
- **Linear Models**: LinearRegressionModel, RidgeModel, LassoModel, PolynomialRegressionModel
- **Ensemble Models**: RandomForestModel, XGBoostModel, GradientBoostingModel
- **Time Series Models**: ARIMAModel, SARIMAModel
- **Usage**: `from src.ml.models import LinearRegressionModel, XGBoostModel`

#### `src/ml/training.py` - Complete Training System
Contains all training functionality:
- **Validation**: TimeSeriesValidator with comprehensive metrics
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Training Orchestration**: ModelTrainer for single/multi-asset training
- **Usage**: `from src.ml.training import ModelTrainer, TimeSeriesValidator`

#### `src/ml/visualization.py` - All Visualization Classes
Contains all visualization functionality:
- **Performance Plots**: PerformanceVisualizer for metrics and comparisons
- **Forecasting Plots**: ForecastingVisualizer for predictions and residuals
- **Comparison Plots**: ComparisonVisualizer for cross-model analysis
- **Enhanced Plots**: EnhancedForecastingVisualizer for advanced analysis
- **Usage**: `from src.ml.visualization import PerformanceVisualizer, ForecastingVisualizer`

### Multi-Horizon Training
Models are trained for multiple prediction horizons using **RMSE as the primary performance metric**:
- **1-day**: Short-term price movements
- **3-day**: Medium-term trends  
- **7-day**: Weekly forecasts
- **14-day**: Bi-weekly projections
- **30-day**: Monthly forecasts

**Training Commands**:
```bash
# Using unified CLI (recommended)
python src/cli.py train --horizons 7 14 30
python src/cli.py train --models linear xgboost --max-assets 50

# Using new architecture
python src/cli.py full --horizons 7 14 30 --models linear xgboost

# Direct module execution
python -m src.ml.multi_horizon_training --horizons 7 14 30
python -m src.ml.multi_horizon_training --analyze-only
```

## Data Collection Pipeline

### Watch Data Collection Components
Located in `src/collectors/watch/`:

#### Core Components
- **`base_scraper.py`**: Abstract base class with shared scraping utilities
- **`watch_urls.py`**: Discovers 100 watch URLs (10 per brand) from WatchCharts  
- **`price_scraper.py`**: Extracts historical price data using Chart.js injection
- **`batch_scraper.py`**: Manages batch processing with progress tracking
- **`scrape_runner.py`**: User-friendly CLI interface for mass scraping

#### Selenium Integration
- **`selenium_utils.py`**: Selenium utilities with domain-specific placement
- **Features**: Cloudflare bypass, stealth configuration, retry logic
- **Usage**: `from src.collectors.selenium_utils import SeleniumDriverFactory`

### Data Pipeline (`src/pipeline/`)
Advanced data processing with 80+ engineered features:

```bash
# Using unified CLI
python src/cli.py pipeline --max-files 20 --interpolation-method spline

# Using new architecture with full monitoring
python src/cli.py full --max-files 20

# Direct execution  
python -m src.pipeline.run_pipeline --max-files 20
```

**Key Features**:
- **Multiple interpolation methods**: backfill, linear, spline, seasonal, hybrid
- **Sophisticated outlier detection**: IQR, Z-score, Isolation Forest
- **Watch domain expertise**: Brand tiers, seasonality, luxury market dynamics
- **Feature selection**: Multiple algorithms with hybrid approach

## Development Commands

### Using Unified CLI (Recommended)
```bash
# Data collection
python src/cli.py scrape urls                     # Generate watch URLs
python src/cli.py scrape prices                   # Scrape price data

# Data processing  
python src/cli.py pipeline --max-files 20         # Process data
python src/cli.py pipeline --interpolation-method spline --outlier-method isolation_forest

# Model training
python src/cli.py train --horizons 7 14 30        # Train models
python src/cli.py train --models linear xgboost --max-assets 50
python src/cli.py train --analyze-only            # Analyze existing results

# Full pipeline with new architecture
python src/cli.py full --horizons 7 14 30 --models linear xgboost --max-assets 10

# Visualization
python src/cli.py visualize --max-assets 5        # Create visualizations
python src/cli.py visualize --specific-assets "Rolex-Submariner-638"

# Serving
python src/cli.py serve dashboard                  # Start Streamlit dashboard
python src/cli.py serve api                       # Start FastAPI server
```

### Legacy Direct Commands
```bash
# Data collection (legacy)
python -m src.collectors.watch.watch_urls
python -m src.collectors.watch.scrape_runner

# Pipeline processing (legacy)
python -m src.pipeline.run_pipeline --max-files 20

# Model training (legacy)
python -m src.ml.multi_horizon_training --horizons 7 14 30

# Visualization (legacy)
python -m src.ml.create_visualizations --max-assets 5

# Serving (legacy)
python src/scripts/run_dashboard.py
python src/scripts/run_api.py
```

## Data Structure

### Input Data
```
data/scrape/
├── url/watch_targets_100.json          # Watch URLs and metadata
├── prices/{Brand}-{Model}-{ID}.csv     # Historical price data per watch
└── scraping_progress.json              # Progress tracking
```

### Output Data  
```
data/output/
├── featured_data.csv                   # Enhanced dataset (80+ features)
├── brand_summary.csv                   # Performance analysis by brand
├── models/                            # Trained models by horizon
│   ├── models_7_day/
│   ├── models_14_day/
│   └── model_comparison.csv
└── visualizations/                    # Organized visualization structure
    ├── {watch_model}/{ml_model}/      # Individual model plots
    └── aggregate/                     # Cross-model analysis
```

## Technical Infrastructure

### Design Pattern Architecture
The codebase implements multiple design patterns for maintainability and extensibility:

#### Factory Pattern
```python
from src.core.model_factory import ModelFactory

# Create models using factory
model = ModelFactory.create_model('xgboost', config)
available_models = ModelFactory.get_available_models()
```

#### Builder Pattern
```python
from src.core.config_builder import PipelineConfigBuilder

# Build configuration with fluent API
config = (PipelineConfigBuilder()
          .with_data_paths("data", "scrape/prices", "output")
          .with_processing_options("spline", "isolation_forest")
          .with_modeling_options(test_size=0.2, random_state=42)
          .build())
```

#### Strategy Pattern
```python
from src.core.pipeline_strategy import StrategyFactory, PipelineStage

# Create and execute strategies
strategy = StrategyFactory.create_strategy(PipelineStage.DATA_LOADING, config)
result = strategy.execute(input_data, max_files=20)
```

#### Observer Pattern
```python
from src.core.event_system import create_default_event_system, EventType

# Create event system with monitoring
event_bus = create_default_event_system()
event_bus.publish_event(EventType.PIPELINE_STARTED, source="MyComponent")
```

#### Command Pattern
```python
from src.core.command_pattern import CommandFactory, CommandInvoker

# Execute commands with event support
command_invoker = CommandInvoker(event_bus)
command = CommandFactory.create_training_command(args)
result = command_invoker.execute_command(command)
```

#### Pipeline Orchestrator
```python
from src.core.pipeline_orchestrator import create_orchestrator

# Unified pipeline execution with all patterns
orchestrator = create_orchestrator(config)
result = orchestrator.execute_full_pipeline(
    horizons=[7, 14, 30],
    models=['linear', 'xgboost'],
    max_assets=10
)
```

### Centralized Logging System
Loguru-based logging with automatic configuration:

```python
# Training operations
from src.utils.logging_config import get_training_logger
logger = get_training_logger()

# Data validation
from src.utils.logging_config import get_validation_logger
logger = get_validation_logger()

# Web scraping
from src.utils.logging_config import get_scraping_logger
logger = get_scraping_logger()
```

**Log Files**: `./logs/ml_training.log`, `./logs/data_pipeline.log`, `./logs/web_scraping.log`

### Organized Visualization System
Hierarchical structure for all visualizations:

```python
from src.utils.visualization_utils import get_organized_visualization_path

# Individual watch-model visualization
viz_path = get_organized_visualization_path(
    "data/output/visualizations", 
    "Rolex-Submariner-638", 
    "xgboost", 
    "predictions"
)
# → data/output/visualizations/Rolex-Submariner-638/xgboost/predictions.png
```

### Core Services (`src/core/`)
Modern architecture with design patterns:
- **`model_factory.py`**: Factory pattern for model creation
- **`config_builder.py`**: Builder pattern for configuration
- **`data_store.py`**: Centralized data management
- **`pipeline_strategy.py`**: Strategy pattern for modularity
- **`event_system.py`**: Observer pattern for monitoring
- **`command_pattern.py`**: Command pattern for CLI
- **`pipeline_orchestrator.py`**: Unified orchestrator

## Development Standards

### Code Quality Requirements
- **Type Hints**: All functions must use proper type annotations
- **Logging**: Use centralized logging system (see Technical Infrastructure)
- **Error Handling**: Implement specific exception handling with context
- **Documentation**: Add docstrings for all modules and functions

### Legacy Code Removal Policy
**CRITICAL REQUIREMENT**: Always remove legacy, deprecated, and duplicated code after refactoring or implementing new functions.

#### When to Remove Legacy Code:
1. **After implementing design patterns** - Remove old instantiation code
2. **After refactoring modules** - Remove old interfaces and duplicated functions
3. **After consolidating functionality** - Remove scattered implementations
4. **After adding new abstractions** - Remove hard-coded implementations

#### Legacy Code Removal Checklist:
- [ ] Remove old factory functions (e.g., `create_model_factory()` in base.py)
- [ ] Remove duplicated configuration logic
- [ ] Remove hard-coded file paths after introducing DataStore
- [ ] Remove scattered model instantiation after implementing ModelFactory
- [ ] Update imports to use new patterns
- [ ] Remove deprecated command-line interfaces
- [ ] Clean up unused utility functions

### Import Best Practices
Use these import patterns with the current architecture:

```python
# Design Patterns Architecture
from src.core.model_factory import ModelFactory, ModelConfigBuilder
from src.core.config_builder import PipelineConfigBuilder, TrainingConfigBuilder
from src.core.data_store import create_data_store, ScrapedDataManager

# Strategy Pattern
from src.core.pipeline_strategy import (
    StrategyFactory, PipelineStage, create_default_strategies
)

# Observer Pattern
from src.core.event_system import (
    EventBus, EventType, EventPriority, create_default_event_system,
    LoggingObserver, ProgressObserver, MetricsObserver
)

# Command Pattern
from src.core.command_pattern import (
    CommandFactory, CommandInvoker, Command, CompositeCommand
)

# Pipeline Orchestrator (combines all patterns)
from src.core.pipeline_orchestrator import create_orchestrator, PipelineOrchestrator

# ML models (consolidated)
from src.ml.models import LinearRegressionModel, XGBoostModel

# Training (consolidated)  
from src.ml.training import ModelTrainer, TimeSeriesValidator

# Visualization (consolidated)
from src.ml.visualization import PerformanceVisualizer, ForecastingVisualizer

# Collectors (domain-specific)
from src.collectors.selenium_utils import SeleniumDriverFactory

# DEPRECATED - DO NOT USE:
# from src.ml.base import create_model_factory  # Use ModelFactory instead
```

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
4. **Updating this CLAUDE.md file**

#### Data Files Policy
- **NEVER commit files in `data/` directory** unless explicitly requested
- Use `.gitignore` to exclude `data/` by default

## Extending the Platform

### Adding New Asset Types
```bash
# Create new asset module
mkdir -p src/pipeline/assets/stocks
touch src/pipeline/assets/stocks/stock.py
```

### Custom Feature Engineering
```python
from src.pipeline.features import FeatureEngineer

class CustomFeatureEngineer(FeatureEngineer):
    def add_custom_features(self, df, price_column):
        df['custom_indicator'] = df[price_column].rolling(10).mean()
        return df
```

### Custom Models
```python
# Add to src/ml/models.py
from .models import BaseTimeSeriesModel

class CustomModel(BaseTimeSeriesModel):
    def _fit_model(self, X, y):
        # Custom training logic
        pass

# Register with factory
from src.core.model_factory import ModelFactory
ModelFactory.register_model('custom', CustomModel, 'ensemble')
```

### Custom Pipeline Strategies
```python
from src.core.pipeline_strategy import PipelineStrategy, StrategyFactory

class CustomStrategy(PipelineStrategy):
    def execute(self, input_data, **kwargs):
        # Custom pipeline logic
        pass

# Register with factory
StrategyFactory.register_strategy(PipelineStage.DATA_PROCESSING, 'custom', CustomStrategy)
```

### Custom Event Observers
```python
from src.core.event_system import EventObserver

class CustomObserver(EventObserver):
    def handle_event(self, event):
        # Custom event handling
        pass

# Subscribe to event bus
event_bus.subscribe(CustomObserver())
```

## Troubleshooting

### Common Issues
- **Import Errors**: Use new consolidated import paths (see Import Best Practices)
- **Missing Data Files**: Run complete pipeline: `python src/cli.py scrape urls` → `scrape prices` → `pipeline` → `train`
- **CLI Issues**: Use `python src/cli.py --help` for all available commands
- **Path Issues**: Run commands from project root directory
- **Dependencies**: Ensure `pip install -r requirements.txt`

### Debug Information
- **Centralized Logs**: Check `./logs/` directory for component-specific logs
- **Training Logs**: Multi-horizon training creates timestamped log files
- **Progress Tracking**: Scraping progress in `data/scrape/scraping_progress.json`
- **Event Monitoring**: Use event system for real-time pipeline visibility
- **Visualization**: Use `--verbose` flag with CLI commands for detailed output

### New Architecture Debugging
- **Event System**: Monitor events in real-time with built-in observers
- **Strategy Pattern**: Swap strategies for different pipeline behaviors
- **Command Pattern**: Review command execution history and results
- **Orchestrator**: Use comprehensive monitoring and metrics collection

## Repository Information
- **Remote Origin**: https://github.com/AurevIa-Capital/aurevia-capital.git
- **Main Branch**: main  
- **Development Practice**: Commit regularly with descriptive messages