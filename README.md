# Luxury Watch Price Forecasting Platform

A streamlined, scalable platform for luxury watch price forecasting with multi-asset support capabilities.

## Project Structure

```
aurevIa_timepiece/
├── forecasting-platform/       # Main application platform
│   ├── core/                   # Shared abstractions and utilities
│   ├── api/                    # REST API gateway
│   ├── dashboard/              # Plugin-based user interface
│   ├── collectors/             # Data collection modules
│   ├── models/                 # ML models and forecasting
│   ├── utils/                  # Shared utilities
│   └── scripts/                # Entry point scripts
├── data/                       # Data files and outputs
└── tests/                      # Test files
```

## Quick Start

### Prerequisites
```bash
cd aurevIa_timepiece
pip install -e .
```

### Running the Applications

#### Dashboard
```bash
# Using entry point
watch-forecaster

# Or directly
python forecasting-platform/scripts/run_dashboard.py
```

#### API Gateway
```bash
# Using entry point
watch-api

# Or directly
python forecasting-platform/scripts/run_api.py
```

## Features

### Current Capabilities
- **Watch Price Forecasting**: ML-powered predictions using 5 different models
- **Market Analysis**: Technical analysis with moving averages and volatility
- **Data Collection**: Automated web scraping with Cloudflare bypass
- **Interactive Dashboard**: Plugin-based Streamlit interface
- **REST API**: FastAPI-based programmatic access

### Models Available
- Linear Regression (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting) 
- ARIMA (time series)
- SARIMA (seasonal time series)

## Installation Options

### Core Platform
```bash
pip install -e .
```

### With Web Scraping
```bash
pip install -e .[scraping]
```

### With API Gateway
```bash
pip install -e .[api]
```

### Development Setup
```bash
pip install -e .[dev]
```

### Everything
```bash
pip install -e .[all]
```

## Key Directories

### `forecasting-platform/core/`
Shared abstractions for multi-asset support:
- Base asset classes
- Data schemas (PricePoint, Forecast)
- Event system
- Dependency injection

### `forecasting-platform/dashboard/`
Plugin-based Streamlit dashboard with multi-asset support architecture

### `forecasting-platform/collectors/`
Data collection modules:
- **`watch/`**: Watch-specific scrapers with Cloudflare bypass

### `forecasting-platform/models/`
Machine learning pipeline:
- **`forecasting/`**: Time series models and data preparation

## Usage Examples

### Data Analysis
```bash
# Prepare data
python forecasting-platform/models/forecasting/data_prep.py

# Train models
python forecasting-platform/models/forecasting/modelling.py
```

### Data Collection  
```bash
# Scrape 100 watches
python forecasting-platform/scripts/scrape_100_watches.py

# Single watch scraping
python -m forecasting-platform.collectors.watch.scraper
```

## API Endpoints

When running the API gateway (`watch-api`):

- **Assets**: `GET /api/v1/assets/watch`
- **Forecasts**: `GET /api/v1/forecasts/watch/{asset_id}`
- **History**: `GET /api/v1/assets/watch/{asset_id}/history`
- **Documentation**: `http://localhost:8000/docs`

## Data Format

Historical price data format:
```csv
date,price(SGD)
2024-05-11,15463
2024-05-14,15453
```

## Development

### Code Quality
```bash
# Format code
black forecasting-platform/
isort forecasting-platform/

# Type checking  
mypy forecasting-platform/

# Run tests
pytest tests/
```

### Project History
1. **Phase 1**: Core infrastructure and plugin architecture ✅
2. **Refactoring**: Code modernization and dependency cleanup ✅  
3. **Consolidation**: Unified directory structure ✅
4. **Streamlining**: Legacy code removal and simplification ✅
5. **Phase 2**: Existing code integration (Planned)
6. **Phase 3**: Multi-asset expansion (Planned)

## Documentation

- **Full Documentation**: See `CLAUDE.md` for comprehensive development guide
- **API Docs**: Available at `/docs` when running API gateway
- **Plugin Development**: See `forecasting-platform/dashboard/` for examples

## Contributing

1. Follow the established directory structure
2. Use type hints throughout
3. Add tests for new functionality
4. Update documentation
5. Run code quality checks before committing

## License

See repository license for details.

## Support

- **Issues**: Use GitHub issues for bug reports
- **Documentation**: Check `CLAUDE.md` for detailed guidance
- **API Reference**: Visit `/docs` endpoint when running API