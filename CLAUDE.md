# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modern Streamlit-based luxury watch price forecasting application that analyzes historical price data and generates predictions using multiple machine learning models. The system includes automated data collection via web scraping and comprehensive price analysis tools.

## Quick Start

```bash
# Run the main application
python run_app.py

# For mass scraping (100 watches)
pip install -r requirements_scraping.txt
python scrape_100_watches.py
```

## Project Architecture

### Core Application Structure
- **Entry Point**: `run_app.py` → configures Python path and launches Streamlit app
- **Main App**: `src/app/app.py` → application configuration, routing, and page orchestration
- **Pages**: `src/app/pages/` → individual page implementations (overview, analysis)
- **Components**: `src/app/components/components.py` → reusable UI components and visualizations
- **Utils**: `src/app/utils/` → data loading, formatting, styling utilities

### Data Processing Pipeline
1. **Web Scraping**: `src/dev/scraper.py` → Cloudflare-bypass scraping system
2. **Data Analysis**: `src/dev/data_prep.py` → time series analysis and visualization
3. **Model Training**: `src/dev/modelling.py` → ML model training and evaluation
4. **Data Storage**: `data/watches/*.csv` → historical price data
5. **Results**: `data/output/` → model predictions and analysis results

### Machine Learning Models
The system employs 5 forecasting models:
- **Linear Regression**: Baseline linear trend analysis
- **Random Forest**: Ensemble method for complex patterns
- **XGBoost**: Gradient boosting for high accuracy
- **ARIMA**: Classical time series forecasting
- **SARIMA**: Seasonal time series with trend decomposition

## Development Commands

### Application Development
```bash
# Run development server
python run_app.py

# Data analysis workflow
python src/dev/data_prep.py
python src/dev/modelling.py
```

### Mass Scraping System
```bash
# Scrape 100 watches across 10 brands
python scrape_100_watches.py

# Individual components
python src/dev/watch_discovery.py  # Discover watch URLs
python src/dev/scraper.py          # Cloudflare-bypass scraping engine
python src/dev/mass_scraper.py     # Orchestrate mass scraping
```

## Development Workflow Standards

### Code Quality & Maintenance
- **Refactoring Policy**: Always refactor and update this CLAUDE.md when making major changes
- **No Backwards Compatibility**: Remove deprecated code immediately, use modern libraries only
- **Type Hints**: All functions must use proper type annotations
- **Logging**: Use structured logging instead of print statements
- **Error Handling**: Implement specific exception handling with context
- **Warning Management**: Use targeted warning suppression, never global

### Git Commit Strategy
**CRITICAL**: Commit checkpoints regularly throughout development:

```bash
# After completing any significant change or feature
git add .
git commit -m "descriptive message"

# Push regularly to maintain backup
git push origin main
```

**Commit Triggers:**
- After completing any file refactoring
- When adding new features or components
- After fixing bugs or issues
- Before starting complex changes
- When updating dependencies or configurations
- After updating this CLAUDE.md file

### Selenium Infrastructure
- **Shared Utilities**: `src/utils/selenium_utils.py` contains reusable driver factories
- **Cloudflare Bypass**: Stealth browser configuration with anti-detection
- **Resource Management**: Automatic driver cleanup and error handling

## Data Formats & Conventions

### File Naming
- **Watch Data**: `{watch-id}-{brand-model}.csv`
- **Model Outputs**: `{watch-id}_{model}_{type}.{ext}`
- **Progress Files**: `scraping_progress.json`, `watch_targets.json`

### Data Structure
```csv
# Historical price data format (data/watches/*.csv)
date,price(SGD)
2024-05-11,15463
2024-05-14,15453

# Model outputs (data/output/)
{watch-id}_{model-type}_predictions.png
{watch-id}_price_forecast.csv
model_summary.csv
featured_data.csv
```

## Technical Standards

### Plotting & Visualization
- Uses modern `sns.set_theme()` instead of deprecated seaborn syntax
- Matplotlib configured with `plt.rcParams` for consistency
- All plots use consistent styling and color schemes

### Web Scraping Considerations
- **Rate Limiting**: Built-in delays (10-20 seconds) between requests
- **Parallel Processing**: Maximum 2 concurrent workers for politeness
- **Progress Persistence**: All scraping progress saved continuously
- **Resume Capability**: Can restart from interruption points
- **Cloudflare Handling**: Automatic detection and bypass mechanisms

### Dependencies
- **Core**: streamlit, pandas, numpy, matplotlib, seaborn
- **ML**: scikit-learn, xgboost, statsmodels
- **Scraping**: selenium, beautifulsoup4, requests, webdriver-manager

## Testing & Data Pipeline

### Required Data Files
The application expects these files to exist:
- `data/output/model_summary.csv` → model performance metrics
- `data/output/featured_data.csv` → processed visualization data
- `data/watches/*.csv` → historical price data

### Data Pipeline Setup
If data files don't exist, run the complete pipeline:
1. Scrape data: `python scrape_100_watches.py`
2. Process data: `python src/dev/data_prep.py`
3. Train models: `python src/dev/modelling.py`

## Troubleshooting

### Common Issues
- **Missing Data Files**: Run the complete data pipeline
- **Scraping Failures**: Check ChromeDriver installation and Cloudflare status
- **Import Errors**: Ensure all requirements are installed
- **Path Issues**: Run commands from project root directory

### Debug Logging
All components use Python logging. Set `logging.basicConfig(level=logging.DEBUG)` for detailed output.

### Browser Dependencies
Mass scraping requires Chrome/Chromium and ChromeDriver (automatically managed by webdriver-manager).

## Repository Information
- **Remote Origin**: https://github.com/simplysindy/ACTP.git
- **Main Branch**: main
- **Commit regularly and push frequently for backup and collaboration**