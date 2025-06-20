# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modern luxury watch price forecasting application that analyzes historical price data and generates predictions using multiple machine learning models. The system includes a comprehensive two-stage data collection pipeline via web scraping with Cloudflare bypass capabilities.

## Quick Start

```bash
# Stage 1: Generate watch URLs (100 watches, 10 per brand)
python -m src.collectors.watch.watch_urls

# Stage 2: Scrape price data 
python -m src.collectors.watch.scrape_runner

# Run the dashboard
python src/scripts/run_dashboard.py
```

## Current Architecture

### Core Application Structure
- **Entry Point**: `src/scripts/run_dashboard.py` → launches Streamlit dashboard
- **Data Collection**: `src/collectors/watch/` → watch price scraping pipeline
- **ML Models**: `src/models/forecasting/` → time series analysis and prediction
- **Utils**: `src/utils/` → shared utilities (Selenium, helpers)

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
│   ├── model_summary.csv             # ML model performance metrics
│   ├── featured_data.csv             # Processed visualization data
│   └── {watch-id}_{model}_{type}.{ext}
└── images/                           # Watch reference images
```

### File Naming Standards
- **Watch Targets**: `watch_targets_100.json`
- **Price Data**: `{Brand}-{Model}-{watch_id}.csv` (e.g., `Rolex-Submariner-638.csv`)
- **Progress Files**: `scraping_progress.json`

### Machine Learning Models
The system employs 5 forecasting models:
- **Linear Regression**: Baseline linear trend analysis
- **Random Forest**: Ensemble method for complex patterns
- **XGBoost**: Gradient boosting for high accuracy
- **ARIMA**: Classical time series forecasting
- **SARIMA**: Seasonal time series with trend decomposition

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

### ML Pipeline
```bash
# Data analysis and visualization
python src/models/forecasting/data_prep.py

# Model training and evaluation
python src/models/forecasting/modelling.py

# Run dashboard
python src/scripts/run_dashboard.py
```

## Development Standards

### Code Quality Requirements
- **Type Hints**: All functions must use proper type annotations
- **Logging**: Use structured logging instead of print statements
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

### Web Scraping Capabilities
- **Cloudflare Bypass**: Stealth browser configuration with anti-detection
- **Rate Limiting**: Built-in delays (10-20 seconds) between requests
- **Progress Persistence**: All scraping progress saved continuously
- **Resume Capability**: Can restart from interruption points
- **Incremental Updates**: Only scrapes newer data points
- **Error Handling**: Comprehensive retry logic and error screenshots

### Dependencies
- **Core**: streamlit, pandas, numpy, matplotlib, seaborn
- **ML**: scikit-learn, xgboost, statsmodels
- **Scraping**: selenium, beautifulsoup4, webdriver-manager

### Required Data Files
The application expects these files to exist:
- `data/output/model_summary.csv` → model performance metrics
- `data/output/featured_data.csv` → processed visualization data
- `data/scrape/prices/*.csv` → historical price data

## Troubleshooting

### Common Issues
- **Missing Data Files**: Run the complete data pipeline (URL generation → scraping → ML processing)
- **Scraping Failures**: Check ChromeDriver installation and Cloudflare status
- **Import Errors**: Ensure all requirements are installed
- **Path Issues**: Run commands from project root directory

### Debug Information
- All components use Python logging with structured output
- Error screenshots saved to `data/scrape/prices/error/` directory
- Progress tracking in `data/scrape/scraping_progress.json`

## Repository Information
- **Remote Origin**: https://github.com/simplysindy/ACTP.git
- **Main Branch**: main
- **Development Practice**: Commit regularly and push frequently for backup and collaboration