"""
Setup configuration for the luxury watch price forecasting platform.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Core dependencies
core_requirements = [
    "streamlit>=1.28.0,<2.0.0",
    "pandas>=2.1.0,<3.0.0",
    "numpy>=1.25.0,<2.0.0",
    "matplotlib>=3.7.0,<4.0.0",
    "seaborn>=0.12.0,<1.0.0",
    "plotly>=5.17.0,<6.0.0",
    "scikit-learn>=1.3.0,<2.0.0",
    "xgboost>=2.0.0,<3.0.0",
    "statsmodels>=0.14.0,<1.0.0",
    "requests>=2.31.0,<3.0.0",
    "beautifulsoup4>=4.12.0,<5.0.0",
    "pydantic>=2.5.0,<3.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
]

# Optional dependencies
scraping_requirements = [
    "selenium>=4.15.0,<5.0.0",
    "webdriver-manager>=4.0.0,<5.0.0",
]

api_requirements = [
    "fastapi>=0.104.0,<1.0.0",
    "uvicorn[standard]>=0.24.0,<1.0.0",
    "python-multipart>=0.0.6,<1.0.0",
    "python-jose[cryptography]>=3.3.0,<4.0.0",
    "passlib[bcrypt]>=1.7.4,<2.0.0",
    "aiofiles>=23.2.0,<24.0.0",
    "httpx>=0.25.0,<1.0.0",
]

dev_requirements = [
    "pytest>=7.4.0,<8.0.0",
    "black>=23.0.0,<24.0.0",
    "isort>=5.12.0,<6.0.0",
    "mypy>=1.6.0,<2.0.0",
    "flake8>=6.0.0,<7.0.0",
    "pre-commit>=3.0.0,<4.0.0",
]

setup(
    name="luxury-watch-forecaster",
    version="1.0.0",
    author="Asset Forecasting Platform",
    author_email="noreply@example.com",
    description="Multi-asset price forecasting platform for luxury watches",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simplysindy/ACTP",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require={
        "scraping": scraping_requirements,
        "api": api_requirements,
        "dev": dev_requirements,
        "all": scraping_requirements + api_requirements + dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "watch-forecaster=scripts.run_dashboard:main",
            "watch-api=scripts.run_api:main", 
            "watch-scraper=collectors.watch.scraper:main",
            "watch-modeller=models.forecasting.modelling:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
    },
    zip_safe=False,
)