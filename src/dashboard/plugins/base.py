"""Base plugin class for asset-specific UI components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AssetPlugin(ABC):
    """Base class for asset-specific UI plugins."""
    
    def __init__(self, asset_type: str, config: Optional[Dict[str, Any]] = None):
        self.asset_type = asset_type
        self.config = config or {}
        self.api_base_url = self.config.get('api_base_url', 'http://localhost:8000/api/v1')
    
    @abstractmethod
    def get_display_name(self) -> str:
        """Return human-readable display name for the asset type."""
        pass
    
    @abstractmethod
    def get_icon(self) -> str:
        """Return emoji or icon for the asset type."""
        pass
    
    @abstractmethod
    def render_overview(self) -> None:
        """Render asset-specific overview page."""
        pass
    
    @abstractmethod
    def render_analysis(self) -> None:
        """Render detailed analysis view."""
        pass
    
    @abstractmethod
    def render_forecasting(self) -> None:
        """Render forecasting interface."""
        pass
    
    def render_settings(self) -> None:
        """Render settings page (default implementation)."""
        st.subheader(f"{self.get_display_name()} Settings")
        
        with st.expander("Data Sources"):
            st.write("Configure data collection sources and parameters.")
            
        with st.expander("Model Configuration"):
            st.write("Configure forecasting models and parameters.")
            
        with st.expander("Display Options"):
            st.write("Configure display preferences.")
    
    @abstractmethod
    def get_custom_metrics(self) -> List[str]:
        """Return asset-specific metrics."""
        pass
    
    def get_supported_pages(self) -> List[str]:
        """Return list of supported page names."""
        return ["Overview", "Analysis", "Forecasting", "Settings"]
    
    def fetch_assets(self) -> List[Dict[str, Any]]:
        """Fetch available assets for this type."""
        try:
            # This would make actual API calls
            # For now, return mock data
            if self.asset_type == "watch":
                return [
                    {
                        "asset_id": "21813-rolex-submariner-124060",
                        "name": "Rolex Submariner 124060",
                        "brand": "Rolex",
                        "current_price": 15000.0
                    },
                    {
                        "asset_id": "326-tudor-black-bay-58-79030n",
                        "name": "Tudor Black Bay 58",
                        "brand": "Tudor",
                        "current_price": 4500.0
                    }
                ]
            return []
        except Exception as e:
            logger.error(f"Error fetching assets: {e}")
            return []
    
    def fetch_asset_history(self, asset_id: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical data for an asset."""
        try:
            # This would make actual API calls
            # For now, return mock data
            import numpy as np
            from datetime import datetime, timedelta
            
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=days),
                end=datetime.now(),
                freq='D'
            )
            
            # Generate mock price data
            base_price = 15000 if "rolex" in asset_id.lower() else 4500
            prices = base_price + np.cumsum(np.random.randn(len(dates)) * 50)
            
            return pd.DataFrame({
                'date': dates,
                'price': prices
            })
        except Exception as e:
            logger.error(f"Error fetching asset history: {e}")
            return pd.DataFrame()
    
    def render_common_filters(self) -> Dict[str, Any]:
        """Render common filter controls."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_range = st.selectbox(
                "Time Range",
                ["7 days", "30 days", "90 days", "1 year"],
                index=1
            )
        
        with col2:
            currency = st.selectbox(
                "Currency",
                ["SGD", "USD", "EUR"],
                index=0
            )
        
        with col3:
            chart_type = st.selectbox(
                "Chart Type",
                ["Line", "Candlestick", "Area"],
                index=0
            )
        
        return {
            "time_range": time_range,
            "currency": currency,
            "chart_type": chart_type
        }
    
    def render_asset_selector(self) -> Optional[str]:
        """Render asset selection widget."""
        assets = self.fetch_assets()
        
        if not assets:
            st.warning(f"No {self.asset_type} assets available.")
            return None
        
        asset_options = {
            f"{asset['name']} (${asset['current_price']:,.0f})": asset['asset_id']
            for asset in assets
        }
        
        selected_display = st.selectbox(
            f"Select {self.get_display_name()}:",
            list(asset_options.keys())
        )
        
        return asset_options[selected_display] if selected_display else None