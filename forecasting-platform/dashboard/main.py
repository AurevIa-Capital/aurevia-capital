"""
Multi-Asset Forecasting Dashboard

Plugin-based dashboard for multiple asset types.
"""

import streamlit as st
import logging
from typing import Dict, Any

from src.core.plugin_manager import PluginManager
from src.core.config import DashboardConfig
from src.components.sidebar import render_sidebar
from src.components.header import render_header

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Asset Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'plugin_manager' not in st.session_state:
    st.session_state.plugin_manager = PluginManager()
    st.session_state.config = DashboardConfig()

def main():
    """Main dashboard application."""
    try:
        # Render header
        render_header()
        
        # Render sidebar
        selected_asset_type, selected_page = render_sidebar(
            st.session_state.plugin_manager
        )
        
        # Get the appropriate plugin
        plugin = st.session_state.plugin_manager.get_plugin(selected_asset_type)
        
        if plugin:
            # Render the selected page using the plugin
            if selected_page == "Overview":
                plugin.render_overview()
            elif selected_page == "Analysis":
                plugin.render_analysis()
            elif selected_page == "Forecasting":
                plugin.render_forecasting()
            elif selected_page == "Settings":
                plugin.render_settings()
        else:
            st.error(f"Plugin not found for asset type: {selected_asset_type}")
            
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()