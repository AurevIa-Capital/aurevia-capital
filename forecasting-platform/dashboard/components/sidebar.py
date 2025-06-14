"""Sidebar component for navigation."""

import streamlit as st
from typing import Tuple

from ..core.plugin_manager import PluginManager


def render_sidebar(plugin_manager: PluginManager) -> Tuple[str, str]:
    """Render sidebar navigation."""
    
    with st.sidebar:
        st.title("📈 Asset Forecasting")
        
        # Asset type selection
        st.subheader("Asset Type")
        available_types = plugin_manager.get_available_asset_types()
        
        if not available_types:
            st.error("No asset plugins available")
            return "", ""
        
        # Create display options
        display_options = {}
        for asset_type in available_types:
            plugin_info = plugin_manager.get_plugin_info(asset_type)
            if plugin_info:
                display_name = f"{plugin_info['icon']} {plugin_info['display_name']}"
                display_options[display_name] = asset_type
        
        selected_display = st.selectbox(
            "Select Asset Type:",
            list(display_options.keys()),
            key="asset_type_selector"
        )
        
        selected_asset_type = display_options[selected_display]
        
        # Page selection
        st.subheader("Navigation")
        plugin_info = plugin_manager.get_plugin_info(selected_asset_type)
        
        if plugin_info:
            supported_pages = plugin_info['supported_pages']
            selected_page = st.radio(
                "Select Page:",
                supported_pages,
                key="page_selector"
            )
        else:
            selected_page = "Overview"
        
        # Plugin information
        st.subheader("Plugin Info")
        if plugin_info:
            st.write(f"**Type:** {plugin_info['display_name']}")
            st.write(f"**Pages:** {len(plugin_info['supported_pages'])}")
            st.write(f"**Metrics:** {len(plugin_info['custom_metrics'])}")
            
            with st.expander("Custom Metrics"):
                for metric in plugin_info['custom_metrics']:
                    st.write(f"• {metric.replace('_', ' ').title()}")
        
        # System status
        st.subheader("System Status")
        stats = plugin_manager.get_plugin_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Plugins", stats['total_registered'])
        with col2:
            st.metric("Loaded", stats['total_loaded'])
        
        # Settings
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.show_settings = True
        
        # Help
        if st.button("❓ Help", use_container_width=True):
            st.session_state.show_help = True
    
    return selected_asset_type, selected_page