"""Header component for the dashboard."""

import streamlit as st
from datetime import datetime


def render_header():
    """Render the dashboard header."""
    
    # Main header
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.markdown("# 📈 Asset Forecasting Platform")
        st.markdown("*Multi-asset price prediction and analysis*")
    
    with col2:
        # Status indicators
        st.markdown("### System Status")
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown("🟢 **API**")
        with col_b:
            st.markdown("🟢 **Data**")
        with col_c:
            st.markdown("🟢 **Models**")
    
    with col3:
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"**Time:** {current_time}")
        st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    
    # Divider
    st.divider()
    
    # Show alerts or announcements if any
    if st.session_state.get('show_alerts', True):
        st.info("🚀 **New Feature:** Multi-asset forecasting is now available!")
    
    # Show help modal if requested
    if st.session_state.get('show_help', False):
        render_help_modal()
        st.session_state.show_help = False
    
    # Show settings modal if requested
    if st.session_state.get('show_settings', False):
        render_settings_modal()
        st.session_state.show_settings = False


def render_help_modal():
    """Render help modal."""
    with st.expander("📚 Help & Documentation", expanded=True):
        st.markdown("""
        ## Getting Started
        
        1. **Select Asset Type**: Choose the type of asset you want to analyze (Watches, Gold, Crypto, etc.)
        2. **Navigate Pages**: Use the sidebar to switch between Overview, Analysis, and Forecasting
        3. **Select Assets**: Choose specific assets from the dropdown menus
        4. **Customize Views**: Use filters and settings to customize your analysis
        
        ## Features
        
        - **Overview**: Market overview and key metrics
        - **Analysis**: Detailed technical analysis and price history
        - **Forecasting**: AI-powered price predictions
        - **Settings**: Configure data sources and display preferences
        
        ## Keyboard Shortcuts
        
        - `R`: Refresh data
        - `S`: Open settings
        - `H`: Show/hide help
        - `ESC`: Close modals
        
        ## Support
        
        For technical support or feature requests, please contact the development team.
        """)


def render_settings_modal():
    """Render settings modal."""
    with st.expander("⚙️ Dashboard Settings", expanded=True):
        st.markdown("### Display Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            theme = st.selectbox(
                "Theme",
                ["Light", "Dark", "Auto"],
                index=0
            )
            
            currency = st.selectbox(
                "Default Currency",
                ["SGD", "USD", "EUR", "GBP"],
                index=0
            )
        
        with col2:
            refresh_rate = st.selectbox(
                "Auto Refresh",
                ["Disabled", "1 minute", "5 minutes", "15 minutes"],
                index=2
            )
            
            items_per_page = st.number_input(
                "Items per Page",
                min_value=10,
                max_value=100,
                value=50
            )
        
        st.markdown("### Data Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cache_enabled = st.checkbox("Enable Caching", value=True)
            real_time_updates = st.checkbox("Real-time Updates", value=False)
        
        with col2:
            data_quality_check = st.checkbox("Data Quality Checks", value=True)
            auto_backup = st.checkbox("Auto Backup", value=True)
        
        if st.button("Save Settings"):
            st.success("Settings saved successfully!")
        
        if st.button("Reset to Defaults"):
            st.info("Settings reset to defaults.")