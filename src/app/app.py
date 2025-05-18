import warnings

import streamlit as st

from app.components.components import create_header, create_sidebar_navigation
from app.pages.overview import show_overview_page
from app.utils.data_loader import load_data, load_watch_images
from app.utils.styling import create_footer, load_css

# Other pages will be imported here as they are developed

# Suppress warnings
warnings.filterwarnings("ignore")


def main():
    """Main function to run the Streamlit app."""
    # Set page config
    st.set_page_config(
        page_title="Luxury Watch Price Forecaster",
        page_icon="⌚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load CSS
    load_css()

    # Create header
    create_header()

    # Create navigation
    page = create_sidebar_navigation()

    # Use a relative path from the project root (where run_app.py is located)
    # Since the script is run from aurevIa_timepiece directory
    images_dir = "data/images"

    # Load data with spinner
    with st.spinner("Loading data..."):
        summary_df, watch_data = load_data()
        watch_images = load_watch_images(images_dir)

    # Remove debugging lines later
    if not any(not path.startswith("http") for path in watch_images.values()):
        st.sidebar.warning("No local images were found. Using online fallback images.")

    # Show selected page
    if page == "Overview":
        show_overview_page(summary_df, watch_data, watch_images)
    elif page == "Watch Analysis":
        st.markdown("# Watch Analysis")
        st.info("This page is under development.")
    elif page == "Compare Watches":
        st.markdown("# Compare Watches")
        st.info("This page is under development.")
    elif page == "Market Insights":
        st.markdown("# Market Insights")
        st.info("This page is under development.")

    # Create footer
    create_footer()


if __name__ == "__main__":
    main()
