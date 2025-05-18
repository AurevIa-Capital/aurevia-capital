import plotly.graph_objects as go
import streamlit as st

from ..utils.formatting import (
    format_currency,
    format_percent,
    format_watch_name,
    get_trend_arrow,
    get_trend_color,
)


def create_header():
    """Create the main header."""
    st.markdown(
        "<h1 class='main-header'>Luxury Watch Price Forecaster</h1>",
        unsafe_allow_html=True,
    )


def create_sidebar_navigation():
    """Create sidebar navigation."""
    st.sidebar.image("https://www.svgrepo.com/show/491067/watch.svg", width=50)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Watch Analysis", "Compare Watches", "Market Insights"],
    )
    return page


def create_metric_card(title, value, change=None, prefix=None):
    """Create a metric card with title, value, and optional change percentage."""
    if prefix:
        formatted_value = f"{prefix}{value}"
    else:
        formatted_value = value

    if change is not None:
        trend_class = get_trend_color(change)
        trend_arrow = get_trend_arrow(change)
        formatted_change = format_percent(change)

        st.markdown(
            f"""
        <div class="card">
            <h3>{title}</h3>
            <p class="metric-value">{formatted_value}</p>
            <p class="{trend_class}">{formatted_change} {trend_arrow}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="card">
            <h3>{title}</h3>
            <p class="metric-value">{formatted_value}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )


def create_watch_card(watch_name, watch_data, image_url, summary_data=None):
    """Create a card displaying a watch's info."""
    # Format watch name
    display_name = format_watch_name(watch_name)

    # Extract data
    current_price = (
        watch_data["historical"]["price(SGD)"].iloc[-1]
        if "price(SGD)" in watch_data["historical"].columns
        else None
    )
    forecast_30d = (
        watch_data["forecast"]["forecasted_price"].iloc[-1]
        if "forecasted_price" in watch_data["forecast"].columns
        else None
    )

    # Calculate percentage change
    if current_price is not None and forecast_30d is not None and current_price != 0:
        price_change = ((forecast_30d / current_price) - 1) * 100
    else:
        price_change = None

    # Get model info if available
    best_model = "N/A"
    if summary_data is not None:
        model_row = summary_data[summary_data["Watch Model"] == watch_name]
        if not model_row.empty:
            best_model = model_row.iloc[0]["Best Model"]

    # Create card
    col1, col2 = st.columns([1, 3])

    with col1:
        # Check if image_url is a local path or a URL
        if image_url.startswith("http"):
            # Use the URL directly
            st.image(image_url, width=150)
        else:
            # Try to load from local file
            try:
                import os

                from PIL import Image

                if os.path.exists(image_url):
                    image = Image.open(image_url)
                    st.image(image, width=150)
                else:
                    # Fallback to default image
                    st.image("https://www.svgrepo.com/show/491067/watch.svg", width=150)
                    st.caption("Image not found")
            except Exception as e:
                st.warning(f"Unable to load image: {e}")
                st.image("https://www.svgrepo.com/show/491067/watch.svg", width=150)

    with col2:
        trend_class = get_trend_color(price_change)

        st.markdown(
            f"""
        <div class="card">
            <h3>{display_name}</h3>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p><strong>Current Price:</strong> {format_currency(current_price)}</p>
                    <p><strong>30-Day Forecast:</strong> {format_currency(forecast_30d)}</p>
                </div>
                <div>
                    <p><strong>Best Model:</strong> {best_model}</p>
                    <p><strong>Expected Change:</strong> <span class="{trend_class}">{format_percent(price_change)} {get_trend_arrow(price_change)}</span></p>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def create_market_trend_chart(watch_data):
    """Create a market trend chart showing all watches."""
    fig = go.Figure()

    for watch_name, data in watch_data.items():
        # Get display name
        display_name = format_watch_name(watch_name)

        # Historical data
        if "price(SGD)" in data["historical"].columns:
            fig.add_trace(
                go.Scatter(
                    x=data["historical"].index,
                    y=data["historical"]["price(SGD)"],
                    mode="lines",
                    name=f"{display_name} (Historical)",
                    line=dict(width=2),
                )
            )

        # Forecast data
        if "forecasted_price" in data["forecast"].columns:
            fig.add_trace(
                go.Scatter(
                    x=data["forecast"].index,
                    y=data["forecast"]["forecasted_price"],
                    mode="lines",
                    name=f"{display_name} (Forecast)",
                    line=dict(dash="dash", width=2),
                )
            )

    fig.update_layout(
        title="Historical Prices and Forecasts",
        xaxis_title="Date",
        yaxis_title="Price (SGD)",
        legend_title="Watch Models",
        template="plotly_white",
        height=500,
    )

    return fig
