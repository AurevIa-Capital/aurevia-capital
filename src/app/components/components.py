import pandas as pd
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
    # st.sidebar.image("https://www.svgrepo.com/show/491067/watch.svg", width=50)
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


def create_watch_card(
    watch_name, watch_data, image_url, summary_data=None, show_model=True
):
    """Create a card displaying a watch's info."""
    # Format watch name
    display_name = format_watch_name(watch_name)

    # Extract data from watch_data
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

    # Try to get 7-day forecast from watch_data
    try:
        forecast_7d = (
            watch_data["forecast"]["forecasted_price"].iloc[6]
            if len(watch_data["forecast"]) > 6
            else None
        )
    except (KeyError, IndexError):
        forecast_7d = None

    # If summary_data is provided, use it as the source of truth for forecasts
    if summary_data is not None:
        model_row = summary_data[summary_data["Watch Model"] == watch_name]
        if not model_row.empty:
            # Get forecast values from summary data (more accurate)
            if "Forecasted Price (7 days)" in model_row.columns:
                forecast_7d = model_row.iloc[0]["Forecasted Price (7 days)"]

            if "Forecasted Price (30 days)" in model_row.columns:
                forecast_30d = model_row.iloc[0]["Forecasted Price (30 days)"]

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
        # Calculate 7-day price change if 7-day forecast and current price are available
        if current_price is not None and forecast_7d is not None and current_price != 0:
            price_change_7d = ((forecast_7d / current_price) - 1) * 100
            trend_class_7d = get_trend_color(price_change_7d)
            change_7d_html = f'<span class="{trend_class_7d}">{format_percent(price_change_7d)} {get_trend_arrow(price_change_7d)}</span>'
        else:
            change_7d_html = "N/A"

            # Calculate 7-day price change if 7-day forecast and current price are available
        if (
            current_price is not None
            and forecast_30d is not None
            and current_price != 0
        ):
            price_change_30d = ((forecast_30d / current_price) - 1) * 100
            trend_class_30d = get_trend_color(price_change_30d)
            change_30d_html = f'<span class="{trend_class_30d}">{format_percent(price_change_30d)} {get_trend_arrow(price_change_30d)}</span>'
        else:
            change_30d_html = "N/A"

        # Create HTML for card content
        html_content = f"""
        <div class="card">
            <h3>{display_name}</h3>
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <p><strong>Current Price:</strong> {format_currency(current_price)}</p>
                    <p><strong>7-Day Forecast:</strong> {format_currency(forecast_7d)} ({change_7d_html})</p>
                    <p><strong>30-Day Forecast:</strong> {format_currency(forecast_30d)} ({change_30d_html})</p>
                </div>
                <div>
        """

        # # Only add best model if show_model is True
        # if show_model and best_model:
        #     html_content += f"<p><strong>Best Model:</strong> {best_model}</p>"

        # html_content += f"""
        #             <p><strong>30-Day Change:</strong> <span class="{trend_class}">{format_percent(price_change)} {get_trend_arrow(price_change)}</span></p>
        #         </div>
        #     </div>
        # </div>
        # """

        st.markdown(html_content, unsafe_allow_html=True)


def create_market_trend_chart(watch_data, show_average_only=False):
    """
    Create a market trend chart.

    Parameters:
    -----------
    watch_data : dict
        Dictionary of watch data
    show_average_only : bool
        If True, only show the average of all watches, otherwise show individual watches
    """
    fig = go.Figure()

    if show_average_only:
        # Create empty DataFrames to store all historical and forecast data
        all_historical = pd.DataFrame()
        all_forecast = pd.DataFrame()

        # Collect all data from all watches
        for watch_name, data in watch_data.items():
            # Historical data
            if "price(SGD)" in data["historical"].columns:
                # Add price column renamed with watch name to identify it
                historical_df = pd.DataFrame(index=data["historical"].index)
                historical_df[watch_name] = data["historical"]["price(SGD)"]
                all_historical = pd.concat([all_historical, historical_df], axis=1)

            # Forecast data
            if "forecasted_price" in data["forecast"].columns:
                # Add forecasted price column renamed with watch name
                forecast_df = pd.DataFrame(index=data["forecast"].index)
                forecast_df[watch_name] = data["forecast"]["forecasted_price"]
                all_forecast = pd.concat([all_forecast, forecast_df], axis=1)

        # Calculate average across all watches for each date
        historical_avg = all_historical.mean(axis=1)
        forecast_avg = all_forecast.mean(axis=1)

        # Add historical average trace
        fig.add_trace(
            go.Scatter(
                x=historical_avg.index,
                y=historical_avg.values,
                mode="lines",
                name="Market Average (Historical)",
                line=dict(width=3, color="#1E3A8A"),  # Thicker line with blue color
            )
        )

        # Add forecast average trace with a different color
        fig.add_trace(
            go.Scatter(
                x=forecast_avg.index,
                y=forecast_avg.values,
                mode="lines",
                name="Market Average (Forecast)",
                line=dict(
                    dash="dash", width=3, color="#F59E0B"
                ),  # Dashed line with amber/orange color
            )
        )

        title = "Market Average Price Trend"
        legend_title = "Market Average"
    else:
        # Original code for showing individual watches
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
                        line=dict(
                            dash="dash", width=2, color="#F59E0B"
                        ),  # Dashed line with amber color
                    )
                )

        # First, update all the legend names to shorter versions
        if len(fig.data) >= 6:  # Only do this if we have at least 6 traces
            fig.data[0].name = "Rolex Sub (Historical)"
            fig.data[1].name = "Rolex Sub (Forecast)"
            fig.data[2].name = "Omega Speedmaster (Historical)"
            fig.data[3].name = "Omega Speedmaster (Forecast)"
            fig.data[4].name = "Tudor Black Bay (Historical)"
            fig.data[5].name = "Tudor Black Bay (Forecast)"

            fig.data[0].legendgroup = "Rolex"
            fig.data[1].legendgroup = "Rolex"
            fig.data[2].legendgroup = "Omega"
            fig.data[3].legendgroup = "Omega"
            fig.data[4].legendgroup = "Tudor"
            fig.data[5].legendgroup = "Tudor"

        title = "Historical Prices and Forecasts"
        legend_title = "Watch Models"

    # Update the layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (SGD)",
        legend_title=legend_title,
        template="plotly_white",
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.3 if show_average_only else -0.5,  # Position below the plot
            xanchor="center",
            x=0.5 if show_average_only else 0.4,
            traceorder="grouped"
            if not show_average_only
            else None,  # This ensures legend groups stay together
        ),
    )

    return fig
