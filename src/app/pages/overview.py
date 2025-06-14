import numpy as np
import pandas as pd
import streamlit as st

from app.components.components import (
    create_market_trend_chart,
    create_metric_card,
    create_watch_card,
)
from app.utils.formatting import (
    format_currency,
    format_percent,
    get_trend_arrow,
    get_trend_color,
)


def show_overview_page(summary_df, watch_data, watch_images):
    """Display the market overview page."""
    st.markdown("<h2 class='sub-header'>Market Overview</h2>", unsafe_allow_html=True)

    # Check if we have data
    if summary_df.empty and not watch_data:
        st.error("No data available for analysis")
        return

    # Calculate market statistics
    current_prices = []
    forecast_prices = []
    for watch_name, data in watch_data.items():
        try:
            if (
                "price(SGD)" in data["historical"].columns
                and "forecasted_price" in data["forecast"].columns
            ):
                current_price = data["historical"]["price(SGD)"].iloc[-1]
                forecast_price = data["forecast"]["forecasted_price"].iloc[-1]

                if not pd.isna(current_price) and not pd.isna(forecast_price):
                    current_prices.append(current_price)
                    forecast_prices.append(forecast_price)
        except (KeyError, IndexError):
            continue

    if current_prices and forecast_prices:
        avg_current = np.mean(current_prices)
        avg_forecast = np.mean(forecast_prices)
        avg_change = ((avg_forecast / avg_current) - 1) * 100

        # Key metrics row
        col1, col2, col3 = st.columns(3)

        with col1:
            create_metric_card("Average Current Price", format_currency(avg_current))

        with col2:
            create_metric_card("Average 30-Day Forecast", format_currency(avg_forecast))

        with col3:
            _trend_class = get_trend_color(avg_change)
            create_metric_card(
                "Average Price Change",
                format_percent(avg_change) + " " + get_trend_arrow(avg_change),
                avg_change,
            )
    else:
        st.warning("Insufficient data to calculate market statistics")

    st.markdown(
        "<h3 class='sub-header'>Market Trend Visualization</h3>", unsafe_allow_html=True
    )

    if watch_data:
        try:
            fig = create_market_trend_chart(watch_data, show_average_only=True)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating market trend chart: {e}")
    else:
        st.warning("No watch data available for visualization")

    # Watch summary cards
    st.markdown(
        "<h3 class='sub-header'>Watch Forecast Summary</h3>", unsafe_allow_html=True
    )

    # Display watch cards for each watch
    if not summary_df.empty:
        for _, row in summary_df.iterrows():
            watch_name = row["Watch Model"]
            if watch_name in watch_data:
                watch_info = watch_data[watch_name]
                image_url = watch_images.get(
                    watch_name, "https://www.svgrepo.com/show/491067/watch.svg"
                )

                create_watch_card(watch_name, watch_info, image_url, summary_df)
    else:
        # If no summary data, show cards for all watches in watch_data
        for watch_name, watch_info in watch_data.items():
            image_url = watch_images.get(
                watch_name, "https://www.svgrepo.com/show/491067/watch.svg"
            )
            create_watch_card(watch_name, watch_info, image_url)
