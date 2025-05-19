import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.formatting import (
    format_currency,
    format_percent,
    format_watch_name,
)


def show_analysis_page(summary_df, watch_data, watch_images):
    """Display the watch analysis page."""
    st.markdown("<h2 class='sub-header'>Watch Analysis</h2>", unsafe_allow_html=True)

    # Check if we have data
    if not watch_data:
        st.error("No data available for analysis")
        return

    # Watch selector
    st.markdown("### Select a Watch for Analysis")
    watch_options = list(watch_data.keys())
    selected_watch = st.selectbox(
        "Choose a watch model",
        options=watch_options,
        format_func=format_watch_name,
    )

    # Get selected watch data
    if selected_watch not in watch_data:
        st.error(f"Data for {selected_watch} not available")
        return

    watch_info = watch_data[selected_watch]
    image_url = watch_images.get(
        selected_watch, "https://www.svgrepo.com/show/491067/watch.svg"
    )

    # Display watch image and basic info
    col1, col2 = st.columns([1, 3])

    with col1:
        # Display watch image
        if image_url.startswith("http"):
            st.image(image_url, width=200)
        else:
            try:
                import os

                from PIL import Image

                if os.path.exists(image_url):
                    image = Image.open(image_url)
                    st.image(image, width=200)
                else:
                    st.image("https://www.svgrepo.com/show/491067/watch.svg", width=200)
            except Exception as e:
                st.warning(f"Unable to load image: {e}")
                st.image("https://www.svgrepo.com/show/491067/watch.svg", width=200)

    with col2:
        # Display watch details
        st.markdown(f"## {format_watch_name(selected_watch)}")

        # Extract price metrics
        current_price = None
        forecast_7d = None
        forecast_30d = None

        # Get current price from historical data
        if "price(SGD)" in watch_info["historical"].columns:
            current_price = watch_info["historical"]["price(SGD)"].iloc[-1]

        # Get forecast prices
        if "forecasted_price" in watch_info["forecast"].columns:
            forecast_df = watch_info["forecast"]
            # 7-day forecast (if available)
            if len(forecast_df) > 6:
                forecast_7d = forecast_df["forecasted_price"].iloc[6]
            # 30-day forecast (or the last available forecast)
            forecast_30d = forecast_df["forecasted_price"].iloc[-1]

        # Check if we should override with summary data
        if not summary_df.empty:
            model_row = summary_df[summary_df["Watch Model"] == selected_watch]
            if not model_row.empty:
                if "Forecasted Price (7 days)" in model_row.columns:
                    forecast_7d = model_row.iloc[0]["Forecasted Price (7 days)"]
                if "Forecasted Price (30 days)" in model_row.columns:
                    forecast_30d = model_row.iloc[0]["Forecasted Price (30 days)"]

        # Calculate price changes
        price_change_7d = (
            ((forecast_7d / current_price) - 1) * 100
            if current_price and forecast_7d
            else None
        )
        price_change_30d = (
            ((forecast_30d / current_price) - 1) * 100
            if current_price and forecast_30d
            else None
        )

        # Display metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric(
                "Current Price",
                format_currency(current_price) if current_price else "N/A",
            )

        with metrics_col2:
            st.metric(
                "7-Day Forecast",
                format_currency(forecast_7d) if forecast_7d else "N/A",
                format_percent(price_change_7d)
                if price_change_7d is not None
                else "N/A",
                delta_color="normal"
                if price_change_7d is None
                else "inverse"
                if price_change_7d < 0
                else "normal",
            )

        with metrics_col3:
            st.metric(
                "30-Day Forecast",
                format_currency(forecast_30d) if forecast_30d else "N/A",
                format_percent(price_change_30d)
                if price_change_30d is not None
                else "N/A",
                delta_color="normal"
                if price_change_30d is None
                else "inverse"
                if price_change_30d < 0
                else "normal",
            )

    # Historical & Forecast Chart with confidence intervals
    st.markdown("### Price History and Forecast")

    fig = create_detailed_forecast_chart(watch_info, selected_watch)
    st.plotly_chart(fig, use_container_width=True)

    # Two-column layout for distribution and stats
    dist_col, stats_col = st.columns([3, 2])

    with dist_col:
        # Price Distribution Analysis
        st.markdown("### Price Distribution")
        dist_fig = create_price_distribution(watch_info)
        st.plotly_chart(dist_fig, use_container_width=True)

    with stats_col:
        # Statistical Summary
        st.markdown("### Statistical Summary")
        stats_table = create_statistical_summary(watch_info)
        st.table(stats_table)

    # Feature Importance
    st.markdown("### Feature Importance")
    st.info(
        "Feature importance data is not yet available. This will show which factors most influence price predictions."
    )

    # Placeholder for feature importance - would be populated with real data in production
    # For now, create a sample visualization
    create_sample_feature_importance()


def create_detailed_forecast_chart(watch_info, watch_name):
    """
    Create a detailed forecast chart with confidence intervals.

    Parameters:
    ----------
    watch_info : dict
        Dictionary containing historical and forecast data
    watch_name : str
        Name of the watch

    Returns:
    -------
    fig : plotly figure
        Time series visualization
    """
    fig = go.Figure()

    # Historical data
    if "price(SGD)" in watch_info["historical"].columns:
        fig.add_trace(
            go.Scatter(
                x=watch_info["historical"].index,
                y=watch_info["historical"]["price(SGD)"],
                mode="lines",
                name="Historical Prices",
                line=dict(width=2, color="#1E3A8A"),
            )
        )

    # Forecast data
    if "forecasted_price" in watch_info["forecast"].columns:
        forecast_df = watch_info["forecast"]

        # Simulate confidence intervals (in a real app, these would come from the model)
        # For demonstration, we'll create 95% confidence intervals at ±5% of the forecasted price
        forecast_df["upper_ci"] = forecast_df["forecasted_price"] * 1.05
        forecast_df["lower_ci"] = forecast_df["forecasted_price"] * 0.95

        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["forecasted_price"],
                mode="lines",
                name="Price Forecast",
                line=dict(width=2, color="#F59E0B", dash="dash"),
            )
        )

        # Add confidence interval (shaded area)
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                y=forecast_df["upper_ci"].tolist()
                + forecast_df["lower_ci"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.2)",  # Semi-transparent amber
                line=dict(color="rgba(255, 255, 255, 0)"),  # Transparent line
                name="95% Confidence Interval",
                showlegend=True,
            )
        )

    # Last historical point
    if "price(SGD)" in watch_info["historical"].columns:
        last_historical_date = watch_info["historical"].index[-1]
        last_historical_price = watch_info["historical"]["price(SGD)"].iloc[-1]

        fig.add_trace(
            go.Scatter(
                x=[last_historical_date],
                y=[last_historical_price],
                mode="markers",
                marker=dict(size=10, color="#1E3A8A"),
                name="Last Recorded Price",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"{format_watch_name(watch_name)} - Price History and Forecast",
        xaxis_title="Date",
        yaxis_title="Price (SGD)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def create_price_distribution(watch_info):
    """
    Create a price distribution histogram.

    Parameters:
    ----------
    watch_info : dict
        Dictionary containing historical and forecast data

    Returns:
    -------
    fig : plotly figure
        Histogram visualization
    """
    if "price(SGD)" not in watch_info["historical"].columns:
        # Return empty figure if no price data
        fig = go.Figure()
        fig.update_layout(title="No price data available")
        return fig

    # Get historical prices
    prices = watch_info["historical"]["price(SGD)"].dropna()

    # Create histogram
    fig = px.histogram(
        prices,
        nbins=20,
        title="Historical Price Distribution",
        labels={"value": "Price (SGD)", "count": "Frequency"},
        color_discrete_sequence=["#1E3A8A"],
    )

    # Add a vertical line for the current price
    current_price = prices.iloc[-1] if not prices.empty else None

    if current_price is not None:
        fig.add_vline(
            x=current_price,
            line_width=2,
            line_dash="dash",
            line_color="#F59E0B",
            annotation_text="Current Price",
            annotation_position="top right",
        )

    # Update layout
    fig.update_layout(
        xaxis_title="Price (SGD)",
        yaxis_title="Frequency",
        template="plotly_white",
    )

    return fig


def create_statistical_summary(watch_info):
    """
    Create a statistical summary of the watch prices.

    Parameters:
    ----------
    watch_info : dict
        Dictionary containing historical and forecast data

    Returns:
    -------
    df : pandas DataFrame
        Formatted statistics table
    """
    if "price(SGD)" not in watch_info["historical"].columns:
        # Return empty DataFrame if no price data
        return pd.DataFrame({"Statistics": ["No data available"]})

    # Get historical prices
    prices = watch_info["historical"]["price(SGD)"].dropna()

    if prices.empty:
        return pd.DataFrame({"Statistics": ["No data available"]})

    # Calculate statistics
    stats = {
        "Metric": [
            "Current Price",
            "Mean Price",
            "Median Price",
            "Min Price",
            "Max Price",
            "Price Range",
            "Standard Deviation",
            "Volatility (%)",
        ],
        "Value": [
            format_currency(prices.iloc[-1]),
            format_currency(prices.mean()),
            format_currency(prices.median()),
            format_currency(prices.min()),
            format_currency(prices.max()),
            format_currency(prices.max() - prices.min()),
            format_currency(prices.std()),
            f"{(prices.std() / prices.mean() * 100):.2f}%",
        ],
    }

    # Create DataFrame
    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.set_index("Metric")

    return stats_df


def create_sample_feature_importance():
    """Create a sample feature importance chart for demonstration purposes."""
    # Sample data for feature importance
    features = [
        "Market Sentiment",
        "Seasonality",
        "Supply Constraints",
        "Demand Trends",
        "Brand Reputation",
    ]

    importance = [0.35, 0.25, 0.20, 0.15, 0.05]

    # Create bar chart
    fig = px.bar(
        x=importance,
        y=features,
        orientation="h",
        title="Feature Importance (Sample Data)",
        labels={"x": "Relative Importance", "y": "Feature"},
        color=importance,
        color_continuous_scale=["#1E3A8A", "#F59E0B"],
    )

    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
