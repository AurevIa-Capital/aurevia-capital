import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st

# Configure warnings - suppress only specific pandas/numpy warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def load_data():
    """Load watch data and forecasts with comprehensive error handling."""
    # Define paths
    output_dir = "data/output/"
    summary_path = os.path.join(output_dir, "model_summary.csv")
    featured_data_path = os.path.join(output_dir, "featured_data.csv")

    # Check if output directory exists
    if not os.path.exists(output_dir):
        st.error(f"Output directory not found: {output_dir}")
        st.info("Creating directory...")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to create directory: {e}")

    # Load summary data
    summary_df = pd.DataFrame()
    if os.path.exists(summary_path):
        try:
            summary_df = pd.read_csv(summary_path)
            # st.success("Successfully loaded summary data")
        except Exception as e:
            st.error(f"Error loading summary data: {e}")
    else:
        st.warning(f"Summary file not found: {summary_path}")

    # Load featured data
    featured_df = pd.DataFrame()
    if os.path.exists(featured_data_path):
        try:
            featured_df = pd.read_csv(featured_data_path)
            featured_df["date"] = pd.to_datetime(featured_df["date"])
            # st.success("Successfully loaded featured data")
        except Exception as e:
            st.error(f"Error loading featured data: {e}")
    else:
        st.warning(f"Featured data file not found: {featured_data_path}")

    # Process summary data - ensure numeric columns
    if not summary_df.empty:
        for col in summary_df.columns:
            if "Price" in col:
                summary_df[col] = summary_df[col].apply(safe_numeric_conversion)
            if "Change" in col:
                summary_df[col] = summary_df[col].apply(safe_numeric_conversion)

    # Create watch data dictionary
    watch_data = {}

    # If we have featured data, process it
    if not featured_df.empty and "watch_model" in featured_df.columns:
        # Group by watch model
        for watch_name, group in featured_df.groupby("watch_model"):
            # Create a copy of the group data
            watch_group = group.copy()

            # Set date as index for historical data
            historical = watch_group.set_index("date")

            # Check if forecast file exists
            forecast_path = os.path.join(output_dir, f"{watch_name}_price_forecast.csv")
            forecast = None

            if os.path.exists(forecast_path):
                try:
                    forecast = pd.read_csv(forecast_path)
                    forecast["date"] = pd.to_datetime(forecast["date"])
                    forecast.set_index("date", inplace=True)
                    # st.success(f"Loaded forecast for {watch_name}")
                except Exception as e:
                    st.error(f"Error loading forecast for {watch_name}: {e}")
                    forecast = None

            # If no forecast data, create empty dataframe
            if forecast is None:
                # Create placeholder forecast
                last_date = historical.index.max()
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1), periods=30
                )
                forecast = pd.DataFrame(index=forecast_dates)
                forecast["forecasted_price"] = np.nan
                st.warning(f"Created placeholder forecast for {watch_name}")

            # Store in watch data dictionary
            watch_data[watch_name] = {"historical": historical, "forecast": forecast}

    # If no watch data, create placeholder
    if not watch_data:
        st.warning("No watch data found. Creating placeholder...")

        # Create placeholder watch data
        watches = ["Tudor Black Bay", "Rolex Submariner", "Omega Speedmaster"]

        for i, watch in enumerate(watches):
            # Create dates
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100)

            # Create random price data
            base_price = 5000 * (i + 1)
            prices = np.linspace(
                base_price * 0.9, base_price, len(dates)
            ) + np.random.normal(0, base_price * 0.02, len(dates))

            # Create historical dataframe
            historical = pd.DataFrame(
                {"price(SGD)": prices, "watch_model": watch}, index=dates
            )

            # Create forecast dataframe
            forecast_dates = pd.date_range(
                start=dates[-1] + pd.Timedelta(days=1), periods=30
            )
            forecast_prices = np.linspace(
                prices[-1], prices[-1] * 1.05, len(forecast_dates)
            ) + np.random.normal(0, prices[-1] * 0.01, len(forecast_dates))

            forecast = pd.DataFrame(
                {"forecasted_price": forecast_prices}, index=forecast_dates
            )

            # Add to watch data
            watch_data[watch] = {"historical": historical, "forecast": forecast}

        # Create placeholder summary data
        if summary_df.empty:
            summary_rows = []

            for watch in watches:
                historical = watch_data[watch]["historical"]
                forecast = watch_data[watch]["forecast"]

                current_price = historical["price(SGD)"].iloc[-1]
                forecast_7d = (
                    forecast["forecasted_price"].iloc[6] if len(forecast) > 6 else None
                )
                forecast_30d = (
                    forecast["forecasted_price"].iloc[-1]
                    if len(forecast) >= 30
                    else None
                )

                price_change = (
                    ((forecast_30d / current_price) - 1) * 100
                    if forecast_30d is not None
                    else None
                )

                summary_rows.append(
                    {
                        "Watch Model": watch,
                        "Best Model": np.random.choice(
                            ["Linear Regression", "SARIMA", "Random Forest"]
                        ),
                        "RMSE": np.random.uniform(50, 200),
                        "Current Price": current_price,
                        "Forecasted Price (7 days)": forecast_7d,
                        "Forecasted Price (30 days)": forecast_30d,
                        "Price Change %": price_change,
                    }
                )

            summary_df = pd.DataFrame(summary_rows)

    return summary_df, watch_data


def load_watch_images(image_dir="data/images"):
    """
    Load watch images from local directory.

    Parameters:
    -----------
    image_dir : str
        Path to directory containing watch images

    Returns:
    --------
    dict
        Dictionary mapping watch_name to image path
    """
    import os

    import streamlit as st

    # Create a dictionary to store watch images
    watch_images = {}

    # Check if directory exists
    if not os.path.exists(image_dir):
        st.warning(f"Image directory not found: {image_dir}")
        return watch_images

    # Map watch models to their expected image filenames
    watch_model_map = {
        "30921-omega-speedmaster-professional-moonwatch-310-30-42-50-01-002": [
            "omega",
            "speedmaster",
            "moonwatch",
        ],
        "21813-rolex-submariner-124060": ["rolex", "submariner"],
        "326-tudor-black-bay-58-79030n": ["tudor", "black-bay", "black_bay"],
        "Tudor Black Bay": ["tudor", "black-bay", "black_bay"],
        "Rolex Submariner": ["rolex", "submariner"],
        "Omega Speedmaster": ["omega", "speedmaster", "moonwatch"],
    }

    # List all image files in directory
    try:
        image_files = os.listdir(image_dir)

        # For each watch model, try to find a matching image
        for watch_name, keywords in watch_model_map.items():
            for image_file in image_files:
                # Check if any keyword is in the image filename (case insensitive)
                if any(keyword.lower() in image_file.lower() for keyword in keywords):
                    watch_images[watch_name] = os.path.join(image_dir, image_file)
                    break
    except Exception as e:
        st.error(f"Error loading images from directory: {e}")

    return watch_images


def safe_numeric_conversion(value):
    """Safely convert various formats to numeric values."""
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float)):
        return value

    if not isinstance(value, str):
        return np.nan

    # Handle currency format
    value = value.strip()
    if value.startswith("$"):
        value = value[1:]

    # Handle percentage format
    if value.endswith("%"):
        value = value[:-1]

    # Remove thousand separators
    value = value.replace(",", "")

    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan
