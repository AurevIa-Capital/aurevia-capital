import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# Set style for plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("viridis")
plt.rcParams["figure.figsize"] = [12, 6]


def ensure_output_dir(output_dir="data/output/"):
    """
    Ensure that the output directory exists, create it if it doesn't

    Parameters:
    ----------
    output_dir : str, default='data/output/'
        Path to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_watch_data(file_paths):
    """
    Load watch data from CSV files and return a dictionary of dataframes

    Parameters:
    ----------
    file_paths : list
        List of file paths to CSV files

    Returns:
    -------
    dict
        Dictionary of dataframes with watch names as keys
    """
    watch_data = {}
    for file_path in file_paths:
        # Extract watch name from file path properly
        # Get just the filename without directory structure
        filename = os.path.basename(file_path)
        # Remove extension
        watch_name = filename.split(".")[0]

        # Load data
        df = pd.read_csv(file_path)

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Set date as index
        df.set_index("date", inplace=True)

        # Store dataframe in dictionary
        watch_data[watch_name] = df

    return watch_data


def preprocess_data(watch_data_dict, freq="D"):
    """
    Preprocess watch data by resampling to daily frequency and filling missing values

    Parameters:
    ----------
    watch_data_dict : dict
        Dictionary of dataframes with watch names as keys
    freq : str, default='D'
        Frequency for resampling (D: daily, W: weekly, etc.)

    Returns:
    -------
    dict
        Dictionary of preprocessed dataframes
    """
    preprocessed_data = {}

    for watch_name, df in watch_data_dict.items():
        # Create full date range (daily)
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)

        # Reindex dataframe with full date range
        df_reindexed = df.reindex(date_range)

        # Fill missing values with backward fill (as suggested by user)
        df_filled = df_reindexed.bfill()

        # Store preprocessed dataframe
        preprocessed_data[watch_name] = df_filled

    return preprocessed_data


def perform_eda(watch_data_dict, output_dir="data/output/"):
    """
    Perform exploratory data analysis on watch data and save plots

    Parameters:
    ----------
    watch_data_dict : dict
        Dictionary of dataframes with watch names as keys
    output_dir : str, default='data/output/'
        Directory to save output plots
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)

    # Create a figure for plotting all watch prices
    plt.figure(figsize=(14, 8))

    for watch_name, df in watch_data_dict.items():
        # Plot price over time
        plt.plot(df.index, df["price(SGD)"], label=watch_name)

    plt.title("Watch Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (SGD)")
    plt.legend()
    plt.tight_layout()
    # Save plot instead of showing it
    plt.savefig(os.path.join(output_dir, "all_watch_prices.png"))
    plt.close()

    # Individual watch analysis
    for watch_name, df in watch_data_dict.items():
        print(f"\n--- Analysis for {watch_name} ---")

        # Basic statistics
        print("\nDescriptive Statistics:")
        print(df.describe())

        # Check for stationarity
        print("\nStationarity Test (Augmented Dickey-Fuller):")
        adf_result = adfuller(df["price(SGD)"].dropna())
        print(f"ADF Statistic: {adf_result[0]}")
        print(f"p-value: {adf_result[1]}")
        print(f"Critical Values: {adf_result[4]}")
        if adf_result[1] <= 0.05:
            print("Result: The time series is stationary")
        else:
            print("Result: The time series is not stationary")

        # Plot individual time series
        plt.figure(figsize=(14, 10))

        # Price plot
        plt.subplot(3, 1, 1)
        plt.plot(df.index, df["price(SGD)"])
        plt.title(f"{watch_name} - Price Over Time")
        plt.ylabel("Price (SGD)")

        # Autocorrelation plot
        plt.subplot(3, 1, 2)
        plot_acf(df["price(SGD)"].dropna(), lags=30, ax=plt.gca())
        plt.title("Autocorrelation Function (ACF)")

        # Partial Autocorrelation plot
        plt.subplot(3, 1, 3)
        plot_pacf(df["price(SGD)"].dropna(), lags=30, ax=plt.gca())
        plt.title("Partial Autocorrelation Function (PACF)")

        plt.tight_layout()
        # Save plot instead of showing it
        plt.savefig(os.path.join(output_dir, f"{watch_name}_analysis.png"))
        plt.close()

        # Seasonal decomposition if we have enough data
        if len(df) >= 14:  # Need at least 2*period data points
            try:
                # Try weekly seasonality (7 days)
                decomposition = seasonal_decompose(
                    df["price(SGD)"], model="additive", period=7
                )

                plt.figure(figsize=(14, 10))
                plt.subplot(4, 1, 1)
                plt.plot(decomposition.observed)
                plt.title("Observed")

                plt.subplot(4, 1, 2)
                plt.plot(decomposition.trend)
                plt.title("Trend")

                plt.subplot(4, 1, 3)
                plt.plot(decomposition.seasonal)
                plt.title("Seasonality")

                plt.subplot(4, 1, 4)
                plt.plot(decomposition.resid)
                plt.title("Residuals")

                plt.tight_layout()
                # Save plot instead of showing it
                plt.savefig(os.path.join(output_dir, f"{watch_name}_decomposition.png"))
                plt.close()
            except ValueError:
                print("Could not perform seasonal decomposition - may need more data")


def engineer_features(watch_data_dict):
    """
    Engineer features for time series forecasting

    Parameters:
    ----------
    watch_data_dict : dict
        Dictionary of dataframes with watch names as keys

    Returns:
    -------
    dict
        Dictionary of dataframes with engineered features
    """
    featured_data = {}

    for watch_name, df in watch_data_dict.items():
        # Create a copy of the dataframe
        df_featured = df.copy()

        # Extract date features
        df_featured["day_of_week"] = df_featured.index.dayofweek
        df_featured["day_of_month"] = df_featured.index.day
        df_featured["month"] = df_featured.index.month
        df_featured["quarter"] = df_featured.index.quarter
        df_featured["year"] = df_featured.index.year
        df_featured["is_weekend"] = df_featured.index.dayofweek >= 5

        # Add lagged features
        df_featured["price_lag_1"] = df_featured["price(SGD)"].shift(1)
        df_featured["price_lag_2"] = df_featured["price(SGD)"].shift(2)
        df_featured["price_lag_3"] = df_featured["price(SGD)"].shift(3)
        df_featured["price_lag_7"] = df_featured["price(SGD)"].shift(7)

        # Add rolling mean features
        df_featured["rolling_mean_3"] = (
            df_featured["price(SGD)"].rolling(window=3).mean()
        )
        df_featured["rolling_mean_7"] = (
            df_featured["price(SGD)"].rolling(window=7).mean()
        )
        df_featured["rolling_mean_14"] = (
            df_featured["price(SGD)"].rolling(window=14).mean()
        )

        # Add rolling std features
        df_featured["rolling_std_3"] = df_featured["price(SGD)"].rolling(window=3).std()
        df_featured["rolling_std_7"] = df_featured["price(SGD)"].rolling(window=7).std()

        # Add momentum indicators
        df_featured["momentum_1"] = df_featured["price(SGD)"].diff(1)
        df_featured["momentum_3"] = df_featured["price(SGD)"].diff(3)
        df_featured["momentum_7"] = df_featured["price(SGD)"].diff(7)

        # Add target variable (next day's price) for supervised learning
        df_featured["target"] = df_featured["price(SGD)"].shift(-1)

        # Drop NaN values
        df_featured.dropna(inplace=True)

        # Store featured dataframe
        featured_data[watch_name] = df_featured

    return featured_data


def main():
    """
    Main function to run the time series forecasting pipeline
    """
    # Define output directory
    output_dir = "data/output/"
    ensure_output_dir(output_dir)

    # Define file paths
    file_paths = [
        "data/watches/326-tudor-black-bay-58-79030n.csv",
        "data/watches/21813-rolex-submariner-124060.csv",
        "data/watches/30921-omega-speedmaster-professional-moonwatch-310-30-42-50-01-002.csv",
    ]

    print("\n--- Loading Data ---")
    watch_data = load_watch_data(file_paths)

    print("\n--- Preprocessing Data ---")
    preprocessed_data = preprocess_data(watch_data)

    print("\n--- Performing Exploratory Data Analysis ---")
    perform_eda(preprocessed_data, output_dir)

    print("\n--- Engineering Features ---")
    featured_data = engineer_features(preprocessed_data)

    # Fixed the CSV output with proper column names
    # First, prepare each DataFrame with proper multi-index
    dfs_with_names = []
    for watch_name, df in featured_data.items():
        # Reset index to get date as a column
        df_reset = df.reset_index()
        # Rename the index column to be clear
        df_reset.rename(columns={"index": "date"}, inplace=True)
        # Add watch name as a column
        df_reset.insert(0, "watch_model", watch_name)
        dfs_with_names.append(df_reset)

    # Concatenate all DataFrames
    combined_df = pd.concat(dfs_with_names)

    # Save with all proper column names
    combined_df.to_csv(os.path.join(output_dir, "featured_data.csv"), index=False)


if __name__ == "__main__":
    main()
