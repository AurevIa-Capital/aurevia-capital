import os
import warnings
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Configure warnings - only suppress specific warnings we know are safe
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Set modern plotting style
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = [12, 6]
plt.rcParams["font.size"] = 10


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


def train_test_split_ts(df, test_size=0.2):
    """
    Split data into training and testing sets for time series

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    test_size : float, default=0.2
        Proportion of data to use for testing

    Returns:
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Define features and target
    X = df.drop(["price(SGD)", "target"], axis=1)
    y = df["target"]

    # Get the split point
    split_idx = int(len(df) * (1 - test_size))

    # Split the data
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate model performance

    Parameters:
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model

    Returns:
    -------
    dict
        Dictionary with evaluation metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Print metrics
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return {"model": model_name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def plot_predictions(y_true, y_pred, dates, model_name, output_dir, watch_name):
    """
    Plot actual vs predicted values

    Parameters:
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    dates : array-like
        Dates for x-axis
    model_name : str
        Name of the model
    output_dir : str
        Directory to save output plots
    watch_name : str
        Name of the watch
    """
    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_true, label="Actual", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted", linestyle="--")
    plt.title(f"{watch_name} - {model_name} Predictions")
    plt.xlabel("Date")
    plt.ylabel("Price (SGD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{watch_name}_{model_name}_predictions.png"))
    plt.close()


def build_linear_regression(
    X_train, X_test, y_train, y_test, dates_test, watch_name, output_dir
):
    """
    Build and evaluate a linear regression model

    Parameters:
    ----------
    X_train, X_test, y_train, y_test : pandas.DataFrame
        Train and test data
    dates_test : array-like
        Dates for test data
    watch_name : str
        Name of the watch
    output_dir : str
        Directory to save output plots

    Returns:
    -------
    dict
        Evaluation metrics
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Use log transformation for price data to prevent negative predictions
    # Add a small constant (1) to avoid log(0)
    y_train_log = np.log(y_train + 1)

    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train_log)

    # Make predictions in log space
    y_pred_log = model.predict(X_test_scaled)

    # Transform back to original space
    y_pred = np.exp(y_pred_log) - 1

    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, "Linear Regression")

    # Plot predictions
    plot_predictions(
        y_test, y_pred, dates_test, "Linear Regression", output_dir, watch_name
    )

    # Feature importance
    feature_importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": np.abs(model.coef_)}
    )
    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance["Feature"][:10], feature_importance["Importance"][:10])
    plt.title(f"{watch_name} - Linear Regression Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{watch_name}_linear_feature_importance.png"))
    plt.close()

    return metrics


def build_random_forest(
    X_train, X_test, y_train, y_test, dates_test, watch_name, output_dir
):
    """
    Build and evaluate a random forest model

    Parameters:
    ----------
    X_train, X_test, y_train, y_test : pandas.DataFrame
        Train and test data
    dates_test : array-like
        Dates for test data
    watch_name : str
        Name of the watch
    output_dir : str
        Directory to save output plots

    Returns:
    -------
    dict
        Evaluation metrics
    """
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, "Random Forest")

    # Plot predictions
    plot_predictions(
        y_test, y_pred, dates_test, "Random Forest", output_dir, watch_name
    )

    # Feature importance
    feature_importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model.feature_importances_}
    )
    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance["Feature"][:10], feature_importance["Importance"][:10])
    plt.title(f"{watch_name} - Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{watch_name}_rf_feature_importance.png"))
    plt.close()

    return metrics


def build_xgboost(X_train, X_test, y_train, y_test, dates_test, watch_name, output_dir):
    """
    Build and evaluate an XGBoost model

    Parameters:
    ----------
    X_train, X_test, y_train, y_test : pandas.DataFrame
        Train and test data
    dates_test : array-like
        Dates for test data
    watch_name : str
        Name of the watch
    output_dir : str
        Directory to save output plots

    Returns:
    -------
    dict
        Evaluation metrics
    """
    # Train model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    metrics = evaluate_model(y_test, y_pred, "XGBoost")

    # Plot predictions
    plot_predictions(y_test, y_pred, dates_test, "XGBoost", output_dir, watch_name)

    # Feature importance
    feature_importance = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": model.feature_importances_}
    )
    feature_importance = feature_importance.sort_values("Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance["Feature"][:10], feature_importance["Importance"][:10])
    plt.title(f"{watch_name} - XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{watch_name}_xgb_feature_importance.png"))
    plt.close()

    return metrics


def build_arima(df, test_size, watch_name, output_dir):
    """
    Build and evaluate an ARIMA model

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    test_size : float
        Proportion of data to use for testing
    watch_name : str
        Name of the watch
    output_dir : str
        Directory to save output plots

    Returns:
    -------
    dict
        Evaluation metrics
    """
    # Extract the price series
    price_series = df["price(SGD)"]

    # Split into train and test sets
    split_idx = int(len(price_series) * (1 - test_size))
    train, test = price_series[:split_idx], price_series[split_idx:]

    # Fit ARIMA model - try different orders if this doesn't work well
    try:
        # Auto ARIMA could be used here if available
        model = ARIMA(train, order=(5, 1, 2))
        model_fit = model.fit()

        # Make predictions
        predictions = model_fit.forecast(steps=len(test))

        # Evaluate model
        metrics = evaluate_model(test, predictions, "ARIMA")

        # Plot predictions
        plot_predictions(test, predictions, test.index, "ARIMA", output_dir, watch_name)

        # Plot model diagnostics
        plt.figure(figsize=(14, 10))
        model_fit.plot_diagnostics(fig=plt.gcf())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{watch_name}_arima_diagnostics.png"))
        plt.close()

        return metrics
    except Exception:
        print(
            f"ARIMA model failed for {watch_name}. Try different parameters or more data."
        )
        return None


def build_sarima(df, test_size, watch_name, output_dir):
    """
    Build and evaluate a SARIMA model

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    test_size : float
        Proportion of data to use for testing
    watch_name : str
        Name of the watch
    output_dir : str
        Directory to save output plots

    Returns:
    -------
    dict
        Evaluation metrics
    """
    # Extract the price series
    price_series = df["price(SGD)"]

    # Split into train and test sets
    split_idx = int(len(price_series) * (1 - test_size))
    train, test = price_series[:split_idx], price_series[split_idx:]

    try:
        # Fit SARIMA model - try different orders if this doesn't work well
        # order=(p,d,q), seasonal_order=(P,D,Q,s)
        model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))
        model_fit = model.fit(disp=False)

        # Make predictions
        predictions = model_fit.forecast(steps=len(test))

        # Evaluate model
        metrics = evaluate_model(test, predictions, "SARIMA")

        # Plot predictions
        plot_predictions(
            test, predictions, test.index, "SARIMA", output_dir, watch_name
        )

        # Plot model diagnostics
        plt.figure(figsize=(14, 10))
        model_fit.plot_diagnostics(fig=plt.gcf())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{watch_name}_sarima_diagnostics.png"))
        plt.close()

        return metrics
    except Exception:
        print(
            f"SARIMA model failed for {watch_name}. Try different parameters or more data."
        )
        return None


def forecast_future_prices(df, best_model_name, future_days, watch_name, output_dir):
    """
    Forecast future prices for a watch

    Parameters:
    ----------
    df : pandas.DataFrame
        DataFrame with time series data
    best_model_name : str
        Name of the best performing model
    future_days : int
        Number of days to forecast
    watch_name : str
        Name of the watch
    output_dir : str
        Directory to save output plots

    Returns:
    -------
    pandas.DataFrame
        DataFrame with forecasted prices
    """
    # Create a dataframe with future dates
    last_date = df.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days
    )

    # Calculate min and max historical prices to use as constraints
    min_historical_price = df["price(SGD)"].min()
    max_historical_price = df["price(SGD)"].max()
    # Calculate the historical price range for sanity checks
    price_range = max_historical_price - min_historical_price
    # Get the last observed price
    last_observed_price = df["price(SGD)"].iloc[-1]

    print(f"\nHistorical price info for {watch_name}:")
    print(f"Min price: {min_historical_price:.2f} SGD")
    print(f"Max price: {max_historical_price:.2f} SGD")
    print(f"Current price: {last_observed_price:.2f} SGD")

    # Force use of SARIMA for longer forecasts (more stable) if best model is linear regression
    if best_model_name == "Linear Regression" and future_days > 14:
        print(
            f"Note: Switching from Linear Regression to SARIMA for {watch_name} long-term forecast"
        )
        best_model_name = "SARIMA"

    if best_model_name in ["ARIMA", "SARIMA"]:
        # For ARIMA/SARIMA, we can use the model's built-in forecast
        price_series = df["price(SGD)"]

        if best_model_name == "ARIMA":
            model = ARIMA(price_series, order=(5, 1, 2))
        else:  # SARIMA
            model = SARIMAX(price_series, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7))

        model_fit = model.fit(disp=False)
        future_forecast = model_fit.forecast(steps=future_days)

        # Apply constraints to ensure reasonable forecasts
        # Ensure forecasts don't drop below min_historical_price * 0.9
        # or exceed max_historical_price * 1.1
        lower_bound = min_historical_price * 0.9
        upper_bound = max_historical_price * 1.1

        # Apply constraints
        future_forecast = np.clip(future_forecast, lower_bound, upper_bound)

        # Create forecast dataframe
        forecast_df = pd.DataFrame(
            {"date": future_dates, "forecasted_price": future_forecast}
        )
        forecast_df.set_index("date", inplace=True)

    else:
        # For ML models, we need to create future features
        # Start with the last 14 days (or more) to have enough data for lag features
        historical_data = df.iloc[-14:].copy()
        forecast_df = pd.DataFrame(index=future_dates)

        # Add date features
        forecast_df["day_of_week"] = forecast_df.index.dayofweek
        forecast_df["day_of_month"] = forecast_df.index.day
        forecast_df["month"] = forecast_df.index.month
        forecast_df["quarter"] = forecast_df.index.quarter
        forecast_df["year"] = forecast_df.index.year
        forecast_df["is_weekend"] = forecast_df.index.dayofweek >= 5

        # Initialize with the last known price
        last_price = df["price(SGD)"].iloc[-1]
        forecast_prices = []

        # Define reasonable price bounds (allow 10% outside historical range)
        lower_bound = min_historical_price * 0.9
        upper_bound = max_historical_price * 1.1

        # Define a maximum allowed price change per day (as % of price range)
        max_daily_change_pct = 0.05  # 5% of the historical range per day
        max_daily_change = price_range * max_daily_change_pct

        # For each future day
        for i in range(future_days):
            # Get the current date we're forecasting
            current_date = future_dates[i]

            # Create a temporary dataframe for the last 14 days plus days we've already forecasted
            temp_df = pd.concat([historical_data, forecast_df.iloc[:i]])

            # Add the last known or forecasted price
            if i == 0:
                forecast_df.loc[current_date, "price(SGD)"] = last_price
            else:
                forecast_df.loc[current_date, "price(SGD)"] = forecast_prices[-1]

            # Update lag features - with extra care for data boundaries
            forecast_df.loc[current_date, "price_lag_1"] = temp_df["price(SGD)"].iloc[
                -1
            ]
            forecast_df.loc[current_date, "price_lag_2"] = (
                temp_df["price(SGD)"].iloc[-2]
                if len(temp_df) >= 2
                else temp_df["price(SGD)"].iloc[-1]
            )
            forecast_df.loc[current_date, "price_lag_3"] = (
                temp_df["price(SGD)"].iloc[-3]
                if len(temp_df) >= 3
                else temp_df["price(SGD)"].iloc[-1]
            )
            forecast_df.loc[current_date, "price_lag_7"] = (
                temp_df["price(SGD)"].iloc[-7]
                if len(temp_df) >= 7
                else temp_df["price(SGD)"].iloc[-1]
            )

            # Calculate rolling means - with better handling of small windows
            if len(temp_df) >= 3:
                forecast_df.loc[current_date, "rolling_mean_3"] = (
                    temp_df["price(SGD)"].iloc[-3:].mean()
                )
            else:
                forecast_df.loc[current_date, "rolling_mean_3"] = temp_df[
                    "price(SGD)"
                ].mean()

            if len(temp_df) >= 7:
                forecast_df.loc[current_date, "rolling_mean_7"] = (
                    temp_df["price(SGD)"].iloc[-7:].mean()
                )
            else:
                forecast_df.loc[current_date, "rolling_mean_7"] = temp_df[
                    "price(SGD)"
                ].mean()

            if len(temp_df) >= 14:
                forecast_df.loc[current_date, "rolling_mean_14"] = (
                    temp_df["price(SGD)"].iloc[-14:].mean()
                )
            else:
                forecast_df.loc[current_date, "rolling_mean_14"] = temp_df[
                    "price(SGD)"
                ].mean()

            # Calculate rolling std - with minimum values to avoid division by zero
            if len(temp_df) >= 3 and temp_df["price(SGD)"].iloc[-3:].std() > 0:
                forecast_df.loc[current_date, "rolling_std_3"] = (
                    temp_df["price(SGD)"].iloc[-3:].std()
                )
            else:
                forecast_df.loc[current_date, "rolling_std_3"] = (
                    1.0  # Default minimal std
                )

            if len(temp_df) >= 7 and temp_df["price(SGD)"].iloc[-7:].std() > 0:
                forecast_df.loc[current_date, "rolling_std_7"] = (
                    temp_df["price(SGD)"].iloc[-7:].std()
                )
            else:
                forecast_df.loc[current_date, "rolling_std_7"] = (
                    1.0  # Default minimal std
                )

            # Calculate momentum - with better handling of small windows
            if len(temp_df) >= 2:
                forecast_df.loc[current_date, "momentum_1"] = (
                    temp_df["price(SGD)"].iloc[-1] - temp_df["price(SGD)"].iloc[-2]
                )
            else:
                forecast_df.loc[current_date, "momentum_1"] = 0

            if len(temp_df) >= 4:
                forecast_df.loc[current_date, "momentum_3"] = (
                    temp_df["price(SGD)"].iloc[-1] - temp_df["price(SGD)"].iloc[-4]
                )
            else:
                forecast_df.loc[current_date, "momentum_3"] = 0

            if len(temp_df) >= 8:
                forecast_df.loc[current_date, "momentum_7"] = (
                    temp_df["price(SGD)"].iloc[-1] - temp_df["price(SGD)"].iloc[-8]
                )
            else:
                forecast_df.loc[current_date, "momentum_7"] = 0

            # Use the best ML model to predict the next price
            if best_model_name == "Linear Regression":
                # Use log transformation for linear regression to prevent negative prices
                X = df.drop(["price(SGD)", "target"], axis=1)
                # Create log of target (adding small constant to avoid log(0))
                y_log = np.log(df["target"] + 1)

                model = LinearRegression()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y_log)

                # Predict next price (in log space)
                forecast_features = forecast_df.loc[current_date].drop(["price(SGD)"])
                forecast_features_scaled = scaler.transform(
                    forecast_features.values.reshape(1, -1)
                )
                next_price_log = model.predict(forecast_features_scaled)[0]

                # Convert back from log space to normal
                next_price = np.exp(next_price_log) - 1

            elif best_model_name == "Random Forest":
                # Train a random forest model on all available data
                X = df.drop(["price(SGD)", "target"], axis=1)
                y = df["target"]
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)

                # Predict next price
                forecast_features = forecast_df.loc[current_date].drop(["price(SGD)"])
                next_price = model.predict(forecast_features.values.reshape(1, -1))[0]

            elif best_model_name == "XGBoost":
                # Train an XGBoost model on all available data
                X = df.drop(["price(SGD)", "target"], axis=1)
                y = df["target"]
                model = xgb.XGBRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42
                )
                model.fit(X, y)

                # Predict next price
                forecast_features = forecast_df.loc[current_date].drop(["price(SGD)"])
                next_price = model.predict(forecast_features.values.reshape(1, -1))[0]

            elif best_model_name == "LSTM":
                # For LSTM, let's use a simpler approach here
                # Fall back to XGBoost for simplicity
                X = df.drop(["price(SGD)", "target"], axis=1)
                y = df["target"]
                model = xgb.XGBRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42
                )
                model.fit(X, y)

                # Predict next price
                forecast_features = forecast_df.loc[current_date].drop(["price(SGD)"])
                next_price = model.predict(forecast_features.values.reshape(1, -1))[0]

            else:
                # Default to simple moving average
                next_price = forecast_df.loc[current_date, "rolling_mean_7"]

            # Get previous forecasted price (or last known price for first day)
            prev_price = last_price if i == 0 else forecast_prices[-1]

            # Limit daily price change to prevent extreme swings
            price_change = next_price - prev_price
            if abs(price_change) > max_daily_change:
                direction = 1 if price_change > 0 else -1
                next_price = prev_price + (direction * max_daily_change)

            # Apply global constraints to ensure reasonable forecasts
            next_price = np.clip(next_price, lower_bound, upper_bound)

            # Store the forecasted price
            forecast_prices.append(next_price)

            # Update the forecasted price for the next iteration
            if i < future_days - 1:  # Don't need to update the last day
                forecast_df.loc[future_dates[i + 1], "price(SGD)"] = next_price

        # Update the forecasted prices column
        forecast_df["forecasted_price"] = forecast_prices

    # Perform a final sanity check on the forecasted prices
    print(
        f"Forecast range for {watch_name}: {forecast_df['forecasted_price'].min():.2f} to {forecast_df['forecasted_price'].max():.2f} SGD"
    )

    # Plot the forecast
    plt.figure(figsize=(14, 7))

    # Plot historical prices (last 30 days)
    historical_period = min(30, len(df))
    plt.plot(
        df.index[-historical_period:],
        df["price(SGD)"][-historical_period:],
        label="Historical",
        linewidth=2,
    )

    # Plot forecasted prices
    plt.plot(
        forecast_df.index,
        forecast_df["forecasted_price"],
        label="Forecasted",
        linestyle="--",
    )

    # Add confidence intervals (simplified, using standard deviation)
    if "rolling_std_7" in forecast_df.columns:
        std = forecast_df["rolling_std_7"].mean()
        plt.fill_between(
            forecast_df.index,
            forecast_df["forecasted_price"] - 1.96 * std,
            forecast_df["forecasted_price"] + 1.96 * std,
            alpha=0.2,
            label="95% Confidence Interval",
        )

    plt.title(f"{watch_name} - Price Forecast ({future_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Price (SGD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{watch_name}_price_forecast.png"))
    plt.close()

    # Add this code to save the forecast data to CSV
    # Reset the index to make 'date' a column
    forecast_csv = forecast_df.reset_index()
    # Keep only the 'date' and 'forecasted_price' columns
    forecast_csv = forecast_csv[["date", "forecasted_price"]]
    # Save to CSV
    forecast_csv.to_csv(
        os.path.join(output_dir, f"{watch_name}_price_forecast.csv"), index=False
    )

    return forecast_df


def compare_models(watch_name, metrics_list, output_dir):
    """
    Compare different models and select the best one

    Parameters:
    ----------
    watch_name : str
        Name of the watch
    metrics_list : list
        List of dictionaries with model metrics
    output_dir : str
        Directory to save output plots

    Returns:
    -------
    str
        Name of the best performing model
    """
    # Create a dataframe with model metrics
    metrics_df = pd.DataFrame(metrics_list)

    # Sort by RMSE (lower is better)
    metrics_df = metrics_df.sort_values("rmse")

    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.barplot(x="model", y="rmse", data=metrics_df)
    plt.title("RMSE Comparison")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.barplot(x="model", y="r2", data=metrics_df)
    plt.title("R² Comparison")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{watch_name}_model_comparison.png"))
    plt.close()

    # Print best model
    best_model = metrics_df.iloc[0]["model"]
    print(f"\n--- Best Model for {watch_name} ---")
    print(f"Model: {best_model}")
    print(f"RMSE: {metrics_df.iloc[0]['rmse']:.4f}")
    print(f"R²: {metrics_df.iloc[0]['r2']:.4f}")

    return best_model


def run_models(
    featured_data, output_dir="data/output/", test_size=0.2, forecast_days=30
):
    """
    Run all models on the featured data

    Parameters:
    ----------
    featured_data : dict
        Dictionary of dataframes with engineered features
    output_dir : str, default='data/output/'
        Directory to save output plots
    test_size : float, default=0.2
        Proportion of data to use for testing
    forecast_days : int, default=30
        Number of days to forecast
    """
    ensure_output_dir(output_dir)

    # Store results for all watches
    all_results = {}

    # Process each watch
    for watch_name, df in featured_data.items():
        print(f"\n=== Modeling for {watch_name} ===")

        # Regular ML models
        X_train, X_test, y_train, y_test = train_test_split_ts(df, test_size)

        # Get test dates for plotting
        dates_test = df.index[int(len(df) * (1 - test_size)) :]

        # Initialize metrics list
        metrics_list = []

        # Linear Regression
        lr_metrics = build_linear_regression(
            X_train, X_test, y_train, y_test, dates_test, watch_name, output_dir
        )
        metrics_list.append(lr_metrics)

        # Random Forest
        rf_metrics = build_random_forest(
            X_train, X_test, y_train, y_test, dates_test, watch_name, output_dir
        )
        metrics_list.append(rf_metrics)

        # XGBoost
        xgb_metrics = build_xgboost(
            X_train, X_test, y_train, y_test, dates_test, watch_name, output_dir
        )
        metrics_list.append(xgb_metrics)

        # ARIMA
        arima_metrics = build_arima(df, test_size, watch_name, output_dir)
        if arima_metrics:
            metrics_list.append(arima_metrics)

        # SARIMA
        sarima_metrics = build_sarima(df, test_size, watch_name, output_dir)
        if sarima_metrics:
            metrics_list.append(sarima_metrics)

        # # LSTM (if enough data)
        # if len(df) >= 100:  # Arbitrary threshold, adjust as needed
        #     lstm_metrics = build_lstm(
        #         X_train, X_test, y_train, y_test, dates_test, watch_name, output_dir
        #     )
        #     metrics_list.append(lstm_metrics)

        # Compare models and select the best
        best_model = compare_models(watch_name, metrics_list, output_dir)

        # Forecast future prices
        future_forecast = forecast_future_prices(
            df, best_model, forecast_days, watch_name, output_dir
        )

        # Store results
        all_results[watch_name] = {
            "metrics": metrics_list,
            "best_model": best_model,
            "forecast": future_forecast,
        }

    # Create a summary table
    summary_table = []
    for watch_name, results in all_results.items():
        summary_table.append(
            {
                "Watch Model": watch_name,
                "Best Model": results["best_model"],
                "RMSE": [
                    m["rmse"]
                    for m in results["metrics"]
                    if m["model"] == results["best_model"]
                ][0],
                "Current Price": featured_data[watch_name]["price(SGD)"].iloc[-1],
                "Forecasted Price (7 days)": results["forecast"][
                    "forecasted_price"
                ].iloc[6]
                if len(results["forecast"]) > 6
                else None,
                "Forecasted Price (30 days)": results["forecast"][
                    "forecasted_price"
                ].iloc[-1]
                if len(results["forecast"]) >= forecast_days
                else None,
                "Price Change %": (
                    (
                        results["forecast"]["forecasted_price"].iloc[-1]
                        / featured_data[watch_name]["price(SGD)"].iloc[-1]
                    )
                    - 1
                )
                * 100
                if len(results["forecast"]) >= forecast_days
                else None,
            }
        )

    summary_df = pd.DataFrame(summary_table)

    # Format price columns as currency and percentage column
    for col in [
        "Current Price",
        "Forecasted Price (7 days)",
        "Forecasted Price (30 days)",
    ]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else None
            )

    if "Price Change %" in summary_df.columns:
        summary_df["Price Change %"] = summary_df["Price Change %"].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else None
        )

    # Save to CSV
    summary_df.to_csv(os.path.join(output_dir, "model_summary.csv"), index=False)

    print("\n=== Model Summary ===")
    print(summary_df.to_string())

    return all_results


# Main function to call from the pipeline
def main():
    """
    Main function to run the time series forecasting models
    """
    # Define output directory
    output_dir = "data/output/"
    ensure_output_dir(output_dir)

    # Load the featured data created by the preprocessing script
    featured_data_path = os.path.join(output_dir, "featured_data.csv")

    if os.path.exists(featured_data_path):
        # Load the data
        featured_df = pd.read_csv(featured_data_path)

        # Convert date to datetime
        featured_df["date"] = pd.to_datetime(featured_df["date"])

        # Set date as index
        featured_df.set_index("date", inplace=True)

        # Split data by watch model
        watch_data = {}
        for watch_name, group in featured_df.groupby("watch_model"):
            watch_data[watch_name] = group.drop("watch_model", axis=1)

        # Run models
        run_models(watch_data, output_dir, test_size=0.2, forecast_days=30)

        print("\n--- Completed modeling and forecasting ---")
    else:
        print(f"Featured data file not found at {featured_data_path}")
        print("Please run the data preprocessing script first")


if __name__ == "__main__":
    main()
