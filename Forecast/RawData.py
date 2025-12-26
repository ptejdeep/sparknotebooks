# Databricks notebook source
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# --- 1. Raw Data Generation (Same as before) ---

def generate_jfk_traffic_data(
    start_date: str = '2024-10-01',
    end_date: str = '2024-12-31',
    daily_vehicle_ranges: dict = None,
    locations: list = None
):
    """
    Generates synthetic historical vehicle movement data for JFK Airport.
    (This function remains the same as previously provided)
    """
    if daily_vehicle_ranges is None:
        daily_vehicle_ranges = {
            'Car': (15000, 24000),
            'Taxi/RideShare': (10000, 40000),
            'Bus': (1000, 5000),
            'Van': (800, 3000),
            'PickupTruck': (500, 2500)
        }

    if locations is None:
        locations = [
            'ARR Entering AR1 - West', 'ARR Entering AR1 - East', 'DEP door c',
            'DEP door B', 'ARR door a', 'ARR door b', 'DEP Curbside - Terminal 4',
            'ARR Curbside - Terminal 5', 'Cargo Entrance - North', 'Cargo Entrance - South'
        ]

    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))
    data = []
    vehicleseqid_counter = 1

    peak_hours_morning = [6, 7, 8, 9]
    peak_hours_afternoon = [15, 16, 17, 18, 19]
    hourly_distribution = {
        0: 0.02, 1: 0.01, 2: 0.005, 3: 0.005, 4: 0.01, 5: 0.02,
        6: 0.06, 7: 0.08, 8: 0.09, 9: 0.08, 10: 0.06, 11: 0.05,
        12: 0.05, 13: 0.05, 14: 0.05, 15: 0.06, 16: 0.07, 17: 0.08,
        18: 0.07, 19: 0.06, 20: 0.04, 21: 0.03, 22: 0.02, 23: 0.01
    }
    total_hourly_dist = sum(hourly_distribution.values())
    hourly_distribution = {k: v / total_hourly_dist for k, v in hourly_distribution.items()}


    for current_date in date_range:
        day_of_week = current_date.dayofweek
        is_weekend = day_of_week >= 5

        daily_multipliers = {
            'Car': 0.9 if is_weekend else 1.0,
            'Taxi/RideShare': 0.95 if is_weekend else 1.0,
            'Bus': 0.8 if is_weekend else 1.0,
            'Van': 0.9 if is_weekend else 1.0,
            'PickupTruck': 0.7 if is_weekend else 1.0
        }

        daily_counts_by_type = {}
        for v_type, (min_count, max_count) in daily_vehicle_ranges.items():
            adjusted_min = int(min_count * daily_multipliers[v_type])
            adjusted_max = int(max_count * daily_multipliers[v_type])
            daily_counts_by_type[v_type] = random.randint(adjusted_min, adjusted_max)

        for v_type, daily_count in daily_counts_by_type.items():
            hourly_counts = {hour: int(daily_count * prob) for hour, prob in hourly_distribution.items()}
            remaining = daily_count - sum(hourly_counts.values())
            for _ in range(remaining):
                random_hour = random.choice(list(hourly_counts.keys()))
                hourly_counts[random_hour] += 1

            for hour, count_in_hour in hourly_counts.items():
                for _ in range(count_in_hour):
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    movement_time = current_date.replace(hour=hour, minute=minute, second=second)

                    location = random.choice(locations)
                    v_type_id = list(daily_vehicle_ranges.keys()).index(v_type) + 1
                    loc_id = locations.index(location) + 1

                    data.append({
                        'datetime': movement_time,
                        'vehicletype': v_type,
                        'entryexitlocationname': location,
                        'vehicletypeid': v_type_id,
                        'entryexitlocationid': loc_id,
                        'vehicleseqid': vehicleseqid_counter
                    })
                    vehicleseqid_counter += 1

    df = pd.DataFrame(data)
    df = df.sort_values(by='datetime').reset_index(drop=True)
    return df

print("Generating synthetic JFK traffic data...")
raw_df = generate_jfk_traffic_data()
print("Raw data generated. Sample head:")
print(raw_df.head())
print(f"\nTotal records generated: {len(raw_df)}")
print(f"Data period: {raw_df['datetime'].min()} to {raw_df['datetime'].max()}")

# COMMAND ----------


spark_df = spark.createDataFrame(raw_df)
spark_df.coalesce(1).write.mode("overwrite").option("header", True).csv("/Workspace/rawdata/jfk_traffic_data.csv")

# COMMAND ----------

datadf = spark.read.csv("/Workspace/rawdata/jfk_traffic_data.csv",header=True)
display(datadf)

# COMMAND ----------

from datetime import datetime, timedelta
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# --- 2. Forecast Model (Slightly modified for better plotting output) ---

def create_traffic_forecast_model(df: pd.DataFrame):
    """
    Builds and evaluates a vehicle traffic forecast model.
    (Includes minor modifications for cleaner plotting preparation)
    """
    print("\n--- Starting Forecast Model Building ---")

    # --- 2.1. Data Preprocessing and Aggregation ---
    print("Preprocessing and aggregating data...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    df_agg = df.groupby([
        pd.Grouper(freq='H'),
        'vehicletype',
        'entryexitlocationname'
    ]).size().unstack(fill_value=0)

    df_agg['Total_Vehicles'] = df_agg.sum(axis=1)
    if df_agg.empty:
        print("Aggregated DataFrame is empty. Check data generation or date range.")
        return None, None, None, None

    # Correctly access the timestamp from the MultiIndex
    min_date = df_agg.index.get_level_values(0).min()
    max_date = df_agg.index.get_level_values(0).max()
    full_hourly_index = pd.date_range(start=min_date, end=max_date, freq='H')
    df_agg = df_agg.reset_index()

    # --- 2.2. Feature Engineering ---
    print("Engineering features...")

    df_agg['datetime'] = pd.to_datetime(df_agg['datetime'])
    df_agg.set_index('datetime', inplace=True)

    df_agg['hour'] = df_agg.index.hour
    df_agg['dayofweek'] = df_agg.index.dayofweek
    df_agg['dayofyear'] = df_agg.index.dayofyear
    df_agg['month'] = df_agg.index.month
    df_agg['year'] = df_agg.index.year
    df_agg['weekofyear'] = df_agg.index.isocalendar().week.astype(int)
    df_agg['is_weekend'] = (df_agg.index.dayofweek >= 5).astype(int)
    df_agg['quarter'] = df_agg.index.quarter

    df_agg['Total_Vehicles_lag_1h'] = df_agg['Total_Vehicles'].shift(1)
    df_agg['Total_Vehicles_lag_24h'] = df_agg['Total_Vehicles'].shift(24)
    df_agg['Total_Vehicles_lag_168h'] = df_agg['Total_Vehicles'].shift(168)

    df_agg['Total_Vehicles_rolling_24h_mean'] = df_agg['Total_Vehicles'].rolling(window=24, closed='left').mean()
    df_agg['Total_Vehicles_rolling_7d_mean'] = df_agg['Total_Vehicles'].rolling(window=24*7, closed='left').mean()

    df_agg.fillna(0, inplace=True)

    # --- 2.3. Model Selection & Training ---
    print("Splitting data and training model...")

    target = 'Total_Vehicles'
    features = [
        'hour', 'dayofweek', 'dayofyear', 'month', 'year', 'weekofyear', 'is_weekend', 'quarter',
        'Total_Vehicles_lag_1h', 'Total_Vehicles_lag_24h', 'Total_Vehicles_lag_168h',
        'Total_Vehicles_rolling_24h_mean', 'Total_Vehicles_rolling_7d_mean'
    ]

    df_model = df_agg.dropna(subset=features + [target])

    X = df_model[features]
    y = df_model[target]

    test_size_days = 30
    test_start_date = pd.to_datetime(df_model.index.max()) - pd.Timedelta(days=test_size_days)

    X_train = X[X.index < test_start_date]
    y_train = y[y.index < test_start_date]
    X_test = X[X.index >= test_start_date]
    y_test = y[y.index >= test_start_date]

    print(f"Train data period: {X_train.index.min()} to {X_train.index.max()}")
    print(f"Test data period: {X_test.index.min()} to {X_test.index.max()}")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)
    trained_model = model

    # --- 2.4. Prediction ---
    print("Making predictions...")
    predictions_test = model.predict(X_test)
    predictions_test[predictions_test < 0] = 0

    # Create a DataFrame for actuals and predictions on the test set
    forecast_df_test = pd.DataFrame({
        'Actual': y_test,
        'Predicted': predictions_test
    }, index=y_test.index)

    # --- 2.5. Evaluation ---
    print("Evaluating model performance...")
    mae = mean_absolute_error(y_test, predictions_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    mape = np.mean(np.abs((y_test - predictions_test) / y_test)) * 100
    mape = mape if not np.isinf(mape) else 0

    evaluation_metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    print(f"Evaluation Metrics:\n{evaluation_metrics}")

    # --- 2.6. Future Forecasting ---
    print("\nGenerating future forecast (example for next 7 days)...")

    last_known_date = df_agg.index.max()
    future_dates = pd.date_range(start=last_known_date + pd.Timedelta(hours=1), periods=24*7, freq='H')
    future_df = pd.DataFrame(index=future_dates)

    combined_df = pd.concat([df_agg, future_df], sort=False)
    combined_df['hour'] = combined_df.index.hour
    combined_df['dayofweek'] = combined_df.index.dayofweek
    combined_df['dayofyear'] = combined_df.index.dayofyear
    combined_df['month'] = combined_df.index.month
    combined_df['year'] = combined_df.index.year
    combined_df['weekofyear'] = combined_df.index.isocalendar().week.astype(int)
    combined_df['is_weekend'] = (combined_df.index.dayofweek >= 5).astype(int)
    combined_df['quarter'] = combined_df.index.quarter

    # Re-calculate lags and rolling means on the combined dataframe for future prediction
    # This requires an iterative approach or careful handling of NaNs if not all features are known
    # For simplicity, we will fill NaNs in lagged/rolling features in the future section.
    # A more robust approach involves a loop where each predicted value feeds into the next lag.
    # For this example, we'll use the available data and fill NaNs.
    combined_df['Total_Vehicles_lag_1h'] = combined_df['Total_Vehicles'].shift(1)
    combined_df['Total_Vehicles_lag_24h'] = combined_df['Total_Vehicles'].shift(24)
    combined_df['Total_Vehicles_lag_168h'] = combined_df['Total_Vehicles'].shift(168)
    combined_df['Total_Vehicles_rolling_24h_mean'] = combined_df['Total_Vehicles'].rolling(window=24, closed='left').mean()
    combined_df['Total_Vehicles_rolling_7d_mean'] = combined_df['Total_Vehicles'].rolling(window=24*7, closed='left').mean()

    # Iterative prediction for future:
    # This loop is crucial for accurate multi-step-ahead forecasting with ML models
    # where future lag features depend on previous predictions.
    future_predictions_list = []
    current_features_for_prediction = combined_df.loc[last_known_date][features].values.reshape(1, -1)

    for i, date in enumerate(future_dates):
        # Create a DataFrame for the current future time step
        current_step_df = pd.DataFrame(index=[date])
        current_step_df['hour'] = date.hour
        current_step_df['dayofweek'] = date.dayofweek
        current_step_df['dayofyear'] = date.dayofyear
        current_step_df['month'] = date.month
        current_step_df['year'] = date.year
        current_step_df['weekofyear'] = date.isocalendar().week
        current_step_df['is_weekend'] = int(date.dayofweek >= 5)
        current_step_df['quarter'] = date.quarter

        # Get lag features from either actuals (if available) or previous predictions
        # For lag_1h, use the last actual/predicted value
        if i == 0: # First step, use last actual from df_agg
            current_step_df['Total_Vehicles_lag_1h'] = df_agg['Total_Vehicles'].iloc[-1]
        else: # Subsequent steps, use previous prediction
            current_step_df['Total_Vehicles_lag_1h'] = future_predictions_list[-1]

        # For longer lags (24h, 168h), get from df_agg or previous predictions if the lag falls within forecast window
        if (date - timedelta(hours=24)) in combined_df.index:
            current_step_df['Total_Vehicles_lag_24h'] = combined_df.loc[date - timedelta(hours=24)]['Total_Vehicles'] if (date - timedelta(hours=24)) <= last_known_date else future_predictions_list[i-24] if i>=24 else 0
        else:
            current_step_df['Total_Vehicles_lag_24h'] = 0

        if (date - timedelta(hours=168)) in combined_df.index:
            current_step_df['Total_Vehicles_lag_168h'] = combined_df.loc[date - timedelta(hours=168)]['Total_Vehicles'] if (date - timedelta(hours=168)) <= last_known_date else future_predictions_list[i-168] if i>=168 else 0
        else:
            current_step_df['Total_Vehicles_lag_168h'] = 0

        # Rolling means are tricky for future. For simplicity, we'll use the last known rolling mean from df_agg,
        current_step_df['Total_Vehicles_rolling_24h_mean'] = combined_df.loc[last_known_date]['Total_Vehicles_rolling_24h_mean'] if i==0 else future_predictions_list[-100:].mean() if len(future_predictions_list) >= 100 else 0
        current_step_df['Total_Vehicles_rolling_7d_mean'] = combined_df.loc[last_known_date]['Total_Vehicles_rolling_7d_mean'] if i==0 else future_predictions_list[-500:].mean() if len(future_predictions_list) >= 500 else 0

        current_step_df.fillna(0, inplace=True)

        # Predict
        pred = model.predict(current_step_df[features])[0]
        future_predictions_list.append(max(0, pred))

    forecast_future_df = pd.DataFrame({
        'Predicted': future_predictions_list
    }, index=future_dates)

    # Combine historical predictions with future predictions for plotting
    full_forecast_df = pd.concat([forecast_df_test, forecast_future_df])
    # Add actuals from the training period for a complete view
    full_forecast_df['Actual'] = pd.concat([y_train, y_test, pd.Series(index=future_dates, dtype='float64')]).reindex(full_forecast_df.index)

    print("Forecast completed.")
    return trained_model, full_forecast_df, evaluation_metrics, df_agg, X_test.index.min(), future_dates.min()

# Run the forecast model
trained_model, full_forecast_df, eval_metrics, df_agg, test_start_date_actual, forecast_start_date_actual = create_traffic_forecast_model(raw_df.copy())