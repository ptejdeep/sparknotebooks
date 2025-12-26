# Databricks notebook source
# MAGIC %pip install scikit-learn
# MAGIC %pip install sktime
# MAGIC %pip install prophet
# MAGIC %pip install pmdarima

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sktime.forecasting.arima import AutoARIMA
from sklearn.model_selection import train_test_split
from sktime.forecasting.model_selection import temporal_train_test_split
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.compose import make_reduction
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# COMMAND ----------

from pyspark.sql.functions import collect_list, struct
from pyspark.sql.functions import col, hour, to_timestamp, date_trunc,collect_list, struct
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import hour, date_trunc

# COMMAND ----------

parquer = spark.read.parquet("dbfs:/FileStore/vehicletraffic.parquet")

# COMMAND ----------

display(parquer)

# COMMAND ----------

"""datadf = spark.read.csv("/Workspace/rawdata/jfk_traffic_data.csv",header=True)
display(datadf)"""

# COMMAND ----------


# Reload data as Spark DataFrame
raw_df_spark = spark.read.csv("/Workspace/rawdata/jfk_traffic_data.csv", header=True, inferSchema=True)

# Ensure timestamp column is in correct format
timestamp_col = 'timestamp' if 'timestamp' in raw_df_spark.columns else raw_df_spark.columns[0]
raw_df_spark = raw_df_spark.withColumn('event_time', to_timestamp(col(timestamp_col)))


# Enable Arrow-based columnar data transfers
# spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Prepare hourly Spark DataFrame
hourly_df = (
    raw_df_spark
    .withColumn('hour', hour('event_time'))
    .withColumn('date_hour', date_trunc('hour', 'event_time'))
    .groupBy('date_hour', 'vehicletype')
    .count()
    .orderBy('date_hour', 'vehicletype')
)

#display(hourly_df)

# Convert to Pandas for modeling
hourly_pd = hourly_df.toPandas()
hourly_pd['date_hour'] = pd.to_datetime(hourly_pd['date_hour'])
hourly_pd = hourly_pd.sort_values(['vehicletype', 'date_hour'])

# COMMAND ----------

hourly_pd.head()

# COMMAND ----------

for vtype in hourly_pd['vehicletype'].unique():
    df_vtype = hourly_pd[hourly_pd['vehicletype'] == vtype][['date_hour', 'count']].set_index('date_hour')
    df_vtype.index.freq = 'H'  # Explicitly set the frequency
    y = df_vtype['count']
    y_train, y_test = temporal_train_test_split(y, test_size=24*7)

    # ARIMA
    arima = AutoARIMA(sp=24, maxiter=1000)  # Increase the number of iterations
    arima.fit(y_train)
    y_pred_arima = arima.predict(fh=list(range(1, len(y_test)+1)))

    # SARIMA
    sarima = AutoARIMA(sp=24, seasonal=True, maxiter=1000)  # Increase the number of iterations
    sarima.fit(y_train)
    y_pred_sarima = sarima.predict(fh=list(range(1, len(y_test)+1)))

    # Prophet
    # prophet_df = y.reset_index().rename(columns={'date_hour': 'ds', 'count': 'y'})
    # prophet_train = prophet_df.iloc[:len(y_train)]
    # m = Prophet()
    # m.fit(prophet_train)
    # future = m.make_future_dataframe(periods=len(y_test), freq='H')
    # forecast = m.predict(future)
    # y_pred_prophet = forecast['yhat'].iloc[-len(y_test):].values

    # Plot


# COMMAND ----------

    plt.figure(figsize=(12, 5))
    plt.plot(y, label='Actual')
    plt.plot(y_test.index, y_pred_arima, label='ARIMA Forecast')
    plt.plot(y_test.index, y_pred_sarima, label='SARIMA Forecast')
    plt.plot(y_test.index, y_pred_prophet, label='Prophet Forecast')
    plt.title(f'Hourly Forecast for Vehicle Type: {vtype}')
    plt.xlabel('Date Hour')
    plt.ylabel('Count')
    plt.legend()
    plt.show()