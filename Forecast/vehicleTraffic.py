# Databricks notebook source
filepath =  '/dbfs/FileStore/vehicletraffic.parquet'

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# Read the Parquet file into a pandas DataFrame
df = pd.read_parquet("/dbfs:/FileStore/vehicletraffic.parquet")

# Display the DataFrame
display(df)

# COMMAND ----------

/dbfs/FileStore/vehicletraffic.parquet