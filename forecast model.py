# Databricks notebook source
output_base_path = "/Volumes/workspace/processed/files/"
file_path = f"{output_base_path}vehicletrafficraw"

df = spark.read.parquet(file_path)
parquet_file_path = "/Volumes/workspace/processed/files/vehicletrafficraw.parquet" 

print(f"\nSaving raw_df to a single Parquet file at: {parquet_file_path}")

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Reduce the number of partitions to 1 before writing
# Use .coalesce(1) for minimal data shuffle if reducing partitions
# Use .repartition(1) for a full shuffle to ensure even distribution (might be slower for large data)
# For a single file, either works, but coalesce(1) is generally preferred if you're only reducing.
spark_df.coalesce(1).write.mode("overwrite").parquet(parquet_file_path)

# COMMAND ----------

# Assuming 'spark' session is already available in Databricks notebook environment

output_base_path = "/Volumes/workspace/processed/files/"
file_path = f"{output_base_path}vehicletrafficraw" # This is the directory containing multiple parquet files

print(f"Reading multi-part Parquet files from: {file_path}")
df = spark.read.parquet(file_path) # df is now a Spark DataFrame

# Define the full path for your single Parquet file
# This will be the name of the single output file
parquet_single_file_path = "/Volumes/workspace/processed/files/vehicletrafficraw.parquet" 

print(f"\nSaving the Spark DataFrame to a single Parquet file at: {parquet_single_file_path}")

# Reduce the number of partitions to 1 and then write to a single file
# Using df directly, as it's already a Spark DataFrame
df.coalesce(1).write.mode("overwrite").parquet(parquet_single_file_path)

print("Data saved successfully as a single Parquet file.")

# --- Optional: Verify by reading the single file back ---
print(f"\nVerifying by reading the single Parquet file back from: {parquet_single_file_path}")
df_read_back_single = spark.read.parquet(parquet_single_file_path)

print("Schema of the read-back Spark DataFrame (single file):")
df_read_back_single.printSchema()
print("\nSample data from the read-back Spark DataFrame (single file):")
df_read_back_single.show(5, truncate=False)

# To confirm it's a single file, you can check the number of partitions:
print(f"\nNumber of partitions in the read-back single file DataFrame: {df_read_back_single.rdd.getNumPartitions()}")