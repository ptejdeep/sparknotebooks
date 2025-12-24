# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, lit

# Initialize Spark session
spark = SparkSession.builder.appName("CricketMatchFlatten").getOrCreate()

# Load JSON file into DataFrame
file_path = "dbfs:/FileStore/1383534.json"  # Update path if needed
df = spark.read.option("multiline", "true").json(file_path)

# COMMAND ----------

# Extract match metadata
match_info = df.selectExpr("info.match_type as Match_ID", "info.dates[0] as Date", "info.venue as Venue",
                           "info.teams[0] as Team1", "info.teams[1] as Team2",
                           "info.outcome.winner as Winner")

# COMMAND ----------

# Extract match metadata
match_info = df.selectExpr("info.match_type as Match_ID", "info.dates[0] as Date", "info.venue as Venue",
                           "info.teams[0] as Team1", "info.teams[1] as Team2",
                           "info.outcome.winner as Winner")

# Explode innings to access nested delivery data
innings_df = df.select(explode("innings").alias("inning")).selectExpr("inning.team as Team", "inning.overs")
overs_df = innings_df.select("Team", explode("overs").alias("over_data"))
deliveries_df = overs_df.select("Team", col("over_data.over").alias("Over"), explode("over_data.deliveries").alias("delivery"))

# Extract ball-by-ball details
final_df = deliveries_df.select(
    lit(match_info.collect()[0][0]).alias("Match_ID"),
    lit(match_info.collect()[0][1]).alias("Date"),
    lit(match_info.collect()[0][2]).alias("Venue"),
    "Team",
    "Over",
    col("delivery.batter").alias("Batter"),
    col("delivery.bowler").alias("Bowler"),
    col("delivery.non_striker").alias("Non_Striker"),
    col("delivery.runs.batter").alias("Runs_Batter"),
    col("delivery.runs.extras").alias("Runs_Extras"),
    col("delivery.runs.total").alias("Runs_Total"),
    col("delivery.wickets.player_out")[0].alias("Wicket_Player_Out"),
    col("delivery.wickets.kind")[0].alias("Wicket_Kind")
)

# Show result
final_df.show(truncate=False)

# Save to CSV (modify path as needed)
final_df.write.mode("overwrite").csv("flattened_match_data.csv", header=True)


# COMMAND ----------

filecsv = "dbfs:/flattened_match_data.csv"

# COMMAND ----------

csv_df = spark.read.option("header", "true").csv(filecsv)
display(csv_df)

# COMMAND ----------

json_file_path = "dbfs:/FileStore/1383534.json" 

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, explode, get_json_object
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType, MapType

def process_nested_json(spark, json_file_path, schema):
    """
    Reads a multi-line nested JSON file, applies a schema, and transforms the data.

    Args:
        spark: SparkSession instance.
        json_file_path: Path to the multi-line JSON file.
        schema: Spark StructType schema for the JSON data.

    Returns:
        DataFrame: Transformed Spark DataFrame.
    """

    df = spark.read.text(json_file_path)  # Read as text to handle multi-line JSON

    # Parse each line as JSON. Errors are handled by setting mode to PERMISSIVE.
    df = df.withColumn("json_data", from_json(col("value"), schema, mode="PERMISSIVE"))

    # Explode the array if the root element is an array.
    # Check if schema is an array.
    if isinstance(schema, ArrayType):
      df = df.withColumn("exploded_data", explode(col("json_data"))).select("exploded_data.*")
      # If the schema is not an array, select the json_data column directly.
    else:
      df = df.select("json_data.*")

    # Example transformations (replace with your specific logic):
    # 1. Access nested fields:
    # df = df.withColumn("nested_field", col("address.city"))

    # 2. Explode nested arrays (if any):
    # df = df.withColumn("items", explode(col("items"))).select("*", "items.*")  # Assuming 'items' is an array

    # 3. Rename columns (optional):
    # df = df.withColumnRenamed("old_name", "new_name")

    # 4. Filter data (optional):
    # df = df.filter(col("age") > 18)

    # 5. Select relevant columns:
    # df = df.select("name", "nested_field", "items.id", "items.value")


    return df


# Example usage:
if __name__ == "__main__":
    spark = SparkSession.builder.appName("NestedJsonProcessor").getOrCreate()

    json_file_path = ""dbfs:/FileStore/1383534.json"   # Replace with your file path

    # Define your schema (replace with your actual schema)
    schema = StructType([
        StructField("name", StringType()),
        StructField("age", IntegerType()),
        StructField("address", StructType([
            StructField("street", StringType()),
            StructField("city", StringType())
        ])),
        StructField("items", ArrayType(StructType([
            StructField("id", IntegerType()),
            StructField("value", StringType())
        ]))),
        StructField("details", MapType(StringType(), StringType())) # Example of Map type
    ])

    # Example of schema if the root element of your JSON is an array:
    # schema = ArrayType(StructType([
    #     StructField("name", StringType()),
    #     StructField("age", IntegerType()),
    #     # ... other fields
    # ]))


    transformed_df = process_nested_json(spark, json_file_path, schema)

    transformed_df.printSchema()
    transformed_df.show(truncate=False)

    spark.stop()