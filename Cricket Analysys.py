# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/1405316-1.json"
file_type = "json"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "1405316-1_json"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `1405316-1_json`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "1405316-1_json"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# Load the JSON file
file_path = "/FileStore/tables/1405316-1.json"
data = spark.read.option("multiline", "true").json(file_path)

# COMMAND ----------

# 1. Extract Metadata Table
metadata_df = data.selectExpr(
    "meta.data_version AS data_version",
    "meta.created AS created",
    "meta.revision AS revision"
)
metadata_df.write.format("delta").saveAsTable("metadata_table")

# COMMAND ----------

# 2. Extract Match Information Table
match_info_df = data.selectExpr(
    "info.city AS city",
    "info.dates[0] AS match_date",
    "info.event.name AS event_name",
    "info.event.match_number AS match_number",
    "info.event.group AS group",
    "info.gender AS gender",
    "info.match_type AS match_type",
    "info.match_type_number AS match_type_number",
    "info.season AS season",
    "info.team_type AS team_type",
    "info.venue AS venue",
    "info.toss.winner AS toss_winner",
    "info.toss.decision AS toss_decision",
    "info.outcome.winner AS winner",
    "info.outcome.by.runs AS win_by_runs"
)
match_info_df.write.format("delta").saveAsTable("match_information_table")


# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Extract Officials Table
# MAGIC officials_df = data.selectExpr(
# MAGIC     "explode(info.officials.match_referees) AS match_referee",
# MAGIC     "explode(info.officials.reserve_umpires) AS reserve_umpire",
# MAGIC     "explode(info.officials.umpires) AS umpire"
# MAGIC )
# MAGIC officials_df.write.format("delta").saveAsTable("officials_table")

# COMMAND ----------

# 3. Extract Officials Table
# Explode match_referees
match_referees_df = data.selectExpr("explode(info.officials.match_referees) AS match_referee")

# Explode reserve_umpires
reserve_umpires_df = data.selectExpr("explode(info.officials.reserve_umpires) AS reserve_umpire")

# Explode umpires
umpires_df = data.selectExpr("explode(info.officials.umpires) AS umpire")

# Combine into a single DataFrame (union can be adjusted based on requirements)
from pyspark.sql.functions import lit
officials_df = (
    match_referees_df.withColumn("role", lit("match_referee"))
    .union(reserve_umpires_df.withColumn("role", lit("reserve_umpire")))
    .union(umpires_df.withColumn("role", lit("umpire")))
)

# Save as a Delta table
officials_df.write.format("delta").saveAsTable("officials_table")

# COMMAND ----------

# 4. Extract Players Table
players_df = data.selectExpr(
    "explode(info.players.Hong_Kong) AS player_name",
    "lit('Hong Kong') AS team_name"
).union(
    data.selectExpr(
        "explode(info.players.Kuwait) AS player_name",
        "lit('Kuwait') AS team_name"
    )
)
players_df.write.format("delta").saveAsTable("players_table")


# COMMAND ----------

from pyspark.sql.functions import lit, explode

# Extract players from the "Hong Kong" team
hong_kong_players_df = data.select(
    explode(data["info"]["players"]["Hong Kong"]).alias("player_name")
).withColumn("team_name", lit("Hong Kong"))

# Extract players from the "Kuwait" team
kuwait_players_df = data.select(
    explode(data["info"]["players"]["Kuwait"]).alias("player_name")
).withColumn("team_name", lit("Kuwait"))

# Combine both DataFrames
players_df = hong_kong_players_df.union(kuwait_players_df)

# Write the combined DataFrame to a Delta table
players_df.write.format("delta").saveAsTable("players_table")


# COMMAND ----------

from pyspark.sql.functions import col, explode, lit, create_map

# Dynamically extract the field names from the "people" struct
field_names = [field.name for field in data.select("info.registry.people").schema.fields[0].dataType.fields]

# Create a map of player_name and player_id dynamically
registry_map = create_map(
    *[lit(field).alias("key") for field in field_names for lit_key]
)

# Explode the map into key-value pairs
registry_df = (
    registry.querySelector()
)
# convert this