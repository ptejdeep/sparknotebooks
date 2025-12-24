# Databricks notebook source
 from pyspark.sql.types import StructType, IntegerType, StringType, DoubleType

# COMMAND ----------

schema = StructType() \
 .add("ProductID", IntegerType(), True) \
 .add("ProductName", StringType(), True) \
 .add("Category", StringType(), True) \
 .add("ListPrice", DoubleType(), True)


# COMMAND ----------



 df = spark.read.format("csv").option("header","true").schema(schema).load("/data")
 # df now is a Spark DataFrame containing CSV data from "Files/products/products.csv".
 display(df)