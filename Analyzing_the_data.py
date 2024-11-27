# Databricks notebook source
#Counting the number of rows in the given parquet file

from pyspark.sql import SparkSession

# Create or get a Spark Session
spark = SparkSession.builder \
    .appName("Read Parquet File from S3") \
    .getOrCreate()




# COMMAND ----------

file_path = "s3a://raw-zip/Parquet/cleaned_parquet_files/part-00000-tid-5056588444897740595-f5662673-05b5-4c7e-a38e-57ac76a4f48b-3196-1.c000.snappy.parquet"
# Read the Parquet file
df = spark.read.parquet(file_path)

# Count the total number of rows
total_rows = df.count()
print(f"Total number of rows in the file: {total_rows}")
