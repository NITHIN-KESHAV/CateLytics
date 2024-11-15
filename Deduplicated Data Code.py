# Databricks notebook source
from pyspark.sql import SparkSession

# Initialize a Spark session with a different name
spark = SparkSession.builder \
    .appName("DeduplicateReviewData") \
    .getOrCreate()

# S3 path for input Parquet files
input_parquet_path = "s3a://raw-zip/Parquet/cleaned_parquet_files/"

# Load Parquet file into a DataFrame
df = spark.read.parquet(input_parquet_path)

# Drop duplicates based on the combination of columns: reviewerID, asin, unixReviewTime
df_deduplicated = df.dropDuplicates(["reviewerID", "asin", "unixReviewTime"])

# S3 path for output Parquet files
output_parquet_path = "s3a://raw-zip/Parquet/deduplicated_parquet_files/"

# Save the deduplicated DataFrame back to Parquet
df_deduplicated.write.mode("overwrite").parquet(output_parquet_path)

# Stop the Spark session
spark.stop()


# COMMAND ----------

from pyspark.sql.functions import col

# Define the columns to check for duplicates
duplicate_columns = ["reviewerID", "asin", "unixReviewTime"]

# Group by the specified columns and count occurrences
duplicates = df.groupBy(duplicate_columns).count().filter(col("count") > 1)

# Show duplicate rows and their counts
duplicates.show(truncate=False)

# Count the number of duplicate rows
duplicate_count = duplicates.count()
print(f"Number of duplicate rows: {duplicate_count}")

