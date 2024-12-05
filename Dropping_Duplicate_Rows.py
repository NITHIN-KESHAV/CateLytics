# Databricks notebook source
# MAGIC %md
# MAGIC **Checking for the number of duplicate rows based on composite fields {reviewerID, asin, unixReviewTime}**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("DisplayDuplicatesAsTable") \
    .getOrCreate()

# S3 path for input Parquet files
input_parquet_path = "s3://raw-zip-final/Parquet/flattened_transformed_files/"

# Load Parquet file into a DataFrame
df = spark.read.parquet(input_parquet_path)

# Group by the specified columns and count occurrences
df_duplicates = df.groupBy("reviewerID", "asin", "unixReviewTime") \
    .agg(count("*").alias("count")) \
    .filter(col("count") > 1)

# Showing duplicates in a table format
df_duplicates.select("reviewerID", "asin", "unixReviewTime", "count").show(truncate=False)

# Counting the number of duplicated rows
total_duplicated_count = df_duplicates.count()
print(f"Total number of duplicated rows: {total_duplicated_count}")



# COMMAND ----------

# MAGIC %md
# MAGIC **Dropping the duplicated rows**

# COMMAND ----------

# MAGIC %md
# MAGIC The data after dropping all duplicate rows is saved in s3 under the following folder:
# MAGIC
# MAGIC s3://raw-zip-final/Parquet/deduplicated_files/

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql import SparkSession

# Initializing a Spark session
spark = SparkSession.builder \
    .appName("DropDuplicatesAndCheckMissingValues") \
    .getOrCreate()

# S3 paths for input and output Parquet files
input_parquet_path = "s3://raw-zip-final/Parquet/flattened_transformed_files/"
output_parquet_path = "s3://raw-zip-final/Parquet/deduplicated_files2/"

# Load Parquet file into a DataFrame
df = spark.read.parquet(input_parquet_path)

# Checking for missing values in each column
missing_reviewerID = df.filter(col("reviewerID").isNull() | (col("reviewerID") == "")).count()
missing_asin = df.filter(col("asin").isNull() | (col("asin") == "")).count()
missing_unixReviewTime = df.filter(col("unixReviewTime").isNull() | (col("unixReviewTime") == "")).count()

print(f"Missing values in 'reviewerID': {missing_reviewerID}")
print(f"Missing values in 'asin': {missing_asin}")
print(f"Missing values in 'unixReviewTime': {missing_unixReviewTime}")

# Counting duplicates based on specified columns
duplicates_count = df.groupBy("reviewerID", "asin", "unixReviewTime") \
    .count() \
    .filter("count > 1") \
    .count()

print(f"Number of duplicate rows before deduplication: {duplicates_count}")

# Dropping duplicates based on specified columns
df_deduplicated = df.dropDuplicates(["reviewerID", "asin", "unixReviewTime"])

# to validate deduplication by counting duplicates in deduplicated DataFrame
deduplicated_duplicates_count = df_deduplicated.groupBy("reviewerID", "asin", "unixReviewTime") \
    .count() \
    .filter("count > 1") \
    .count()

print(f"Number of duplicate rows after deduplication: {deduplicated_duplicates_count}")

# Saving deduplicated DataFrame to S3 in Parquet format
df_deduplicated.write.mode("overwrite").parquet(output_parquet_path)
print(f"Deduplicated data saved to: {output_parquet_path}")


