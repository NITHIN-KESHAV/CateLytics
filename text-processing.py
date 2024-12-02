# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Assuming the SparkSession is already initialized as `spark`

# Load your dataset
file_path = "s3://raw-zip/Parquet/deduplicated_parquet_files/"
df = spark.read.parquet(file_path)

# Function to check duplicates in each column
def check_duplicates_per_column(df):
    columns = df.columns
    duplicates = {}

    for column in columns:
        # Group by the column and count duplicates
        duplicate_count = (
            df.groupBy(column)
            .agg(count("*").alias("count"))
            .filter(col("count") > 1)
            .agg(count("*").alias("duplicate_count"))
            .collect()[0][0]
        )
        duplicates[column] = duplicate_count if duplicate_count else 0

    return duplicates

# Check duplicates for all columns
duplicates = check_duplicates_per_column(df)

# Display the results
for column, duplicate_count in duplicates.items():
    print(f"Column '{column}' has {duplicate_count} duplicate values.")


# COMMAND ----------

from pyspark.sql.functions import col, sum as _sum

# Assuming the SparkSession is already initialized and the DataFrame is loaded as `df`

# Check for null values in each column
def check_null_values_per_column(df):
    null_counts = (
        df.select(
            [
                _sum(col(column).isNull().cast("int")).alias(column)
                for column in df.columns
            ]
        )
        .collect()[0]
        .asDict()
    )
    return null_counts

# Check for null values
null_counts = check_null_values_per_column(df)

# Display the results
print("Null values per column:")
for column, null_count in null_counts.items():
    print(f"Column '{column}' has {null_count} null values.")


# COMMAND ----------

display(df)

# COMMAND ----------


