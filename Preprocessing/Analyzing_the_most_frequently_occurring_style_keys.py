# Databricks notebook source
# MAGIC %pip install fsspec

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import fsspec
print("fsspec installed successfully!")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Initializing the Spark session
spark = SparkSession.builder \
    .appName("StyleColumnNonNullCounts") \
    .getOrCreate()

# Defining the path to the S3 bucket containing the deduplicated Parquet files
parquet_path = "s3://raw-zip-final/Parquet/deduplicated_files/"

# Loading the Parquet data
df = spark.read.parquet(parquet_path)

# Identifying the columns related to "style"
style_columns = [col_name for col_name in df.columns if col_name.startswith("style_")]

# Counting the non-null values for each style column
non_null_counts = (
    df.select(*[count(col(c)).alias(c) for c in style_columns])
    .toPandas()  
    .transpose()
    .reset_index()
)

# Renaming the columns for better undersatnding
non_null_counts.columns = ["style_column", "non_null_count"]

# Sorting the results by non-null counts in descending order
non_null_counts = non_null_counts.sort_values(by="non_null_count", ascending=False)

print("Style columns sorted by non-null counts:")
print(non_null_counts.to_string(index=False))

# COMMAND ----------

# Initializing the Spark session
spark = SparkSession.builder \
    .appName("FilterLowPopulationStyleColumns") \
    .getOrCreate()

# Defining the path to the S3 bucket containing the Parquet files
input_parquet_path = "s3://raw-zip-final/Parquet/deduplicated_files/"
output_parquet_path = "s3://raw-zip-final/Parquet/filtered_parquet_files/"

# Loading the Parquet data
df = spark.read.parquet(input_parquet_path)

# Identifying columns related to "style"
style_columns = [col_name for col_name in df.columns if col_name.startswith("style_")]

# Counting non-null values for each style column
non_null_counts = (
    df.select(*[count(col(c)).alias(c) for c in style_columns])
    .collect()[0]
)

# Filtering out columns with fewer than 10,000 non-null values
filtered_style_columns = [col_name for col_name in style_columns if non_null_counts[col_name] >= 10000]

# Creating a new DataFrame with only the filtered style columns and the original non-style columns
non_style_columns = [col_name for col_name in df.columns if not col_name.startswith("style_")]
filtered_columns = non_style_columns + filtered_style_columns

filtered_df = df.select(*filtered_columns)

# Writing the filtered DataFrame back to S3 in Parquet format
filtered_df.write.mode("overwrite").parquet(output_parquet_path)

print(f"Filtered dataset saved to: {output_parquet_path}")

# COMMAND ----------

filtered_data_path = "s3://raw-zip-final/Parquet/filtered_parquet_files/"
filtered_df = spark.read.parquet(filtered_data_path)

# Displaying the first few rows
filtered_df.display(10, truncate=False)

# COMMAND ----------

# Printing the schema of the filtered DataFrame
print("Schema of the filtered DataFrame:")
filtered_df.printSchema()
