# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import re

# Initialize Spark session
spark = SparkSession.builder.appName("Check Null Columns by Pattern").getOrCreate()

# Define the S3 folder path containing Parquet files
parquet_folder_path = "s3a://raw-zip/Parquet/cleaned_parquet_files/"

# Load all Parquet files from the folder
df = spark.read.format("parquet").load(parquet_folder_path)

# Define the regex pattern for column names
pattern = r"^style_dup\d+$"  # Matches column names like style_dup1, style_dup2, etc.

# Filter columns based on the pattern
matching_columns = [col_name for col_name in df.columns if re.match(pattern, col_name)]

# Function to check if columns are entirely null
def check_null_columns(dataframe, column_list):
    null_status = {}
    for column in column_list:
        # Count non-null values for each column
        non_null_count = dataframe.filter(col(column).isNotNull()).count()
        null_status[column] = non_null_count == 0  # True if column is entirely null
    return null_status

# Check null status for all matching columns
null_status = check_null_columns(df, matching_columns)

# Print the results
print("Null status of matching columns:")
for column, status in null_status.items():
    print(f"Column {column}: {'All Null' if status else 'Contains Non-Null Values'}")
