# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import re

# Initializing the Spark session
spark = SparkSession.builder.appName("Check Null Columns by Pattern").getOrCreate()

# Defining the S3 folder path containing the Cleaned Parquet files after dropping the "images" and "reviewTime" key-values
parquet_folder_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Loading all the Parquet files from the folder
df = spark.read.format("parquet").load(parquet_folder_path)

# Defining the regex pattern for column names
pattern = r"^style_dup\d+$"  

# Filtering columns based on the pattern
matching_columns = [col_name for col_name in df.columns if re.match(pattern, col_name)]

# Function to check if columns are entirely null
def check_null_columns(dataframe, column_list):
    null_status = {}
    for column in column_list:
        # Counting the number of non-null values for each column
        non_null_count = dataframe.filter(col(column).isNotNull()).count()
        null_status[column] = non_null_count == 0  # True if column is entirely null
    return null_status

# Checking null status for all matching columns
null_status = check_null_columns(df, matching_columns)

# Printing the results
print("Null status of matching columns:")
for column, status in null_status.items():
    print(f"Column {column}: {'All Null' if status else 'Contains Non-Null Values'}")
