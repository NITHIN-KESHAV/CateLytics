# Databricks notebook source
from pyspark.sql import SparkSession
import re

# Initializing the Spark session
spark = SparkSession.builder.appName("Drop Specified Columns from Parquet").getOrCreate()

# Defining the S3 folder path containing Parquet files
input_folder_path = "s3a://raw-zip/Parquet/cleaned_parquet_files/"
output_folder_path = "s3a://raw-zip/Parquet/updated_parquet_files/"

# Loading all the Parquet files from the folder
df = spark.read.format("parquet").load(input_folder_path)

# Defining the regex pattern for column names to drop
pattern = r"^style_dup\d+$"  

# Filtering the columns to drop based on the pattern
columns_to_drop = [col_name for col_name in df.columns if re.match(pattern, col_name)]

# Dropping the matching columns
updated_df = df.drop(*columns_to_drop)

# Writing the updated DataFrame back to the new folder in Parquet format
updated_df.write.mode("overwrite").parquet(output_folder_path)

print(f"Dropped columns: {columns_to_drop}")
print(f"Updated files saved to {output_folder_path}")

# COMMAND ----------

from pyspark.sql import SparkSession

# Initializing the Spark session
spark = SparkSession.builder.appName("Print Schema of Parquet Files").getOrCreate()

# Defining the S3 folder path containing Parquet files
parquet_folder_path = "s3a://raw-zip/Parquet/updated_parquet_files/"

# Loading all Parquet files from the folder
df = spark.read.format("parquet").load(parquet_folder_path)

# Printing the schema of the DataFrame
print("Schema of Parquet Files:")
df.printSchema()
