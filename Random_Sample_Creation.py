# Databricks notebook source
import random
from pyspark.sql import SparkSession

# Initializing the Spark session
spark = SparkSession.builder.appName("Sample Rows from Multiple Parquet Files").getOrCreate()

# Defining the S3 folder path containing Parquet files
parquet_folder_path = "s3a://raw-zip/Parquet/cleaned_parquet_files/"

# Loading all valid Parquet files and extracting file paths
files_df = spark.read.format("parquet").load(parquet_folder_path).select("_metadata.file_path").distinct()
file_paths = [row["file_path"] for row in files_df.collect()]

# Randomly selecting at least 100 files
selected_files = random.sample(file_paths, min(len(file_paths), 100))

# Initializing an empty DataFrame for combining samples
final_sample_df = None

# Processing each file
for file_path in selected_files:
    # Loading each Parquet file
    df = spark.read.format("parquet").load(file_path)
    
    # Sampling rows from each file
    sample_df = df.sample(withReplacement=False, fraction=0.1, seed=42).limit(10)  # Adjust fraction to ensure sampling
    
    # Combining the sampled rows
    if final_sample_df is None:
        final_sample_df = sample_df
    else:
        final_sample_df = final_sample_df.union(sample_df)

# Limiting the final sample to 1000 rows (if more rows are sampled)
final_sample_df = final_sample_df.limit(1000)

display(final_sample_df)
