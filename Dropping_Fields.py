# Databricks notebook source
from pyspark.sql import SparkSession

# Initializing Spark session
spark = SparkSession.builder.appName("Drop Fields from Parquet").getOrCreate()

# Defining the file input path (S3 path)
input_file_path = "s3a://raw-zip/Parquet/processed-data/All_Amazon_Review.parquet/"

# Reading the Parquet files into a DataFrame
raw_df = spark.read.parquet(input_file_path)

# Dropping the fields 'reviewTime' and 'image'
cleaned_df = raw_df.drop("reviewTime", "image")

# to verify the schema after dropping the fields
cleaned_df.printSchema()

# Defining the output path for the cleaned Parquet files
output_parquet_path = "s3a://raw-zip/Parquet//cleaned_parquet_files/"

# Writing the cleaned DataFrame to Parquet format in the S3 bucket
cleaned_df.write.mode("overwrite").parquet(output_parquet_path)

print("Parquet files successfully written to:", output_parquet_path)

