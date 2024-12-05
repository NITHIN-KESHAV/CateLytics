# Databricks notebook source
# MAGIC %restart_python 

# COMMAND ----------

# MAGIC %md
# MAGIC pcsk_6jhjsF_HAh7SofvpDDnzk67gJRaV6e6wxHPDY4WiZypGmfe5ujvnNdAb9v871yyUoTPDL8
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##generate embeddings

# COMMAND ----------

# MAGIC %pip install sentence_transformers

# COMMAND ----------

import boto3
import pandas as pd
import pyarrow.parquet as pq
import os
import uuid
from sentence_transformers import SentenceTransformer

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# AWS S3 setup
aws_access_key_id = "AKIAVYV52E7ZRAA4ISFA"  # Replace with your AWS access key
aws_secret_access_key = "ZUJ4ABufRGFUhgBJFxyjRwE+G5yHfY2UjeKzGjk5"  # Replace with your AWS secret key
aws_region = "us-west-1"  # Replace with your region
bucket_name = "raw-zip-final"  # Replace with your bucket name
s3_input_folder = "Parquet/Final_Data/"  # Folder containing input Parquet files
s3_output_folder = "Parquet/Embeddings/"  # Folder to save output files

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)

# Create a local temp folder for processing files
local_temp_folder = "temp_folder"
os.makedirs(local_temp_folder, exist_ok=True)

# List all Parquet files in the S3 input folder
response = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_input_folder)
parquet_files = [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".parquet")]

if not parquet_files:
    print("No Parquet files found in the specified S3 folder.")
    exit()

print(f"Found {len(parquet_files)} Parquet files in S3 folder: {s3_input_folder}")

# Process each Parquet file
for file_key in parquet_files:
    print(f"Processing file: {file_key}")

    # Download file from S3
    local_file_path = os.path.join(local_temp_folder, os.path.basename(file_key))
    s3.download_file(Bucket=bucket_name, Key=file_key, Filename=local_file_path)

    # Read Parquet file into a DataFrame
    table = pq.read_table(local_file_path)
    df = table.to_pandas()

    # Generate UUID if not present
    if "UUID" not in df.columns:
        df["UUID"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Generate embeddings for `reviewText_cleaned` and `summary_cleaned`
    print("Generating embeddings...")
    df["reviewText_embedding"] = embedding_model.encode(df["reviewText_cleaned"].fillna("").tolist()).tolist()
    df["summary_embedding"] = embedding_model.encode(df["summary_cleaned"].fillna("").tolist()).tolist()

    # Save the updated DataFrame back to Parquet
    updated_local_file_path = os.path.join(local_temp_folder, f"processed_{os.path.basename(file_key)}")
    table_with_embeddings = pq.Table.from_pandas(df)
    pq.write_table(table_with_embeddings, updated_local_file_path)

    # Upload the updated file back to S3
    output_s3_key = f"{s3_output_folder}{os.path.basename(file_key)}"
    s3.upload_file(updated_local_file_path, bucket_name, output_s3_key)

    print(f"Processed file saved to S3: {output_s3_key}")

# Clean up local temp folder
for file in os.listdir(local_temp_folder):
    os.remove(os.path.join(local_temp_folder, file))
os.rmdir(local_temp_folder)

print("All files processed and uploaded to S3 successfully!")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Removing specific columns

# COMMAND ----------

from pyspark.sql import SparkSession

# Initializing the Spark session
spark = SparkSession.builder.appName("to check schema").getOrCreate()

# Defining the input Parquet folder path
parquet_folder_path = "s3://raw-zip-final/Parquet/To_Embeddings/"

# Loading the Parquet files
df = spark.read.format("parquet").load(parquet_folder_path)

# COMMAND ----------

columns_to_drop = ["style", "reviewText","summary"]  # Replace with the columns you want to drop
df_cleaned = df.drop(*columns_to_drop)

# COMMAND ----------

from pyspark.sql.functions import count

# Group by 'asin' and count occurrences
duplicates_df = df.groupBy("asin").agg(count("asin").alias("count")).filter("count > 1")

# Show duplicate ASINs
duplicates_df.show()

# Count the total number of duplicate ASINs
total_duplicates = duplicates_df.count()
print(f"Total duplicate ASINs: {total_duplicates}")


# COMMAND ----------

from pyspark.sql.functions import col, count

# Group by the combination of 'asin' and 'unixReviewTime', and count occurrences
duplicates_df = (
    df_cleaned.groupBy("asin", "unixReviewTime")
    .agg(count("*").alias("count"))
    .filter(col("count") > 1)  # Filter for rows where count > 1
)

# Show duplicates
duplicates_df.show(truncate=False)

# Optional: Count the total number of duplicate groups
total_duplicates = duplicates_df.count()
print(f"Total duplicate groups based on asin+unixReviewTime: {total_duplicates}")


# COMMAND ----------

# from pyspark.sql.window import Window
# from pyspark.sql.functions import row_number

# # Define a window partitioned by 'asin' and 'unixReviewTime'
# window_spec = Window.partitionBy("asin", "unixReviewTime").orderBy("unixReviewTime")

# # Add a row_number column to identify duplicates
# df_with_row_number = df.withColumn("row_number", row_number().over(window_spec))

# # Filter to keep only the first occurrence of each duplicate group
# df_no_duplicates = df_with_row_number.filter(col("row_number") == 1).drop("row_number")

# # Verify the result by checking if duplicates are removed
# df_no_duplicates.show()

# Optional: Save the cleaned DataFrame back to S3 or another location
output_path = "s3://raw-zip-final/Parquet/To_Embeddings/"
df_cleaned.write.mode("overwrite").parquet(output_path)

print(f"Cleaned data written to {output_path}")


# COMMAND ----------

# # Define the specific ASIN to check
# specific_asin = "B000WIX6YC"  # Replace with your ASIN

# # Filter the DataFrame for the specific ASIN
# asin_data = df.filter(df["asin"] == specific_asin)

# # Show the data for the specific ASIN
# asin_data.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## generate mebeddings

# COMMAND ----------

# MAGIC %pip install sentence_transformers

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, lit
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the Spark session
spark = SparkSession.builder.appName("Generate Embeddings").getOrCreate()

# Load the Parquet files
parquet_folder_path = "s3://raw-zip-final/Parquet/To_Embeddings/"
df = spark.read.format("parquet").load(parquet_folder_path)

# Step 1: Limit the dataset to 1 million rows
df_limited = df.limit(1000000)

# Step 2: Combine 'reviewText' and 'summary' columns
df_combined = df_limited.withColumn("combined_text", concat_ws(" ", df_limited.reviewText_cleaned, df_limited.summary_cleaned))

# Step 3: Convert Spark DataFrame to Pandas for embedding generation
df_pandas = df_combined.select("combined_text").toPandas()

# Step 4: Generate embeddings using SentenceTransformer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(df_pandas["combined_text"].tolist(), batch_size=64)

# Step 5: Add embeddings back to the DataFrame
df_pandas["embeddings"] = list(embeddings)

# Convert back to Spark DataFrame
df_with_embeddings = spark.createDataFrame(df_pandas)

# # Step 6: Save the result to S3 as Parquet
# output_path = "s3://raw-zip-final/Parquet/Embeddings_Output/"
# df_with_embeddings.write.mode("overwrite").parquet(output_path)

# print(f"Embeddings generated and saved to {output_path}")


# COMMAND ----------

df_with_embeddings.printSchema()

# COMMAND ----------

import pandas as pd
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import torch

# Initialize Spark session with optimized memory settings
spark = SparkSession.builder \
    .appName("Generate Embeddings") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "1000") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
    .getOrCreate()

# Initialize model with CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Define the UDF with larger batch size
@pandas_udf(ArrayType(FloatType()))
def generate_embeddings(texts: pd.Series) -> pd.Series:
    """Generate embeddings for a series of texts."""
    embeddings = model.encode(
        texts.fillna("").tolist(),
        batch_size=256,  # Increased batch size for better performance
        show_progress_bar=True,
        device=device
    )
    return pd.Series([embed.tolist() for embed in embeddings])

# S3 paths
s3_input_folder = "s3://raw-zip-final/Parquet/To_Embeddings/"

print(f"Reading data from {s3_input_folder}")

# Read 1 million rows
df = spark.read.parquet(s3_input_folder)
df = df.limit(1000000)

# Cache the sampled dataframe
df.cache()
total_rows = df.count()
print(f"Processing {total_rows} rows...")

# Combine text columns
df_combined = df.withColumn(
    "combined_text",
    concat_ws(" ", col("reviewText_cleaned"), col("summary_cleaned"))
)

print("Generating embeddings... This will take some time.")
# Generate embeddings using the UDF
df_with_embeddings = df_combined \
    .withColumn("combined_embedding", generate_embeddings("combined_text"))

# Trigger the computation and show progress
processed_count = df_with_embeddings.count()
print(f"\nProcessed {processed_count} rows successfully!")

# Show a sample of 5 rows to verify
print("\nSample of processed data (5 rows):")
df_with_embeddings.select("combined_text", "combined_embedding").show(5, truncate=True)

# Print statistics about the embeddings
print("\nEmbedding statistics:")
first_row = df_with_embeddings.select("combined_embedding").first()
if first_row:
    print(f"Embedding vector length: {len(first_row[0])}")
print(f"Total rows processed: {processed_count}")

# COMMAND ----------

df_with_embeddings = df_with_embeddings.drop(columns=['combined_text'])


# COMMAND ----------

import weaviate

# Connect to the Weaviate instance
client = weaviate.Client(
    url="http://50.18.99.196:8080",  
)

# Define the schema
schema = {
    "classes": [
        {
            "class": "Review",
            "description": "Schema for storing Amazon review data with metadata and embeddings",
            "vectorIndexConfig": {
                "distance": "cosine",
                "ef": 100,
                "efConstruction": 128,
                "maxConnections": 64
            },
            "properties": [
                {"name": "asin", "dataType": ["string"]},
                {"name": "overall", "dataType": ["number"]},
                {"name": "reviewerID", "dataType": ["string"]},
                {"name": "reviewerName", "dataType": ["string"]},
                {"name": "unixReviewTime", "dataType": ["int"]},
                {"name": "verified", "dataType": ["boolean"]},
                {"name": "vote", "dataType": ["string"]},
                {"name": "style_size", "dataType": ["string"]},
                {"name": "style_flavor", "dataType": ["string"]},
                {"name": "style_style", "dataType": ["string"]},
                {"name": "style_color", "dataType": ["string"]},
                {"name": "style_edition", "dataType": ["string"]},
                {"name": "style_package_type", "dataType": ["string"]},
                {"name": "style_number_of_items", "dataType": ["string"]},
                {"name": "style_size_per_pearl", "dataType": ["string"]},
                {"name": "style_color_name", "dataType": ["string"]},
                {"name": "style_size_name", "dataType": ["string"]},
                {"name": "style_metal_type", "dataType": ["string"]},
                {"name": "style_style_name", "dataType": ["string"]},
                {"name": "style_flavor_name", "dataType": ["string"]},
                {"name": "style_length", "dataType": ["string"]},
                {"name": "style_package_quantity", "dataType": ["string"]},
                {"name": "style_design", "dataType": ["string"]},
                {"name": "style_platform", "dataType": ["string"]},
                {"name": "style_item_package_quantity", "dataType": ["string"]},
                {"name": "style_wattage", "dataType": ["string"]},
                {"name": "style_pattern", "dataType": ["string"]},
                {"name": "style_material_type", "dataType": ["string"]},
                {"name": "style_team_name", "dataType": ["string"]},
                {"name": "style_shape", "dataType": ["string"]},
                {"name": "style_hand_orientation", "dataType": ["string"]},
                {"name": "style_flex", "dataType": ["string"]},
                {"name": "style_material", "dataType": ["string"]},
                {"name": "style_shaft_material", "dataType": ["string"]},
                {"name": "style_configuration", "dataType": ["string"]},
                {"name": "style_capacity", "dataType": ["string"]},
                {"name": "style_product_packaging", "dataType": ["string"]},
                {"name": "style_offer_type", "dataType": ["string"]},
                {"name": "style_model", "dataType": ["string"]},
                {"name": "style_overall_length", "dataType": ["string"]},
                {"name": "style_width", "dataType": ["string"]},
                {"name": "style_grip_type", "dataType": ["string"]},
                {"name": "style_gem_type", "dataType": ["string"]},
                {"name": "style_scent_name", "dataType": ["string"]},
                {"name": "style_model_number", "dataType": ["string"]},
                {"name": "style_item_shape", "dataType": ["string"]},
                {"name": "style_connectivity", "dataType": ["string"]},
                {"name": "style_digital_storage_capacity", "dataType": ["string"]},
                {"name": "style_subscription_length", "dataType": ["string"]},
                {"name": "style_primary_stone_gem_type", "dataType": ["string"]},
                {"name": "style_item_display_weight", "dataType": ["string"]},
                {"name": "style_gift_amount", "dataType": ["string"]},
                {"name": "style_format", "dataType": ["string"]},
                {"name": "reviewText_cleaned", "dataType": ["text"]},
                {"name": "summary_cleaned", "dataType": ["text"]},
                {"name": "UUID", "dataType": ["string"]},
                {"name": "combined_embedding", "dataType": ["number[]"]} 
            ],
        }
    ]
}

# Delete existing schema if it exists
try:
    client.schema.delete_class("Review")
    print("Existing schema deleted.")
except:
    print("No existing schema to delete.")

# Create the schema in Weaviate
client.schema.create(schema)
print("Schema created successfully.")

