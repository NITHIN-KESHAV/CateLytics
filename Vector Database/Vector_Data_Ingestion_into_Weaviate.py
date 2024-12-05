# Databricks notebook source
# MAGIC %pip install sentence_transformers

# COMMAND ----------

!pip install --upgrade protobuf
!pip install weaviate-client sentence-transformers tqdm

# COMMAND ----------

# MAGIC %pip install protobuf==3.20.0
# MAGIC %pip install weaviate-client

# COMMAND ----------

# MAGIC %restart_python 

# COMMAND ----------

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


# COMMAND ----------

# MAGIC %md
# MAGIC **Embeddings Generation**

# COMMAND ----------

# MAGIC %pip install sentence_transformers

# COMMAND ----------

# MAGIC %md
# MAGIC Removing specific columns

# COMMAND ----------

from pyspark.sql import SparkSession

# Initializing the Spark session
spark = SparkSession.builder.appName("to check schema").getOrCreate()

# Defining the input Parquet folder path
parquet_folder_path = "s3://raw-zip-final/Parquet/To_Embeddings/"

# Loading the Parquet files
df = spark.read.format("parquet").load(parquet_folder_path)

# COMMAND ----------

from pyspark.sql.functions import count

# Grouping by 'asin' and count occurrences
duplicates_df = df.groupBy("asin").agg(count("asin").alias("count")).filter("count > 1")

# Showing the duplicate ASINs
duplicates_df.show()

# Checking total number of duplicate ASINs
total_duplicates = duplicates_df.count()
print(f"Total duplicate ASINs: {total_duplicates}")


# COMMAND ----------

from pyspark.sql.functions import col, count

# Grouping by the combination of 'asin' and 'unixReviewTime', and count occurrences
duplicates_df = (
    df.groupBy("asin", "unixReviewTime")
    .agg(count("*").alias("count"))
    .filter(col("count") > 1)  
)

# Showing duplicates
duplicates_df.show(truncate=False)

total_duplicates = duplicates_df.count()
print(f"Total duplicate groups based on asin+unixReviewTime: {total_duplicates}")


# COMMAND ----------

import pandas as pd
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, pandas_udf
from pyspark.sql.types import ArrayType, FloatType
import torch

# Initializing the Spark session with optimized memory settings
spark = SparkSession.builder \
    .appName("Generate Embeddings") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.sql.shuffle.partitions", "1000") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000") \
    .getOrCreate()

# Initializing the model with CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

# Defining the UDF with larger batch size
@pandas_udf(ArrayType(FloatType()))
def generate_embeddings(texts: pd.Series) -> pd.Series:
    """Generate embeddings for a series of texts."""
    embeddings = model.encode(
        texts.fillna("").tolist(),
        batch_size=256,  
        show_progress_bar=True,
        device=device
    )
    return pd.Series([embed.tolist() for embed in embeddings])

s3_input_folder = "s3://raw-zip-final/Parquet/To_Embeddings/"

print(f"Reading data from {s3_input_folder}")

# Reading 1 million rows
df = spark.read.parquet(s3_input_folder)
df = df.limit(1000000)

# Caching the sampled dataframe
df.cache()
total_rows = df.count()
print(f"Processing {total_rows} rows...")

# Combining the text columns
df_combined = df.withColumn(
    "combined_text",
    concat_ws(" ", col("reviewText_cleaned"), col("summary_cleaned"))
)

print("Generating embeddings... This will take some time.")
# Generating the embeddings using the UDF
df_with_embeddings = df_combined \
    .withColumn("combined_embedding", generate_embeddings("combined_text"))

# Triggering the computation and show progress
processed_count = df_with_embeddings.count()
print(f"\nProcessed {processed_count} rows successfully!")

# Showing a sample of 5 rows to verify
print("\nSample of processed data (5 rows):")
df_with_embeddings.select("combined_text", "combined_embedding").show(5, truncate=True)

# Printing some statistics about the embeddings
print("\nEmbedding statistics:")
first_row = df_with_embeddings.select("combined_embedding").first()
if first_row:
    print(f"Embedding vector length: {len(first_row[0])}")
print(f"Total rows processed: {processed_count}")

# COMMAND ----------

df_with_embeddings = df_with_embeddings.drop(columns=['combined_text'])


# COMMAND ----------

import weaviate

# Connecting to the Weaviate instance
client = weaviate.Client(
    url="http://50.18.99.196:8080",  
)

# Defining the schema
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

# Deleting the existing schema if it exists
try:
    client.schema.delete_class("Review")
    print("Existing schema deleted.")
except:
    print("No existing schema to delete.")

# Creating the schema in Weaviate
client.schema.create(schema)
print("Schema created successfully.")


# COMMAND ----------

# Cjecking the classes in Weavaiate
classes = client.schema.get()
print("Available classes:", [c['class'] for c in classes['classes']])

# COMMAND ----------

# MAGIC %md
# MAGIC Data Ingestion into Weaviate

# COMMAND ----------

import time
import pandas as pd
from tqdm import tqdm

def process_data_sequentially(df_with_embeddings, client, batch_size=1000):
    """
    Process and ingest data into Weaviate in sequential batches
    """
    print("Starting sequential processing...")
    total_rows = df_with_embeddings.count()
    print(f"Total rows to process: {total_rows}")
    
    processed = 0
    start_time = time.time()
    
    try:
        while processed < total_rows:
            # Getting the batch of data
            batch_df = (df_with_embeddings
                       .offset(processed)
                       .limit(batch_size)
                       .toPandas())
            
            with client.batch(batch_size=100) as batch:
                for idx, row in batch_df.iterrows():
                    # Extracting the embedding vector
                    vector = row['combined_embedding']
                    
                    # Creating the properties dict with proper type conversion
                    properties = {}
                    for col in batch_df.columns:
                        if col != 'combined_embedding' and pd.notna(row[col]):  
                            if col == 'unixReviewTime':
                                properties[col] = int(row[col])
                            elif col == 'overall':
                                properties[col] = float(row[col])
                            elif col == 'verified':
                                properties[col] = bool(row[col])
                            else:
                                properties[col] = str(row[col])
                    
                    # Adding the object to batch
                    batch.add_data_object(
                        data_object=properties,
                        class_name="Review",
                        vector=vector  
                    )
            
            # Updating the progress
            processed += len(batch_df)
            elapsed = time.time() - start_time
            rate = processed / elapsed
            
            # Progress update
            print(f"\nProgress Update:")
            print(f"- Processed: {processed:,}/{total_rows:,} ({processed/total_rows*100:.4f}%)")
            print(f"- Time elapsed: {elapsed/3600:.2f} hours")
            print(f"- Rate: {rate:.2f} rows/second")
            print(f"- Estimated remaining: {(total_rows-processed)/rate/3600:.2f} hours")
            
            # Small delay to prevent overwhelming the system
            time.sleep(2)
            
    except Exception as e:
        print(f"Error occurred at row {processed}: {str(e)}")
        print("Last successful batch processed up to row:", processed - batch_size)
        raise
    
    print("\nProcessing completed!")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Final row count: {processed:,}")

if __name__ == "__main__":    
    process_data_sequentially(
        df_with_embeddings=df_with_embeddings,
        client=client,
        batch_size=1000
    )
