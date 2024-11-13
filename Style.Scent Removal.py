# Databricks notebook source
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder.appName("DropNestedKey").getOrCreate()

# Define the file input path (S3 path)
input_file_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

# Read the JSON file as a DataFrame with each row as a JSON string without schema inference
raw_df = spark.read.text(input_file_path)

# Function to drop 'Scent' and 'SCENT' from the 'style' object in the JSON string
def drop_nested_key(json_string):
    try:
        data = json.loads(json_string)
        
        # Check if 'style' is a dictionary and remove 'Scent'  if they exist
        if "style" in data and isinstance(data["style"], dict):
            data["style"].pop("Scent", None)  # Remove 'Scent' key if it exists
        
        return json.dumps(data)
    except json.JSONDecodeError:
        return json_string

# Register the UDF to drop 'style.Scent' 
drop_nested_key_udf = udf(drop_nested_key, StringType())

# Apply the UDF to transform the raw JSON data
cleaned_df = raw_df.withColumn("value", drop_nested_key_udf(col("value")))

# Show the cleaned DataFrame
cleaned_df.show(truncate=False)


# COMMAND ----------

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StringType

# Initialize Spark session
spark = SparkSession.builder.appName("Extracting Ordered Nested Keys").getOrCreate()

# Define a recursive function to extract all keys with full nested paths
def extract_keys_from_json(json_string):
    try:
        data = json.loads(json_string)
        return list(_extract_keys(data))
    except json.JSONDecodeError:
        return []

def _extract_keys(data, prefix=""):
    keys = set()
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.add(full_key)
            keys.update(_extract_keys(value, prefix=full_key))  # Recurse into nested structures
    elif isinstance(data, list):
        for item in data:
            keys.update(_extract_keys(item, prefix=prefix))  # Recurse into list items
    return keys

# Register the UDF
extract_keys_udf = udf(extract_keys_from_json, ArrayType(StringType()))

# Apply the UDF to extract keys from 'cleaned_df'
keys_df = cleaned_df.withColumn("keys", extract_keys_udf(col("value")))

# Explode the list of keys to get one key per row, then get distinct keys
unique_keys_df = keys_df.select(explode(col("keys")).alias("key")).distinct()

# Collect all unique keys to the driver and sort them
unique_keys = sorted([row["key"] for row in unique_keys_df.collect()])

# Function to display keys in a nested, ordered format
def display_nested_keys(keys):
    current_indent = ""
    for key in keys:
        parts = key.split(".")
        for i in range(len(parts)):
            prefix = ".".join(parts[:i])
            if prefix != current_indent:
                current_indent = prefix
                print("  " * i + parts[i])
            elif i == len(parts) - 1:  
                print("  " * i + parts[i])

# Print all unique keys in a structured, ordered format
print("All unique keys in a structured, ordered format:")
display_nested_keys(unique_keys)

