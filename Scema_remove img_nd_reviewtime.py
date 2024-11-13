# Databricks notebook source
import json
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StringType

# Define the input path
input_file_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

# Read JSON file as a DataFrame with each row as a JSON string (no schema inference)
raw_df = spark.read.text(input_file_path)

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
            # Recurse into nested dictionaries or lists of dictionaries
            keys.update(_extract_keys(value, prefix=full_key))
    elif isinstance(data, list):
        for item in data:
            keys.update(_extract_keys(item, prefix=prefix))
    return keys

# Register the UDF
extract_keys_udf = udf(extract_keys_from_json, ArrayType(StringType()))

# Apply the UDF to extract keys
keys_df = raw_df.withColumn("keys", extract_keys_udf(col("value")))

# Explode the list of keys to get one key per row, then get distinct keys
unique_keys_df = keys_df.select(explode(col("keys")).alias("key")).distinct()

# Collect all unique keys to the driver and sort them
unique_keys = sorted([row["key"] for row in unique_keys_df.collect()])

# Function to display the keys in a nested, ordered format
def display_nested_keys(keys):
    current_indent = ""
    for key in keys:
        # Split the key by "." to check the nested levels
        parts = key.split(".")
        for i in range(len(parts)):
            # Determine if we need to change indentation
            prefix = ".".join(parts[:i])
            if prefix != current_indent:
                current_indent = prefix
                print("  " * i + parts[i])
            elif i == len(parts) - 1:  # Print the final part of the key
                print("  " * i + parts[i])

# Print all keys in a structured format
print("All unique keys in a structured, ordered format:")
display_nested_keys(unique_keys)

# COMMAND ----------

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder.appName("Remove Nested Fields").getOrCreate()

# Define the input path
input_file_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

# Read JSON file as a DataFrame with each row as a JSON string (no schema inference)
raw_df = spark.read.text(input_file_path)

# Define a function to remove the 'image' and 'reviewTime' fields from the JSON string
def remove_nested_fields(json_string):
    try:
        data = json.loads(json_string)  # Parse JSON
        # Remove 'image' and 'reviewTime' if they exist at any level
        _remove_keys(data, ["image", "reviewTime"])
        return json.dumps(data)  # Return cleaned JSON as string
    except json.JSONDecodeError:
        return json_string  # If there's an error, return the original string

def _remove_keys(data, keys_to_remove):
    if isinstance(data, dict):
        for key in keys_to_remove:
            data.pop(key, None)  # Remove key if it exists
        for key, value in data.items():
            _remove_keys(value, keys_to_remove)  # Recurse into nested dictionaries
    elif isinstance(data, list):
        for item in data:
            _remove_keys(item, keys_to_remove)  # Recurse into list items

# Register the UDF
remove_fields_udf = udf(remove_nested_fields, StringType())

# Apply the UDF to remove 'image' and 'reviewTime' fields from each JSON row
cleaned_json_df = raw_df.withColumn("value", remove_fields_udf(col("value")))

# Convert the cleaned JSON strings back to a DataFrame by inferring schema
cleaned_df = spark.read.json(cleaned_json_df.rdd.map(lambda row: row["value"]))

# Show the resulting DataFrame to verify the fields have been removed
print("DataFrame after removing nested 'image' and 'reviewTime' fields:")
cleaned_df.show(truncate=False)

# Stop the Spark session
spark.stop()

