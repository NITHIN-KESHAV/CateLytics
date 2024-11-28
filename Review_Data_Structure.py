# Databricks notebook source
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StringType

# Initializing the Spark session
spark = SparkSession.builder.appName("Extract Ordered Nested Keys").getOrCreate()

# Defining the input path
input_file_path = "s3://raw-zip-final/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

# Reading the JSON file as a DataFrame with each row as a JSON string
raw_df = spark.read.text(input_file_path)

# Defining a recursive function to extract all keys with full nested paths
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

# Registering the UDF
extract_keys_udf = udf(extract_keys_from_json, ArrayType(StringType()))

# Applying the UDF to extract keys
keys_df = raw_df.withColumn("keys", extract_keys_udf(col("value")))

# Exploding the list of keys to get one key per row, then get distinct keys
unique_keys_df = keys_df.select(explode(col("keys")).alias("key")).distinct()

# Collecting all unique keys to the driver and sort them
unique_keys = sorted([row["key"] for row in unique_keys_df.collect()])

# Function to display the keys in a nested, ordered format
def display_nested_keys(keys):
    current_indent = ""
    for key in keys:
        # Splitting the key by "." to check the nested levels
        parts = key.split(".")
        for i in range(len(parts)):
            # Determining if we need to change the indentation
            prefix = ".".join(parts[:i])
            if prefix != current_indent:
                current_indent = prefix
                print("  " * i + parts[i])
            elif i == len(parts) - 1:  
                print("  " * i + parts[i])

print("All unique keys in a structured, ordered format:")
display_nested_keys(unique_keys)

# COMMAND ----------


