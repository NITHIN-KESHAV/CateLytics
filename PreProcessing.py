# Databricks notebook source
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StringType

# Initializing Spark session
spark = SparkSession.builder.appName("Extracting Ordered Nested Keys").getOrCreate()

# Defining the file input path (S3 path)
input_file_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

# Reading the JSON file as a DataFrame with each row as a JSON string without schema inference.
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
            # Recursing into nested dictionaries or lists of dictionaries
            keys.update(_extract_keys(value, prefix=full_key))
    elif isinstance(data, list):
        for item in data:
            keys.update(_extract_keys(item, prefix=prefix))
    return keys

# Registering the UDF (User-Defined Function)
extract_keys_udf = udf(extract_keys_from_json, ArrayType(StringType()))

# Applying the UDF to extract keys
keys_df = raw_df.withColumn("keys", extract_keys_udf(col("value")))

# Exploding the list of keys to get one key per row, then get distinct keys
unique_keys_df = keys_df.select(explode(col("keys")).alias("key")).distinct()

# Collecting all unique keys to the driver and sorting them
unique_keys = sorted([row["key"] for row in unique_keys_df.collect()])

# Function to display the keys in a nested, ordered format
def display_nested_keys(keys):
    current_indent = ""
    for key in keys:
        # Splitting the key by "." to check the nested levels
        parts = key.split(".")
        for i in range(len(parts)):
            # Determining if we need to change indentation
            prefix = ".".join(parts[:i])
            if prefix != current_indent:
                current_indent = prefix
                print("  " * i + parts[i])
            elif i == len(parts) - 1:  
                print("  " * i + parts[i])

# Printing all keys in a structured format
print("All unique keys in a structured, ordered format:")
display_nested_keys(unique_keys)

# COMMAND ----------

file_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

raw_df = spark.read.text(file_path)
raw_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC Steps to Infer Schema Automatically
# MAGIC Read the JSON File with Schema Inference Enabled: Spark can infer the schema automatically if you read the JSON file directly with spark.read.json. This will allow Spark to explore the nested structure and generate a proper schema.
# MAGIC
# MAGIC Optimize for Large Files: For large files, such as your 120GB JSON file, it’s often helpful to sample a subset of the data to infer the schema. This way, Spark doesn’t need to load the entire dataset just to get the schema.
# MAGIC
# MAGIC Save and Reuse the Inferred Schema: Once the schema is inferred, you can save it to avoid re-inferring it every time, which can save processing time for large files.

# COMMAND ----------

from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("InferJSONSchema").getOrCreate()

# Sample the file to infer schema (helps with large files)
sample_df = spark.read.option("multiline", "true").json("s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json")
schema = sample_df.schema

# Display the inferred schema using the DataFrame's printSchema method
print("Inferred Schema:")
sample_df.printSchema()  # This will print the schema in a tree format

# Alternatively, you can print the schema as JSON directly
print("Schema in JSON format:")
print(schema.json())

# Save schema to a file if needed (optional)
with open("inferred_schema.json", "w") as schema_file:
    schema_file.write(schema.json())


# COMMAND ----------

# MAGIC %md
# MAGIC For a recursive schema inference approach, you can use libraries like spark-schema-inference, or alternatively, you can implement a recursive method to generate the schema programmatically. Unfortunately, spark-schema-inference isn’t directly available in PySpark. However, you can try using the json library to recursively extract the schema and then convert it into a StructType schema for Spark.
# MAGIC
# MAGIC Here’s how to create a recursive schema inference function in PySpark, which dynamically infers a schema from nested JSON data and applies it in Spark:
# MAGIC
# MAGIC Step 1: Recursive Function to Generate Schema
# MAGIC This function inspects nested structures in a sample JSON record to define a PySpark schema (StructType). You can run this on a small sample of your JSON data for efficiency.
# MAGIC

# COMMAND ----------

import json
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType, ArrayType

def infer_schema_recursively(data):
    """Recursively infers PySpark schema from nested JSON data."""
    if isinstance(data, dict):
        fields = []
        for key, value in data.items():
            field_type = infer_schema_recursively(value)
            fields.append(StructField(key, field_type, True))
        return StructType(fields)
    elif isinstance(data, list):
        if len(data) > 0:
            # Assuming all items in the list have the same structure
            element_type = infer_schema_recursively(data[0])
            return ArrayType(element_type, True)
        else:
            return ArrayType(StringType(), True)  # Default type if empty
    elif isinstance(data, str):
        return StringType()
    elif isinstance(data, bool):
        return BooleanType()
    elif isinstance(data, int):
        return LongType()
    elif isinstance(data, float):
        return DoubleType()
    else:
        return StringType()

# Load a small sample for schema inference
sample_df = spark.read.text("s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json")
sample_json = sample_df.limit(10000000).collect()[0]["value"]
sample_data = json.loads(sample_json)

# Generate schema
inferred_schema = infer_schema_recursively(sample_data)
print("Inferred Recursive Schema:")
print(inferred_schema)


# COMMAND ----------

# MAGIC %md
# MAGIC Steps to Generate a Schema
# MAGIC Extract Unique Keys: Use the provided code to get all unique nested keys.
# MAGIC Parse Keys into Schema Levels: Convert each key (like style.Color Name) into a nested structure within StructType.
# MAGIC Create Recursive Function for Schema Generation: Build StructType objects recursively based on the unique keys.
# MAGIC

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, LongType
import json

# Sample function to map key names to Spark data types
def infer_data_type(key):
    if "id" in key.lower():
        return StringType()
    elif "time" in key.lower() or "date" in key.lower():
        return LongType()
    elif "verified" == key.lower():
        return BooleanType()
    elif "overall" == key.lower():
        return DoubleType()
    else:
        return StringType()

# Function to build schema from unique keys
def build_schema_from_keys(keys):
    schema_dict = {}
    
    # Construct a nested dictionary representing the schema structure
    for key in keys:
        parts = key.split(".")
        current_level = schema_dict
        for part in parts[:-1]:  # Go through each part except the last
            if part not in current_level:
                current_level[part] = {}
            elif not isinstance(current_level[part], dict):
                # Conflict resolution: If a field already has a type, replace it with a dict
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[-1]] = infer_data_type(parts[-1])  # Assign data type at the last part

    # Recursive function to convert the nested dictionary to StructType
    def dict_to_struct(d):
        fields = []
        for key, value in d.items():
            if isinstance(value, dict):  # Nested field
                fields.append(StructField(key, dict_to_struct(value), True))
            else:  # Leaf field with data type
                fields.append(StructField(key, value, True))
        return StructType(fields)

    return dict_to_struct(schema_dict)

# Generate schema from unique keys
unique_keys = sorted([row["key"] for row in unique_keys_df.collect()])  # Assuming unique_keys_df is generated as before
generated_schema = build_schema_from_keys(unique_keys)

# Display the generated schema
print("Generated Schema:")
print(generated_schema)


# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, LongType
import json

# Sample function to map key names to Spark data types
def infer_data_type(key):
    if "id" in key.lower():
        return StringType()
    elif "time" in key.lower() or "date" in key.lower():
        return LongType()
    elif "verified" == key.lower():
        return BooleanType()
    elif "overall" == key.lower():
        return DoubleType()
    else:
        return StringType()

# Function to build schema from unique keys and handle duplicates
def build_schema_from_keys(keys):
    schema_dict = {}
    duplicate_count = {}

    # Construct a nested dictionary representing the schema structure
    for key in keys:
        parts = key.split(".")
        current_level = schema_dict
        for part in parts[:-1]:  # Traverse each part except the last
            if part not in current_level:
                current_level[part] = {}
            elif not isinstance(current_level[part], dict):
                # Conflict resolution: If a field already has a type, replace it with a dict
                current_level[part] = {}
            current_level = current_level[part]
        
        # Handle duplicates by renaming
        final_part = parts[-1]
        if final_part in current_level:
            # If duplicate, append a counter
            duplicate_count[final_part] = duplicate_count.get(final_part, 1) + 1
            final_part = f"{final_part}_{duplicate_count[final_part]}"
        current_level[final_part] = infer_data_type(parts[-1])

    # Recursive function to convert the nested dictionary to StructType
    def dict_to_struct(d):
        fields = []
        for key, value in d.items():
            if isinstance(value, dict):  # Nested field
                fields.append(StructField(key, dict_to_struct(value), True))
            else:  # Leaf field with data type
                fields.append(StructField(key, value, True))
        return StructType(fields)

    return dict_to_struct(schema_dict)

# Generate schema from unique keys
unique_keys = sorted([row["key"] for row in unique_keys_df.collect()])  # Assuming unique_keys_df is generated as before
generated_schema = build_schema_from_keys(unique_keys)

# Display the generated schema
print("Generated Schema with Duplicates Resolved:")
print(generated_schema)


# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType, LongType
import json

# Sample function to map key names to Spark data types
def infer_data_type(key):
    if "id" in key.lower():
        return StringType()
    elif "time" in key.lower() or "date" in key.lower():
        return LongType()
    elif "verified" == key.lower():
        return BooleanType()
    elif "overall" == key.lower():
        return DoubleType()
    else:
        return StringType()

# Function to build schema from unique keys and resolve duplicates by renaming
def build_schema_from_keys(keys):
    schema_dict = {}
    duplicate_tracker = {}

    # Construct a nested dictionary representing the schema structure
    for key in keys:
        parts = key.split(".")
        current_level = schema_dict
        for part in parts[:-1]:  # Traverse each part except the last
            if part not in current_level:
                current_level[part] = {}
            elif not isinstance(current_level[part], dict):
                # Conflict resolution: If a field already has a type, replace it with a dict
                current_level[part] = {}
            current_level = current_level[part]
        
        # Handle duplicates by renaming
        final_part = parts[-1]
        if final_part in current_level:
            # If duplicate, append a suffix to make it unique
            duplicate_tracker[final_part] = duplicate_tracker.get(final_part, 0) + 1
            final_part = f"{final_part}_dup{duplicate_tracker[final_part]}"
        current_level[final_part] = infer_data_type(parts[-1])

    # Recursive function to convert the nested dictionary to StructType
    def dict_to_struct(d):
        fields = []
        for key, value in d.items():
            if isinstance(value, dict):  # Nested field
                fields.append(StructField(key, dict_to_struct(value), True))
            else:  # Leaf field with data type
                fields.append(StructField(key, value, True))
        return StructType(fields)

    return dict_to_struct(schema_dict)

# Generate schema from unique keys
unique_keys = sorted([row["key"] for row in unique_keys_df.collect()])  # Assuming unique_keys_df is generated as before
generated_schema = build_schema_from_keys(unique_keys)

# Display the generated schema with duplicates resolved
print("Generated Schema with Duplicate Columns Resolved:")
print(generated_schema)


# COMMAND ----------

# Load JSON with the generated schema, now with duplicate columns resolved
raw_df = spark.read.schema(generated_schema).json("s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json")

# Display the schema to verify that all fields are included
raw_df.printSchema()

# Show a sample of the data to verify that it loads correctly
raw_df.show(5, truncate=False)

