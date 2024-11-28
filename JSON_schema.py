# Databricks notebook source
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, udf
from pyspark.sql.types import ArrayType, StringType

# Initializing Spark session
spark = SparkSession.builder.appName("Extracting Ordered Nested Keys").getOrCreate()

# Defining the file input path (S3 path)
input_file_path = "s3a://raw-zip-final/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

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

# MAGIC %md
# MAGIC The persistent [COLUMN_ALREADY_EXISTS] error indicates that duplicate field names are still not fully resolved. This can happen if Spark encounters identical column names within nested structures, even after appending suffixes. Let's apply a stricter, more comprehensive solution by ensuring that every instance of a duplicate field receives a unique name across the schema.
# MAGIC
# MAGIC Solution: Uniquely Name Every Field Instance
# MAGIC This solution will ensure each field has a globally unique name, regardless of its nested position, by keeping track of each encountered field and its path.
# MAGIC
# MAGIC Updated Schema-Building Code
# MAGIC Here’s the revised code to guarantee unique names by using the full path as a reference and renaming every duplicate consistently:

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

# Function to build schema from unique keys and ensure all column names are unique
def build_unique_schema_from_keys(keys):
    schema_dict = {}
    unique_name_tracker = {}

    # Helper function to get a unique name based on the full path
    def get_unique_name(name, path):
        full_name = f"{path}.{name}" if path else name
        if full_name in unique_name_tracker:
            # Append a suffix to make it unique
            count = unique_name_tracker[full_name]
            unique_name_tracker[full_name] += 1
            return f"{name}_dup{count}"
        else:
            unique_name_tracker[full_name] = 1
            return name

    # Construct a nested dictionary with unique field names
    for key in keys:
        parts = key.split(".")
        current_level = schema_dict
        current_path = ""
        for part in parts[:-1]:  # Traverse each part except the last
            unique_part = get_unique_name(part, current_path)
            current_path = f"{current_path}.{unique_part}" if current_path else unique_part
            if unique_part not in current_level:
                current_level[unique_part] = {}
            current_level = current_level[unique_part]
        
        # Final part (the actual field) with data type and unique naming
        final_part = get_unique_name(parts[-1], current_path)
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
generated_schema = build_unique_schema_from_keys(unique_keys)

# Display the generated schema with unique column names
print("Generated Schema with Fully Unique Column Names:")
print(generated_schema)


# COMMAND ----------

# Load JSON data with the generated schema
raw_df = spark.read.schema(generated_schema).json("s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json")

# Verify the schema and data
raw_df.printSchema()
raw_df.show(5, truncate=False)

