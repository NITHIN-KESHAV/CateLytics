# Databricks notebook source
# Dropping the reviewTime and image fields and putting inside cleaned_parquet_files from All_Amazon_Review.parqetfiles

# from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("Read and Process Parquet Data") \
    .getOrCreate()

# Path to the Parquet folder
s3_path = 's3://raw-zip-final/Parquet/processed-data/All_Amazon_Review.parquet/'

# Read the Parquet files into a DataFrame
df = spark.read.parquet(s3_path)

# Display the original schema
print("Original Schema:")
df.printSchema()

# Drop the columns 'reviewTime' and 'image'
cleaned_df = df.drop("reviewTime", "image")

# Display the schema after dropping the columns
print("Schema After Dropping Columns:")
cleaned_df.printSchema()

# Display the first few rows of the cleaned DataFrame
cleaned_df.show(5, truncate=False)


# COMMAND ----------

# Defining the output path for the cleaned Parquet files
output_parquet_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Writing the cleaned DataFrame to Parquet format in the S3 bucket
cleaned_df.write.mode("overwrite").parquet(output_parquet_path)

print("Parquet files successfully written to:", output_parquet_path)

# COMMAND ----------

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

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder \
    .appName("Dynamic Flattening of Nested Fields") \
    .getOrCreate()

# Path to the Parquet file
parquet_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Read the Parquet file
df = spark.read.parquet(parquet_path)

# Display the original schema
print("Original Schema:")
df.printSchema()

# Extract and dynamically flatten all fields within 'style'
style_fields = df.schema["style"].dataType.fields  # Get all fields in the 'style' struct
flattened_columns = [
    col(f"style.{field.name}").alias(f"style.{field.name.replace(' ', '_')}")
    for field in style_fields
]

# Create a new DataFrame with flattened fields
flattened_df = df.select(*flattened_columns)

# Display the schema of the flattened DataFrame
print("Flattened Schema:")
flattened_df.printSchema()

# Show some sample data
flattened_df.show(truncate=False)


# COMMAND ----------

from pyspark.sql.types import StructType

# Function to check if a column is of StructType
def check_struct_fields(df):
    for field in df.schema.fields:
        if isinstance(field.dataType, StructType):
            print(f"Field '{field.name}' is a StructType")
        else:
            print(f"Field '{field.name}' is NOT a StructType")

# Example usage with your DataFrame
check_struct_fields(df)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create a Spark session
spark = SparkSession.builder \
    .appName("Dynamic Flattening of Nested Fields") \
    .getOrCreate()

# Path to the Parquet file
parquet_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Read the Parquet file
df = spark.read.parquet(parquet_path)

# Display the original schema
print("Original Schema:")
df.printSchema()

# Function to flatten all StructType fields dynamically
def flatten_struct_fields(df):
    flattened_columns = []
    
    # Loop through all fields in the schema
    for field in df.schema.fields:
        if isinstance(field.dataType, StructType):
            # For each StructType field, flatten its nested fields
            for nested_field in field.dataType.fields:
                new_col_name = f"{field.name}.{nested_field.name.replace(' ', '_')}"
                flattened_columns.append(col(f"{field.name}.{nested_field.name}").alias(new_col_name))
        else:
            # If the field is not a StructType, retain it as is
            flattened_columns.append(col(field.name))
    
    # Create a new DataFrame with the flattened columns
    return df.select(*flattened_columns)

# Apply the function to flatten the DataFrame
flattened_df = flatten_struct_fields(df)

# Display the schema of the flattened DataFrame
print("Flattened Schema:")
flattened_df.printSchema()

# Show some sample data
flattened_df.show(truncate=False)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField
import re

# Create a Spark session
spark = SparkSession.builder \
    .appName("Dynamic Flattening and Null Validation") \
    .getOrCreate()

# Path to the Parquet file
parquet_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Read the Parquet file
df = spark.read.parquet(parquet_path)

# Function to flatten all StructType fields dynamically
def flatten_struct_fields(df):
    flattened_columns = []

    # Loop through all fields in the schema
    for field in df.schema.fields:
        if isinstance(field.dataType, StructType):
            # For each StructType field, flatten its nested fields
            for nested_field in field.dataType.fields:
                nested_field_name = nested_field.name.replace(" ", "_")
                new_col_name = f"{field.name}_{nested_field_name}"  # Flatten column name
                flattened_columns.append(col(f"`{field.name}`.`{nested_field.name}`").alias(new_col_name))
                print(f"Flattening column: {new_col_name}")
        else:
            # If the field is not a StructType, retain it as is
            flattened_columns.append(col(f"`{field.name}`"))

    # Create a new DataFrame with the flattened columns
    return df.select(*flattened_columns)

# Apply the function to flatten the DataFrame
flattened_df = flatten_struct_fields(df)

# Display the schema of the flattened DataFrame
print("Flattened Schema:")
flattened_df.printSchema()

# Show some sample data to ensure the flattening is happening
flattened_df.show(5, truncate=False)

# Function to check if columns are entirely null
def check_null_columns(dataframe, column_list):
    null_status = {}
    for column in column_list:
        # Use backticks around column names with special characters
        non_null_count = dataframe.filter(col(f"`{column}`").isNotNull()).count()
        null_status[column] = non_null_count == 0  # True if column is entirely null
    return null_status

# Extract column names matching the pattern for style_dup fields
pattern = r"^style_dup\d+_.+"  # Updated to match flattened column names
matching_columns = [col_name for col_name in flattened_df.columns if re.match(pattern, col_name)]

# Checking null status for all matching columns
null_status = check_null_columns(flattened_df, matching_columns)

# Printing the results
print("\nNull status of matching columns:")
for column, status in null_status.items():
    print(f"Column {column}: {'All Null' if status else 'Contains Non-Null Values'}")


# COMMAND ----------

from pyspark.sql import SparkSession

# Step 1: Create a Spark session
spark = SparkSession.builder \
    .appName("Inspect style Column") \
    .getOrCreate()

# Step 2: Define the Parquet path
parquet_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Step 3: Read the Parquet file
df = spark.read.parquet(parquet_path)

# Step 4: Check if the 'style' column exists
if 'style' in df.columns:
    print("Schema of 'style' column:")
    print(df.schema['style'].dataType)

    print("\nSample Data for 'style' column:")
    df.select("style").show(truncate=False, n=10)  # Adjust `n` to display more rows
else:
    print("The 'style' column does not exist in the DataFrame.")


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import split, regexp_replace, when, col, expr
from pyspark.sql.types import StructType, StructField, StringType

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Style Data Processor") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Read the parquet file
parquet_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"
df = spark.read.parquet(parquet_path)

# Get all column names
all_columns = df.columns

# Filter out style_dup columns
style_dup_columns = [col_name for col_name in all_columns if 'style_dup' in col_name.lower()]
columns_to_keep = [col_name for col_name in all_columns if 'style_dup' not in col_name.lower()]

# Drop all style_dup columns
df = df.select(*columns_to_keep)

# Function to extract style attributes
def extract_style_attributes(df):
    # First, get all possible style attributes from the dataset
    # This helps us handle cases where some rows might not have all attributes
    sample_styles = df.select("Style").rdd.flatMap(lambda x: x[0].split() if x[0] else []).distinct().collect()
    style_keys = set()
    
    for style in sample_styles:
        if ':' in style:
            key = style.split(':')[0]
            style_keys.add(key)
    
    # Create columns for each style attribute
    for key in style_keys:
        # Create a regexp pattern to match the specific key-value pair
        pattern = f"{key}:([^\\s]+)"
        
        # Extract the value using regexp_replace
        # If the pattern is not found, it will return null
        df = df.withColumn(
            key.replace('.', '_'),  # Replace dots with underscores for valid column names
            regexp_replace(
                regexp_replace(
                    expr(f"regexp_extract(Style, '{pattern}', 1)"),
                    '^$',
                    None
                ),
                '^null$',
                None
            )
        )
    
    return df

# Process the dataframe
processed_df = extract_style_attributes(df)

# Print number of dropped columns for verification
print(f"Number of style_dup columns dropped: {len(style_dup_columns)}")
print("Dropped columns:", style_dup_columns)

# Show the schema to verify the new columns
processed_df.printSchema()

# Optional: Cache the processed dataframe if you'll be using it multiple times
processed_df.cache()

# Example usage:
# You can now access specific style attributes as columns
# For example, if you had Style.color in your data:
# processed_df.select("Style_color").show()

# Write the processed data back to parquet if needed
# processed_df.write.mode("overwrite").parquet("output_path")

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import MapType, StringType

# Step 1: Create a Spark session
spark = SparkSession.builder \
    .appName("Drop style_dup and Extract Format from style Column") \
    .getOrCreate()

# Step 2: Define the Parquet path
parquet_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Step 3: Read the Parquet file
df = spark.read.parquet(parquet_path)

# Step 4: Drop all columns starting with 'style_dup'
style_dup_columns = [col_name for col_name in df.columns if col_name.startswith("style_dup")]
df = df.drop(*style_dup_columns)
print(f"Dropped Columns: {style_dup_columns}")

# Step 5: Check if the 'style' column exists and is a string (JSON-like)
if 'style' in df.columns:
    # Define the schema for the 'style' column as a MapType (key-value pairs)
    style_schema = MapType(StringType(), StringType())

    # Step 6: Parse the 'style' column as JSON
    df_parsed = df.withColumn("style_parsed", from_json(col("style"), style_schema))

    # Step 7: Extract fields from the parsed 'style' column
    # Assuming we're interested in extracting the key "Format:"
    df_extracted = df_parsed.withColumn("Format", col("style_parsed")["Format:"])

    # Step 8: Drop the original 'style' column and the parsed one if not needed
    df_extracted = df_extracted.drop("style", "style_parsed")

    # Step 9: Show the updated schema and sample data
    print("Updated Schema:")
    df_extracted.printSchema()

    print("\nSample Data:")
    df_extracted.show(truncate=False, n=10)  # Display top 10 rows for inspection

else:
    print("The 'style' column does not exist in the DataFrame.")


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, split, explode, expr, when
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Spark Session
spark = SparkSession.builder.appName("Style Processor").getOrCreate()

# Define the input path
input_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Read data
df = spark.read.parquet(input_path)

# Drop style_dup columns if they exist
style_dup_columns = [c for c in df.columns if c.startswith("style_dup")]
df = df.drop(*style_dup_columns)

# Clean Style column by removing braces and quotes
df = df.withColumn("Style_cleaned", regexp_replace(col("Style"), "[{}\"]", ""))

# Extract unique keys using DataFrame operations and avoid collecting data to the driver
keys_df = (
    df
    .select(explode(split(col("Style_cleaned"), ",")).alias("style_pair"))
    .filter(col("style_pair").contains(":"))
    .select(split(col("style_pair"), ":").getItem(0).alias("key"))
    .distinct()
)

# Count keys for debugging purposes
keys_count = keys_df.count()
logging.debug(f"Found {keys_count} unique style keys")

# Collect keys (consider using .collect() only if the number of keys is small)
style_keys = [row.key.strip() for row in keys_df.collect()]

# Process each key and add as new column
result_df = df
for key in style_keys:
    # Create safe column name
    safe_key = re.sub(r'[^a-zA-Z0-9]', '_', key)
    if safe_key[0].isdigit():
        safe_key = f"n{safe_key}"

    logging.debug(f"Processing key: {key} -> safe key: {safe_key}")
    
    try:
        # Escape special characters in key for regexp_extract
        escaped_key = re.escape(key)
        expr_str = f"trim(regexp_extract(Style_cleaned, '{escaped_key}:([^,]+)', 1))"
        logging.debug(f"Generated expression for key '{key}': {expr_str}")
        
        # Create column with extracted value or None
        result_df = result_df.withColumn(
            f"style_{safe_key}",
            when(
                col("Style_cleaned").contains(f"{key}:"),
                expr(expr_str)
            ).otherwise(None)
        )
    except Exception as e:
        logging.error(f"Error processing key '{key}': {e}")
        logging.warning(f"Skipping key '{key}' due to error: {e}")
        continue

# Clean up temporary column and show a limited number of rows
result_df = result_df.drop("Style_cleaned")
logging.debug("Displaying processed DataFrame")
result_df.show(10)

