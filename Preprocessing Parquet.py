# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Check Parquet Schema").getOrCreate()
parquet_path = "s3://raw-zip-final/Parquet/processed-data/All_Amazon_Review.parquet/"
df = spark.read.parquet(parquet_path)
df.printSchema()


# COMMAND ----------

df.show(25, truncate=False)

# COMMAND ----------

from pyspark.sql.functions import col, sum

# Load the Parquet file
parquet_path = "s3://raw-zip-final/Parquet/processed-data/All_Amazon_Review.parquet/"
df = spark.read.parquet(parquet_path)

# Check for null values in each column
null_counts = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])

# Show the null value counts
null_counts.display()


# COMMAND ----------

display(df
    )

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_json, lit
from pyspark.sql.types import MapType, StringType

# Initialize the Spark session
spark = SparkSession.builder.appName("Convert to Long Format").getOrCreate()

# Step 1: Load the Parquet file
parquet_path = "s3://raw-zip-final/Parquet/processed-data/All_Amazon_Review.parquet/"
df = spark.read.parquet(parquet_path)

# Step 2: Drop fully null columns
non_null_columns = [col_name for col_name in df.columns if df.select(col_name).distinct().count() > 1]
df = df.select(*non_null_columns)

# Step 3: Explode the 'style' column into key-value pairs (if it exists)
if 'style' in df.columns:
    # Parse the 'style' column as a MapType
    df = df.withColumn(
        "style",
        from_json(col("style"), MapType(StringType(), StringType())).alias("style")
    )
    
    # Explode the 'style' column into key-value pairs
    df_long = df.select(
        "asin",  # Include other relevant columns like `asin` or `reviewText`
        explode(col("style")).alias("Feature Key", "Feature Value")
    )
else:
    raise ValueError("The 'style' column does not exist in the dataset.")

# Step 4: Save the resulting DataFrame in long format
output_path = "s3://raw-zip-final/Parquet2/processed-data-long-format/All_Amazon_Review_long_format/"
df_long.write.mode("overwrite").parquet(output_path)

print(f"Long format data saved to {output_path}")


# COMMAND ----------

# Load the long format Parquet file (if saved previously)
long_format_path = "s3://raw-zip-final/Parquet2/processed-data-long-format/All_Amazon_Review_long_format/"
df_long = spark.read.parquet(long_format_path)

# Display the schema to verify the structure
df_long.printSchema()

# Show a sample of the data
print("Sample rows from the long format data:")
df_long.show(truncate=False)

# Count the total number of rows
total_rows = df_long.count()
print(f"Total rows in the long format dataset: {total_rows}")

# Count unique values in key column
distinct_keys = df_long.select("Feature Key").distinct().count()
print(f"Number of unique keys in the dataset: {distinct_keys}")

# Display a few unique keys for review
print("Sample unique keys:")
df_long.select("Feature Key").distinct().show(10, truncate=False)

# Count rows for a specific key (optional)
key_to_check = "Format"  # Example key
key_count = df_long.filter(col("Feature Key") == key_to_check).count()
print(f"Rows with 'Feature Key' = '{key_to_check}': {key_count}")

# Check for null or empty values in key or value columns
null_keys = df_long.filter(col("Feature Key").isNull() | col("Feature Value").isNull()).count()
print(f"Rows with null keys or values: {null_keys}")


# COMMAND ----------

# Display 20 rows of the data
df_long.show(20, truncate=False)

# Display distinct Feature Keys
df_long.select("Feature Key").distinct().show(150, truncate=False)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
import ast

# Start a Spark session
spark = SparkSession.builder.appName("Process Amazon Reviews").getOrCreate()

# Load the Parquet file from S3
parquet_path = "s3://raw-zip-final/Parquet/processed-data/All_Amazon_Review.parquet/"
df = spark.read.parquet(parquet_path)

# Remove columns with names containing "style_dup"
columns_to_keep = [col_name for col_name in df.columns if not col_name.startswith("style_dup")]
df_filtered = df.select(*columns_to_keep)

# Define a UDF to parse the dictionary-like strings in the 'style' column
def parse_style(style_str):
    try:
        if style_str is not None:
            return ast.literal_eval(style_str).get("Format:", None)
    except:
        return None

parse_style_udf = udf(parse_style)

# Apply the UDF to extract the 'Format' from the 'style' column
df_final = df_filtered.withColumn("Format", parse_style_udf(col("style"))).drop("style")

# Show the resulting DataFrame schema and some rows
df_final.printSchema()
df_final.show(truncate=False)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, StringType
import ast

# Start a Spark session
spark = SparkSession.builder.appName("Process Amazon Reviews").getOrCreate()

# Load the Parquet file from S3
parquet_path = "s3://raw-zip-final/Parquet/processed-data/All_Amazon_Review.parquet/"
df = spark.read.parquet(parquet_path)

# Remove columns with names containing "style_dup"
columns_to_keep = [col_name for col_name in df.columns if not col_name.startswith("style_dup")]
df_filtered = df.select(*columns_to_keep)

# Define a UDF to parse the dictionary-like strings in the 'style' column and return the keys
def parse_style_keys(style_str):
    try:
        if style_str is not None:
            return list(ast.literal_eval(style_str).keys())
    except:
        return []
    return []

parse_style_keys_udf = udf(parse_style_keys, ArrayType(StringType()))

# Apply the UDF to extract keys from the 'style' column into an array
df_keys = df_filtered.withColumn("style_keys", parse_style_keys_udf(col("style")))

# Explode the 'style_keys' column to get individual keys and find distinct keys
distinct_keys_df = df_keys.select(explode(col("style_keys")).alias("key")).distinct()

# Collect the unique keys
unique_keys = [row["key"] for row in distinct_keys_df.collect()]

# Define a UDF to parse the dictionary-like strings in the 'style' column to extract values by key
def parse_style_value(style_str, key):
    try:
        if style_str is not None:
            return ast.literal_eval(style_str).get(key, None)
    except:
        return None

parse_style_value_udf = udf(parse_style_value, StringType())

# Create a column for each unique key found in the dictionaries
for key in unique_keys:
    df_filtered = df_filtered.withColumn(key, parse_style_value_udf(col("style"), udf(lambda: key, StringType())()))

# Drop the original 'style' column
df_final = df_filtered.drop("style")

# Show the resulting DataFrame schema and some rows
df_final.printSchema()
df_final.show(truncate=False)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, explode
from pyspark.sql.types import ArrayType, StringType
import ast

# Start a Spark session
spark = SparkSession.builder.appName("Process Amazon Reviews").getOrCreate()

# Load the Parquet file from S3
parquet_path = "s3://raw-zip-final/Parquet/processed-data/All_Amazon_Review.parquet/"
df = spark.read.parquet(parquet_path)

# Remove columns with names containing "style_dup"
columns_to_keep = [col_name for col_name in df.columns if not col_name.startswith("style_dup")]
df_filtered = df.select(*columns_to_keep)

# Define a UDF to parse the dictionary-like strings in the 'style' column and return the keys
def parse_style_keys(style_str):
    try:
        if style_str is not None:
            return list(ast.literal_eval(style_str).keys())
    except:
        return []
    return []

parse_style_keys_udf = udf(parse_style_keys, ArrayType(StringType()))

# Apply the UDF to extract keys from the 'style' column into an array
df_keys = df_filtered.withColumn("style_keys", parse_style_keys_udf(col("style")))

# Explode the 'style_keys' column to get individual keys and find distinct keys
distinct_keys_df = df_keys.select(explode(col("style_keys")).alias("key")).distinct()

# Collect the unique keys
unique_keys = [row["key"] for row in distinct_keys_df.collect()]

# Define a UDF to parse the dictionary-like strings in the 'style' column to extract values by key
def parse_style_value(style_str, key):
    try:
        if style_str is not None:
            return ast.literal_eval(style_str).get(key, None)
    except:
        return None

parse_style_value_udf = udf(parse_style_value, StringType())

# Create a column for each unique key found in the dictionaries
for key in unique_keys:
    df_filtered = df_filtered.withColumn(key, parse_style_value_udf(col("style"), udf(lambda: key, StringType())()))

# Drop the original 'style' column
df_final = df_filtered.drop("style")

# Define S3 path for saving the transformed data
output_parquet_path = "s3://raw-zip-final/Parquet/processed-data/Transformed_Amazon_Review.parquet/"

# Write the transformed DataFrame to S3 as Parquet
df_final.write.mode("overwrite").parquet(output_parquet_path)


# COMMAND ----------

from pyspark.sql import SparkSession

# Start a Spark session
spark = SparkSession.builder.appName("Display Transformed Amazon Reviews").getOrCreate()

# Load the transformed Parquet file from S3
transformed_parquet_path = "s3://raw-zip-final/Parquet/processed-data/Transformed_Amazon_Review.parquet/"
df_transformed = spark.read.parquet(transformed_parquet_path)

# Display the first few rows of the transformed DataFrame
display(df_transformed)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count

# Start a Spark session
spark = SparkSession.builder.appName("Check Null Values in Columns").getOrCreate()

# Load the transformed Parquet file from S3
transformed_parquet_path = "s3://raw-zip-final/Parquet/processed-data/Transformed_Amazon_Review.parquet/"
df_transformed = spark.read.parquet(transformed_parquet_path)

# Count null values for each column
null_counts = df_transformed.select([count(when(col(c).isNull(), c)).alias(c) for c in df_transformed.columns])

# Show the result
null_counts.show()


# COMMAND ----------

# Calculate null counts for each column and display as a list
null_counts_df = df_transformed.select([count(when(col(c).isNull(), c)).alias(c) for c in df_transformed.columns])

# Collecting the data into a dictionary for easier display
null_counts = null_counts_df.collect()[0].asDict()

# Sorting by null counts for easier analysis
sorted_null_counts = dict(sorted(null_counts.items(), key=lambda item: item[1], reverse=True))

# Displaying the null counts in a more readable format
for column, null_count in sorted_null_counts.items():
    print(f"Column '{column}': {null_count} null values")

