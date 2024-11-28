# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Initialize Spark session
# spark = SparkSession.builder.appName("ProductRecommendationEDA").getOrCreate()

# Define your S3 path to the Parquet files
s3_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Read all parquet files from the S3 directory
df = spark.read.parquet(s3_path)




# COMMAND ----------

# Verify the schema before processing
df.printSchema()


# COMMAND ----------

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, when

# # Initialize Spark session
# spark = SparkSession.builder.appName("FlattenStyleAttributes").getOrCreate()

# # Read the Parquet files
# df = spark.read.parquet("s3://your-bucket-path/Parquet/cleaned_parquet_files/")

# Function to rename and flatten the struct fields
def flatten_style_columns(df):
    style_columns = []

    # Loop through the 117 style_dup columns (adjust this range as needed)
    for i in range(1, 118):  # Assuming style_dup1 to style_dup117
        style_struct_cols = df.select(f"style_dup{i}.*").columns
        for field in style_struct_cols:
            # Create unique column names for each attribute
            new_column_name = f"style_{i}_{field.replace(' ', '_').lower()}"
            style_columns.append(col(f"style_dup{i}.{field}").alias(new_column_name))
    
    # Select and keep the original columns along with flattened style columns
    return df.select("asin", "overall", "reviewText", "reviewerID", "reviewerName", "summary", "unixReviewTime", "verified", "vote", *style_columns)

# Flatten the style columns
flattened_df = flatten_style_columns(df)

# # Display the schema of the flattened DataFrame
# flattened_df.printSchema()

# # Handle missing data: Fill missing attributes with NULL
# flattened_df = flattened_df.select(
#     [when(col(c).isNull(), None).otherwise(col(c)).alias(c) for c in flattened_df.columns]
# )

# # Verify the data (this will show a sample of the flattened data)
# flattened_df.show(5)

# # Save the transformed DataFrame into a new S3 folder in Parquet format
# output_path = "s3://your-bucket-path/processed-data/"
# flattened_df.write.mode("overwrite").parquet(output_path)

# # Stop the Spark session
# spark.stop()



# COMMAND ----------

# # Display the schema of the flattened DataFrame
flattened_df.printSchema()




# COMMAND ----------

# Handle missing data: Fill missing attributes with NULL
flattened_df = flattened_df.select(
    [when(col(c).isNull(), None).otherwise(col(c)).alias(c) for c in flattened_df.columns]
)

flattened_df = flattened_df.limit(1000)

display(flattened_df)
# # Verify the data (this will show a sample of the flattened data)
# flattened_df.show(5)

# # Save the transformed DataFrame into a new S3 folder in Parquet format
# output_path = "s3://your-bucket-path/processed-data/"
# flattened_df.write.mode("overwrite").parquet(output_path)

# # Stop the Spark session
# spark.stop()

# COMMAND ----------

display(filtered_df)

# COMMAND ----------

from pyspark.sql.functions import col

# List of style_dup columns (as you have 117 style_dup columns)
style_columns = [f"style_dup{i}" for i in range(1, 118)]  # Adjust based on your dataset

# Flatten the style_dup fields into separate columns
for col_name in style_columns:
    # Use getField to access each attribute inside the struct and create new columns
    flattened_df = flattened_df.withColumn(f"style_{col_name.split('_')[1]}", col(f"{col_name}.*"))

# Show the flattened columns (to verify the results)
flattened_df.show(5, truncate=False)


# COMMAND ----------

# # Databricks notebook cell 1: Import statements
# from pyspark.sql.functions import col, when

# # Databricks automatically creates a SparkSession for you, so you don't need to manually create one
# # Instead, you can use the predefined spark session

# # Read the Parquet files
# df = spark.read.parquet("s3://raw-zip-final/Parquet/cleaned_parquet_files/")

# Function to dynamically flatten struct columns
def flatten_style_columns(dataframe):
    # Identify all columns that start with 'style_dup'
    style_dup_columns = [col_name for col_name in dataframe.columns if col_name.startswith('style_dup')]
    
    # Collect all possible attributes across all style_dup columns
    all_attributes = set()
    for col_name in style_dup_columns:
        # Check if the column is a struct type
        col_type = dataframe.schema[col_name].dataType
        if isinstance(col_type, StructType):
            all_attributes.update([field.name for field in col_type.fields])
    
    # Create a list to store new columns
    new_columns = []
    
    # Add original non-style columns
    non_style_columns = [col(c) for c in dataframe.columns if not c.startswith('style_dup')]
    new_columns.extend(non_style_columns)
    
    # Flatten each style_dup column
    for i, col_name in enumerate(style_dup_columns, 1):
        for attr in all_attributes:
            # Create unique column name for each attribute
            new_column_name = f"style_{i}_{attr.replace(' ', '_').lower()}"
            
            # Safely extract the attribute, handling potential null values
            new_columns.append(
                when(col(f"{col_name}.{attr}").isNotNull(), 
                     col(f"{col_name}.{attr}"))
                .otherwise(None)
                .alias(new_column_name)
            )
    
    # Select the new flattened columns
    flattened_df = dataframe.select(new_columns)
    
    return flattened_df

# Flatten the style columns
try:
    # Verify the input DataFrame
    print("Input DataFrame Columns:")
    print(df.columns)
    
    # Attempt to flatten
    flattened_df = flatten_style_columns(df)
    
    # Display the schema of the flattened DataFrame
    print("\nFlattened DataFrame Schema:")
    flattened_df.printSchema()
    
    # Verify the data (this will show a sample of the flattened data)
    print("\nFlattened DataFrame Sample:")
    flattened_df.show(5, truncate=False)
    
    # # Save the transformed DataFrame into a new S3 folder in Parquet format
    # output_path = "s3://your-bucket-path/processed-data/"
    # flattened_df.write.mode("overwrite").parquet(output_path)

except Exception as e:
    print(f"An error occurred: {e}")
    # Print the full stack trace
    import traceback
    traceback.print_exc()

# COMMAND ----------

from pyspark.sql import SparkSession

# Explicitly create or get the Spark session
spark = SparkSession.builder \
    .appName("StyleAttributesFlatten") \
    .getOrCreate()

# Try to read your Parquet files
try:
    df = spark.read.parquet("s3://your-bucket-path/Parquet/cleaned_parquet_files/")
    
    # Print columns to verify
    print("DataFrame Columns:")
    print(df.columns)
    
    # Print schema to understand structure
    print("\nDataFrame Schema:")
    df.printSchema()

except Exception as e:
    print(f"Error reading Parquet files: {e}")

# COMMAND ----------


