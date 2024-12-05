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


# COMMAND ----------

display(filtered_df)

# COMMAND ----------


