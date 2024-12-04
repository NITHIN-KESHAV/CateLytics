# Databricks notebook source
from pyspark.sql import SparkSession

# Initializing the Spark session
spark = SparkSession.builder.appName("Drop style_dup Columns and Print Schema").getOrCreate()

# Defining the input Parquet folder path
parquet_folder_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Loading the Parquet files
df = spark.read.format("parquet").load(parquet_folder_path)

# Identifying the columns the columns that need to be dropped - columns starting with 'style_dup')
columns_to_drop = [col_name for col_name in df.columns if col_name.startswith("style_dup")]

# Dropping the identified columns
updated_df = df.drop(*columns_to_drop)

# Printing the schema of the updated DataFrame
updated_df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col, from_json, get_json_object, expr, trim
from pyspark.sql.types import StructType, StructField, StringType
import json

def extract_all_keys_values(updated_df):
    try:
        # Defining the schema based on what we see in the data
        json_schema = StructType([
            StructField("Format:", StringType(), True),
            StructField("Size:", StringType(), True),
            StructField("Color:", StringType(), True)
        ])
        
        # Parsing the JSON and creating new columns
        result_df = updated_df.withColumn(
            'parsed_json',
            from_json(col('style'), json_schema)
        ).withColumn(
            'style_Format',
            trim(col('parsed_json.Format:'))
        ).withColumn(
            'style_Size',
            trim(col('parsed_json.Size:'))
        ).withColumn(
            'style_Color',
            trim(col('parsed_json.Color:'))
        )
        
        # Dropping the intermediate parsed_json column
        result_df = result_df.drop('parsed_json')
        
        return result_df
        
    except Exception as e:
        print(f"Error in extraction: {str(e)}")
        return updated_df

try:
    # Applying the transformation
    transformed_df = extract_all_keys_values(updated_df)
    
    # Let us see a few rows that have Size and Color
    print("\nRows with Size and Color:")
    transformed_df.filter(col("style").contains("Size")).select(
        "style", "style_Size", "style_Color"
    ).show(5, truncate=False)
    
    # Seeing a few rows that have Format
    print("\nRows with Format:")
    transformed_df.filter(col("style").contains("Format")).select(
        "style", "style_Format"
    ).show(5, truncate=False)
    
except Exception as e:
    print(f"Error during transformation: {str(e)}")

# COMMAND ----------

from pyspark.sql.functions import col, get_json_object, trim, lit
import json

def extract_all_keys_values(updated_df):
    try:
        # Collecting distinct style column values to extract all possible keys
        style_rows = updated_df.select("style").distinct().collect()
        
        unique_keys = {}

        # Iterating through rows to find all the keys
        for row in style_rows:
            if row.style and row.style.strip():
                try:
                    json_dict = json.loads(row.style)
                    for k in json_dict.keys():
                        unique_keys[k] = k  
                except Exception as parse_err:
                    print(f"Error parsing JSON: {parse_err}")
                    continue
        
        print("\nExtracted unique keys:", sorted(unique_keys.keys()))
        
        # Starting with the original DataFrame
        result_df = updated_df

        # Creating columns for each unique key
        for original_key in unique_keys.values():
            # Cleaning the column name to make it safe
            safe_key = original_key.replace(":", "").replace(" ", "_").lower().strip()
            # Extracting the value using get_json_object
            result_df = result_df.withColumn(
                f'style_{safe_key}',
                trim(get_json_object(col('style'), f'$.{original_key}'))
            )
        
        return result_df

    except Exception as e:
        print(f"Error in extraction process: {str(e)}")
        return updated_df

try:
    # Applying the function to transform the DataFrame
    transformed_df = extract_all_keys_values(updated_df)

    print("\nTransformed data with extracted values:")
    transformed_df.display(truncate=False)

except Exception as e:
    print(f"Error during transformation: {str(e)}")

# COMMAND ----------

from pyspark.sql.functions import col

# Filtering the rows where `style_color` is not NULL and not empty
filtered_df = transformed_df.filter(col("style_color").isNotNull() & (col("style_color") != ""))

# Displaying all the columns for the filtered rows
filtered_df.display(truncate=False)


# COMMAND ----------

from pyspark.sql.functions import col, sum, when

# Calculating the number of NULL values in each column
null_counts = transformed_df.select(
    [
        sum(when(col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in transformed_df.columns
    ]
)

null_counts.display(truncate=False)

# COMMAND ----------

# Printing the schema of the DataFrame
transformed_df.printSchema()


# COMMAND ----------

# Defining the S3 path for writing the transformed and flattened data
output_s3_path = "s3://raw-zip-final/Parquet/flattened_transformed_files/"

# Writing the DataFrame to the S3 bucket in Parquet format
transformed_df.write.mode("overwrite").parquet(output_s3_path)

print(f"Flattened and transformed data successfully written to {output_s3_path}")

# COMMAND ----------

# Defining the S3 path for reading the flattened transformed files
output_s3_path = "s3://raw-zip-final/Parquet/flattened_transformed_files/"

# Reading the Parquet files from the S3 bucket
verified_df = spark.read.parquet(output_s3_path)

# Displaying the first few rows to verify the data
verified_df.display(n=10, truncate=False)
