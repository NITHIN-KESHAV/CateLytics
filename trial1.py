# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, first
from pyspark.sql.types import StructType

# Initialize Spark session
spark = SparkSession.builder.appName("FlattenStyleColumns").getOrCreate()

# Read the parquet files
df = spark.read.parquet("s3://raw-zip-final/Parquet/cleaned_parquet_files/")

# Function to flatten the style_dup* columns
def parse_style_columns(df):
    # Get all columns that start with 'style_dup'
    style_columns = [col for col in df.columns if col.startswith('style_dup')]
    
    # Iterate over each style_dup column and extract the inner fields
    for style_column in style_columns:
        # Get the name of the style (e.g., style_dup1, style_dup2)
        style_name = style_column.split("_")[-1]  # e.g., "1", "2", etc.
        
        # Handle each field in the struct inside the style column
        # For example: style_dup1.Bare Outside Diameter String
        for field in df.select(f"{style_column}.*").columns:
            # Create a new column with a name like "style_1_bare_outside_diameter_string"
            new_column_name = f"style_{style_name}_{field.replace(' ', '_').lower()}"
            df = df.withColumn(new_column_name, col(f"{style_column}.{field}"))
        
    return df

def create_specific_columns(df):
    # After flattening all the style fields, drop the original style_dup* columns
    style_columns_to_drop = [col for col in df.columns if col.startswith('style_dup')]
    df = df.drop(*style_columns_to_drop)
    
    return df

# Parse and flatten the style columns
parsed_df = parse_style_columns(df)

# Create the final DataFrame with all the extracted columns
final_df = create_specific_columns(parsed_df)

# Show the first few rows to verify
final_df.show(5, truncate=False)

# Print schema to check column creation
final_df.printSchema()

# # Save the transformed DataFrame into a new S3 folder in Parquet format (without overwriting)
# output_path = "s3://your-bucket/your-output-folder/"
# final_df.write.mode("append").parquet(output_path)

# # Stop the Spark session
# spark.stop()


# COMMAND ----------

final_df = final_df.limit(1000)

display(final_df)

# COMMAND ----------

parsed_df = parsed_df.limit(1000)

display(parsed_df)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when, regexp_extract, udf
from pyspark.sql.types import StringType, MapType

# Initialize Spark session
# spark = SparkSession.builder.appName("FlattenStyleColumn").getOrCreate()

# Read the parquet files from S3
df = spark.read.parquet("s3://raw-zip-final/Parquet/cleaned_parquet_files/")

# Step 1: Extract key-value pairs from the 'style' column
def extract_key_value_pairs(style_str):
    """
    Extract key-value pairs from the style column (assuming it's a string of key-value pairs).
    This function is just an example; you may need to adjust based on the actual structure of the style column.
    """
    import ast
    try:
        # If the style column is in a string format resembling a dictionary
        style_dict = ast.literal_eval(style_str)
        return style_dict
    except:
        return {}

# Register the UDF (User-Defined Function) to parse the style column
extract_udf = udf(extract_key_value_pairs, MapType(StringType(), StringType()))

# Apply the UDF to parse the style column into key-value pairs
df_parsed = df.withColumn("style_parsed", extract_udf(col("style")))

# Step 2: Extract individual keys and create separate columns for each one
# We will create separate columns for each key in the style column (like style_color, style_diameter, etc.)

style_keys = [
    "Bare Outside Diameter String", "Bore Diameter", "Capacity", "Closed Length String", 
    "Color Name", "Color", "Colorj", "Colour", "Compatible Fastener Description", 
    "Conference Name", "Configuration", "Connectivity", "Connector Type", "Content", 
    "Current Type", "Curvature", "Cutting Diameter", "Denomination", "Department", "Design", 
    "Diameter", "Digital Storage Capacity", "Display Height", "Edition", "Electrical Connector Type","Extended Length","Fits Shaft Diameter","Flavor Name","Flavor","Flex","Format","Free Length","Gauge","Gem Type","Gift Amount","Grip Type","Grit Type","Hand Orientation","Hardware Platform","Head Diameter","Horsepower","Initial","Input Range Description","Inside Diameter","Item Display Length","Item Display Weight","Item Package Quantity","Item Shape","Item Thickness","Item Weight","Lead Type","Length Range","Length","Line Weight","Load Capacity","Loft","Material Type","Material","Maximum Measurement","Measuring Range","Metal Stamp","Metal Type","Model Name","Model Number","Model","Nominal Outside Diameter","Number of Items","Number of Shelves","Offer Type","Options","Outside Diameter","Overall Height","Overall Length","Overall Width","Package Quantity","Package Type","Part Number","Pattern","Pitch Diameter String","Pitch","Platform for Display","Platform","Preamplifier Output Channel Quantity","Primary Stone Gem Type","Processor Description","Product Packaging","Range","SCENT","Scent Name","Scent","Service plan term","Shaft Material Type","Shaft Material","Shape","Size Name","Size per Pearl","Size","Special Features","Stone Shape","Style Name","Style","Subscription Length","Team Name","Temperature Range","Tension Supported","Thickness","Total Diamond Weight","Unit Count","Volume","Wattage","Width Range","Width","Wire Diameter","option","processor_description","style name","style"
    # Add other keys as needed...
]

# For each key, create a column if it exists in the parsed map
for key in style_keys:
    key_column = f"style_{key.replace(' ', '_').lower()}"
    df_parsed = df_parsed.withColumn(
        key_column, 
        when(col("style_parsed").getItem(key).isNotNull(), col("style_parsed").getItem(key)).otherwise(lit(None))
    )

# Step 3: Drop the parsed 'style' column and 'style_parsed' map column
df_final = df_parsed.drop("style", "style_parsed")

# Show the first few rows to verify the result
df_final.show(5, truncate=False)

# Print schema to check the created columns
df_final.printSchema()

df_final = df_final.limit(1000)

display(df_final)

# # Step 4: Save the final DataFrame to a new S3 location
# output_path = "s3://your-bucket/your-output-path/"
# df_final.write.mode("overwrite").parquet(output_path)

# # Stop the Spark session
# spark.stop()


# COMMAND ----------

from pyspark.sql.functions import from_json, col, lit, when, explode_outer, map_keys
from pyspark.sql.types import MapType, StringType, StructType

def split_style_column(df):
    # First, parse the style column into a map
    # Adjust the parsing based on your actual data type
    df_parsed = df.withColumn("style_map", 
        from_json(col("style"), MapType(StringType(), StringType()), 
        options={"mode": "PERMISSIVE"})
    )
    
    # Get unique keys dynamically
    style_keys = df_parsed.select(explode_outer(map_keys(col("style_map")))).distinct().collect()
    
    # Create a function to add columns dynamically
    def add_style_columns(df):
        for key in style_keys:
            key_name = key[0]
            df = df.withColumn(f"style_{key_name}", 
                when(col("style_map").getItem(key_name).isNotNull(), 
                     col("style_map").getItem(key_name))
                .otherwise(lit(None)))
        return df
    
    # Apply the transformation
    df_expanded = add_style_columns(df_parsed)
    
    # Drop the intermediate style_map column
    df_final = df_expanded.drop("style_map")
    
    return df_final

# Read Parquet files from S3
s3_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"
df = spark.read.parquet(s3_path)

# Apply style column splitting
expanded_df = split_style_column(df)

# Show the first 1000 rows with the new columns
display(expanded_df.limit(1000))

# Print the schema to verify new columns
display(expanded_df.printSchema())

# Count the number of new style columns
style_columns = [col for col in expanded_df.columns if col.startswith("style_")]
print(f"Number of new style columns: {len(style_columns)}")

# Optional: Save the expanded DataFrame
# expanded_df.write.mode("overwrite").parquet("s3://your-bucket/path/to/expanded/parquet/files/")

# COMMAND ----------

# Import necessary modules
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# Initialize Spark session (if not already initialized)
spark = SparkSession.builder.appName("CountRowsInParquet").getOrCreate()

# Define the S3 path where your Parquet files are stored
s3_path = "s3://raw-zip-final/Parquet/cleaned_parquet_files/"

# Read all Parquet files from the S3 directory
df = spark.read.parquet(s3_path)

# Count the total number of rows
total_rows = df.count()

# Print the total number of rows
print(f"Total number of rows in all Parquet files: {total_rows}")


# COMMAND ----------


