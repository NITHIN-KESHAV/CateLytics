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

# MAGIC %pip install sentence_transformers

# COMMAND ----------

import boto3
import pandas as pd
import pyarrow.parquet as pq
import uuid
from sentence_transformers import SentenceTransformer

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# AWS S3 setup
aws_access_key_id = "AKIAVYV52E7ZRAA4ISFA"  # Replace with your AWS access key
aws_secret_access_key = "ZUJ4ABufRGFUhgBJFxyjRwE+G5yHfY2UjeKzGjk5"  # Replace with your AWS secret key
aws_region = "us-west-1"  # Replace with your region
bucket_name = "raw-zip-final"  # Replace with your bucket name
s3_input_file = "Parquet/Final_Data/part-00001-tid-4390418886624719263-ec9cbcbd-07dc-4e4c-8c33-c7c581c6155a-22426-1.c000.snappy.parquet"  # Replace with your specific file path in S3

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region,
)

# Local temp folder for processing the file
local_temp_file = "temp.parquet"

# Download the specified Parquet file
s3.download_file(Bucket=bucket_name, Key=s3_input_file, Filename=local_temp_file)

# Read the Parquet file into a DataFrame
table = pq.read_table(local_temp_file)
df = table.to_pandas()

# Sample 1% of the data
df_sampled = df.sample(frac=0.01, random_state=42)  # Sampling 1% of the data

# Generate UUID for each row (if not already present)
if "UUID" not in df_sampled.columns:
    df_sampled["UUID"] = [str(uuid.uuid4()) for _ in range(len(df_sampled))]

# Generate embeddings for `reviewText_cleaned` and `summary_cleaned`
df_sampled["reviewText_embedding"] = embedding_model.encode(df_sampled["reviewText_cleaned"].fillna("").tolist()).tolist()
df_sampled["summary_embedding"] = embedding_model.encode(df_sampled["summary_cleaned"].fillna("").tolist()).tolist()

# Display the sampled DataFrame with embeddings
print(df_sampled.head())  # Display the first few rows


# COMMAND ----------

from pyspark.sql import SparkSession
# ... create spark DataFrame ...
spark_df_sampled = spark.createDataFrame(df_sampled)
spark_df_sampled.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## NEW
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import nltk
import spacy

# Downloading stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Loading spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Creating a set of stopwords for faster lookup
stop_words = set(stopwords.words('english'))

# Initializing the Spark session
spark = SparkSession.builder \
    .appName("StopwordRemovalAndLemmatization") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# COMMAND ----------

# Defining the UDF for text preprocessing (stopword removal and lemmatization)
def preprocess_text(text):
    if text is None:  # Handling null values
        return None
    try:
        # Using spaCy for tokenization, lemmatization, and stopword removal
        doc = nlp(text)
        processed_tokens = [
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct and not token.is_space
        ]
        return " ".join(processed_tokens)
    except Exception as e:
        # Logging and handling any processing errors
        print(f"Error processing text: {str(e)}")
        return None

# Registering the UDF with correct data type handling
preprocess_udf = udf(preprocess_text, StringType())

# COMMAND ----------

# Applying the UDF to the relevant columns and ensuring the column names are handled correctly
try:
    processed_df = (
        df.withColumn("cleaned_reviewText", preprocess_udf(col("reviewText_cleaned")))
          .withColumn("cleaned_summary", preprocess_udf(col("summary_cleaned")))
    )
except Exception as e:
    print(f"Error applying UDF: {str(e)}")

# COMMAND ----------

# Dropping unnecessary columns
columns_to_drop = ["reviewText", "summary", "style", "reviewText_cleaned", "summary_cleaned"]
try:
    processed_df = processed_df.drop(*columns_to_drop)
except Exception as e:
    print(f"Error dropping columns: {str(e)}")

# COMMAND ----------

processed_df.printSchema()

# COMMAND ----------

# MAGIC %pip install sentence_transformers

# COMMAND ----------

import pandas as pd
from sentence_transformers import SentenceTransformer
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws
import uuid

# Initialize Spark session (Databricks provides this by default)
spark = SparkSession.builder.appName("Generate Embeddings").getOrCreate()

# Initialize Sentence Transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Define S3 paths
s3_input_folder = "s3://raw-zip-final/Parquet/To_Embeddings/"  # Folder containing multiple Parquet files
s3_output_path = "s3://raw-zip-final/Parquet/Embeddings_Output1/"

# Step 1: Read all Parquet files in the folder into a Spark DataFrame
df = spark.read.parquet(s3_input_folder)

# Step 2: Limit to 1 million rows
df_sampled = df.limit(500000)

# Step 3: Combine `reviewText_cleaned` and `summary_cleaned`
df_combined = df_sampled.withColumn(
    "combined_text", concat_ws(" ", df_sampled.reviewText_cleaned, df_sampled.summary_cleaned)
)

# Step 4: Convert to Pandas for embedding generation
df_pandas = df_combined.select("combined_text").toPandas()

# Step 5: Generate embeddings for combined text
print("Generating embeddings. This might take a while...")
df_pandas["combined_embedding"] = embedding_model.encode(
    df_pandas["combined_text"].fillna("").tolist(), batch_size=64
).tolist()

# Step 6: Add UUIDs to the Pandas DataFrame
df_pandas["UUID"] = [str(uuid.uuid4()) for _ in range(len(df_pandas))]

# Step 7: Convert back to Spark DataFrame
df_with_embeddings = spark.createDataFrame(df_pandas)



# COMMAND ----------


