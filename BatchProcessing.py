# Databricks notebook source
# MAGIC %pip install nltk
# MAGIC %pip install spacy
# MAGIC !python -m spacy download en_core_web_sm

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

# Loading Parquet files from S3
input_path = "s3://raw-zip-final/Parquet/Final_Data/"
df = spark.read.parquet(input_path)

# Checking the schema to verify columns
df.printSchema()

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

# Checking for missing or erroneous data after processing
processed_df.select("cleaned_reviewText", "cleaned_summary").display(5, truncate=False)

# COMMAND ----------

# Dropping unnecessary columns
columns_to_drop = ["reviewText", "summary", "style", "reviewText_cleaned", "summary_cleaned"]
try:
    processed_df = processed_df.drop(*columns_to_drop)
except Exception as e:
    print(f"Error dropping columns: {str(e)}")

# COMMAND ----------

# MAGIC %pip install protobuf==3.20.0
# MAGIC %pip install weaviate-client

# COMMAND ----------

!pip install --upgrade protobuf
!pip install weaviate-client sentence-transformers tqdm

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import weaviate
client = weaviate.Client(url="http://50.18.99.196:8080")

# Get meta information about the instance
meta = client.get_meta()
print("\nInstance Information:")
print(meta)

# COMMAND ----------

# Check if the class exists
classes = client.schema.get()
print("Available classes:", [c['class'] for c in classes['classes']])

# COMMAND ----------

import weaviate
import time
import pandas as pd

# Use v3 client explicitly
client = weaviate.Client(
    url="http://50.18.99.196:8080",
    timeout_config=(60, 60)
)

def test_batch_behavior(spark_df, batch_size=20):
    try:
        # Get both batches at once to avoid Spark overhead
        all_rows = spark_df.limit(batch_size * 2).toPandas()
        
        # Process first batch
        print("Testing first batch...")
        start_time = time.time()
        
        with client.batch as batch:
            for idx, row in all_rows.iloc[:batch_size].iterrows():
                properties = {
                    col: (int(row[col]) if col == 'unixReviewTime'
                          else float(row[col]) if col == 'overall'
                          else bool(row[col]) if col == 'verified'
                          else str(row[col]))
                    for col in all_rows.columns
                    if pd.notna(row[col])
                }
                batch.add_data_object(
                    data_object=properties,
                    class_name="Data228_Review"
                )
                print(f"First batch: Processed row {idx + 1}")
                
        print(f"First batch completed in {time.time() - start_time:.2f} seconds")
        
        # Short pause
        print("Waiting 5 seconds before second batch...")
        time.sleep(5)
        
        # Process second batch
        print("\nTesting second batch...")
        start_time = time.time()
        
        with client.batch as batch:
            for idx, row in all_rows.iloc[batch_size:].iterrows():
                properties = {
                    col: (int(row[col]) if col == 'unixReviewTime'
                          else float(row[col]) if col == 'overall'
                          else bool(row[col]) if col == 'verified'
                          else str(row[col]))
                    for col in all_rows.columns
                    if pd.notna(row[col])
                }
                batch.add_data_object(
                    data_object=properties,
                    class_name="Data228_Review"
                )
                print(f"Second batch: Processed row {idx + 1}")
                
        print(f"Second batch completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")

# Run the test
test_batch_behavior(processed_df)

# COMMAND ----------

import weaviate
import pandas as pd
import time
import warnings
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from pyspark.sql import SparkSession

def setup_logging():
    logging.basicConfig(
        filename=f'weaviate_import_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

def create_spark_session():
    try:
        spark = (SparkSession.builder
                 .appName("DataBatchImport")
                 .config("spark.sql.shuffle.partitions", "56")
                 .config("spark.default.parallelism", "56")
                 .getOrCreate())
        logging.info("Spark session created successfully")
        return spark
    except Exception as e:
        logging.error(f"Failed to create Spark session: {e}")
        raise

def configure_spark():
    spark = create_spark_session()

def create_weaviate_client():
    try:
        client = weaviate.Client(
            url="http://50.18.99.196:8080",
            timeout_config=(300, 300)
        )
        logging.info("Weaviate client created successfully")
        return client
    except Exception as e:
        logging.error(f"Failed to create Weaviate client: {e}")
        raise

class_obj = {
    "class": "Data228_Review",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {
        "text2vec-transformers": {
            "vectorizeClassName": False,
            "textFields": ["cleaned_reviewText", "cleaned_summary"]
        }
    },
    "properties": [
        {
            "name": "cleaned_reviewText",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-transformers": {
                    "skip": False,
                    "vectorizePropertyName": False
                }
            }
        },
        {
            "name": "cleaned_summary",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-transformers": {
                    "skip": False,
                    "vectorizePropertyName": False
                }
            }
        },
        {"name": "asin", "dataType": ["string"]},
        {"name": "overall", "dataType": ["number"]},
        {"name": "reviewerID", "dataType": ["string"]},
        {"name": "reviewerName", "dataType": ["string"]},
        {"name": "unixReviewTime", "dataType": ["int"]},
        {"name": "verified", "dataType": ["boolean"]},
        {"name": "vote", "dataType": ["string"]},
        {"name": "style_size", "dataType": ["string"]},
        {"name": "style_flavor", "dataType": ["string"]},
        {"name": "style_style", "dataType": ["string"]},
        {"name": "style_color", "dataType": ["string"]},
        {"name": "style_edition", "dataType": ["string"]},
        {"name": "style_package_type", "dataType": ["string"]},
        {"name": "style_number_of_items", "dataType": ["string"]},
        {"name": "style_size_per_pearl", "dataType": ["string"]},
        {"name": "style_color_name", "dataType": ["string"]},
        {"name": "style_size_name", "dataType": ["string"]},
        {"name": "style_metal_type", "dataType": ["string"]},
        {"name": "style_style_name", "dataType": ["string"]},
        {"name": "style_flavor_name", "dataType": ["string"]},
        {"name": "style_length", "dataType": ["string"]},
        {"name": "style_package_quantity", "dataType": ["string"]},
        {"name": "style_design", "dataType": ["string"]},
        {"name": "style_platform", "dataType": ["string"]},
        {"name": "style_item_package_quantity", "dataType": ["string"]},
        {"name": "style_wattage", "dataType": ["string"]},
        {"name": "style_pattern", "dataType": ["string"]},
        {"name": "style_material_type", "dataType": ["string"]},
        {"name": "style_team_name", "dataType": ["string"]},
        {"name": "style_item_shape", "dataType": ["string"]},
        {"name": "style_hand_orientation", "dataType": ["string"]},
        {"name": "style_flex", "dataType": ["string"]},
        {"name": "style_material", "dataType": ["string"]},
        {"name": "style_shaft_material", "dataType": ["string"]},
        {"name": "style_configuration", "dataType": ["string"]},
        {"name": "style_capacity", "dataType": ["string"]},
        {"name": "style_product_packaging", "dataType": ["string"]},
        {"name": "style_offer_type", "dataType": ["string"]},
        {"name": "style_model", "dataType": ["string"]},
        {"name": "style_overall_length", "dataType": ["string"]},
        {"name": "style_width", "dataType": ["string"]},
        {"name": "style_grip_type", "dataType": ["string"]},
        {"name": "style_gem_type", "dataType": ["string"]},
        {"name": "style_scent_name", "dataType": ["string"]},
        {"name": "style_model_number", "dataType": ["string"]},
        {"name": "style_connectivity", "dataType": ["string"]},
        {"name": "style_digital_storage_capacity", "dataType": ["string"]},
        {"name": "style_subscription_length", "dataType": ["string"]},
        {"name": "style_primary_stone_gem_type", "dataType": ["string"]},
        {"name": "style_item_display_weight", "dataType": ["string"]},
        {"name": "style_gift_amount", "dataType": ["string"]},
        {"name": "style_format_type", "dataType": ["string"]},
        {"name": "UUID", "dataType": ["string"]}
    ]
}

def create_schema():
    client.schema.delete_class("Data228_Review")
    client.schema.create_class(class_obj)

def process_data_sequentially():

    print("Starting sequential processing...")
    total_rows = processed_df.count()
    print(f"Total rows to process: {total_rows}")
    
    total_rows = processed_df.count()
    
    batch_size = 1000
    processed = 0
    start_time = time.time()
    
    try:
        while processed < total_rows:
            batch_df = (processed_df
                       .offset(processed)
                       .limit(batch_size)
                       .toPandas())
            
            with client.batch as batch:
                for idx, row in batch_df.iterrows():
                    properties = {
                        col: (int(row[col]) if col == 'unixReviewTime'
                              else float(row[col]) if col == 'overall'
                              else bool(row[col]) if col == 'verified'
                              else str(row[col]))
                        for col in batch_df.columns
                        if pd.notna(row[col])
                    }
                    batch.add_data_object(
                        data_object=properties,
                        class_name="Data228_Review"
                    )
            
            processed += len(batch_df)
            elapsed = time.time() - start_time
            rate = processed / elapsed
            
            print(f"\nProgress Update:")
            print(f"- Processed: {processed:,}/{total_rows:,} ({processed/total_rows*100:.4f}%)")
            print(f"- Time elapsed: {elapsed/3600:.2f} hours")
            print(f"- Rate: {rate:.2f} rows/second")
            print(f"- Estimated remaining: {(total_rows-processed)/rate/3600:.2f} hours")
            time.sleep(5)
            
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise

setup_logging()

spark = create_spark_session() 
configure_spark()

client = create_weaviate_client()

create_schema()

process_data_sequentially()

client.close()
spark.stop()

# COMMAND ----------

def process_data_sequentially():
    print("Starting sequential processing...")
    total_rows = processed_df.count()
    print(f"Total rows to process: {total_rows}")
    
    batch_size = 1000  # Increased from 20 to 1000
    processed = 0
    start_time = time.time()
    
    try:
        while processed < total_rows:
            batch_df = (processed_df
                       .offset(processed)
                       .limit(batch_size)
                       .toPandas())
            
            with client.batch as batch:
                for idx, row in batch_df.iterrows():
                    properties = {
                        col: (int(row[col]) if col == 'unixReviewTime'
                              else float(row[col]) if col == 'overall'
                              else bool(row[col]) if col == 'verified'
                              else str(row[col]))
                        for col in batch_df.columns
                        if pd.notna(row[col])
                    }
                    batch.add_data_object(
                        data_object=properties,
                        class_name="Data228_Review"
                    )
            
            processed += len(batch_df)
            elapsed = time.time() - start_time
            rate = processed / elapsed
            
            print(f"\nProgress Update:")
            print(f"- Processed: {processed:,}/{total_rows:,} ({processed/total_rows*100:.4f}%)")
            print(f"- Time elapsed: {elapsed/3600:.2f} hours")
            print(f"- Rate: {rate:.2f} rows/second")
            print(f"- Estimated remaining: {(total_rows-processed)/rate/3600:.2f} hours")
            time.sleep(5)
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

process_data_sequentially()
