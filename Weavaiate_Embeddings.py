# Databricks notebook source
# MAGIC %md
# MAGIC Stopword Removal and Lemmatization

# COMMAND ----------

# MAGIC %pip install nltk
# MAGIC %pip install spacy
# MAGIC !python -m spacy download en_core_web_sm

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import nltk
import spacy

# Downloading the stopwords
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
    if text is None:  # To handle null values in the data
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

processed_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC **Embeddings Generation**

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

# Getting meta information about the instance
meta = client.get_meta()
print("\nInstance Information:")
print(meta)

# COMMAND ----------

import weaviate
from weaviate.util import generate_uuid5
import pandas as pd
import time

# Connecting to Weaviate
client = weaviate.Client(
    url="http://50.18.99.196:8080",
)

# Creating the schema to generate embeddings
class_obj = {
    "class": "Review",
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

# Delete an existing schema if it exists
try:
    client.schema.delete_class("Review")
    print("Existing schema deleted")
except:
    pass

# Creating the schema
client.schema.create_class(class_obj)
print("Schema created successfully")

def get_embeddings_and_update_df(spark_df, batch_size=2):
    """Generate embeddings and replace text with vectors"""
    
    # Converting the spark dataframe to pandas
    pandas_df = spark_df.limit(batch_size).toPandas()
    print(f"\nProcessing {len(pandas_df)} records")
    
    # Storing the original texts for reference
    original_texts = []
    
    # Uploading the data to get embeddings
    with client.batch as batch:
        batch.batch_size = batch_size
        
        for _, row in pandas_df.iterrows():
            try:
                # Storing the original texts
                original_texts.append({
                    'review': row['cleaned_reviewText'],
                    'summary': row['cleaned_summary']
                })
                
                # Preparing the properties dictionary
                properties = {}
                for col in pandas_df.columns:
                    if pd.notna(row[col]):
                        if col == 'unixReviewTime':
                            properties[col] = int(row[col])
                        elif col == 'overall':
                            properties[col] = float(row[col])
                        elif col == 'verified':
                            properties[col] = bool(row[col])
                        else:
                            properties[col] = str(row[col])
                
                # Generating UUID
                uuid = generate_uuid5(f"{row['reviewerID']}-{row['asin']}")
                
                # Adding the data object to get embeddings
                batch.add_data_object(
                    data_object=properties,
                    class_name="Review",
                    uuid=uuid
                )
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
    
    # Waiting a moment for indexing to be donee
    time.sleep(2)
    
    # Getting the vectors for the uplooded objects
    print("\nRetrieving generated embeddings...")
    result = (
        client.query
        .get("Review")
        .with_additional(['vector'])
        .with_limit(batch_size)
        .do()
    )
    
    # Creating a new DataFrame with embeddings replacing text
    embedded_df = pandas_df.copy()
    
    if 'data' in result and 'Get' in result['data'] and 'Review' in result['data']['Get']:
        records = result['data']['Get']['Review']
        
        for i, (record, orig_text) in enumerate(zip(records, original_texts)):
            if '_additional' in record and 'vector' in record['_additional']:
                vector = record['_additional']['vector']
                
                # Replacing text with vectors in the DataFrame
                embedded_df.at[i, 'cleaned_reviewText'] = vector
                embedded_df.at[i, 'cleaned_summary'] = vector
                
                # Printing information about the conversion
                print(f"\nRecord {i+1}:")
                print(f"Original Review Text: {orig_text['review'][:100]}...")
                print(f"Original Summary: {orig_text['summary'][:100]}...")
                print(f"Generated Vector dimension: {len(vector)}")
                print(f"First 5 values of vector: {vector[:5]}")
    
    return embedded_df

# Testing with 2 reviews
print("\nGenerating embeddings for test records...")
embedded_df = get_embeddings_and_update_df(processed_df, batch_size=2)

print("\nShape of DataFrame with embeddings:", embedded_df.shape)

# COMMAND ----------

# Checking if the class exists
classes = client.schema.get()
print("Available classes:", [c['class'] for c in classes['classes']])

# COMMAND ----------

# Fetching the schema and print it
schema = client.schema.get()

# Printing the schema
print("\nWeaviate Schema:")
for class_obj in schema['classes']:
    print(f"Class: {class_obj['class']}")
    for prop in class_obj['properties']:
        print(f"  Property: {prop['name']}, Type: {prop['dataType']}")
