# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, udf, sha2, concat_ws, trim
from pyspark.sql.types import StringType
import re

# Initialize Spark Session (if not already running)
spark = SparkSession.builder.appName("TextCleaningAndDeterministicUUID").getOrCreate()

# Define input and output paths
input_parquet_path = "s3://raw-zip-final/Parquet/Final_Data/"  # Change to your Parquet file location

# Load the Parquet data
df = spark.read.parquet(input_parquet_path)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, udf, sha2, concat_ws, trim
from pyspark.sql.types import StringType
import re

# Initialize Spark Session (if not already running)
spark = SparkSession.builder.appName("TextCleaningAndDeterministicUUID").getOrCreate()

# Define input and output paths
input_parquet_path = "s3://raw-zip-final/Parquet/filtered_parquet_files/"  # Change to your Parquet file location
output_parquet_path = "s3://raw-zip-final/Parquet/deduplicated_Cleaned_files/"  # Change to your desired output location

# Load the Parquet data
df = spark.read.parquet(input_parquet_path)

# Define a Python function for text cleaning
def clean_text(text):
    if not text:
        return None
    # Remove emojis
    emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Remove URLs
    text = re.sub(r"http\S+|www.\S+", "", text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters (except alphanumeric and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Trim leading/trailing spaces
    return text.strip()

# Register the Python function as a UDF
clean_text_udf = udf(clean_text, StringType())

# Create a deterministic UUID using SHA2 hashing of the concatenated key columns
# Combine reviewerID, asin, and unixReviewTime as a unique identifier
uuid_udf = sha2(concat_ws("||", col("reviewerID"), col("asin"), col("unixReviewTime")), 256)

# Apply text cleaning and add deterministic UUID
cleaned_df = (
    df.withColumn("reviewText_cleaned", clean_text_udf(col("reviewText")))
      .withColumn("summary_cleaned", clean_text_udf(col("summary")))
      .withColumn("UUID", uuid_udf)  # Deterministic UUID based on the 3 columns
)

# Write the cleaned data with UUID back to Parquet
cleaned_df.write.mode("overwrite").parquet(output_parquet_path)

print(f"Cleaned data with UUIDs written to {output_parquet_path}")


# COMMAND ----------

cleaned_df.display()

# COMMAND ----------

from pyspark.sql.functions import col, count

# Check for null values in the reviewText_cleaned column
null_count = cleaned_df.filter(col("UUID").isNull()).count()

print(f"Total null values in reviewText_cleaned column: {null_count}")



# COMMAND ----------

# Replace all null values with 'NaN'
cleaned_df_with_nan = cleaned_df.fillna('NaN')

# COMMAND ----------

cleaned_df_with_nan.display()

# COMMAND ----------

import datetime

# Generate a unique folder name based on the current timestamp
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"s3://raw-zip-final/Parquet/Final_Data/"  # New folder with timestamp

# Save the DataFrame as Parquet files
cleaned_df_with_nan.write.parquet(output_path)

print(f"Data with nulls replaced saved to {output_path}")


# COMMAND ----------

cleaned_df_with_nan.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stopwords and Lemmatizing "Final_Data"

# COMMAND ----------

pip install nltk

# COMMAND ----------

import os

# Install the spaCy model
os.system("python -m spacy download en_core_web_sm")

# COMMAND ----------

pip install spacy

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import nltk
import spacy

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Create a list of stopwords
stop_words = set(stopwords.words('english'))

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StopwordRemovalAndLemmatization") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Load Parquet files from S3
input_path = "s3://raw-zip-final/Parquet/Final_Data/"
df = spark.read.parquet(input_path)

# Define UDF for text preprocessing (stopword removal and lemmatization)
def preprocess_text(text):
    if text:
        # Use spaCy for lemmatization and stopword removal
        doc = nlp(text)
        processed_tokens = [
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct
        ]
        return " ".join(processed_tokens)
    return None

# Register the UDF
preprocess_udf = udf(preprocess_text, StringType())

# Apply the UDF to both reviewText and summary columns
processed_df = df.withColumn("cleaned_reviewText", preprocess_udf(df["reviewText"])) \
                 .withColumn("cleaned_summary", preprocess_udf(df["summary"]))

# Drop the specified columns
columns_to_drop = ["reviewText", "style", "summary", "reviewText_cleaned", "summary_cleaned"]
processed_df = processed_df.drop(*columns_to_drop)

# Save the resulting DataFrame back to S3
output_path = "s3://processed-data-final/Parquet/Final_Processed_Data/"
processed_df.write.mode("overwrite").parquet(output_path)

# Stop Spark session
spark.stop()

print(f"Processing complete! Processed data saved to: {output_path}")



# COMMAND ----------

# Ensure required NLTK resources are downloaded and accessible
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


# COMMAND ----------

# MAGIC %pip install nltk spacy
# MAGIC

# COMMAND ----------

import nltk
nltk.download('stopwords')
nltk.download('punkt')

import spacy
!python -m spacy download en_core_web_sm


# COMMAND ----------

df = spark.read.parquet("s3://raw-zip-final/Parquet/Final_Data/")


# COMMAND ----------

df.display(10)

# COMMAND ----------

!pip install nltk
from nltk.corpus import stopwords
from spacy.lang.en import English
import string

# Load Spacy English tokenizer
nlp = English()
nlp.add_pipe('lemmatizer', config={"mode": "lookup"})  # Use lookup lemmatizer
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and punctuations
    tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    # Lemmatization
    lemmatized = [nlp(token)[0].lemma_ for token in tokens]
    return ' '.join(lemmatized)


# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

clean_udf = udf(clean_text, StringType())

# Applying the UDF
df = df.withColumn("reviewText_cleaned", clean_udf(df["reviewText_cleaned"]))
df = df.withColumn("summary_cleaned", clean_udf(df["summary_cleaned"]))


# COMMAND ----------

import requests
import json
from typing import Dict, Optional, List
import pandas as pd

class WeaviateConnection:
    def __init__(self, ec2_ip: str = "50.18.99.196", port: str = "8080"):
        """
        Initialize connection to Weaviate running in Docker on EC2.
        
        This method sets up the base URL and headers for the Weaviate connection.
        The EC2 IP address and port are used to construct the base URL, which will
        be used for all API calls to Weaviate.
        
        :param ec2_ip: The public IP of the EC2 instance where Weaviate is running.
        :param port: The port on which Weaviate's HTTP API is accessible (default is 8080).
        """
        self.base_url = f"http://{ec2_ip}:{port}"
        self.headers = {
            "Content-Type": "application/json"
        }
        


# COMMAND ----------

connection = WeaviateConnection(ec2_ip="50.18.99.196", port="8080")


# COMMAND ----------

print(connection.base_url)
# Output: http://50.18.99.196:8080

print(connection.headers)
# Output: {'Content-Type': 'application/json'}


# COMMAND ----------


