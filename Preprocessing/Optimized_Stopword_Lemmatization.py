# Databricks notebook source
# MAGIC %pip install nltk

# COMMAND ----------

# MAGIC %pip install spacy

# COMMAND ----------

# MAGIC %pip install numpy==1.23.5

# COMMAND ----------

import numpy
print(numpy.__version__)

# COMMAND ----------

!python -m spacy download en_core_web_sm

# COMMAND ----------

dbutils.library.restartPython()

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
    if text is None: 
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

processed_df.printSchema()

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

# Viewing the execution plan and looking for partitioning information
processed_df.explain(True)

# COMMAND ----------

# Saving the resulting DataFrame back to S3
output_path = "s3://raw-zip-final/Parquet/Final_Processed_Data/"
processed_df.write.mode("overwrite").parquet(output_path)

print(f"Processing complete! Processed data saved to: {output_path}")
