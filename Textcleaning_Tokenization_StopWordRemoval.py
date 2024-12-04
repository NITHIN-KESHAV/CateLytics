# Databricks notebook source
spark.conf.set("fs.s3a.access.key", "AKIAVYV52E7ZRAA4ISFA")
spark.conf.set("fs.s3a.secret.key", "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")
spark.conf.set("fs.s3a.endpoint", "s3.amazonaws.com")


# COMMAND ----------

# Install nltk
%pip install nltk


# COMMAND ----------

import nltk

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# COMMAND ----------

import nltk
print("NLTK is successfully installed and imported.")


# COMMAND ----------

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_and_tokenize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# COMMAND ----------

pip install awscli


# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import boto3
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Initialize Spark session
spark = SparkSession.builder.appName('TextProcessing').getOrCreate()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define S3 bucket and Parquet file path
bucket_name = 'raw-zip-final'
file_key = 'Parquet/deduplicated_files/_committed_5145292485103526270'

# Define schema (you need to adjust this based on your actual Parquet structure)
schema = StructType([
    StructField("text_column", StringType(), True),  # Replace 'text_column' with actual column name
])

# Read the Parquet file from S3 with the defined schema
df = spark.read.schema(schema).parquet(f"s3a://{bucket_name}/{file_key}")

# Function to clean and process text
def clean_and_process_text(text):
    if text is not None:
        # Convert text to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords manually using NLTK
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        return lemmatized_tokens
    else:
        return []

# Register UDF (User Defined Function) to use in Spark DataFrame
clean_and_process_udf = F.udf(clean_and_process_text, ArrayType(StringType()))

# Apply UDF to the DataFrame to process the text column
df_cleaned = df.withColumn('cleaned_text', clean_and_process_udf(F.col('text_column')))

# Show the cleaned DataFrame (with lemmatized text)
df_cleaned.show(truncate=False)

# If you need to perform additional stopword removal step, you can apply it manually:
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Register UDF for stopword removal
remove_stopwords_udf = F.udf(remove_stopwords, ArrayType(StringType()))

# Apply UDF to remove stopwords
df_final = df_cleaned.withColumn('final_cleaned_text', remove_stopwords_udf(F.col('cleaned_text')))

# Show the final cleaned DataFrame with removed stopwords
df_final.select('final_cleaned_text').show(truncate=False)





# COMMAND ----------

from pyspark.sql.functions import concat_ws

# Convert ARRAY<STRING> columns into a single string column
df_final = df_final.withColumn("cleaned_text", concat_ws(" ", df_final["cleaned_text"]))
df_final = df_final.withColumn("final_cleaned_text", concat_ws(" ", df_final["final_cleaned_text"]))

# Now, you can write the DataFrame to a CSV
df_final.write.option("header", "true").csv("/dbfs/tmp/final_cleaned_text.csv")





# COMMAND ----------

# Check if the file exists in the target location
dbutils.fs.ls("/dbfs/tmp/")



# COMMAND ----------

# Write the DataFrame to a CSV file in DBFS (Databricks File System) with overwrite mode
df_final.write.option("header", "true").mode("overwrite").csv("/dbfs/tmp/final_cleaned_text.csv")


# COMMAND ----------

# Copy the file from DBFS to a local directory (adjust path as needed)
dbutils.fs.cp("dbfs:/tmp/final_cleaned_text.csv", "file:/D:/Downloads/final_cleaned_text.csv")


# COMMAND ----------

import boto3
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Initialize Spark session
spark = SparkSession.builder.appName('TextProcessing').getOrCreate()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define S3 bucket and Parquet file path
bucket_name = 'raw-zip-final'
file_key = 'Parquet/deduplicated_files/_committed_5145292485103526270'

# Define schema (you need to adjust this based on your actual Parquet structure)
schema = StructType([
    StructField("reviewText", StringType(), True),  # Column for review text
    StructField("summary", StringType(), True),    # Column for summary
])

# Read the Parquet file from S3 with the defined schema
df = spark.read.schema(schema).parquet(f"s3a://{bucket_name}/{file_key}")

# Function to clean and process text
def clean_and_process_text(text):
    if text is not None:
        # Convert text to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords manually using NLTK
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        return lemmatized_tokens
    else:
        return []

# Register UDF (User Defined Function) to use in Spark DataFrame
clean_and_process_udf = F.udf(clean_and_process_text, ArrayType(StringType()))

# Apply UDF to both 'reviewText' and 'summary' columns
df_cleaned = df.withColumn('cleaned_reviewText', clean_and_process_udf(F.col('reviewText')))
df_cleaned = df_cleaned.withColumn('cleaned_summary', clean_and_process_udf(F.col('summary')))

# Show the cleaned DataFrame (with lemmatized text)
df_cleaned.show(truncate=False)

# If you need to perform additional stopword removal step, you can apply it manually:
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Register UDF for stopword removal
remove_stopwords_udf = F.udf(remove_stopwords, ArrayType(StringType()))

# Apply UDF to remove stopwords from 'cleaned_reviewText' and 'cleaned_summary'
df_final = df_cleaned.withColumn('final_cleaned_reviewText', remove_stopwords_udf(F.col('cleaned_reviewText')))
df_final = df_final.withColumn('final_cleaned_summary', remove_stopwords_udf(F.col('cleaned_summary')))

# Show the final cleaned DataFrame with removed stopwords along with content in rows
df_final.select('reviewText', 'summary', 'final_cleaned_reviewText', 'final_cleaned_summary').show(truncate=False)



# COMMAND ----------

import boto3
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Initialize Spark session
spark = SparkSession.builder.appName('TextProcessing').getOrCreate()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define S3 bucket and Parquet file path
bucket_name = 'raw-zip-final'
file_key = 'Parquet/deduplicated_files/_committed_5145292485103526270'

# Define schema (you need to adjust this based on your actual Parquet structure)
schema = StructType([
    StructField("reviewText", StringType(), True),  # Column for review text
    StructField("summary", StringType(), True),    # Column for summary
])

# Read the Parquet file from S3 with the defined schema
df = spark.read.schema(schema).parquet(f"s3a://{bucket_name}/{file_key}")

# Show the first few rows to check if data is loaded correctly
df.show(5, truncate=False)

# Function to clean and process text
def clean_and_process_text(text):
    if text is not None and len(text) > 0:  # Ensure text is not empty or None
        # Convert text to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords manually using NLTK
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        return lemmatized_tokens
    else:
        return []

# Register UDF (User Defined Function) to use in Spark DataFrame
clean_and_process_udf = F.udf(clean_and_process_text, ArrayType(StringType()))

# Apply UDF to both 'reviewText' and 'summary' columns
df_cleaned = df.withColumn('cleaned_reviewText', clean_and_process_udf(F.col('reviewText')))
df_cleaned = df_cleaned.withColumn('cleaned_summary', clean_and_process_udf(F.col('summary')))

# Show the cleaned DataFrame to check if the processing is working
df_cleaned.show(5, truncate=False)

# If you need to perform additional stopword removal step, you can apply it manually:
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Register UDF for stopword removal
remove_stopwords_udf = F.udf(remove_stopwords, ArrayType(StringType()))

# Apply UDF to remove stopwords from 'cleaned_reviewText' and 'cleaned_summary'
df_final = df_cleaned.withColumn('final_cleaned_reviewText', remove_stopwords_udf(F.col('cleaned_reviewText')))
df_final = df_final.withColumn('final_cleaned_summary', remove_stopwords_udf(F.col('cleaned_summary')))

# Show the final cleaned DataFrame with removed stopwords along with content in rows
df_final.select('reviewText', 'summary', 'final_cleaned_reviewText', 'final_cleaned_summary').show(5, truncate=False)


# COMMAND ----------

import boto3
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType

# Initialize Spark session
spark = SparkSession.builder.appName('TextProcessing').getOrCreate()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define the S3 bucket and the base parquet path
bucket_name = 'raw-zip-final'
parquet_path_prefix = 'Parquet/deduplicated_files/'

# Corrected input path (make sure this points to the correct location)
input_path = 'Parquet/deduplicated_files/'  # Update this to your actual path

# Output path for processed files
output_path = f"s3a://{bucket_name}/processed/textcleaned_deduplicated"

# Read the Parquet files from the input directory
df = spark.read.parquet(f"s3a://{bucket_name}/{input_path}")

# Function to clean and process text
def clean_and_process_text(text):
    if text is not None:
        # Convert text to lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords manually using NLTK
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        return lemmatized_tokens
    else:
        return []

# Register UDF (User Defined Function) to use in Spark DataFrame
clean_and_process_udf = F.udf(clean_and_process_text, ArrayType(StringType()))

# Apply UDF to the DataFrame to process the text columns (e.g., reviewText and summary)
df_cleaned = df.withColumn('cleaned_reviewText', clean_and_process_udf(F.col('reviewText'))) \
               .withColumn('cleaned_summary', clean_and_process_udf(F.col('summary')))

# Show the cleaned DataFrame (with lemmatized text)
df_cleaned.show(truncate=False)

# If you need to perform additional stopword removal step, you can apply it manually:
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Register UDF for stopword removal
remove_stopwords_udf = F.udf(remove_stopwords, ArrayType(StringType()))

# Apply UDF to remove stopwords from the cleaned text
df_final = df_cleaned.withColumn('final_cleaned_reviewText', remove_stopwords_udf(F.col('cleaned_reviewText'))) \
                     .withColumn('final_cleaned_summary', remove_stopwords_udf(F.col('cleaned_summary')))

# Show the final cleaned DataFrame with removed stopwords
df_final.select('final_cleaned_reviewText', 'final_cleaned_summary').show(truncate=False)

# Write the cleaned DataFrame to S3 in the processed directory
df_final.write.parquet(output_path, mode='overwrite')




# COMMAND ----------

import nltk
nltk.download('all')


# COMMAND ----------

import nltk
nltk.data.path.append('/dbfs/path/to/nltk_data')


# COMMAND ----------

pip install --upgrade nltk


# COMMAND ----------

from nltk.corpus import stopwords
stopwords.words('english')


# COMMAND ----------

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print(stopwords.words('english')[:10])


# COMMAND ----------

from pyspark import SparkContext
sc = SparkContext.getOrCreate()

stopwords_broadcast = sc.broadcast(stopwords.words('english'))


# COMMAND ----------

import nltk
nltk.download('punkt')


# COMMAND ----------

nltk.data.path.append('/path/to/nltk_data')


# COMMAND ----------

import nltk
print(nltk.data.path)


# COMMAND ----------

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Check data paths
print(nltk.data.path)

# Your existing processing code continues here...


# COMMAND ----------

import re

def simple_tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize Spark Session (if not already running)
spark = SparkSession.builder.appName("TextCleaningTokenizationAndStopwordRemoval").getOrCreate()

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Register the Python function as a UDF for cleaning text
clean_text_udf = udf(clean_text, StringType())

# Tokenization function: Tokenize text into words
def tokenize_text(text):
    if not text:
        return []
    # Tokenize the text using NLTK's word_tokenize
    return word_tokenize(text)

# Register the tokenization function as a UDF
tokenize_text_udf = udf(tokenize_text, ArrayType(StringType()))

# Remove stopwords function: Remove common stopwords
def remove_stopwords(tokens):
    if not tokens:
        return []
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Register the stopwords removal function as a UDF
remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))

# Lemmatization function: Lemmatize tokens
def lemmatize_tokens(tokens):
    if not tokens:
        return []
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

# Register the lemmatization function as a UDF
lemmatize_tokens_udf = udf(lemmatize_tokens, ArrayType(StringType()))

# Apply text cleaning, tokenization, stopword removal, and lemmatization
cleaned_df = (
    df.withColumn("reviewText_cleaned", clean_text_udf(col("reviewText")))
      .withColumn("summary_cleaned", clean_text_udf(col("summary")))
      .withColumn("reviewText_tokens", tokenize_text_udf(col("reviewText_cleaned")))
      .withColumn("summary_tokens", tokenize_text_udf(col("summary_cleaned")))
      .withColumn("reviewText_no_stopwords", remove_stopwords_udf(col("reviewText_tokens")))
      .withColumn("summary_no_stopwords", remove_stopwords_udf(col("summary_tokens")))
      .withColumn("reviewText_lemmatized", lemmatize_tokens_udf(col("reviewText_no_stopwords")))
      .withColumn("summary_lemmatized", lemmatize_tokens_udf(col("summary_no_stopwords")))
)

# Write the cleaned data back to Parquet
cleaned_df.write.mode("overwrite").parquet(output_parquet_path)

print(f"Cleaned data with tokens, lemmatized text written to {output_parquet_path}")


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize Spark Session
spark = SparkSession.builder.appName("TextCleaningTokenizationAndStopwordRemoval").getOrCreate()

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define input and output S3 paths
input_parquet_path = "s3://raw-zip-final/Parquet/"
output_parquet_path = "s3://raw-zip-final/Parquet/deduplicated_Cleaned_files/"

# Load the Parquet data from S3
df = spark.read.parquet(input_parquet_path)

# Define a Python function for text cleaning
def clean_text(text):
    if not text or text.strip() == "":
        return None
    emoji_pattern = re.compile("[" 
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Remove emojis
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces
    return text.strip()  # Trim spaces

# Register the Python function as a UDF
clean_text_udf = udf(clean_text, StringType())

# Tokenization function
def tokenize_text(text):
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return []

tokenize_text_udf = udf(tokenize_text, ArrayType(StringType()))

# Stopword removal function
def remove_stopwords(tokens):
    if not tokens:
        return []
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))

# Lemmatization function
def lemmatize_tokens(tokens):
    if not tokens:
        return []
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

lemmatize_tokens_udf = udf(lemmatize_tokens, ArrayType(StringType()))

# Apply text cleaning, tokenization, stopword removal, and lemmatization
try:
    cleaned_df = (
        df.withColumn("reviewText_cleaned", clean_text_udf(col("reviewText")))
          .withColumn("summary_cleaned", clean_text_udf(col("summary")))
          .withColumn("reviewText_tokens", tokenize_text_udf(col("reviewText_cleaned")))
          .withColumn("summary_tokens", tokenize_text_udf(col("summary_cleaned")))
          .withColumn("reviewText_no_stopwords", remove_stopwords_udf(col("reviewText_tokens")))
          .withColumn("summary_no_stopwords", remove_stopwords_udf(col("summary_tokens")))
          .withColumn("reviewText_lemmatized", lemmatize_tokens_udf(col("reviewText_no_stopwords")))
          .withColumn("summary_lemmatized", lemmatize_tokens_udf(col("summary_no_stopwords")))
    )

    # Write the cleaned data back to S3 as Parquet
    cleaned_df.write.mode("overwrite").parquet(output_parquet_path)
    print(f"Cleaned data written to {output_parquet_path}")

except Exception as e:
    print(f"Error processing data: {str(e)}")



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize Spark Session with S3 support (ensure Hadoop AWS and AWS SDK are available)
spark = SparkSession.builder \
    .appName("TextCleaningTokenizationAndStopwordRemoval") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0") \
    .getOrCreate()

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define input and output S3 paths
input_parquet_path = "s3://raw-zip-final/Parquet/"
output_parquet_path = "s3://raw-zip-final/Parquet/deduplicated_Cleaned_files/"

# Define the schema explicitly for the expected structure of the Parquet file
schema = StructType([
    StructField("reviewText", StringType(), True),
    StructField("summary", StringType(), True),
    # Add any other fields based on your data structure
])

# Load the Parquet data from S3 with explicit schema
try:
    df = spark.read.schema(schema).parquet(input_parquet_path)
    df.printSchema()  # Check the schema of the loaded DataFrame
except Exception as e:
    print(f"Error reading Parquet data: {str(e)}")

# Define a Python function for text cleaning
def clean_text(text):
    if not text or text.strip() == "":
        return None
    emoji_pattern = re.compile("[" 
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Remove emojis
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces
    return text.strip()  # Trim spaces

# Register the Python function as a UDF
clean_text_udf = udf(clean_text, StringType())

# Tokenization function
def tokenize_text(text):
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return []

tokenize_text_udf = udf(tokenize_text, ArrayType(StringType()))

# Stopword removal function
def remove_stopwords(tokens):
    if not tokens:
        return []
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))

# Lemmatization function
def lemmatize_tokens(tokens):
    if not tokens:
        return []
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

lemmatize_tokens_udf = udf(lemmatize_tokens, ArrayType(StringType()))

# Apply text cleaning, tokenization, stopword removal, and lemmatization
try:
    cleaned_df = (
        df.withColumn("reviewText_cleaned", clean_text_udf(col("reviewText")))
          .withColumn("summary_cleaned", clean_text_udf(col("summary")))
          .withColumn("reviewText_tokens", tokenize_text_udf(col("reviewText_cleaned")))
          .withColumn("summary_tokens", tokenize_text_udf(col("summary_cleaned")))
          .withColumn("reviewText_no_stopwords", remove_stopwords_udf(col("reviewText_tokens")))
          .withColumn("summary_no_stopwords", remove_stopwords_udf(col("summary_tokens")))
          .withColumn("reviewText_lemmatized", lemmatize_tokens_udf(col("reviewText_no_stopwords")))
          .withColumn("summary_lemmatized", lemmatize_tokens_udf(col("summary_no_stopwords")))
    )

    # Write the cleaned data back to S3 as Parquet
    cleaned_df.write.mode("overwrite").parquet(output_parquet_path)
    print(f"Cleaned data written to {output_parquet_path}")

except Exception as e:
    print(f"Error processing data: {str(e)}")



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize Spark Session with S3 support (ensure Hadoop AWS and AWS SDK are available)
spark = SparkSession.builder \
    .appName("TextCleaningTokenizationAndStopwordRemoval") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0") \
    .getOrCreate()

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define input and output S3 paths
input_parquet_path = "s3://raw-zip-final/Parquet/"
output_parquet_path = "s3://raw-zip-final/Parquet/deduplicated_Cleaned_files/"

# Define the schema explicitly for the expected structure of the Parquet file
schema = StructType([
    StructField("reviewText", StringType(), True),
    StructField("summary", StringType(), True),
])

# Load the Parquet data from S3 with explicit schema
try:
    df = spark.read.schema(schema).parquet(input_parquet_path)
    df.printSchema()  # Check the schema of the loaded DataFrame
    df.show(5)  # Show a few records to inspect the data
except Exception as e:
    print(f"Error reading Parquet data: {str(e)}")

# Define a Python function for text cleaning
def clean_text(text):
    if not text or text.strip() == "":
        return None
    emoji_pattern = re.compile("[" 
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)  # Remove emojis
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces
    return text.strip()  # Trim spaces

# Register the Python function as a UDF
clean_text_udf = udf(clean_text, StringType())

# Tokenization function
def tokenize_text(text):
    if not text:
        return []
    try:
        return word_tokenize(text)
    except Exception:
        return []

tokenize_text_udf = udf(tokenize_text, ArrayType(StringType()))

# Stopword removal function
def remove_stopwords(tokens):
    if not tokens:
        return []
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

remove_stopwords_udf = udf(remove_stopwords, ArrayType(StringType()))

# Lemmatization function
def lemmatize_tokens(tokens):
    if not tokens:
        return []
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

lemmatize_tokens_udf = udf(lemmatize_tokens, ArrayType(StringType()))

# Debug: Check if reviewText and summary columns have data
df.select("reviewText", "summary").show(5)

# Apply text cleaning, tokenization, stopword removal, and lemmatization
try:
    cleaned_df = (
        df.withColumn("reviewText_cleaned", clean_text_udf(col("reviewText")))
          .withColumn("summary_cleaned", clean_text_udf(col("summary")))
    )

    # Debug: Check intermediate results after cleaning
    cleaned_df.select("reviewText_cleaned", "summary_cleaned").show(5)

    # Apply further transformations (tokenization, stopword removal, and lemmatization)
    cleaned_df = (
        cleaned_df
          .withColumn("reviewText_tokens", tokenize_text_udf(col("reviewText_cleaned")))
          .withColumn("summary_tokens", tokenize_text_udf(col("summary_cleaned")))
          .withColumn("reviewText_no_stopwords", remove_stopwords_udf(col("reviewText_tokens")))
          .withColumn("summary_no_stopwords", remove_stopwords_udf(col("summary_tokens")))
          .withColumn("reviewText_lemmatized", lemmatize_tokens_udf(col("reviewText_no_stopwords")))
          .withColumn("summary_lemmatized", lemmatize_tokens_udf(col("summary_no_stopwords")))
    )

    # Debug: Check intermediate results after all transformations
    cleaned_df.select("reviewText_lemmatized", "summary_lemmatized").show(5)

    # Write the cleaned data back to S3 as Parquet
    cleaned_df.write.mode("overwrite").parquet(output_parquet_path)
    print(f"Cleaned data written to {output_parquet_path}")

except Exception as e:
    print(f"Error processing data: {str(e)}")


# COMMAND ----------

pip install spacy


# COMMAND ----------

!python -m spacy download en_core_web_sm


# COMMAND ----------

import spacy
nlp = spacy.load("en_core_web_sm")
print("spaCy and the English model loaded successfully!")


# COMMAND ----------

import spacy.cli

# Install the English model
spacy.cli.download("en_core_web_sm")

# Load the model after installation
nlp = spacy.load("en_core_web_sm")


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import nltk
import spacy

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Create a set of stopwords for faster lookup
stop_words = set(stopwords.words('english'))

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StopwordRemovalAndLemmatization") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Load Parquet files from S3
input_path = "s3://raw-zip-final/Parquet/Final_Data/"
df = spark.read.parquet(input_path)

# Check schema to verify columns
df.printSchema()

# Define UDF for text preprocessing (stopword removal and lemmatization)
def preprocess_text(text):
    if text is None:  # Handle null values
        return None
    try:
        # Use spaCy for tokenization, lemmatization, and stopword removal
        doc = nlp(text)
        processed_tokens = [
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct and not token.is_space
        ]
        return " ".join(processed_tokens)
    except Exception as e:
        # Log and handle any processing errors
        print(f"Error processing text: {str(e)}")
        return None

# Register the UDF with correct data type handling
preprocess_udf = udf(preprocess_text, StringType())

# Apply the UDF to the relevant columns and ensure column names are handled correctly
try:
    processed_df = (
        df.withColumn("cleaned_reviewText", preprocess_udf(col("reviewText")))
          .withColumn("cleaned_summary", preprocess_udf(col("summary")))
    )
except Exception as e:
    print(f"Error applying UDF: {str(e)}")

# Check for missing or erroneous data after processing
processed_df.select("cleaned_reviewText", "cleaned_summary").show(5, truncate=False)

# Drop unnecessary columns
columns_to_drop = ["reviewText", "style", "summary"]
try:
    processed_df = processed_df.drop(*columns_to_drop)
except Exception as e:
    print(f"Error dropping columns: {str(e)}")

# Save the resulting DataFrame back to S3
output_path = "s3://raw-zip-final/Parquet/Final_Processed_Data/"
try:
    processed_df.write.mode("overwrite").parquet(output_path)
    print(f"Processing complete! Processed data saved to: {output_path}")
except Exception as e:
    print(f"Error saving data to S3: {str(e)}")

# Stop Spark session
spark.stop()


# COMMAND ----------

processed_df.display(truncate=False, n=20)  

# COMMAND ----------

import spacy
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load the spaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Spark Session
spark = SparkSession.builder.appName("TextProcessing").getOrCreate()

# Define UDF
def preprocess_text(text):
    if text:
        doc = nlp(text)
        return " ".join([
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct
        ])
    return None

preprocess_udf = udf(preprocess_text, StringType())

# Assuming `df` is your DataFrame
processed_df = df.withColumn("cleaned_text", preprocess_udf(df["reviewText"]))

# Display top 20 rows
processed_df.show(truncate=False, n=20)



# COMMAND ----------

from pyspark.sql import SparkSession

# Create or get the existing Spark session
spark = SparkSession.builder \
    .appName("StopwordRemovalAndLemmatization") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Assuming `processed_df` is already defined earlier


processed_df.display(truncate=False, n=20)

# COMMAND ----------

import spacy
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load the spaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Spark Session
spark = SparkSession.builder.appName("TextProcessing").getOrCreate()

# Define UDF
def preprocess_text(text):
    if text:
        doc = nlp(text)
        return " ".join([
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct
        ])
    return None

preprocess_udf = udf(preprocess_text, StringType())

# Assuming `df` is your DataFrame
processed_df = df.withColumn("cleaned_text", preprocess_udf(df["reviewText"]))

# Display top 20 rows
processed_df.display(truncate=False, n=20)


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import nltk
import spacy

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Create a set of stopwords for faster lookup
stop_words = set(stopwords.words('english'))

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StopwordRemovalAndLemmatization") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Load Parquet files from S3
input_path = "s3://raw-zip-final/Parquet/Final_Data/"
df = spark.read.parquet(input_path)

# Check schema to verify columns
df.printSchema()

# Define UDF for text preprocessing (stopword removal and lemmatization)
def preprocess_text(text):
    if text is None:  # Handle null values
        return None
    try:
        # Use spaCy for tokenization, lemmatization, and stopword removal
        doc = nlp(text)
        processed_tokens = [
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct and not token.is_space
        ]
        return " ".join(processed_tokens)
    except Exception as e:
        # Log and handle any processing errors
        print(f"Error processing text: {str(e)}")
        return None

# Register the UDF with correct data type handling
preprocess_udf = udf(preprocess_text, StringType())

# Apply the UDF to the relevant columns and ensure column names are handled correctly
try:
    processed_df = (
        df.withColumn("cleaned_reviewText", preprocess_udf(col("reviewText"))).withColumn("cleaned_summary", preprocess_udf(col("summary")))
    )
except Exception as e:
    print(f"Error applying UDF: {str(e)}")

# Check for missing or erroneous data after processing
processed_df.select("cleaned_reviewText", "cleaned_summary").show(5, truncate=False)

# Drop unnecessary columns
columns_to_drop = ["reviewText", "style", "summary"]
try:
    processed_df = processed_df.drop(*columns_to_drop)
except Exception as e:
    print(f"Error dropping columns: {str(e)}")

# Save the entire processed DataFrame to S3 (in cleaned_final_folder_DATA)
output_path = "s3://raw-zip-final/Parquet/cleaned_final_folder_DATA/"
try:
    processed_df.write.mode("overwrite").parquet(output_path)
    print(f"Processing complete! Processed data saved to: {output_path}")
except Exception as e:
    print(f"Error saving data to S3: {str(e)}")

# Stop Spark session
spark.stop()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import nltk
import spacy

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Create a set of stopwords for faster lookup
stop_words = set(stopwords.words('english'))

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StopwordRemovalAndLemmatization") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Load Parquet files from S3
input_path = "s3://raw-zip-final/Parquet/Final_Data/"
df = spark.read.parquet(input_path)

# Print schema to verify columns
df.printSchema()

# Define UDF for text preprocessing (stopword removal and lemmatization)
def preprocess_text(text):
    if text is None:  # Handle null values
        return None
    try:
        # Use spaCy for tokenization, lemmatization, and stopword removal
        doc = nlp(text)
        processed_tokens = [
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct and not token.is_space
        ]
        return " ".join(processed_tokens)
    except Exception as e:
        print(f"Error processing text: {str(e)}")  # Log any errors
        return None

# Register UDF for Spark
preprocess_udf = udf(preprocess_text, StringType())

# Apply the UDF to relevant columns
processed_df = df \
    .withColumn("cleaned_reviewText", preprocess_udf(col("reviewText"))) \
    .withColumn("cleaned_summary", preprocess_udf(col("summary")))

# Drop unnecessary columns
columns_to_drop = ["reviewText", "style", "summary"]
processed_df = processed_df.drop(*columns_to_drop)

# Save the entire processed DataFrame to S3
output_path = "s3://raw-zip-final/Parquet/cleaned_final_folder_DATA/"
processed_df.write.mode("overwrite").parquet(output_path)

# Print confirmation
print(f"Processing complete! Processed data saved to: {output_path}")

# Stop Spark session
spark.stop()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import nltk
import spacy

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Create a set of stopwords for faster lookup
stop_words = set(stopwords.words('english'))

# Initialize Spark session
spark = SparkSession.builder \
    .appName("StopwordRemovalAndLemmatization") \
    .config("spark.hadoop.fs.s3a.access.key", "<your-access-key>") \
    .config("spark.hadoop.fs.s3a.secret.key", "<your-secret-key>") \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
    .getOrCreate()

# Load Parquet files from S3
input_path = "s3a://raw-zip-final/Parquet/Final_Data/"
df = spark.read.parquet(input_path)

# Check schema to verify columns
df.printSchema()

# Define UDF for text preprocessing (stopword removal and lemmatization)
def preprocess_text(text):
    if text is None:  # Handle null values
        return None
    try:
        # Use spaCy for tokenization, lemmatization, and stopword removal
        doc = nlp(text)
        processed_tokens = [
            token.lemma_ for token in doc 
            if token.text.lower() not in stop_words and not token.is_punct and not token.is_space
        ]
        return " ".join(processed_tokens)
    except Exception as e:
        # Log and handle any processing errors
        print(f"Error processing text: {str(e)}")
        return None

# Register the UDF with correct data type handling
preprocess_udf = udf(preprocess_text, StringType())

# Apply the UDF to the relevant columns and ensure column names are handled correctly
try:
    processed_df = (
        df.withColumn("cleaned_reviewText", preprocess_udf(col("reviewText")))
          .withColumn("cleaned_summary", preprocess_udf(col("summary")))
    )
except Exception as e:
    print(f"Error applying UDF: {str(e)}")

# Check for missing or erroneous data after processing
processed_df.select("cleaned_reviewText", "cleaned_summary").show(5, truncate=False)

# Drop unnecessary columns
columns_to_drop = ["reviewText", "style", "summary"]
try:
    processed_df = processed_df.drop(*columns_to_drop)
except Exception as e:
    print(f"Error dropping columns: {str(e)}")

# Coalesce to a single partition if dataset size is manageable
processed_df = processed_df.coalesce(1)

# Save the entire processed DataFrame to S3 (in cleaned_final_folder_DATA)
output_path = "s3a://raw-zip-final/Parquet/cleaned_final_folder_DATA/"
try:
    processed_df.write.mode("overwrite").parquet(output_path)
    print(f"Processing complete! Processed data saved to: {output_path}")
except Exception as e:
    print(f"Error saving data to S3: {str(e)}")

# Stop Spark session
spark.stop()

