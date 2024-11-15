# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when, length, avg, desc, explode, split
import matplotlib.pyplot as plt
import pandas as pd

# # Initialize Spark session with S3 access (set AWS credentials if required)
# spark = SparkSession.builder \
#     .appName("S3DataAnalysis") \
#     .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
#     .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.DefaultAWSCredentialsProviderChain") \
#     .getOrCreate()

# # Load Data from S3
# s3_path = "s3://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"
# df = spark.read.json(s3_path, multiLine=True)

# COMMAND ----------

# Initialize SparkSession with S3 access
spark = SparkSession.builder \
    .appName("S3DataAnalysis") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .config("spark.hadoop.fs.s3a.access.key", "AKIA5FTY7M3CIARC2UM5 ") \
    .config("spark.hadoop.fs.s3a.secret.key", "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") \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .getOrCreate()

# Load data from S3
s3_path = "s3a://raw-zip/unzipped/All_Amazon_Review.json"
df = spark.read.json(s3_path)
df

# COMMAND ----------

# List contents of your S3 bucket directly
bucket_name = "raw-zip"
path = f"s3a://{bucket_name}/"

# List the contents of the S3 bucket
dbutils.fs.ls(path)


# COMMAND ----------

s3_path = "s3a://raw-zip/All_Amazon_Review.json.gz"  # or whatever the exact path is based on your S3 structure

df = spark.read.json(s3_path)


print("Schema of the Data:")
df.printSchema()


# COMMAND ----------

print("Show Sample Rows:")
df.show(5)




# COMMAND ----------

if 'spark' in locals():
    spark.stop()
    print("Spark session stopped.")


# COMMAND ----------

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Initialize Spark session
# spark = SparkSession.builder.appName("Delete Fields and Save to Parquet").getOrCreate()

# Define the S3 paths
input_file_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"
output_s3_path = "s3a://your-bucket-name/filtered_amazon_reviews.parquet"  # Change to your S3 bucket and path

# List of fields to delete, specify as fully qualified names (for nested fields use dot notation, e.g., "style.Color")
fields_to_delete = ["image", "reviewTime", "reviewerName"]

# Read JSON file as a DataFrame with each row as a JSON string (no schema inference)
raw_df = spark.read.text(input_file_path)

# Function to remove specific keys from JSON data
def delete_fields_from_json(json_string):
    try:
        data = json.loads(json_string)
        for field in fields_to_delete:
            _delete_nested_key(data, field.split("."))
        return json.dumps(data)
    except json.JSONDecodeError:
        return json_string  # Return as-is if there's an issue with JSON parsing

# Recursive helper function to delete a nested key
def _delete_nested_key(data, keys):
    if len(keys) == 1:
        if isinstance(data, dict) and keys[0] in data:
            del data[keys[0]]
    else:
        if isinstance(data, dict) and keys[0] in data:
            _delete_nested_key(data[keys[0]], keys[1:])

# Register the UDF
delete_fields_udf = udf(delete_fields_from_json, StringType())

# Apply the UDF to delete specified fields
filtered_df = raw_df.withColumn("filtered_value", delete_fields_udf(col("value")))

# Parse the JSON strings back into structured JSON columns
final_df = spark.read.json(filtered_df.select("filtered_value").rdd.map(lambda row: row["filtered_value"]))

# Save the final DataFrame to Parquet format in S3
final_df.write.mode("overwrite").parquet(output_s3_path)

print(f"Data saved to {output_s3_path}")

# Stop the Spark session
spark.stop()


# COMMAND ----------

 # 2. Check for Missing Values
print("Count of Missing Values per Column:")
missing_values = df.select([(count(when(col(c).isNull() | isnan(c), c)) / count('*')).alias(c) for c in df.columns])
missing_values.show()



# COMMAND ----------

# 3. Summary Statistics for Numeric Columns
print("Summary Statistics for Numeric Columns:")
numeric_summary = df.describe(["overall", "vote", "review_length"])  # Include numeric columns as per your dataset
numeric_summary.show()



# COMMAND ----------

# 4. Distribution Analysis - Visualize using Matplotlib

# Convert to Pandas for plotting (only for small samples to avoid memory issues)
df_sample = df.select("overall", "vote", "sentiment", "review_length").sample(fraction=0.1).toPandas()

# Plot Distribution of Overall Rating
plt.figure(figsize=(8, 6))
df_sample["overall"].plot(kind="hist", bins=5, color="skyblue", edgecolor="black")
plt.title("Distribution of Overall Rating")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.show()

# Plot Distribution of Review Length
plt.figure(figsize=(8, 6))
df_sample["review_length"].plot(kind="hist", bins=20, color="orange", edgecolor="black")
plt.title("Distribution of Review Length")
plt.xlabel("Review Length")
plt.ylabel("Frequency")
plt.show()



# COMMAND ----------

# 5. Basic Text Data Analysis (for 'reviewText' Column)
# Calculate Average Length of Reviews
avg_length = df.select(avg(length(col("reviewText"))).alias("average_review_length"))
avg_length.show()

# Show 5 Most Common Words in 'reviewText' (requires Tokenization)
from pyspark.sql.functions import explode, split

# Tokenize the reviewText column and count word frequencies
df_tokens = df.withColumn("word", explode(split(col("reviewText"), " ")))  # Split by space
word_counts = df_tokens.groupBy("word").count().orderBy(desc("count"))
print("Most Common Words:")
word_counts.show(5)

# Optional: Convert word_counts to Pandas for visualization
word_counts_pd = word_counts.limit(20).toPandas()
plt.figure(figsize=(10, 6))
plt.barh(word_counts_pd["word"], word_counts_pd["count"], color="teal")
plt.gca().invert_yaxis()
plt.xlabel("Frequency")
plt.title("Top 20 Most Common Words in Review Text")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Word Cloud

# COMMAND ----------


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all reviews into a single string
reviews_text = " ".join(df.select("reviewText").rdd.flatMap(lambda x: x).collect())

# Generate and display the word cloud
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color="white").generate(reviews_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Words in Reviews")
plt.show()



# COMMAND ----------

# MAGIC %md
# MAGIC Top N Most Common Words
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import explode, split, desc
import pandas as pd

# Tokenize and explode the review text column
df_tokens = df.withColumn("word", explode(split(col("reviewText"), " ")))

# Count occurrences of each word and order by frequency
word_counts = df_tokens.groupBy("word").count().orderBy(desc("count")).limit(20)
word_counts_pd = word_counts.toPandas()

# Plot the top 20 most common words
plt.figure(figsize=(10, 6))
plt.barh(word_counts_pd["word"], word_counts_pd["count"], color="skyblue")
plt.xlabel("Frequency")
plt.title("Top 20 Most Common Words in Reviews")
plt.gca().invert_yaxis()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Distribution of Review Lengths

# COMMAND ----------

from pyspark.sql.functions import length

# Calculate the length of each review
df = df.withColumn("review_length", length(col("reviewText")))

# Convert to Pandas for plotting
review_lengths = df.select("review_length").toPandas()

# Plot histogram of review lengths
plt.figure(figsize=(10, 6))
plt.hist(review_lengths["review_length"], bins=30, color="purple", edgecolor="black")
plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length (characters)")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Sentiment Analysis Distribution

# COMMAND ----------

from textblob import TextBlob
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType

# Define a UDF to calculate sentiment
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

sentiment_udf = udf(get_sentiment, FloatType())
df = df.withColumn("sentiment", sentiment_udf(col("reviewText")))

# Convert sentiment scores to Pandas for plotting
sentiment_scores = df.select("sentiment").toPandas()

# Plot histogram of sentiment scores
plt.figure(figsize=(10, 6))
plt.hist(sentiment_scores["sentiment"], bins=30, color="orange", edgecolor="black")
plt.title("Distribution of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Frequency")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Top N Bigrams or Trigrams

# COMMAND ----------

from pyspark.sql.functions import concat_ws
from nltk import ngrams

# Tokenize the reviews and generate bigrams
df_tokens = df.withColumn("tokens", split(col("reviewText"), " "))

# Define a UDF to get bigrams
def get_bigrams(tokens):
    bigrams = ngrams(tokens, 2)
    return [" ".join(bigram) for bigram in bigrams]

bigrams_udf = udf(get_bigrams, ArrayType(StringType()))
df_bigrams = df_tokens.withColumn("bigrams", bigrams_udf(col("tokens")))

# Explode bigrams column and get counts
df_bigrams = df_bigrams.withColumn("bigram", explode("bigrams"))
bigram_counts = df_bigrams.groupBy("bigram").count().orderBy(desc("count")).limit(20)
bigram_counts_pd = bigram_counts.toPandas()

# Plot top 20 bigrams
plt.figure(figsize=(10, 6))
plt.barh(bigram_counts_pd["bigram"], bigram_counts_pd["count"], color="green")
plt.xlabel("Frequency")
plt.title("Top 20 Most Common Bigrams in Reviews")
plt.gca().invert_yaxis()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Sentiment Over Time

# COMMAND ----------

from pyspark.sql.functions import to_date

# Convert date column to DateType and extract the year or month
df = df.withColumn("review_date", to_date(col("reviewTime"), "MM dd, yyyy"))

# Calculate average sentiment by month
monthly_sentiment = df.groupBy("review_date").avg("sentiment").orderBy("review_date")
monthly_sentiment_pd = monthly_sentiment.toPandas()

# Plot sentiment over time
plt.figure(figsize=(12, 6))
plt.plot(monthly_sentiment_pd["review_date"], monthly_sentiment_pd["avg(sentiment)"], color="purple")
plt.title("Average Sentiment Score Over Time")
plt.xlabel("Date")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------


