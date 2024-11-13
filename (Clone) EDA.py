# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, isnan, when, length, avg, desc, explode, split
import matplotlib.pyplot as plt
import pandas as pd

# COMMAND ----------



# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder.appName("Handle Duplicate Columns").getOrCreate()

# Define the input path
input_file_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"

# Read JSON file into DataFrame
df = spark.read.json(input_file_path)

# Normalize column names: Assuming direct access to nested fields might require additional handling
# if nested JSON parsing is needed, adjust the following approach:
df = df.select([col(c).alias(c.lower()) for c in df.columns])  # Lowercase all column names to avoid case issues

# Flatten nested columns if needed and ensure all columns have unique names
# This example assumes you have a way to handle nested structures if necessary:
# E.g., renaming 'style.scent' to 'style_scent'
for field in df.schema.fields:
    if "struct" in str(field.dataType):  # Simple check for a StructType
        for nested_field in field.dataType.names:
            nested_col_name = f"{field.name}.{nested_field}".lower().replace('.', '_')
            df = df.withColumn(nested_col_name, col(f"{field.name}.{nested_field}"))
        df = df.drop(field.name)

# Exclude specific columns directly
columns_to_exclude = ["image", "reviewtime"]  # Assuming lowercase names after normalization
filtered_df = df.drop(*columns_to_exclude)

# Define the output path
output_path = "s3a://raw-zip/output_parquet"

# Save the filtered DataFrame as a Parquet file
filtered_df.write.mode("overwrite").parquet(output_path)

# Optionally, display the filtered DataFrame
filtered_df.show(truncate=False)

# Stop the Spark session
spark.stop()


# COMMAND ----------

# List contents of your S3 bucket directly
bucket_name = "raw-zip"
path = f"s3a://{bucket_name}/"

# List the contents of the S3 bucket
dbutils.fs.ls(path)



try:
    # Load JSON data from S3
    print("Reading JSON file from S3...")
    df = spark.read.json(path)
    
    # Repartition data to optimize parallel processing (adjust number of partitions based on your cluster size)
    df = df.repartition(200)

    # Print schema to verify the structure
    print("Schema of the Data:")
    df.printSchema()

    # Show some data to verify the load
    df.show(5)

except Exception as e:
    print(f"Error loading JSON file: {e}")

finally:
    # Stop the Spark session
    spark.stop()



# COMMAND ----------



# COMMAND ----------



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


