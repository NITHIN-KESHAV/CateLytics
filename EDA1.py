# Databricks notebook source
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("ParquetEDA").getOrCreate()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan

# Initialize Spark session
spark = SparkSession.builder.appName("AmazonReviewEDA").getOrCreate()




# COMMAND ----------

# Define the S3 path to your Parquet folder
parquet_s3_path = "s3://raw-zip/Parquet/cleaned_parquet_files/"


# COMMAND ----------

# Read all Parquet files in the specified S3 folder
df = spark.read.parquet(parquet_s3_path)

# Read top 100 rows and display all columns in table format
df.limit(100).show(truncate=False)



# COMMAND ----------

display(df.limit(100))

# COMMAND ----------

# Check the schema to understand the structure of the DataFrame
df.printSchema()



# COMMAND ----------

# Select only the numeric columns for summary statistics
numeric_columns = [field.name for field in df.schema.fields if field.dataType.typeName() in ["integer", "double", "float", "long", "short"]]
df.select(numeric_columns).describe().show()


# COMMAND ----------

df.count()


# COMMAND ----------



from pyspark.sql.functions import col, isnan, when, count

# Count missing values for each column
missing_values = df.select([(count(when(col(c).isNull() | isnan(col(c)), c)) / df.count()).alias(c) for c in df.columns])
missing_values.show()


# COMMAND ----------

duplicate_count = df.groupBy(df.columns).count().filter("count > 1").count()
print(f"Duplicate Rows: {duplicate_count}")


# COMMAND ----------

# Group by 'asin' and count occurrences to find duplicates
duplicate_asin_df = df.groupBy("reviewText").count().filter("count > 1")

# Display rows with duplicate 'asin' values
display(duplicate_asin_df)


# COMMAND ----------

from pyspark.sql.functions import col, count

# Group by the combination of reviewerID, asin, and unixReviewTime and count occurrences
duplicate_check = df.groupBy("reviewerID", "asin", "unixReviewTime") \
                    .agg(count("*").alias("count")) \
                    .filter(col("count") > 1)

# Show the rows that have duplicates
duplicate_check.show()

# Count the total number of duplicate rows
total_duplicates = duplicate_check.agg({"count": "sum"}).collect()[0][0]
print(f"Total duplicate rows based on reviewerID + asin + unixReviewTime: {total_duplicates}")



# COMMAND ----------

from pyspark.sql.functions import col, count

# Step 1: Identify duplicates based on reviewerID, asin, and unixReviewTime
duplicate_check = df.groupBy("reviewerID", "asin", "unixReviewTime") \
                    .agg(count("*").alias("count")) \
                    .filter(col("count") > 1)

# Step 2: Extract keys (reviewerID, asin, unixReviewTime) that have duplicates
duplicates_keys = duplicate_check.select("reviewerID", "asin", "unixReviewTime").distinct()

# Step 3: Anti-join to keep only unique rows
df_no_duplicates = df.join(duplicates_keys, ["reviewerID", "asin", "unixReviewTime"], "anti")

# Step 4: Save the resulting DataFrame without duplicates to a new Parquet file
output_path = "s3://your-bucket/cleaned_parquet_data_no_duplicates/"
df_no_duplicates.write.mode("overwrite").parquet(output_path)

print(f"DataFrame without duplicates has been saved to {output_path}")


# COMMAND ----------

from pyspark.sql.functions import col, count

# Group by the combination of reviewerID, asin, unixReviewTime, and reviewText, then count occurrences
duplicate_check = df.groupBy("reviewerID", "asin", "unixReviewTime", "reviewText") \
                    .agg(count("*").alias("count")) \
                    .filter(col("count") > 1)

# Show the rows that have duplicates
duplicate_check.show(truncate=False)

# Count the total number of duplicate rows
total_duplicates = duplicate_check.agg({"count": "sum"}).collect()[0][0]
print(f"Total duplicate rows based on reviewerID + asin + unixReviewTime + reviewText: {total_duplicates}")


# COMMAND ----------

numeric_columns = [field.name for field in df.schema.fields if field.dataType.typeName() in ["integer", "double", "float", "long", "short"]]
df.select(numeric_columns).describe().show()


# COMMAND ----------

from pyspark.sql.functions import size, split

# Calculate word counts for reviewText and summary
df.withColumn("reviewText_word_count", size(split(col("reviewText"), " "))) \
  .withColumn("summary_word_count", size(split(col("summary"), " "))) \
  .select("reviewText_word_count", "summary_word_count") \
  .describe().show()




# COMMAND ----------

pip install sparknlp


# COMMAND ----------



