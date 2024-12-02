# Databricks notebook source
# Replace with your S3 bucket and folder path
s3_path = "s3://raw-zip/Parquet/deduplicated_parquet_files/"

# Load the Parquet files into a Spark DataFrame
df = spark.read.parquet(s3_path)

# Display the DataFrame to verify the data
display(df)


# COMMAND ----------

# Path to your Parquet files stored in S3 or any other storage
parquet_path = "s3://raw-zip/Parquet/deduplicated_parquet_files/"

# Read all Parquet files into a DataFrame
df = spark.read.parquet(parquet_path)

# Check the schema and total number of rows
df.printSchema()
print(f"Total Rows: {df.count()}")


# COMMAND ----------

# Select only the ASIN column
asin_df = df.select("asin")


# COMMAND ----------

asin_df.write.csv("s3://raw-zip/Parquet/", header=True, mode="overwrite")


# COMMAND ----------

# Read the CSV file into a Spark DataFrame
csv_path = "s3://raw-zip/Parquet/asin-csv/"
read_asin_df = spark.read.csv(csv_path, header=True, inferSchema=True)

# Show the first few rows
read_asin_df.show(20, truncate=False)  # Display 20 rows without truncating columns


# COMMAND ----------

# Count total rows in the ASIN column
total_rows = read_asin_df.count()

print(f"Total Rows in ASIN: {total_rows}")


# COMMAND ----------

# Count distinct ASIN values
distinct_asin_count = read_asin_df.select("asin").distinct().count()

print(f"Distinct ASIN Count: {distinct_asin_count}")


# COMMAND ----------

pip install requests


# COMMAND ----------

# MAGIC %restart_python
