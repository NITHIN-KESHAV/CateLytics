# Databricks notebook source
# Define the S3 path
bucket_name = "raw-zip"
file_path = "All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"
s3_path = f"s3a://{bucket_name}/{file_path}"

# Load a sample of the JSON data with Spark, limiting to 10 rows
df = spark.read.json(s3_path).limit(10)

# Show the first 10 rows
df.show(truncate=False)


# COMMAND ----------

# Define the S3 path to your JSON file
bucket_name = "raw-zip"
file_path = "All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"
s3_path = f"s3a://{bucket_name}/{file_path}"

# Read the JSON file (just enough to infer the schema)
df = spark.read.json(s3_path)

# Show the schema of the JSON file
df.printSchema()


# COMMAND ----------

# Read a small sample of the JSON file to infer schema
df = spark.read.option("samplingRatio", 0.01).json(s3_path)  # Adjust sampling ratio as needed

# Show the schema of the JSON file
df.printSchema()


# COMMAND ----------

# Define the S3 path to your JSON file
bucket_name = "raw-zip"
file_path = "All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"
s3_path = f"s3a://{bucket_name}/{file_path}"

# Set up Spark to read only a small sample of the JSON file
# Adjust samplingRatio as needed (between 0.01 and 0.1) for best performance
try:
    df = spark.read.option("samplingRatio", 0.01).json(s3_path)
    
    # Show the inferred schema of the JSON file
    df.printSchema()
    
    # Display the first 10 rows to verify the structure of the data
    df.show(10, truncate=False)

except Exception as e:
    print("An error occurred:", e)


# COMMAND ----------

# Define the S3 path to your JSON file
bucket_name = "raw-zip"
file_path = "All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"
s3_path = f"s3a://{bucket_name}/{file_path}"

# Read JSON file as plain text
df_text = spark.read.text(s3_path)

# Show the first 10 lines to confirm data is being read
df_text.show(10, truncate=False)

