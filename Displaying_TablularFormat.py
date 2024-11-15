# Databricks notebook source
# Import necessary library
from pyspark.sql import SparkSession

# Create or get a Spark Session
spark = SparkSession.builder \
    .appName("Read Parquet File from S3") \
    .getOrCreate()



# COMMAND ----------

# Read a Parquet file
file_path = "s3://raw-zip/Parquet/cleaned_parquet_files/part-00000-tid-6655891077008087027-658a9fdb-e68e-4d5c-9d4b-ad402bb7d1a9-8118-1.c000.snappy.parquet"
df = spark.read.parquet(file_path)

# Convert the Spark DataFrame to a Pandas DataFrame
# Using limit to safely convert a manageable amount of data
pandas_df = df.limit(10000).toPandas()

# Use display to show the DataFrame in Jupyter Notebooks
display(pandas_df)



# COMMAND ----------

print (pandas_df.count())

# COMMAND ----------

nums_rows = len(pandas_df)

print (nums_rows)

# COMMAND ----------

pandas_df2 = df.toPandas()



# COMMAND ----------

num_rows = len(pandas_df2)
print (num_rows)
