# Databricks notebook source
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Check Parquet Schema").getOrCreate()
parquet_path = "s3a://raw-zip/Parquet/processed-data/All_Amazon_Review.parquet"
df = spark.read.parquet(parquet_path)
df.printSchema()


# COMMAND ----------

df.show(25, truncate=False)
