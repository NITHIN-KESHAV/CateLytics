# Databricks notebook source
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, concat_ws

# Tokenize reviewText_cleaned and summary_cleaned columns
tokenizer = Tokenizer(inputCol="reviewText_cleaned", outputCol="reviewText_tokens")
cleaned_df = tokenizer.transform(cleaned_df)

tokenizer_summary = Tokenizer(inputCol="summary_cleaned", outputCol="summary_tokens")
cleaned_df = tokenizer_summary.transform(cleaned_df)

# Remove stopwords
stopword_remover = StopWordsRemover(inputCol="reviewText_tokens", outputCol="reviewText_filtered")
cleaned_df = stopword_remover.transform(cleaned_df)

stopword_remover_summary = StopWordsRemover(inputCol="summary_tokens", outputCol="summary_filtered")
cleaned_df = stopword_remover_summary.transform(cleaned_df)

# Optionally join tokens back into a single string for storage
cleaned_df = cleaned_df.withColumn("reviewText_final", concat_ws(" ", col("reviewText_filtered")))
cleaned_df = cleaned_df.withColumn("summary_final", concat_ws(" ", col("summary_filtered")))

# Drop intermediate columns if no longer needed
columns_to_drop = ["reviewText_tokens", "summary_tokens", "reviewText_filtered", "summary_filtered"]
cleaned_df = cleaned_df.drop(*columns_to_drop)

# Show the results
cleaned_df.select("reviewText_cleaned", "reviewText_final", "summary_cleaned", "summary_final").show(truncate=False)

# Save the final preprocessed data back to Parquet
output_path_preprocessed = "s3://raw-zip-final/Parquet/Preprocessed_Data/"
cleaned_df.write.mode("overwrite").parquet(output_path_preprocessed)

print(f"Preprocessed data saved to {output_path_preprocessed}")

