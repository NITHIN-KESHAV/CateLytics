# Databricks notebook source
pip install sentence-transformers pinecone-client pandas


# COMMAND ----------

from sentence_transformers import SentenceTransformer
import pinecone
import pandas as pd
import numpy as np


# COMMAND ----------

# MAGIC %md
# MAGIC pcsk_6jhjsF_HAh7SofvpDDnzk67gJRaV6e6wxHPDY4WiZypGmfe5ujvnNdAb9v871yyUoTPDL8
# MAGIC

# COMMAND ----------

pip install --upgrade pinecone-client


# COMMAND ----------



import pinecone

# Step 1: Initialize Pinecone with your API key and environment
pinecone.init(
    api_key="pcsk_6jhjsF_HAh7SofvpDDnzk67gJRaV6e6wxHPDY4WiZypGmfe5ujvnNdAb9v871yyUoTPDL8",  # Replace with your Pinecone API key
    environment="us-east-1"  # Replace with your environment (e.g., "us-east-1")
)

# Step 2: Define the index name
index_name = "vectordb"

# Step 3: Create the index if it doesn't already exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384)  # 384 for all-MiniLM-L6-v2 embeddings

# Step 4: Connect to the index
index = pinecone.Index(index_name)

# Step 5: Print confirmation
print(f"Successfully connected to index: {index_name}")




# COMMAND ----------


