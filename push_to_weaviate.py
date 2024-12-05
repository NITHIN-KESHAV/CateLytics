# Databricks notebook source
# MAGIC %md
# MAGIC ### Embeddings Generation

# COMMAND ----------

# MAGIC %pip install protobuf==3.20.0
# MAGIC %pip install weaviate-client

# COMMAND ----------

!pip install --upgrade protobuf
!pip install weaviate-client sentence-transformers tqdm

# COMMAND ----------

import weaviate
client = weaviate.Client(url="http://50.18.99.196:8080")

# Get meta information about the instance
meta = client.get_meta()
print("\nInstance Information:")
print(meta)

# COMMAND ----------

pip install --upgrade "pydantic>=2.0.0"

# COMMAND ----------

import weaviate
from weaviate.util import generate_uuid5
import pandas as pd
import time

# Connect to Weaviate
client = weaviate.Client(
    url="http://50.18.99.196:8080",
)

# Create schema to generate embeddings
class_obj = {
    "class": "Review",
    "vectorizer": "text2vec-transformers",
    "moduleConfig": {
        "text2vec-transformers": {
            "vectorizeClassName": False,
            "textFields": ["cleaned_reviewText", "cleaned_summary"]
        }
    },
    "properties": [
        {
            "name": "cleaned_reviewText",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-transformers": {
                    "skip": False,
                    "vectorizePropertyName": False
                }
            }
        },
        {
            "name": "cleaned_summary",
            "dataType": ["text"],
            "moduleConfig": {
                "text2vec-transformers": {
                    "skip": False,
                    "vectorizePropertyName": False
                }
            }
        },
        {"name": "asin", "dataType": ["string"]},
        {"name": "overall", "dataType": ["number"]},
        {"name": "reviewerID", "dataType": ["string"]},
        {"name": "reviewerName", "dataType": ["string"]},
        {"name": "unixReviewTime", "dataType": ["int"]},
        {"name": "verified", "dataType": ["boolean"]},
        {"name": "vote", "dataType": ["string"]},
        {"name": "style_size", "dataType": ["string"]},
        {"name": "style_flavor", "dataType": ["string"]},
        {"name": "style_style", "dataType": ["string"]},
        {"name": "style_color", "dataType": ["string"]},
        {"name": "style_edition", "dataType": ["string"]},
        {"name": "style_package_type", "dataType": ["string"]},
        {"name": "style_number_of_items", "dataType": ["string"]},
        {"name": "style_size_per_pearl", "dataType": ["string"]},
        {"name": "style_color_name", "dataType": ["string"]},
        {"name": "style_size_name", "dataType": ["string"]},
        {"name": "style_metal_type", "dataType": ["string"]},
        {"name": "style_style_name", "dataType": ["string"]},
        {"name": "style_flavor_name", "dataType": ["string"]},
        {"name": "style_length", "dataType": ["string"]},
        {"name": "style_package_quantity", "dataType": ["string"]},
        {"name": "style_design", "dataType": ["string"]},
        {"name": "style_platform", "dataType": ["string"]},
        {"name": "style_item_package_quantity", "dataType": ["string"]},
        {"name": "style_wattage", "dataType": ["string"]},
        {"name": "style_pattern", "dataType": ["string"]},
        {"name": "style_material_type", "dataType": ["string"]},
        {"name": "style_team_name", "dataType": ["string"]},
        {"name": "style_item_shape", "dataType": ["string"]},
        {"name": "style_hand_orientation", "dataType": ["string"]},
        {"name": "style_flex", "dataType": ["string"]},
        {"name": "style_material", "dataType": ["string"]},
        {"name": "style_shaft_material", "dataType": ["string"]},
        {"name": "style_configuration", "dataType": ["string"]},
        {"name": "style_capacity", "dataType": ["string"]},
        {"name": "style_product_packaging", "dataType": ["string"]},
        {"name": "style_offer_type", "dataType": ["string"]},
        {"name": "style_model", "dataType": ["string"]},
        {"name": "style_overall_length", "dataType": ["string"]},
        {"name": "style_width", "dataType": ["string"]},
        {"name": "style_grip_type", "dataType": ["string"]},
        {"name": "style_gem_type", "dataType": ["string"]},
        {"name": "style_scent_name", "dataType": ["string"]},
        {"name": "style_model_number", "dataType": ["string"]},
        {"name": "style_connectivity", "dataType": ["string"]},
        {"name": "style_digital_storage_capacity", "dataType": ["string"]},
        {"name": "style_subscription_length", "dataType": ["string"]},
        {"name": "style_primary_stone_gem_type", "dataType": ["string"]},
        {"name": "style_item_display_weight", "dataType": ["string"]},
        {"name": "style_gift_amount", "dataType": ["string"]},
        {"name": "style_format_type", "dataType": ["string"]},
        {"name": "UUID", "dataType": ["string"]}
    ]
}

# Delete existing schema if it exists
try:
    client.schema.delete_class("Review")
    print("Existing schema deleted")
except:
    pass

# Create the schema
client.schema.create_class(class_obj)
print("Schema created successfully")

def get_embeddings_and_update_df(spark_df, batch_size=2):
    """Generate embeddings and replace text with vectors"""
    
    # Convert spark dataframe to pandas (just 2 rows for testing)
    pandas_df = spark_df.limit(batch_size).toPandas()
    print(f"\nProcessing {len(pandas_df)} records")
    
    # Store original texts for reference
    original_texts = []
    
    # Upload the data to get embeddings
    with client.batch as batch:
        batch.batch_size = batch_size
        
        for _, row in pandas_df.iterrows():
            try:
                # Store original texts
                original_texts.append({
                    'review': row['cleaned_reviewText'],
                    'summary': row['cleaned_summary']
                })
                
                # Prepare properties dictionary
                properties = {}
                for col in pandas_df.columns:
                    if pd.notna(row[col]):
                        if col == 'unixReviewTime':
                            properties[col] = int(row[col])
                        elif col == 'overall':
                            properties[col] = float(row[col])
                        elif col == 'verified':
                            properties[col] = bool(row[col])
                        else:
                            properties[col] = str(row[col])
                
                # Add data object to get embeddings
                batch.add_data_object(
                    data_object=properties,
                    class_name="Review"
                )
                
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
    
    # Wait a moment for indexing
    time.sleep(2)
    
    # Get the vectors for uploaded objects
    print("\nRetrieving generated embeddings...")
    result = (
        client.query
        .get("Review")
        .with_additional(['vector'])
        .with_limit(batch_size)
        .do()
    )
    
    # Create new DataFrame with embeddings replacing text
    embedded_df = pandas_df.copy()
    
    if 'data' in result and 'Get' in result['data'] and 'Review' in result['data']['Get']:
        records = result['data']['Get']['Review']
        
        for i, (record, orig_text) in enumerate(zip(records, original_texts)):
            if '_additional' in record and 'vector' in record['_additional']:
                vector = record['_additional']['vector']
                
                # Replace text with vectors in the DataFrame
                embedded_df.at[i, 'cleaned_reviewText'] = vector
                embedded_df.at[i, 'cleaned_summary'] = vector
                
                # Print information about the conversion
                print(f"\nRecord {i+1}:")
                print(f"Original Review Text: {orig_text['review'][:100]}...")
                print(f"Original Summary: {orig_text['summary'][:100]}...")
                print(f"Generated Vector dimension: {len(vector)}")
                print(f"First 5 values of vector: {vector[:5]}")
    
    return embedded_df

# Test with 2 records
print("\nGenerating embeddings for test records...")
embedded_df = get_embeddings_and_update_df(processed_df, batch_size=2)

print("\nShape of DataFrame with embeddings:", embedded_df.shape)

# COMMAND ----------

# Check if the class exists
classes = client.schema.get()
print("Available classes:", [c['class'] for c in classes['classes']])

# COMMAND ----------

# Fetch and print the schema
schema = client.schema.get()

# Print the schema in a readable format
print("\nWeaviate Schema:")
for class_obj in schema['classes']:
    print(f"Class: {class_obj['class']}")
    for prop in class_obj['properties']:
        print(f"  Property: {prop['name']}, Type: {prop['dataType']}")

# COMMAND ----------

import pandas as pd

# Assuming processed_df is your DataFrame and it has a column named 'UUID'
# Replace the value in `uuid_to_check` with the given value
uuid_to_check = "f61a9d001bffafffca13df58f868768a63cd98a59ab079d80e63e090bb837a45"

# Method 1: Using filter
if processed_df.filter(processed_df.UUID == uuid_to_check).count() > 0:
    print("Match found!")
else:
    print("No match found.")
