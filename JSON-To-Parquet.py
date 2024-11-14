# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, BooleanType

# Define the full schema with all unique column names
schema = StructType([
    StructField('asin', StringType(), True),
    StructField('image', StringType(), True),
    StructField('overall', DoubleType(), True),
    StructField('reviewText', StringType(), True),
    StructField('reviewTime', LongType(), True),
    StructField('reviewerID', StringType(), True),
    StructField('reviewerName', StringType(), True),
    StructField('style', StringType(), True),
    # Add all style_dup fields with their specific nested structure
    StructField('style_dup1', StructType([StructField('Bare Outside Diameter String:', StringType(), True)]), True),
    StructField('style_dup2', StructType([StructField('Bore Diameter:', StringType(), True)]), True),
    StructField('style_dup3', StructType([StructField('Capacity:', StringType(), True)]), True),
    StructField('style_dup4', StructType([StructField('Closed Length String:', StringType(), True)]), True),
    StructField('style_dup5', StructType([StructField('Color Name:', StringType(), True)]), True),
    StructField('style_dup6', StructType([StructField('Color:', StringType(), True)]), True),
    StructField('style_dup7', StructType([StructField('Colorj:', StringType(), True)]), True),
    StructField('style_dup8', StructType([StructField('Colour:', StringType(), True)]), True),
    StructField('style_dup9', StructType([StructField('Compatible Fastener Description:', StringType(), True)]), True),
    StructField('style_dup10', StructType([StructField('Conference Name:', StringType(), True)]), True),
    StructField('style_dup11', StructType([StructField('Configuration:', StringType(), True)]), True),
    StructField('style_dup12', StructType([StructField('Connectivity:', StringType(), True)]), True),
    StructField('style_dup13', StructType([StructField('Connector Type:', StringType(), True)]), True),
    StructField('style_dup14', StructType([StructField('Content:', StringType(), True)]), True),
    StructField('style_dup15', StructType([StructField('Current Type:', StringType(), True)]), True),
    StructField('style_dup16', StructType([StructField('Curvature:', StringType(), True)]), True),
    StructField('style_dup17', StructType([StructField('Cutting Diameter:', StringType(), True)]), True),
    StructField('style_dup18', StructType([StructField('Denomination:', StringType(), True)]), True),
    StructField('style_dup19', StructType([StructField('Department:', StringType(), True)]), True),
    StructField('style_dup20', StructType([StructField('Design:', StringType(), True)]), True),
    StructField('style_dup21', StructType([StructField('Diameter:', StringType(), True)]), True),
    StructField('style_dup22', StructType([StructField('Digital Storage Capacity:', StringType(), True)]), True),
    StructField('style_dup23', StructType([StructField('Display Height:', StringType(), True)]), True),
    StructField('style_dup24', StructType([StructField('Edition:', StringType(), True)]), True),
    StructField('style_dup25', StructType([StructField('Electrical Connector Type:', StringType(), True)]), True),
    StructField('style_dup26', StructType([StructField('Extended Length:', StringType(), True)]), True),
    StructField('style_dup27', StructType([StructField('Fits Shaft Diameter:', StringType(), True)]), True),
    StructField('style_dup28', StructType([StructField('Flavor Name:', StringType(), True)]), True),
    StructField('style_dup29', StructType([StructField('Flavor:', StringType(), True)]), True),
    StructField('style_dup30', StructType([StructField('Flex:', StringType(), True)]), True),
    StructField('style_dup31', StructType([StructField('Format:', StringType(), True)]), True),
    StructField('style_dup32', StructType([StructField('Free Length:', StringType(), True)]), True),
    StructField('style_dup33', StructType([StructField('Gauge:', StringType(), True)]), True),
    StructField('style_dup34', StructType([StructField('Gem Type:', StringType(), True)]), True),
    StructField('style_dup35', StructType([StructField('Gift Amount:', StringType(), True)]), True),
    StructField('style_dup36', StructType([StructField('Grip Type:', StringType(), True)]), True),
    StructField('style_dup37', StructType([StructField('Grit Type:', StringType(), True)]), True),
    StructField('style_dup38', StructType([StructField('Hand Orientation:', StringType(), True)]), True),
    StructField('style_dup39', StructType([StructField('Hardware Platform:', StringType(), True)]), True),
    StructField('style_dup40', StructType([StructField('Head Diameter:', StringType(), True)]), True),
    StructField('style_dup41', StructType([StructField('Horsepower:', StringType(), True)]), True),
    StructField('style_dup42', StructType([StructField('Initial:', StringType(), True)]), True),
    StructField('style_dup43', StructType([StructField('Input Range Description:', StringType(), True)]), True),
    StructField('style_dup44', StructType([StructField('Inside Diameter:', StringType(), True)]), True),
    StructField('style_dup45', StructType([StructField('Item Display Length:', StringType(), True)]), True),
    StructField('style_dup46', StructType([StructField('Item Display Weight:', StringType(), True)]), True),
    StructField('style_dup47', StructType([StructField('Item Package Quantity:', StringType(), True)]), True),
    StructField('style_dup48', StructType([StructField('Item Shape:', StringType(), True)]), True),
    StructField('style_dup49', StructType([StructField('Item Thickness:', StringType(), True)]), True),
    StructField('style_dup50', StructType([StructField('Item Weight:', StringType(), True)]), True),
    StructField('style_dup51', StructType([StructField('Lead Type:', StringType(), True)]), True),
    StructField('style_dup52', StructType([StructField('Length Range:', StringType(), True)]), True),
    StructField('style_dup53', StructType([StructField('Length:', StringType(), True)]), True),
    StructField('style_dup54', StructType([StructField('Line Weight:', StringType(), True)]), True),
    StructField('style_dup55', StructType([StructField('Load Capacity:', StringType(), True)]), True),
    StructField('style_dup56', StructType([StructField('Loft:', StringType(), True)]), True),
    StructField('style_dup57', StructType([StructField('Material Type:', StringType(), True)]), True),
    StructField('style_dup58', StructType([StructField('Material:', StringType(), True)]), True),
    StructField('style_dup59', StructType([StructField('Maximum Measurement:', StringType(), True)]), True),
    StructField('style_dup60', StructType([StructField('Measuring Range:', StringType(), True)]), True),
    StructField('style_dup61', StructType([StructField('Metal Stamp:', StringType(), True)]), True),
    StructField('style_dup62', StructType([StructField('Metal Type:', StringType(), True)]), True),
    StructField('style_dup63', StructType([StructField('Model Name:', StringType(), True)]), True),
    StructField('style_dup64', StructType([StructField('Model Number:', StringType(), True)]), True),
    StructField('style_dup65', StructType([StructField('Model:', StringType(), True)]), True),
    StructField('style_dup66', StructType([StructField('Nominal Outside Diameter:', StringType(), True)]), True),
    StructField('style_dup67', StructType([StructField('Number of Items:', StringType(), True)]), True),
    StructField('style_dup68', StructType([StructField('Number of Shelves:', StringType(), True)]), True),
    StructField('style_dup69', StructType([StructField('Offer Type:', StringType(), True)]), True),
    StructField('style_dup70', StructType([StructField('Options:', StringType(), True)]), True),
    StructField('style_dup71', StructType([StructField('Outside Diameter:', StringType(), True)]), True),
    StructField('style_dup72', StructType([StructField('Overall Height:', StringType(), True)]), True),
    StructField('style_dup73', StructType([StructField('Overall Length:', StringType(), True)]), True),
    StructField('style_dup74', StructType([StructField('Overall Width:', StringType(), True)]), True),
    StructField('style_dup75', StructType([StructField('Package Quantity:', StringType(), True)]), True),
    StructField('style_dup76', StructType([StructField('Package Type:', StringType(), True)]), True),
    StructField('style_dup77', StructType([StructField('Part Number:', StringType(), True)]), True),
    StructField('style_dup78', StructType([StructField('Pattern:', StringType(), True)]), True),
    StructField('style_dup79', StructType([StructField('Pitch Diameter String:', StringType(), True)]), True),
    StructField('style_dup80', StructType([StructField('Pitch:', StringType(), True)]), True),
    StructField('style_dup81', StructType([StructField('Platform for Display:', StringType(), True)]), True),
    StructField('style_dup82', StructType([StructField('Platform:', StringType(), True)]), True),
    StructField('style_dup83', StructType([StructField('Preamplifier Output Channel Quantity:', StringType(), True)]), True),
    StructField('style_dup84', StructType([StructField('Primary Stone Gem Type:', StringType(), True)]), True),
    StructField('style_dup85', StructType([StructField('Processor Description:', StringType(), True)]), True),
    StructField('style_dup86', StructType([StructField('Product Packaging:', StringType(), True)]), True),
    StructField('style_dup87', StructType([StructField('Range:', StringType(), True)]), True),
    StructField('style_dup88', StructType([StructField('SCENT:', StringType(), True)]), True),
    StructField('style_dup89', StructType([StructField('Scent Name:', StringType(), True)]), True),
    StructField('style_dup90', StructType([StructField('Scent:', StringType(), True)]), True),
    StructField('style_dup91', StructType([StructField('Service plan term:', StringType(), True)]), True),
    StructField('style_dup92', StructType([StructField('Shaft Material Type:', StringType(), True)]), True),
    StructField('style_dup93', StructType([StructField('Shaft Material:', StringType(), True)]), True),
    StructField('style_dup94', StructType([StructField('Shape:', StringType(), True)]), True),
    StructField('style_dup95', StructType([StructField('Size Name:', StringType(), True)]), True),
    StructField('style_dup96', StructType([StructField('Size per Pearl:', StringType(), True)]), True),
    StructField('style_dup97', StructType([StructField('Size:', StringType(), True)]), True),
    StructField('style_dup98', StructType([StructField('Special Features:', StringType(), True)]), True),
    StructField('style_dup99', StructType([StructField('Stone Shape:', StringType(), True)]), True),
    StructField('style_dup100', StructType([StructField('Style Name:', StringType(), True)]), True),
    StructField('style_dup101', StructType([StructField('Style:', StringType(), True)]), True),
    StructField('style_dup102', StructType([StructField('Subscription Length:', StringType(), True)]), True),
    StructField('style_dup103', StructType([StructField('Team Name:', StringType(), True)]), True),
    StructField('style_dup104', StructType([StructField('Temperature Range:', StringType(), True)]), True),
    StructField('style_dup105', StructType([StructField('Tension Supported:', StringType(), True)]), True),
    StructField('style_dup106', StructType([StructField('Thickness:', StringType(), True)]), True),
    StructField('style_dup107', StructType([StructField('Total Diamond Weight:', StringType(), True)]), True),
    StructField('style_dup108', StructType([StructField('Unit Count:', StringType(), True)]), True),
    StructField('style_dup109', StructType([StructField('Volume:', StringType(), True)]), True),
    StructField('style_dup110', StructType([StructField('Wattage:', StringType(), True)]), True),
    StructField('style_dup111', StructType([StructField('Width Range:', StringType(), True)]), True),
    StructField('style_dup112', StructType([StructField('Width:', StringType(), True)]), True),
    StructField('style_dup113', StructType([StructField('Wire Diameter:', StringType(), True)]), True),
    StructField('style_dup114', StructType([StructField('option:', StringType(), True)]), True),
    StructField('style_dup115', StructType([StructField('processor_description:', StringType(), True)]), True),
    StructField('style_dup116', StructType([StructField('style name:', StringType(), True)]), True),
    StructField('style_dup117', StructType([StructField('style:', StringType(), True)]), True),
    StructField('summary', StringType(), True),
    StructField('unixReviewTime', LongType(), True),
    StructField('verified', BooleanType(), True),
    StructField('vote', StringType(), True)
])

# Initialize Spark session (adjust for your environment)
spark = SparkSession.builder \
    .appName("JSON to Parquet Conversion") \
    .getOrCreate()

# Input and output paths
input_path = "s3a://raw-zip/All_Amazon_Review.json.gz/unzipped/All_Amazon_Review.json"
output_path = "s3a://raw-zip/Parquet/processed-data/All_Amazon_Review.parquet"

# Read the JSON file with the predefined schema
df = spark.read.schema(schema).json(input_path)

# Write the DataFrame to Parquet format in S3
df.write \
    .mode("overwrite") \
    .parquet(output_path)

print("Conversion to Parquet completed and stored at:", output_path)


# COMMAND ----------

# MAGIC %md
# MAGIC 46.6gb 949 obj
