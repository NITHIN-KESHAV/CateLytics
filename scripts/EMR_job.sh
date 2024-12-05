#!/bin/bash

INPUT_PATH="s3://raw-zip-final/All_Amazon_Review.json.gz"
OUTPUT_PATH="s3://raw-zip-final/decompressed-output/"

hadoop jar /usr/lib/hadoop/hadoop-streaming.jar \
    -input $INPUT_PATH \
    -output $OUTPUT_PATH \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py"