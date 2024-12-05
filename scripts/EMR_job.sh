hadoop jar /usr/lib/hadoop/hadoop-streaming.jar \
    -input s3://raw-zip-final/All_Amazon_Review.json.gz \
    -output s3://raw-zip-final/decompressed-output/ \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py"