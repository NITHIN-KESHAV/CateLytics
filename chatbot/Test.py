import weaviate

client = weaviate.Client("http://50.18.99.196:8080")

result = client.query.get(
    "Review",
    ["asin", "cleaned_reviewText", "overall"]
).with_where({
    "path": ["asin"],
    "operator": "Equal",
    "valueString": "B000E5KJDE"
}).do()

print(result)