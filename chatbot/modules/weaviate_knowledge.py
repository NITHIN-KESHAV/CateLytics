import weaviate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeaviateKnowledge:
    def __init__(self, weaviate_url):
        self.client = weaviate.Client(weaviate_url)
        logging.info(f"Initialized Weaviate client with URL: {weaviate_url}")

    def query_database(self, query, class_name="Review", properties=None, limit=5):
        if properties is None:
            properties = ["cleaned_reviewText", "cleaned_summary", "overall"]
        try:
            result = (
                self.client.query
                .get(class_name, properties)
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            logging.info(f"Successfully queried database for: {query}")
            return result['data']['Get'][class_name]
        except Exception as e:
            logging.error(f"Error querying database: {str(e)}")
            return []

    def add_data(self, class_name, data):
        try:
            with self.client.batch as batch:
                for item in data:
                    batch.add_data_object(item, class_name)
            logging.info(f"Successfully added {len(data)} items to {class_name}")
        except Exception as e:
            logging.error(f"Error adding data to Weaviate: {str(e)}")