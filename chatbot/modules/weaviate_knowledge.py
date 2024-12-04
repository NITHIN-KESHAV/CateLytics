import weaviate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WeaviateKnowledge:
    def __init__(self, weaviate_url):
        self.client = weaviate.Client(weaviate_url)
        logging.info(f"Initialized Weaviate client with URL: {weaviate_url}")

    def query_database(self, query=None, class_name="Review", properties=None, asin=None, limit=5):
        """
        Queries the Weaviate database for reviews based on a query, ASIN, or both.

        :param query: The textual query to match against review data.
        :param class_name: The name of the class in the Weaviate schema (default is 'Review').
        :param properties: List of properties to retrieve from the class objects.
        :param asin: The ASIN identifier to filter results.
        :param limit: Maximum number of results to retrieve.
        :return: List of matched results or an empty list if no results are found.
        """
        if properties is None:
            properties = ["reviewText_cleaned", "summary_cleaned", "overall", "asin"]

        try:
            # Initialize the query builder
            query_builder = (
                self.client.query
                .get(class_name, properties)
                .with_limit(limit)
            )

            # Add an ASIN filter if provided
            if asin:
                query_builder = query_builder.with_where({
                    "path": ["asin"],
                    "operator": "Equal",
                    "valueString": asin.upper()  # Convert ASIN to uppercase for case-insensitive matching
                })

            # Add text-based search if query is provided and ASIN is not used
            if query and not asin:
                query_builder = query_builder.with_near_text({"concepts": [query]})

            # Execute the query
            result = query_builder.do()
            logging.info(f"Successfully queried database for query: '{query}' with ASIN: '{asin}'")
            return result['data']['Get'][class_name]
        except Exception as e:
            logging.error(f"Error querying database: {str(e)}")
            return []

    def add_data(self, class_name, data):
        """
        Adds new data to the Weaviate database.

        :param class_name: The name of the class in the Weaviate schema.
        :param data: List of data objects to add to the class.
        """
        try:
            with self.client.batch as batch:
                for item in data:
                    batch.add_data_object(item, class_name)
            logging.info(f"Successfully added {len(data)} items to {class_name}")
        except Exception as e:
            logging.error(f"Error adding data to Weaviate: {str(e)}")
