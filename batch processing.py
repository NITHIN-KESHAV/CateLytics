# Databricks notebook source
def batch_insert_reviews(self, reviews_df, batch_size: int = None) -> int:
    """
    Insert reviews into Weaviate in batches with retry logic and progress tracking.
    
    :param reviews_df: PySpark DataFrame containing reviews
    :param batch_size: Number of reviews to process in each batch (defaults to self.batch_size)
    :return: Total number of reviews inserted
    """
    # Initialize variables
    batch_size = batch_size or self.batch_size
    total_reviews = reviews_df.count()
    successful_inserts = 0
    remaining_reviews_df = reviews_df  # Track remaining rows for batch subtraction

    logging.info(f"Starting batch insertion of {total_reviews} reviews with batch size {batch_size}")

    # Process reviews in batches
    for batch_num, start_idx in enumerate(range(0, total_reviews, batch_size), start=1):
        batch_df = remaining_reviews_df.limit(batch_size).collect()
        if not batch_df:
            logging.info(f"No more rows to process. Exiting at batch {batch_num}.")
            break
        
        success = self._process_single_batch(batch_num, batch_df)
        if success:
            successful_inserts += len(batch_df)
        remaining_reviews_df = remaining_reviews_df.subtract(remaining_reviews_df.limit(batch_size))
        
    logging.info(f"Batch insertion completed. Total successful inserts: {successful_inserts}/{total_reviews}")
    return successful_inserts

def _process_single_batch(self, batch_num: int, batch_df: list) -> bool:
    """
    Process a single batch with retry logic.
    
    :param batch_num: Current batch number
    :param batch_df: List of rows in the current batch
    :return: True if the batch is successfully processed, False otherwise
    """
    for attempt in range(self.retry_attempts):
        try:
            logging.info(f"Processing batch {batch_num}, attempt {attempt + 1}/{self.retry_attempts}")
            
            batch_objects = [self.prepare_review_object(row.asDict()) for row in batch_df]
            response = requests.post(
                f"{self.base_url}/v1/objects",
                headers=self.headers,
                json={"objects": batch_objects}
            )
            
            if response.status_code == 200:
                logging.info(f"Successfully inserted batch {batch_num} ({len(batch_objects)} reviews)")
                return True
            
            logging.warning(f"Batch {batch_num} failed with status {response.status_code}. Response: {response.text}")
        
        except Exception as e:
            logging.error(f"Error processing batch {batch_num}: {str(e)}")
        
        if attempt < self.retry_attempts - 1:
            logging.info(f"Retrying batch {batch_num} in {self.retry_delay} seconds...")
            time.sleep(self.retry_delay)
        else:
            logging.error(f"Batch {batch_num} failed after {self.retry_attempts} attempts.")
    
    return False

