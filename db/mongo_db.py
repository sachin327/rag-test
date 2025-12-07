import os
from dotenv import load_dotenv
from pymongo import MongoClient

from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class MongoDB:
    def __init__(
        self,
        uri,
        db_name,
    ):
        """Initializes the MongoDB client.

        Args:
            uri: MongoDB connection URI (defaults to MONGO_URI from .env)
            db_name: Database name (defaults to MONGO_DB_NAME from .env)
        """
        self.uri = uri
        self.db_name = db_name

        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.server_info()
            logger.info(f"Connected to MongoDB at {self.uri}, database: {self.db_name}")

        except Exception as e:
            logger.exception(f"Failed to connect to MongoDB: {e}")
            raise

    def get_collection(self, collection_name):
        """Returns the collection for metadata storage."""
        return self.db[collection_name]

    def close(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    # Example Usage
    try:
        mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
        topics = mongo.get_collection("topic-ai-service")

        # # Create index for class_id, chapter_id, subject_id, topic_name, topic_relavence_score
        # topics.create_index(
        #     [
        #         ("class_id", ASCENDING),
        #         ("chapter_id", ASCENDING),
        #         ("subject_id", ASCENDING),
        #         ("topic_name", ASCENDING),
        #         ("topic_relavence_score", DESCENDING),
        #     ]
        # )

        # # Add topic to the collection
        # topics.insert_one(
        #     {
        #         "class_id": "9th",
        #         "chapter_id": "chapter_1",
        #         "subject_id": "subject_1",
        #         "topic_name": "Newtons Law of Motion",
        #         "slug": "newtons-law-of-motion",
        #         "topic_relavence_score": 1.0,
        #     }
        # )

        # # Add topic to the collection
        # topics.insert_one(
        #     {
        #         "class_id": "9th",
        #         "chapter_id": "chapter_1",
        #         "subject_id": "subject_1",
        #         "topic_name": "The Human Body",
        #         "slug": "the-human-body",
        #         "topic_relavence_score": 0.9,
        #     }
        # )

        # # Add topic to the collection
        # topics.insert_one(
        #     {
        #         "class_id": "9th",
        #         "chapter_id": "chapter_1",
        #         "subject_id": "subject_1",
        #         "topic_name": "Thermo dynamics",
        #         "slug": "thermo-dynamics",
        #         "topic_relavence_score": 0.8,
        #     }
        # )

        # Test 1: Search by class_id only
        logger.info("\n--- Test 1: Search by class_id only ---")
        results = list(topics.find({"class_id": "9th"}))
        logger.info(f"Found {len(results)} topics for document '9th'")

        # Test 2: Search by class_id AND chapter_id
        logger.info("\n--- Test 2: Search by class_id AND chapter_id ---")
        results = list(topics.find({"class_id": "9th", "chapter_id": "chapter_1"}))
        logger.info(f"Found {len(results)} topics for chapter_1")

        # Test 3: Search by class_id AND chapter_id AND subject_id
        logger.info(
            "\n--- Test 3: Search by class_id AND chapter_id AND subject_id ---"
        )
        results = list(
            topics.find(
                {
                    "class_id": "9th",
                    "chapter_id": "chapter_1",
                    "subject_id": "subject_1",
                }
            )
        )
        logger.info(results)
        logger.info(f"Found {len(results)} topics for chapter_1 and subject_1")

    except Exception as e:
        logger.exception(f"MongoDB test failed: {e}")
