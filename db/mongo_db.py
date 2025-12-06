import os
from typing import Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient

from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class MongoDB:
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        """Initializes the MongoDB client.

        Args:
            uri: MongoDB connection URI (defaults to MONGO_URI from .env)
            db_name: Database name (defaults to MONGO_DB_NAME from .env)
        """
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.db_name = db_name or os.getenv("MONGO_DB_NAME", "rag_db")

        try:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.server_info()
            logger.info(f"Connected to MongoDB at {self.uri}, database: {self.db_name}")

            # Create indexes
            self._create_indexes()
        except Exception as e:
            logger.exception(f"Failed to connect to MongoDB: {e}")
            raise

    def _create_indexes(self):
        """Creates necessary indexes for the questions collection."""
        try:
            questions = self.db.questions

            # Compound index for document and chapter
            questions.create_index(
                [("class_id", ASCENDING), ("chapter_id", ASCENDING)],
                name="doc_chapter_idx",
            )

            # Multikey index for topic_keys
            questions.create_index("topic_keys", name="topics_idx")

            # Index for created_at
            questions.create_index([("created_at", DESCENDING)], name="created_at_idx")

            # Index for origin
            questions.create_index("origin", name="origin_idx")

            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Error creating indexes (may already exist): {e}")

    def get_questions_collection(self):
        """Returns the questions collection."""
        return self.db.questions

    def get_chapters_collection(self):
        """Returns the chapters collection for metadata storage."""
        return self.db.chapters

    def search_questions(
        self,
        class_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        topic_keys: Optional[list] = None,
        difficulty: Optional[str] = None,
        limit: int = 20,
        sort_by: str = "created_at",
        sort_order: int = DESCENDING,
    ) -> list:
        """Search questions with flexible filtering. All conditions are
        combined with AND logic. None values are ignored.

        Args:
            class_id: Filter by document ID (optional)
            chapter_id: Filter by chapter ID (optional)
            topic_keys: Filter by topics - matches if ANY topic in the list matches (optional)
            difficulty: Filter by difficulty level (optional)
            limit: Maximum number of results to return
            sort_by: Field to sort by (default: "created_at")
            sort_order: Sort order - ASCENDING or DESCENDING (default: DESCENDING)

        Returns:
            List of question documents matching the criteria

        Example:
            # Search for questions in a specific document and chapter with any of the topics
            results = mongo.search_questions(
                class_id="9th_ncert",
                chapter_id="chapter_1",
                topic_keys=["fluids", "pressure"],
                difficulty="easy",
                limit=10
            )
        """
        questions = self.get_questions_collection()

        # Build query dynamically - only include non-None conditions
        query = {}

        if class_id is not None:
            query["class_id"] = class_id

        if chapter_id is not None:
            query["chapter_id"] = chapter_id

        if topic_keys is not None and len(topic_keys) > 0:
            # Use $in to match ANY of the provided topics
            query["topic_keys"] = {"$in": topic_keys}

        if difficulty is not None:
            query["difficulty"] = difficulty

        try:
            # Execute query with sorting and limit
            results = list(questions.find(query).sort(sort_by, sort_order).limit(limit))

            # Convert ObjectId to string for JSON serialization
            for result in results:
                if "_id" in result:
                    result["_id"] = str(result["_id"])

            logger.info(f"Search query: {query}, found {len(results)} results")
            return results

        except Exception as e:
            logger.exception(f"Search failed: {e}")
            return []

    def close(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")


if __name__ == "__main__":
    # Example Usage
    try:
        mongo = MongoDB()
        questions = mongo.get_questions_collection()

        # Test insert multiple questions
        test_questions = [
            {
                "class_id": "9th_ncert",
                "chapter_id": "chapter_1",
                "question_text": "What is RAG?",
                "answer": "Retrieval-Augmented Generation",
                "topic_keys": ["rag", "ai"],
                "source_chunks": [1, 2],
                "difficulty": "easy",
                "type": "fact",
                "created_at": 1764269614.9116824,
                "status": "draft",
                "origin": "generated",
            },
            {
                "class_id": "9th_ncert",
                "chapter_id": "chapter_1",
                "question_text": "Explain fluid pressure",
                "answer": "Pressure in fluids acts equally in all directions",
                "topic_keys": ["fluids", "pressure"],
                "source_chunks": [5, 6],
                "difficulty": "medium",
                "type": "conceptual",
                "created_at": 1764269615.0,
                "status": "draft",
                "origin": "generated",
            },
            {
                "class_id": "9th_ncert",
                "chapter_id": "chapter_2",
                "question_text": "What are states of matter?",
                "answer": "Solid, liquid, and gas",
                "topic_keys": ["states_of_matter", "physics"],
                "source_chunks": [10, 11],
                "difficulty": "easy",
                "type": "fact",
                "created_at": 1764269616.0,
                "status": "published",
                "origin": "generated",
            },
        ]

        # Insert test questions
        result = questions.insert_many(test_questions)
        logger.info(f"Inserted {len(result.inserted_ids)} test questions")

        # Test 1: Search by class_id only
        logger.info("\n--- Test 1: Search by class_id only ---")
        results = mongo.search_questions(class_id="9th_ncert")
        logger.info(f"Found {len(results)} questions for document '9th_ncert'")

        # Test 2: Search by class_id AND chapter_id
        logger.info("\n--- Test 2: Search by class_id AND chapter_id ---")
        results = mongo.search_questions(class_id="9th_ncert", chapter_id="chapter_1")
        logger.info(f"Found {len(results)} questions for chapter_1")

        # Test 3: Search by topics (ANY match)
        logger.info("\n--- Test 3: Search by topics ---")
        results = mongo.search_questions(topic_keys=["rag", "fluids", "nlp"])
        logger.info(f"Found {len(results)} questions with topics: rag, fluids, or nlp")
        for r in results:
            logger.info(f"  - {r['question_text'][:50]}... (topics: {r['topic_keys']})")

        # Test 4: Search by document, chapter, topics, AND difficulty
        logger.info("\n--- Test 4: Search with all filters ---")
        results = mongo.search_questions(
            class_id="9th_ncert",
            chapter_id="chapter_1",
            topic_keys=["fluids", "pressure"],
            difficulty="medium",
        )
        logger.info(
            f"Found {len(results)} medium difficulty questions about fluids/pressure in chapter_1"
        )

        # Test 5: Search by difficulty only
        logger.info("\n--- Test 5: Search by difficulty only ---")
        results = mongo.search_questions(difficulty="easy", limit=10)
        logger.info(f"Found {len(results)} easy questions")

        # Test 6: Search with no filters (get all, sorted by created_at)
        logger.info("\n--- Test 6: Get all questions ---")
        results = mongo.search_questions(limit=5)
        logger.info(f"Found {len(results)} questions (limited to 5)")

        # Clean up
        # questions.delete_many({"_id": {"$in": result.inserted_ids}})
        # logger.info(f"\nCleaned up {len(result.inserted_ids)} test questions")

    except Exception as e:
        logger.exception(f"MongoDB test failed: {e}")
