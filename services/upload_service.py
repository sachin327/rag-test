"""Enhanced ingestion service for question generation system.

Uses RAGSystem from rag.py for all chunking, summarization, and
ingestion functionality.
"""

import os
from logger import get_logger
from services.rag import RAGSystem
from dotenv import load_dotenv
from utils.topic_search import topics_exist_semantic
from db.mongo_db import MongoDB

logger = get_logger(__name__)
load_dotenv()


class UploadService:
    """Upload service for document ingestion."""

    def __init__(self):
        """Initialize the upload service."""
        logger.info("Upload Service initialized")

        self.rag_service = RAGSystem()
        self.rag_service.create_payload_index(
            fields=["class_id", "chapter_id", "subject_id"]
        )
        self.mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))

    def is_already_exists(
        self,
        class_id: str,
        chapter_id: str,
        subject_id: str,
    ):
        result = self.rag_service.search_by_filter(
            filters={
                "class_id": class_id,
                "chapter_id": chapter_id,
                "subject_id": subject_id,
            },
            limit=1,
        )
        return result

    def upload_document(
        self,
        file_path: str,
        class_id: str,
        class_name: str,
        chapter_id: str,
        chapter_name: str,
        subject_id: str,
        subject_name: str,
    ):
        """Uploads a document to the Qdrant vector database.

        Args:
            file_path: Path to document file
            class_id: Unique document identifier
            chapter_id: Chapter identifier
            chapter_name: Name of the chapter
            class_name: Name of the class
            subject_name: Name of the subject
            subject_id: ID of the subject
            collection_name: Qdrant collection name

        Returns:
            Dictionary with ingestion statistics
        """

        result = self.is_already_exists(
            class_id=class_id,
            chapter_id=chapter_id,
            subject_id=subject_id,
        )

        # logger.debug("result: ", result)

        if result:
            result = result[0]["payload"]

            topic_flags_mongo = topics_exist_semantic(
                topics_collection=self.mongo.get_collection("chaptertopics"),
                subject_id=subject_id,
                input_topics=result["topic_keys"],
                similarity_threshold=0.6,
            )
            return {
                "success": True,
                "metadata": {
                    "class_id": class_id,
                    "chapter_id": chapter_id,
                    "subject_id": subject_id,
                    "class_name": class_name,
                    "chapter_name": chapter_name,
                    "subject_name": subject_name,
                },
                "topics_extracted": len(result["topic_keys"]),
                "topic_keys": topic_flags_mongo,
                "summary": result["summary"],
                "summary_length": len(result["summary"]),
            }

        result = self.rag_service.add_document(
            file_path=file_path,
            metadata={
                "class_id": class_id,
                "chapter_id": chapter_id,
                "subject_id": subject_id,
                "class_name": class_name,
                "chapter_name": chapter_name,
                "subject_name": subject_name,
            },
        )

        topic_flags_mongo = topics_exist_semantic(
            topics_collection=self.mongo.get_collection("chaptertopics"),
            subject_id=subject_id,
            input_topics=result["topic_keys"],
            similarity_threshold=0.6,
        )
        result["topic_keys"] = topic_flags_mongo
        return result


if __name__ == "__main__":
    # Example usage
    try:
        service = UploadService()

        # Test ingestion
        result = service.upload_document(
            file_path="data/sample.pdf",
            class_id="class_1",
            chapter_id="chapter_1",
            chapter_name="chapter_1",
            class_name="class_1",
            subject_name="subject_1",
            subject_id="subject_1",
        )

        logger.info(f"Upload result: {result}")

    except Exception as e:
        logger.exception(f"Upload test failed: {e}")
