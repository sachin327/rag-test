"""Enhanced ingestion service for question generation system.

Uses RAGSystem from rag.py for all chunking, summarization, and
ingestion functionality.
"""

import os
from typing import Optional

from llm_gemini import LLMService
from logger import get_logger
from mongo_db import MongoDB
from qdrant_db import QdrantDB
from rag import RAGSystem
from dotenv import load_dotenv

logger = get_logger(__name__)
load_dotenv()

class UploadService:
    """Upload service for document ingestion."""

    def __init__(
        self,
        qdrant_db: Optional[QdrantDB] = None,
        mongo_db: Optional[MongoDB] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """Initialize the enhanced ingestion service.

        Args:
            qdrant_db: QdrantDB instance (creates new if None)
            mongo_db: MongoDB instance (creates new if None)
            llm_service: LLMService instance (creates new if None)
        """
        self.qdrant = qdrant_db or QdrantDB(
            host=os.getenv("QDRANT_HOST"),
            port=os.getenv("QDRANT_PORT"),
            grpc_port=os.getenv("QDRANT_GRPC_PORT"),
            prefer_grpc=False,
        )
        self.mongo = mongo_db or MongoDB()
        self.llm = llm_service or LLMService()

        # Initialize RAGSystem which contains all the chunking and processing logic
        self.rag = RAGSystem(
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            host=os.getenv("QDRANT_HOST"),
            port=os.getenv("QDRANT_PORT"),
            grpc_port=os.getenv("QDRANT_GRPC_PORT"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )

        logger.info("Enhanced Ingestion Service initialized")

    def extract_topics_and_summary(self, text: str, max_topics: int = 3):
        """Extracts topics and summary from text using LLM. Delegates to
        RAGSystem.

        Args:
            text: Chunk text
            max_topics: Maximum number of topics to extract

        Returns:
            Dictionary with 'topic_keys', 'summary', and 'importance_score'
        """
        return self.rag.extract_topics_and_summary(text, max_topics)

    def generate_chunk_summary(
        self,
        text: str,
        chapter_name: str = "",
        class_name: str = "",
        subject_name: str = "",
    ) -> str:
        """Generates a short summary (around 200 words) for a large chunk using
        Gemini. Delegates to RAGSystem.

        Args:
            text: Text chunk (up to 4000 chars)
            chapter_name: Name of the chapter (optional context)
            class_name: Name of the class (optional context)
            subject_name: Name of the subject (optional context)

        Returns:
            Summary text (around 200 words)
        """
        return self.rag.generate_chunk_summary(
            text,
            chapter_name=chapter_name,
            class_name=class_name,
            subject_name=subject_name,
        )

    def generate_final_summary_and_topics(
        self,
        combined_summaries: str,
        chapter_name: str = "",
        class_name: str = "",
        subject_name: str = "",
    ):
        """Generates final summary and topic list from concatenated summaries.
        Delegates to RAGSystem.

        Args:
            combined_summaries: All chunk summaries concatenated
            chapter_name: Name of the chapter (optional context)
            class_name: Name of the class (optional context)
            subject_name: Name of the subject (optional context)

        Returns:
            Dictionary with 'final_summary' and 'topic_keys' (5-6 topics)
        """
        return self.rag.generate_final_summary_and_topics(
            combined_summaries,
            chapter_name=chapter_name,
            class_name=class_name,
            subject_name=subject_name,
        )

    def chunk_with_overlap(self, text: str, chunk_size: int = 512, overlap: int = 100):
        """Splits text into chunks with overlap. Delegates to RAGSystem.

        Args:
            text: Full text
            chunk_size: Target chunk size (default: 512)
            overlap: Overlap size (default: 100)

        Returns:
            List of text chunks with overlap
        """
        return self.rag.chunk_with_overlap(text, chunk_size, overlap)

    def ingest_document(
        self,
        file_path: str,
        class_id: str,
        chapter_id: str,
        chapter_name: str,
        class_name: str,
        subject_name: str,
        subject_id: str,
        collection_name: str = "documents",
    ):
        """Ingests a document with enhanced metadata extraction using multi-
        stage processing. Delegates to RAGSystem.

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
        return self.rag.ingest_document(
            file_path=file_path,
            class_id=class_id,
            chapter_id=chapter_id,
            chapter_name=chapter_name,
            class_name=class_name,
            subject_name=subject_name,
            subject_id=subject_id,
            collection_name=collection_name,
        )


if __name__ == "__main__":
    # Example usage
    try:
        service = UploadService()

        # Test ingestion
        result = service.ingest_document(
            file_path="data/sample.pdf",
            class_id="test_doc",
            chapter_id="chapter_1",
            chapter_name="Resume",
            class_name="resume",
            subject_name="resume",
            subject_id="subject_1",
        )

        logger.info(f"Ingestion result: {result}")

    except Exception as e:
        logger.exception(f"Ingestion test failed: {e}")
