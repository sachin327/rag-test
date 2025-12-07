import os
import time
from typing import Dict, List, Optional, Any
from qdrant_client.http import models

from document.document_loader import DocumentLoader
from llm.llm_open_router import LLMService
from logger import get_logger
from db.qdrant_db import QdrantDB

# from utils.common import timing_decorator
from utils.response_format import ResponseSchema, JsonSchema, SummaryResponse

# Initialize logger
logger = get_logger(__name__)


class RAGSystem:
    """A Retrieval-Augmented Generation (RAG) system with Qdrant vector
    database."""

    def __init__(
        self,
        collection_name: str = os.getenv("QDRANT_COLLECTION_NAME"),
        chunk_size: int = int(os.getenv("QDRANT_CHUNK_SIZE", 512)),
        chunk_overlap: int = int(os.getenv("QDRANT_CHUNK_OVERLAP", 100)),
    ):
        """Initializes the RAG system with Qdrant DB.

        Args:
            collection_name: Name of the Qdrant collection.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_service = LLMService()

        # Initialize Qdrant DB with gRPC enabled
        self.db = QdrantDB(
            host=os.getenv("QDRANT_HOST"),
            port=os.getenv("QDRANT_PORT"),
            prefer_grpc=False,
            api_key=os.getenv("QDRANT_API_KEY"),
            vector_size=384,  # Fixed for all-MiniLM-L6-v2
        )

        # Create collection (512 is the vector size for all-MiniLM-L6-v2)
        self.db.create_collection(self.collection_name)

    def split_chunks(
        self, text: str, chunk_size: int = 512, overlap: int = 100
    ) -> List[str]:
        """Splits text into chunks with overlap.

        Args:
            text: Full text
            chunk_size: Target chunk size (default: uses self.chunk_size)
            overlap: Overlap size (default: uses self.chunk_overlap)

        Returns:
            List of text chunks with overlap
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Extend to sentence boundary if possible
            if end < text_length:
                next_period = text.find(".", end)
                next_space = text.find(" ", end)

                # Find the nearest boundary within reasonable distance
                if next_period != -1 and next_period - end < 100:
                    end = next_period + 1
                elif next_space != -1 and next_space - end < 50:
                    end = next_space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position considering overlap
            if end < text_length:
                start = end - overlap
            else:
                start = text_length

        return chunks

    def build_summary_context(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ):
        # Build context string from metadata
        context_parts = []
        if metadata:
            for key, value in metadata.items():
                # Format key to be more readable (e.g., "subject_name" -> "Subject Name")
                readable_key = key.replace("_", " ").title()
                context_parts.append(f"{readable_key}: {value}")

        context_str = "\n".join(context_parts) if context_parts else "General content"

        return context_str + "\nText: " + text

    def add_document(
        self,
        file_path: str,
        metadata: Dict[str, Any],
    ) -> Dict:
        """
        Ingests a document with enhanced metadata extraction using multi-stage processing:
        1. Chunk into 4000 chars and generate summaries
        2. Generate final summary and topics from all summaries
        3. Re-chunk into 512 chars with 100 char overlap for storage

        Args:
            file_path: Path to document file
            metadata: Dictionary containing document metadata (e.g., class_name, subject_name)

        Returns:
            Dictionary with upload document information
        """
        logger.info(f"Starting add document for {file_path}")

        # Step 1: Load document
        raw_text = DocumentLoader.load_document(file_path)

        if not raw_text:
            logger.error(f"No text loaded from {file_path}")
            return {"success": False, "error": "No text loaded"}

        logger.info(f"Loaded document: {len(raw_text)} chars")

        # Step 2: Generate summary and topics
        summary_response_schema = ResponseSchema(
            json_schema=JsonSchema(
                name="summary",
                schema_=SummaryResponse.model_json_schema(),
            )
        )
        if len(raw_text) < 4000:
            logger.info(
                "Document is small (< 4000 chars), generating summary directly."
            )
            final_summary = ""
            topic_keys = []
            for event in self.llm_service.generate_summary(
                raw_text,
                want_topics=True,
                is_final=True,
                response_schema=summary_response_schema,
            ):
                response = event.get("response", {})
                final_summary = response.get("summary", "")
                topic_keys = response.get("topics", [])
            logger.info(f"Generated {len(topic_keys)} topics: {topic_keys}")
            logger.info(f"Final summary: {len(final_summary)} chars")
        else:
            # Chunk into 4000 char chunks for summary generation
            large_chunks = self.split_chunks(raw_text, 4000, 100)
            logger.info(f"Created {len(large_chunks)} large chunks (4000 chars each)")

            # Generate summary for each large chunk
            chunk_summaries = []
            chunk_topics = []
            for i, chunk in enumerate(large_chunks):
                logger.info(f"Generating summary for chunk {i + 1}/{len(large_chunks)}")
                for event in self.llm_service.generate_summary(
                    chunk,
                    want_topics=True,
                    is_final=False,
                    response_schema=summary_response_schema,
                ):
                    # logger.debug(event)
                    response = event.get("response", {})
                    chunk_summaries.append(response.get("summary", ""))
                    chunk_topics.append(
                        "\n".join(topic["name"] for topic in response.get("topics", []))
                    )

            # Concatenate all summaries and topics
            combined_summaries = "\n\n".join(chunk_summaries)
            combined_topics = "\n".join(chunk_topics)
            logger.info(f"Combined summaries: {len(combined_summaries)} chars")
            logger.info(f"Combined topics: {len(combined_topics)} chars")

            # logger.debug(combined_summaries)
            # logger.debug(combined_topics)

            # Generate final summary and topics list (5-6 topics)
            logger.info("Generating final summary and topics list")
            final_summary = ""
            topic_keys = []
            for event in self.llm_service.generate_summary(
                combined_summaries,
                want_topics=True,
                is_final=True,
                response_schema=summary_response_schema,
            ):
                # logger.debug(event)
                response = event.get("response", {})
                final_summary = response.get("summary", "")
                topic_keys = response.get("topics", [])

            logger.info(f"Generated {len(topic_keys)} topics: {topic_keys}")
            logger.info(f"Final summary: {len(final_summary)} chars")

        # Step 6: Re-chunk into 512 chars with 100 char overlap for Qdrant storage
        storage_chunks = self.split_chunks(
            raw_text, chunk_size=self.chunk_size, overlap=self.chunk_overlap
        )
        logger.info(
            f"Created {len(storage_chunks)} storage chunks ({self.chunk_size} chars with {self.chunk_overlap} overlap)"
        )

        # Step 7: Prepare chunks for storage with metadata
        for i, chunk_text in enumerate(storage_chunks):
            # Count tokens (simple approximation: split by whitespace)
            words_count = len(chunk_text.split())

            # Base payload from metadata
            chunk_data = metadata.copy()

            # Add system fields
            chunk_data.update(
                {
                    "index": i,
                    "text": chunk_text,
                    "topic_keys": topic_keys,  # Use the global topics from LLM
                    "source_file": os.path.basename(file_path),
                    "summary": final_summary,
                    "created_at": time.time(),
                    "words_count": words_count,
                    "sentence_count": chunk_text.count("."),
                }
            )

            self.db.add_text(self.collection_name, chunk_text, payload=chunk_data)

        topic_flags_mongo = []
        for topic in topic_keys:
            topic_flags_mongo.append(True)

        return {
            "success": True,
            "metadata": metadata,
            "chunks_processed": len(storage_chunks),
            "topics_extracted": len(topic_keys),
            "topic_keys": topic_keys,
            "topic_flags_mongo": topic_flags_mongo,
            "summary": final_summary,
            "summary_length": len(final_summary),
        }

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        """Searches for relevant text chunks using a query and optional
        filters.

        Args:
            query: The search query text.
            limit: Number of results to return.
            filters: Dictionary of metadata filters (key=value).

        Returns:
            List of search results with scores and metadata.
        """
        filter_conditions = None
        must_conditions = []

        if filters:
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )

        if must_conditions:
            filter_conditions = models.Filter(must=must_conditions)

        return self.db.search_by_text(
            self.collection_name,
            query,
            limit=limit,
            filter_conditions=filter_conditions,
        )

    def close(self):
        """Closes the RAG system connections."""
        if self.db:
            self.db.close()


if __name__ == "__main__":
    rag = RAGSystem(collection_name="test_collection")

    # Add a document (Example usage needs update for new signature)
    result = rag.add_document(
        "data/iesc101.pdf", metadata={"class_name": "class_2", "subject_name": "subj_2"}
    )
    logger.info("\n--- Add Document Result ---")
    logger.info(result)

    # Search
    # results = rag.search("Newtons law", limit=3)
    # logger.info("\n--- Search Results ---")
    # for result in results:
    #     logger.info(f"Score: {result['score']:.4f}")
    #     logger.info(f"Text: {result['payload']['text'][:100]}...")
    #     logger.info(f"Metadata: {result['payload']}")
    #     logger.info("")
