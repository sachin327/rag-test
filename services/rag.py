import os
import time
from typing import Dict, List, Optional, Any
from qdrant_client.http import models

from document.document_loader import DocumentLoader
from llm.llm_open_router import LLMService
from logger import get_logger
from db.qdrant_db import QdrantDB
from utils.embedding import Embedding

# from utils.common import timing_decorator
from utils.response_format import ResponseSchema, JsonSchema, SummaryResponse
from utils.thread_pool import ThreadPoolManager
from utils.chunker import DocumentChunker
from utils.topic_embedder import TopicEmbedder
from utils.topic_search import TopicSearch

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
        summary_chunk_size: int = int(os.getenv("QDRANT_SUMMARY_CHUNK_SIZE", 4000)),
        summary_chunk_overlap: int = int(
            os.getenv("QDRANT_SUMMARY_CHUNK_OVERLAP", 100)
        ),
    ):
        """Initializes the RAG system with Qdrant DB.

        Args:
            collection_name: Name of the Qdrant collection.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.collection_name = collection_name
        self.llm_service = LLMService()
        self.thread_pool = ThreadPoolManager(max_workers=4)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.summary_chunk_size = summary_chunk_size
        self.summary_chunk_overlap = summary_chunk_overlap

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
        # self.db.create_payload_index(self.collection_name, index=models.Index())
        self.embedding = Embedding()
        self.topic_search = TopicSearch()

    def create_payload_index(self, fields: List[str]):
        for field in fields:
            self.db.create_payload_index(
                self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def build_filter(
        self, filter_dict: Optional[Dict[str, Any]] = None
    ) -> Optional[models.Filter]:
        """
        Build a Qdrant models.Filter from a dictionary of filters.

        Rules:
        - If a filter value is a `str` it is added to the `must` conditions
          (exact match).
        - If a filter value is a `list[str]` each element is added as a
          `should` condition for that key. If any `should` conditions exist
          the returned filter will include `min_should=1`.

        Returns None if no valid conditions are provided.
        """
        if not filter_dict:
            return None

        must_conditions: List[models.FieldCondition] = []
        should_conditions: List[models.FieldCondition] = []

        for key, value in filter_dict.items():
            if isinstance(value, str):
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                )
            elif isinstance(value, list):
                for v in value:
                    should_conditions.append(
                        models.FieldCondition(key=key, match=models.MatchValue(value=v))
                    )
            else:
                logger.debug(
                    f"Skipping unsupported filter type for {key}: {type(value)}"
                )

        if should_conditions:
            return models.Filter(must=must_conditions or None, should=should_conditions)

        if must_conditions:
            return models.Filter(must=must_conditions)

        return None

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
                schema=SummaryResponse.model_json_schema(),
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
                # logger.debug(event)
                response = event.get("response", {})
                # logger.debug(response)
                final_summary = response.get("summary", "")
                topic_keys = response.get("topics", [])
            logger.info(f"Generated {len(topic_keys)} topics: {topic_keys}")
            logger.info(f"Final summary: {len(final_summary)} chars")
        else:
            # Chunk into 4000 char chunks for summary generation
            chunker = DocumentChunker(
                target_tokens=self.summary_chunk_size,
                overlap_tokens=self.summary_chunk_overlap,
            )
            large_chunks = chunker.split_chunks(raw_text)
            logger.info(
                f"Created {len(large_chunks)} large chunks ({self.summary_chunk_size} chars each)"
            )

            # Generate summary for each large chunk in parallel
            logger.info(
                f"Generating summaries for {len(large_chunks)} chunks in parallel..."
            )

            def process_chunk(chunk_text):
                chunk_summary = ""
                chunk_topic_list = []
                try:
                    for event in self.llm_service.generate_summary(
                        chunk_text,
                        want_topics=True,
                        is_final=False,
                        response_schema=summary_response_schema,
                    ):
                        response = event.get("response", {})
                        chunk_summary = response.get("summary", "")
                        chunk_topic_list = response.get("topics", [])

                        # logger.info(f"Generated summary for chunk: {chunk_summary}")
                        # logger.info(f"Generated topics for chunk: {chunk_topic_list}")

                    # Format topics as string for the combined list
                    topics_str = "\n".join(topic["name"] for topic in chunk_topic_list)
                    return chunk_summary, topics_str
                except Exception as e:
                    logger.error(f"Error processing chunk summary: {e}")
                    return "", ""

            # Execute in parallel
            results = self.thread_pool.execute(process_chunk, large_chunks)

            chunk_summaries = [r[0] for r in results if r[0]]
            chunk_topics = [r[1] for r in results if r[1]]

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
        chunker = DocumentChunker(
            target_tokens=self.chunk_size, overlap_tokens=self.chunk_overlap
        )
        storage_chunks = chunker.split_chunks(raw_text)
        logger.info(f"Created {len(storage_chunks)} storage chunks")

        # Step 6.5: Embed topic keys
        topic_keys = self.topic_search.topics_exist_semantic(
            subject_id=metadata.get("subject_id"),
            input_topics=topic_keys,
            similarity_threshold=0.6,
        )
        topic_names = [
            f"{topic['name']} {topic['description']}" for topic in topic_keys
        ]
        topic_keys_embeddings = self.embedding.embed(topic_names)
        chunk_keys_embeddings = self.embedding.embed(storage_chunks)

        # Step 7: Prepare chunks for storage with metadata
        for i, chunk_text in enumerate(storage_chunks):
            # Count tokens (simple approximation: split by whitespace)
            words_count = len(chunk_text.split())

            # Search relavent topics
            chunk_embedding = chunk_keys_embeddings[i]
            relevant_topic_keys = TopicEmbedder.get_relevant_topics(
                chunk_embedding, topic_keys_embeddings
            )

            relevant_topics = [topic_keys[i] for i in relevant_topic_keys]

            # Base payload from metadata
            chunk_data = metadata.copy()

            # Add system fields
            chunk_data.update(
                {
                    "index": i,
                    "text": chunk_text,
                    "all_topic_keys": topic_keys,
                    "relevant_topic_keys": relevant_topics,
                    "source_file": os.path.basename(file_path),
                    "summary": final_summary,
                    "created_at": time.time(),
                    "words_count": words_count,
                    "sentence_count": chunk_text.count("."),
                }
            )

            self.db.add_text(self.collection_name, chunk_text, payload=chunk_data)

        return {
            "success": True,
            "metadata": metadata,
            "chunks_processed": len(storage_chunks),
            "topics_extracted": len(topic_keys),
            "topic_keys": topic_keys,
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
        filter_conditions = self.build_filter(filters)
        return self.db.search_by_text(
            self.collection_name,
            query,
            limit=limit,
            filter_conditions=filter_conditions,
        )

    def search_by_filter(
        self, filters: Optional[Dict[str, Any]] = None, limit: int = 5
    ):
        filter_conditions = self.build_filter(filters)
        return self.db.search_by_filter(
            self.collection_name,
            filter_conditions=filter_conditions,
            limit=limit,
        )

    def close(self):
        """Closes the RAG system connections."""
        if self.db:
            self.db.close()


if __name__ == "__main__":
    rag = RAGSystem(collection_name="ai-service")

    # Add a document (Example usage needs update for new signature)
    # result = rag.add_document(
    #     "data/iesc101.pdf", metadata={"class_name": "class_2", "subject_name": "subj_2"}
    # )
    # logger.info("\n--- Add Document Result ---")
    # logger.info(result)

    # Search
    results = rag.search("Newtons law", limit=3)
    logger.info("\n--- Search Results ---")
    for result in results:
        logger.info(f"Score: {result['score']:.4f}")
        logger.info(f"Text: {result['payload']['text'][:100]}...")
        logger.info(f"Metadata: {result['payload']}")
        logger.info("")
