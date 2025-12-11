"""Enhanced ingestion service for question generation system.

Uses RAGSystem from rag.py for all chunking, summarization, and
ingestion functionality.
"""

import os
from typing import List
from logger import get_logger
from services.rag import RAGSystem
from dotenv import load_dotenv
from llm.llm_open_router import LLMService
from utils.response_format import (
    ResponseSchema,
    JsonSchema,
    RagQueryResponse,
)
from db.redis_db import RedisDB

logger = get_logger(__name__)
load_dotenv()


class QueryService:
    """Upload service for document ingestion."""

    def __init__(self):
        """Initialize the upload service."""
        logger.info("Upload Service initialized")

        self.rag_service = RAGSystem()
        self.rag_service.create_payload_index(
            fields=["class_id", "chapter_id", "subject_id"]
        )
        self.llm_service = LLMService()

        self.redis_db = RedisDB(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            channel=os.getenv("REDIS_CHANNEL", "ai-service"),
            user=os.getenv("REDIS_USER", "default"),
            password=os.getenv("REDIS_PASSWORD", ""),
        )

    def query(
        self,
        query: str,
        class_id: str,
        subject_id: str,
        chapter_ids: List[str],
        stream: bool = False,
    ):
        """
        Query documents from Qdrant DB and generate an answer using
        LLM Open Router.

        Args:
            query: The user's query/question
            class_id: Filter by class ID
            subject_id: Filter by subject ID
            chapter_id: Filter by chapter ID
            stream: Whether to stream the response

        Returns:
            LLM response (either complete dict or generator if stream=True)
        """
        try:
            # Search for relevant documents in Qdrant with filters
            logger.info(f"Searching for documents matching: {query}")

            search_filters = {
                "class_id": class_id,
                "subject_id": subject_id,
                "chapter_ids": chapter_ids,
            }

            search_results = self.rag_service.search(
                query=query,
                limit=5,
                filters=search_filters,
            )

            if not search_results:
                logger.warning("No documents found matching the query and filters")
                return {
                    "response": "No relevant documents found for this query.",
                    "sources": [],
                }

            rag_response_schema = ResponseSchema(
                json_schema=JsonSchema(
                    name="answer",
                    schema=RagQueryResponse.model_json_schema(),
                )
            )

            response = self.llm_service.generate_rag_response(
                query=query,
                context=search_results,
                stream=stream,
                response_schema=rag_response_schema,
            )

            # Handle streaming vs non-streaming responses
            if stream:
                for event in response:
                    self.redis_db.publish(event)
            else:
                for event in response:
                    # logger.debug(event)
                    response = event.get("response", {})
                    # logger.debug(response)
                    answer = response.get("answer", "")
                    sources = response.get("sources", [])
                return {"answer": answer, "sources": sources}

        except Exception as e:
            logger.exception(f"Error during query processing: {e}")
            raise e
