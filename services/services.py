import os
from fastapi import HTTPException
from upload_service import UploadService
from llm_gemini import LLMService
from question_generation_service import QuestionGenerationService
from rag import RAGSystem
from embedding import EmbeddingService
from logger import get_logger

logger = get_logger(__name__)

QUESTION_SERVICE = None
UPLOAD_SERVICE = None
RAG_SYSTEM = None
LLM_SERVICE = None
EMBEDDING_SERVICE = None

def get_question_service() -> QuestionGenerationService:
    """Get or create Question Generation service instance."""
    global QUESTION_SERVICE
    if QUESTION_SERVICE is None:
        try:
            QUESTION_SERVICE = QuestionGenerationService()
            logger.info("Question Generation Service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Question Generation service: {e}")
            return None
    return QUESTION_SERVICE


def get_upload_service() -> UploadService:
    """Get or create Upload service instance."""
    global UPLOAD_SERVICE
    if UPLOAD_SERVICE is None:
        try:
            UPLOAD_SERVICE = UploadService()
            logger.info("Upload Service initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Upload service: {e}")
            return None
    return UPLOAD_SERVICE


def get_rag_system() -> RAGSystem:
    """Get or create RAG system instance."""
    # ... (existing code)
    global RAG_SYSTEM
    if RAG_SYSTEM is None:
        try:
            collection_name = os.getenv("QDRANT_COLLECTION_NAME")
            host = os.getenv("QDRANT_HOST")
            port = os.getenv("QDRANT_PORT")
            grpc_port = os.getenv("QDRANT_GRPC_PORT")
            api_key = os.getenv("QDRANT_API_KEY")
            RAG_SYSTEM = RAGSystem(
                collection_name=collection_name,
                host=host,
                port=port,
                grpc_port=grpc_port,
                api_key=api_key,
            )
        except Exception as e:
            logger.exception(f"Failed to connect to Qdrant database: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to Qdrant database. Please ensure Qdrant is running on localhost:6333. Error: {str(e)}",
            )
    return RAG_SYSTEM


def get_llm_service() -> LLMService:
    """Get or create LLM service instance."""
    global LLM_SERVICE
    if LLM_SERVICE is None:
        try:
            LLM_SERVICE = LLMService()
        except Exception as e:
            logger.warning(f"Failed to initialize LLM service: {e}")
            raise e
    return LLM_SERVICE


def get_embedding_service() -> EmbeddingService:
    """Get or create Embedding service instance."""
    global EMBEDDING_SERVICE
    if EMBEDDING_SERVICE is None:
        try:
            EMBEDDING_SERVICE = EmbeddingService()
        except Exception as e:
            logger.warning(f"Failed to initialize Embedding service: {e}")
            raise e
    return EMBEDDING_SERVICE