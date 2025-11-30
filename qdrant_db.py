import os
import uuid
from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import time
from functools import wraps

from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"[TIME] {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class QdrantDB:
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, grpc_port: Optional[int] = None, prefer_grpc: bool = True, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Qdrant client. Model is loaded lazily on first use.
        
        Args:
            host: The hostname of the Qdrant server (defaults to QDRANT_HOST from .env).
            port: The HTTP port of the Qdrant server (defaults to QDRANT_PORT from .env).
            grpc_port: The gRPC port of the Qdrant server (defaults to QDRANT_GRPC_PORT from .env).
            prefer_grpc: Whether to use gRPC.
            model_name: Name of the Sentence Transformer model.
        """
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        # Use environment variables as defaults
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = int(port or os.getenv("QDRANT_PORT", 6333))
        self.grpc_port = int(grpc_port or os.getenv("QDRANT_GRPC_PORT", 6334))
        
        self.client = QdrantClient(host=self.host, port=self.port, grpc_port=self.grpc_port, prefer_grpc=prefer_grpc)
        logger.info(f"Connected to Qdrant at {self.host}:{self.port} (gRPC: {prefer_grpc}, gRPC Port: {self.grpc_port})")
        
        # Lazy loading: only load model when needed
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model on first access."""
        if self._model is None:
            # Try loading FastEmbed first for speed
            try:
                # Suppress ONNX Runtime warnings (like GPU discovery failures on Windows)
                import os
                os.environ["ORT_LOGGING_LEVEL"] = "3"
                
                from fastembed import TextEmbedding
                logger.info(f"Loading FastEmbed model '{self.model_name}'...")
                
                # Map common short names to FastEmbed supported names
                fastembed_model_name = self.model_name
                if self.model_name == "all-MiniLM-L6-v2":
                    fastembed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
                
                try:
                    self._model = TextEmbedding(model_name=fastembed_model_name)
                    self._model_type = "fastembed"
                    logger.info(f"FastEmbed model '{fastembed_model_name}' loaded successfully.")
                except ValueError:
                    logger.warning(f"Model '{fastembed_model_name}' not found in FastEmbed. Switching to default fast model...")
                    # Fallback to default FastEmbed model (usually BAAI/bge-small-en-v1.5) which is very good and fast
                    self._model = TextEmbedding()
                    self._model_type = "fastembed"
                    logger.info(f"Loaded default FastEmbed model (BAAI/bge-small-en-v1.5).")
                    
            except ImportError:
                logger.info("FastEmbed not installed. For faster loading, install it: pip install fastembed")
                logger.info("Falling back to SentenceTransformers...")
                self._load_sentence_transformer()
            except Exception as e:
                logger.warning(f"FastEmbed failed to load: {e}. Falling back to SentenceTransformers...")
                self._load_sentence_transformer()
                
        return self._model

    def _load_sentence_transformer(self):
        """Helper to load SentenceTransformer."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on {device}...")
            self._model = SentenceTransformer(self.model_name, device=device)
            self._model_type = "sentence_transformers"
            logger.info(f"Model loaded successfully.")
        except ImportError:
            raise ImportError("Neither 'fastembed' nor 'sentence-transformers' is installed. Please install one.")

    @timing_decorator
    def create_collection(self, collection_name: str, vector_size: int, distance: Distance = Distance.COSINE):
        """
        Creates a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection.
            vector_size: Dimension of the vectors.
            distance: Distance metric (Cosine, Euclidean, Dot).
        """
        try:
            if self.client.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists.")
            else:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance)
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")

    @timing_decorator
    def add_text(self, collection_name: str, text: str, payload: Optional[Dict[str, Any]] = None):
        """
        Adds a text entry to the collection. Embeddings are generated automatically.
        
        Args:
            collection_name: Name of the collection.
            text: The raw text content.
            payload: Additional metadata/filters.
        """
        if payload is None:
            payload = {}
        
        # Ensure text is in the payload for retrieval
        payload["text"] = text
        
        # Ensure model is loaded before checking type
        model = self.model
        
        # Generate embedding based on the loaded model type
        if getattr(self, '_model_type', '') == 'fastembed':
            # FastEmbed returns a generator of vectors
            embedding = list(model.embed([text]))[0]
        else:
            # SentenceTransformers returns a numpy array or tensor
            embedding = model.encode(text).tolist()
        
        point_id = str(uuid.uuid4())
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            logger.debug(f"Added text to '{collection_name}' with ID {point_id}")
        except Exception as e:
            logger.error(f"Error adding text to '{collection_name}': {e}")

    @timing_decorator
    def search_by_text(self, collection_name: str, query_text: str, limit: int = 5, filter_conditions: Optional[models.Filter] = None) -> List[Dict]:
        """
        Searches for similar texts using a text query (converted to vector).
        
        Args:
            collection_name: Name of the collection.
            query_text: The text to search for.
            limit: Number of results to return.
            filter_conditions: Optional Qdrant filters.
            
        Returns:
            List of search results with scores and payloads.
        """
        # Ensure model is loaded before checking type
        model = self.model
        
        # Generate embedding based on the loaded model type
        if getattr(self, '_model_type', '') == 'fastembed':
            query_vector = list(model.embed([query_text]))[0]
        else:
            query_vector = model.encode(query_text).tolist()
            
        return self.search_by_embedding(collection_name, query_vector, limit, filter_conditions)

    @timing_decorator
    def search_by_embedding(self, collection_name: str, query_vector: List[float], limit: int = 5, filter_conditions: Optional[models.Filter] = None) -> List[Dict]:
        """
        Searches for similar texts using a query embedding.
        
        Args:
            collection_name: Name of the collection.
            query_vector: The embedding vector to search with.
            limit: Number of results to return.
            filter_conditions: Optional Qdrant filters.
            
        Returns:
            List of search results with scores and payloads.
        """
        try:
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=filter_conditions,
                limit=limit
            )
            
            results = []
            # query_points returns QueryResponse with points
            for hit in search_result.points:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            return results
        except Exception as e:
            logger.error(f"Error searching in '{collection_name}': {e}")
            return []

    @timing_decorator
    def search_by_filter(self, collection_name: str, filter_conditions: models.Filter, limit: int = 10) -> List[Dict]:
        """
        Queries the database using only filters (no vector search).
        
        Args:
            collection_name: Name of the collection.
            filter_conditions: Qdrant filter conditions.
            limit: Number of results to return.
            
        Returns:
            List of matching points.
        """
        try:
            # Using scroll to get points based on filter
            scroll_result, _ = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_conditions,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            results = []
            for point in scroll_result:
                results.append({
                    "id": point.id,
                    "payload": point.payload
                })
            return results
        except Exception as e:
            logger.error(f"Error filtering in '{collection_name}': {e}")
            return []

    def close(self):
        """Closes the Qdrant client connection."""
        if self.client:
            self.client.close()
            logger.info("Qdrant client connection closed.")

if __name__ == "__main__":
    # Example Usage
    # Note: This requires a running Qdrant instance on localhost:6334
    
    db = QdrantDB()
    
    # 1. Create Collection
    COLLECTION_NAME = "test_collection"
    VECTOR_SIZE = 384 # Example size for all-MiniLM-L6-v2
    db.create_collection(COLLECTION_NAME, VECTOR_SIZE)
    
    # 2. Add Text
    sample_text = "This is a sample document about AI."
    # sample_embedding is now generated internally
    db.add_text(COLLECTION_NAME, sample_text, payload={"category": "tech"})
    
    # 3. Search by Text
    logger.info("--- Search by Text ---")
    results = db.search_by_text(COLLECTION_NAME, "AI document", limit=1)
    logger.info(results)
    
    # 4. Search by Filter
    logger.info("--- Search by Filter ---")
    f = models.Filter(
        must=[
            models.FieldCondition(
                key="category",
                match=models.MatchValue(value="tech")
            )
        ]
    )
    filter_results = db.search_by_filter(COLLECTION_NAME, f)
    logger.info(filter_results)
