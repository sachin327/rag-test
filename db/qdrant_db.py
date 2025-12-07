import os
import uuid
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from utils.embedding import embedding
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)


class QdrantDB:
    def __init__(
        self,
        host,
        port,
        vector_size,
        prefer_grpc: bool = False,
        model_name: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None,
    ):
        """Initializes the Qdrant client. Model is loaded lazily on first use.

        Args:
            host: The hostname of the Qdrant server (defaults to QDRANT_HOST from .env).
            port: The HTTP port of the Qdrant server (defaults to QDRANT_PORT from .env).
            grpc_port: The gRPC port of the Qdrant server (defaults to QDRANT_GRPC_PORT from .env).
            prefer_grpc: Whether to use gRPC.
            model_name: Name of the Sentence Transformer model.
        """

        self.client = QdrantClient(
            url=f"{host}:{port}",
            api_key=api_key,
            prefer_grpc=prefer_grpc,
        )
        logger.info(f"Connected to Qdrant at {host}:{port} (gRPC: {prefer_grpc})")

        # Lazy loading: only load model when needed
        self.model_name = model_name
        self._model = None
        self.vector_size = vector_size

    def create_collection(
        self,
        collection_name: str,
        distance: Distance = Distance.COSINE,
    ):
        """Creates a new collection in Qdrant.

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
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=distance
                    ),
                )
                logger.info(f"Collection '{collection_name}' created successfully.")
        except Exception as e:
            logger.exception(f"Error creating collection '{collection_name}': {e}")

    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: models.PayloadSchemaType = models.PayloadSchemaType.KEYWORD,
    ):
        """Creates a payload index for a specific field.

        Args:
            collection_name: Name of the collection.
            field_name: Name of the field to index.
            field_schema: Type of index (Keyword, Integer, Float, Geo, Text).
        """
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            logger.info(
                f"Index created for field '{field_name}' in collection '{collection_name}'."
            )
        except Exception as e:
            logger.exception(
                f"Error creating index for field '{field_name}' in '{collection_name}': {e}"
            )

    def add_text(
        self, collection_name: str, text: str, payload: Optional[Dict[str, Any]] = None
    ):
        """Adds a text entry to the collection. Embeddings are generated
        automatically.

        Args:
            collection_name: Name of the collection.
            text: The raw text content.
            payload: Additional metadata/filters.
        """
        if payload is None:
            payload = {}

        # Ensure text is in the payload for retrieval
        payload["text"] = text

        # Generate embedding based on the loaded model type
        embd = embedding.embed([text])[0]

        point_id = str(uuid.uuid4())

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[models.PointStruct(id=point_id, vector=embd, payload=payload)],
            )
            # logger.debug(f"Added text to '{collection_name}' with ID {point_id}")
        except Exception as e:
            logger.exception(f"Error adding text to '{collection_name}': {e}")

    def search_by_text(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 5,
        filter_conditions: Optional[models.Filter] = None,
    ) -> List[Dict]:
        """Searches for similar texts using a text query (converted to vector).

        Args:
            collection_name: Name of the collection.
            query_text: The text to search for.
            limit: Number of results to return.
            filter_conditions: Optional Qdrant filters.

        Returns:
            List of search results with scores and payloads.
        """
        query_vector = embedding.embed([query_text])[0]
        try:
            search_result = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=filter_conditions,
                limit=limit,
            )

            results = []
            # query_points returns QueryResponse with points
            for hit in search_result.points:
                results.append(
                    {"id": hit.id, "score": hit.score, "payload": hit.payload}
                )
            return results
        except Exception as e:
            logger.exception(f"Error searching in '{collection_name}': {e}")
            return []

    def search_by_filter(
        self, collection_name: str, filter_conditions: models.Filter, limit: int = 10
    ) -> List[Dict]:
        """Queries the database using only filters (no vector search).

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
                with_vectors=False,
            )

            results = []
            for point in scroll_result:
                results.append({"id": point.id, "payload": point.payload})
            return results
        except Exception as e:
            logger.exception(f"Error filtering in '{collection_name}': {e}")
            return []

    def close(self):
        """Closes the Qdrant client connection."""
        if self.client:
            self.client.close()
            logger.info("Qdrant client connection closed.")


if __name__ == "__main__":
    # Example Usage
    # Note: This requires a running Qdrant instance on localhost:6334
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    host = os.getenv("QDRANT_HOST")
    port = os.getenv("QDRANT_PORT")
    vector_size = os.getenv("QDRANT_VECTOR_SIZE")
    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", False).lower() == "true"
    model_name = os.getenv("QDRANT_MODEL_NAME")
    api_key = os.getenv("QDRANT_API_KEY")
    db = QdrantDB(
        host=host,
        port=port,
        vector_size=vector_size,
        prefer_grpc=prefer_grpc,
        model_name=model_name,
        api_key=api_key,
    )

    # 1. Create Collection
    COLLECTION_NAME = "test_collection"
    VECTOR_SIZE = 384  # Example size for all-MiniLM-L6-v2
    db.create_collection(COLLECTION_NAME, VECTOR_SIZE)

    # 1.1 Create Payload Index
    db.create_payload_index(
        COLLECTION_NAME, "category", models.PayloadSchemaType.KEYWORD
    )

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
            models.FieldCondition(key="category", match=models.MatchValue(value="tech"))
        ]
    )
    filter_results = db.search_by_filter(COLLECTION_NAME, f)
    logger.info(filter_results)
