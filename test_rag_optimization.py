import os
import sys
import time
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.getcwd())

from services.rag import RAGSystem
from logger import get_logger
from qdrant_client.http import models

logger = get_logger(__name__)
load_dotenv()


def test_rag_optimization():
    rag = RAGSystem(collection_name="test_optimization")
    # Recreate collection
    rag.db.client.delete_collection("test_optimization")
    rag.db.create_collection("test_optimization")

    # Create index for project
    rag.db.create_payload_index(
        rag.collection_name, "project", models.PayloadSchemaType.KEYWORD
    )

    # 1. Test Small File (< 4000 chars)
    print("\n--- Testing Small File (< 4000 chars) ---")
    os.makedirs("data", exist_ok=True)
    small_file = "data/small_doc.txt"
    with open(small_file, "w") as f:
        f.write("This is a small document. " * 100)  # ~2600 chars

    metadata = {"project": "Small Project"}

    try:
        result = rag.add_document(small_file, metadata)
        print(f"Small Doc Result: {result['success']}")
        print(f"Summary Length: {result['summary_length']}")
        # We can't easily assert internal logging, but we can check if it worked.
    except Exception as e:
        logger.exception(f"Small doc ingestion failed: {e}")

    # 2. Test Large File (> 4000 chars)
    print("\n--- Testing Large File (> 4000 chars) ---")
    large_file = "data/large_doc.txt"
    with open(large_file, "w") as f:
        f.write("This is a large document. " * 200)  # ~5200 chars

    metadata_large = {"project": "Large Project"}

    try:
        result = rag.add_document(large_file, metadata_large)
        print(f"Large Doc Result: {result['success']}")
        print(f"Summary Length: {result['summary_length']}")
    except Exception as e:
        logger.exception(f"Large doc ingestion failed: {e}")


if __name__ == "__main__":
    test_rag_optimization()
