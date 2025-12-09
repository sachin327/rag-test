import os
from typing import List, Dict, Any
from pymongo.collection import Collection
import numpy as np
from bson.objectid import ObjectId
from db.mongo_db import MongoDB
from utils.embedding import Embedding
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def _normalize_embeddings(arr: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 2D array of embeddings row-wise.
    Handles zero vectors safely.
    """
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # avoid division by zero
    norms[norms == 0] = 1.0
    return arr / norms


def topics_exist_semantic(
    topics_collection: Collection,
    subject_id: str,
    input_topics: List[Dict[str, Any]],
    similarity_threshold: float = 0.8,
) -> List[bool]:
    """
    Check if each input topic already exists (semantically) in MongoDB
    for the given subject_id using embeddings + cosine similarity.

    Args:
        topics_collection: PyMongo collection object.
        subject_id: Subject identifier to filter topics in Mongo.
        input_topics: List of dicts, each with at least "name" or "topic_name".
        embedding: Object with method `embed(list_of_texts) -> list_of_vectors`.
        similarity_threshold: Cosine similarity threshold in [0, 1].
                              If max similarity >= threshold, topic is "present".

    Returns:
        List[bool] of same length as input_topics.
    """

    for topic in input_topics:
        topic["is_exists"] = False

    # 1. Get existing topics for this subject_id
    pipeline = [
        # 1. Match/Filter: Find documents in chapterTopics by subjectId
        {"$match": {"subjectId": ObjectId(subject_id)}},
        # 2. Lookup/Join: Join chapterTopics with the topics collection
        {
            "$lookup": {
                "from": "topics",  # The collection to join (the 'topics' collection)
                "localField": "topicId",  # Field from the input documents (chapterTopics)
                "foreignField": "_id",  # Field from the documents of the 'from' collection (topics)
                "as": "topicDetails",  # The name for the new array field in the output documents
            }
        },
        # 3. Unwind (Optional but Recommended): Deconstruct the 'topicDetails' array.
        # Since topicId is likely unique, this turns the 'topicDetails' array
        # (which contains one document) into a single object.
        {"$unwind": "$topicDetails"},
        # 4. Project: Shape the final output to include only the fields you need
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "topicId": "$topicId",  # Keep the topicId
                # Get the title from the joined document
                "title": "$topicDetails.title",
            }
        },
    ]

    existing_docs = list(topics_collection.aggregate(pipeline))

    # topic_names = [doc["topicId"] for doc in existing_docs if doc.get("topicId")]

    logger.debug("Input topics: %s", input_topics)
    logger.debug("Existing topics: %s", existing_docs)

    # If no existing topics, nothing can match
    if not existing_docs:
        return input_topics

    existing_names = [doc["title"] for doc in existing_docs if doc.get("title")]

    # If still empty after filtering
    if not existing_names:
        return input_topics

    # 2. Build list of input names (aligned with input_topics)
    input_names: List[str] = []
    for t in input_topics:
        name = t.get("name") or t.get("topic_name") or ""
        input_names.append(name)

    # If all input names are empty, early-return
    if not any(input_names):
        return input_topics

    # 3. Compute embeddings
    #    (One shot for all existing, one shot for all input)
    embedding = Embedding(os.getenv("EMBEDDING_API_URL"))
    logger.debug("Existing names: %s", existing_names)
    logger.debug("Input names: %s", input_names)
    existing_emb_list = embedding.embed(existing_names)  # list of vectors
    input_emb_list = embedding.embed(input_names)  # list of vectors

    # Convert to numpy arrays
    existing_embs = np.array(existing_emb_list, dtype=float)  # (M, D)
    input_embs = np.array(input_emb_list, dtype=float)  # (N, D)

    # 4. Normalize (L2) so cosine = dot product
    existing_norm = _normalize_embeddings(existing_embs)  # (M, D)
    input_norm = _normalize_embeddings(input_embs)  # (N, D)

    # 5. Cosine similarity matrix: (N, M)
    #    sim[i, j] = cosine similarity between input i and existing j
    sim_matrix = np.dot(input_norm, existing_norm.T)  # (N, M)

    # 6. For each input topic, check if any similarity >= threshold

    for i in range(len(input_topics)):
        row = sim_matrix[i]  # shape (M,)
        if row.size > 0:
            best_idx = int(row.argmax())  # index of the max value
            best_sim = float(row[best_idx])
        else:
            best_idx = None
            best_sim = 0.0

        logger.debug(
            f"Input topic '{input_names[i]}' Exist topic: {existing_names[best_idx]} similarity: {best_sim}"
        )

        if best_sim >= similarity_threshold:
            input_topics[i]["is_exists"] = True
            input_topics[i]["name"] = existing_names[best_idx]

        else:
            input_topics[i]["is_exists"] = False

    return input_topics


if __name__ == "__main__":
    mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
    topics_collection = mongo.get_collection("chaptertopics")

    input_topics = [
        {"name": "Metals vs. Non-metals", "relevance": 0.95},
        {"name": "Kinetic Theory of Gases", "relevance": 0.88},
        {"name": "Random LLM Topic", "relevance": 0.5},
    ]

    logger.info("Input topics: %s", input_topics)

    present_flags = topics_exist_semantic(
        topics_collection=topics_collection,
        subject_id="67fed6e64f4451c4718bd135",
        input_topics=input_topics,
        similarity_threshold=0.6,  # tune 0.75–0.85 based on experiments
    )

    print(present_flags)  # e.g. [True, False, False]
