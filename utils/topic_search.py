import os
import re
from typing import List, Dict, Any
from pymongo.collection import Collection
from rapidfuzz import fuzz
from logger import get_logger

logger = get_logger(__name__)


def slugify_topic(name: str) -> str:
    """
    Normalize a topic name into a slug for storage/indices:
    - lowercase
    - remove non-alphanumeric characters (except spaces)
    - collapse whitespace
    - replace spaces with hyphens
    """
    if not isinstance(name, str):
        name = str(name)

    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name.replace(" ", "-")


def normalize_for_fuzzy(name: str) -> str:
    """
    Normalize text for fuzzy matching:
    - lowercase
    - remove most punctuation
    - collapse whitespace
    (keep spaces so token-based matching works well)
    """
    if not isinstance(name, str):
        name = str(name)

    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s]", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def topics_exist_for_subject(
    topics_collection: Collection,
    subject_id: str,
    input_topics: List[Dict[str, Any]],
    similarity_threshold: int = 80,
) -> List[bool]:
    """
    Given a MongoDB collection and a list of topics (dicts with 'name' or 'topic_name'),
    check if each topic already exists for a given subject_id.

    - Fetches existing topics from Mongo by subject_id
    - Uses rapidfuzz.token_set_ratio for fuzzy string similarity
    - Returns list of booleans (True if topic is considered present)

    similarity_threshold:
        0-100. Typical good values: 75-90.
    """

    # 1. Fetch existing topics for this subject_id
    existing_docs = list(
        topics_collection.find(
            {"subject_id": subject_id},
            {"_id": 0, "topic_name": 1},
        )
    )

    logger.info(f"Found {existing_docs} existing topics for subject_id: {subject_id}")

    # 2. Prepare normalized names for fuzzy matching
    existing_norms: List[str] = []
    for doc in existing_docs:
        db_name = doc.get("topic_name")
        if not db_name:
            continue
        existing_norms.append(normalize_for_fuzzy(db_name))

    results: List[bool] = []

    # 3. For each input topic, fuzzy match against existing_norms
    for topic in input_topics:
        topic_name = topic.get("name") or topic.get("topic_name")
        if not topic_name:
            results.append(False)
            continue

        in_norm = normalize_for_fuzzy(topic_name)

        if not existing_norms:
            results.append(False)
            continue

        # Compute best token_set_ratio similarity vs all existing topics
        best_score = max(
            fuzz.token_set_ratio(in_norm, db_norm) for db_norm in existing_norms
        )

        # Debug/logging if you want
        # logger.info(f"Topic '{topic_name}' best score: {best_score}")

        results.append(best_score >= similarity_threshold)

    return results


if __name__ == "__main__":
    from db.mongo_db import MongoDB

    mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
    topics_collection = mongo.get_collection("topic-ai-service")

    input_topics = [
        {"name": "Laws of Motion", "relevance": 0.95},
        {"name": "Kinetic Theory of Gases", "relevance": 0.88},
        {"name": "Random LLM Topic", "relevance": 0.5},
    ]

    logger.info(f"Input topics: {input_topics}")

    exists_flags = topics_exist_for_subject(
        topics_collection=topics_collection,
        subject_id="subject_1",
        input_topics=input_topics,
        similarity_threshold=0.85,
    )

    # e.g. [True, True, False]
    logger.info(exists_flags)
