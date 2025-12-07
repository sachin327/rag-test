import os
import re
from typing import List, Dict, Any
from pymongo.collection import Collection
from rapidfuzz import fuzz, process
from logger import get_logger

logger = get_logger(__name__)


def slugify_topic(name: str) -> str:
    """
    Normalize a topic name into a slug for storage/indexing:
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
    - remove punctuation
    - collapse whitespace
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
    similarity_threshold: int = 85,  # be strict
    logger=None,
) -> List[bool]:
    """
    Check if each input topic already exists for a given subject_id.
    Returns a list[bool] same length as input_topics.
    """

    # 1. Fetch existing topics from Mongo
    existing_docs = list(
        topics_collection.find(
            {"subject_id": subject_id},
            {"_id": 0, "topic_name": 1},
        )
    )

    existing_raw_names: List[str] = []
    existing_norms: List[str] = []

    for doc in existing_docs:
        db_name = doc.get("topic_name")
        if not db_name:
            continue
        existing_raw_names.append(db_name)
        existing_norms.append(normalize_for_fuzzy(db_name))

    if logger:
        logger.info(
            f"Existing topics for subject_id={subject_id}: {existing_raw_names}"
        )

    results: List[bool] = []

    # 2. For each input topic, find best fuzzy match
    for topic in input_topics:
        topic_name = topic.get("name") or topic.get("topic_name")
        if not topic_name:
            results.append(False)
            continue

        in_norm = normalize_for_fuzzy(topic_name)

        if not existing_norms:
            # No topics in DB → nothing can match
            results.append(False)
            if logger:
                logger.info(
                    f"Topic '{topic_name}': no existing topics to compare, result=False"
                )
            continue

        # Find best match using token_set_ratio
        best_match, best_score, best_idx = process.extractOne(
            in_norm,
            existing_norms,
            scorer=fuzz.token_set_ratio,
        )

        exists = best_score >= similarity_threshold
        results.append(exists)

        if logger:
            logger.debug(
                f"Topic '{topic_name}' (norm='{in_norm}') "
                f"best match='{existing_raw_names[best_idx]}' "
                f"(norm='{best_match}') score={best_score} → exists={exists}"
            )

    return results


if __name__ == "__main__":
    from db.mongo_db import MongoDB

    mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
    topics_collection = mongo.get_collection("topic-ai-service")

    input_topics = [
        {"name": "Newtons house", "relevance": 0.95},
        {"name": "Human in the loop", "relevance": 0.88},
        {"name": "Random LLM Topic", "relevance": 0.5},
    ]

    logger.info(f"Input topics: {input_topics}")

    exists_flags = topics_exist_for_subject(
        topics_collection=topics_collection,
        subject_id="subject_1",
        input_topics=input_topics,
        similarity_threshold=70,
        logger=logger,
    )

    logger.info(str(exists_flags))
