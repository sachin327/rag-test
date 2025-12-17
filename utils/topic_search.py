import os
from typing import List, Dict, Any
import numpy as np
from db.mongo_db import MongoDB
from utils.embedding import Embedding
from dotenv import load_dotenv
from logger import get_logger
from utils.mongo_util import get_topics_from_subject_mongo

load_dotenv()
logger = get_logger(__name__)


class TopicSearch:
    def __init__(self):
        self.mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
        self.topics_collection = self.mongo.get_collection(
            os.getenv("MONGO_TOPIC_COLLECTION")
        )
        # Collection where actual topic details (and embeddings) are stored
        self.actual_topics_collection = self.mongo.get_collection("topics")
        self.embedding = Embedding(os.getenv("EMBEDDING_API_URL"))

    def _normalize_embeddings(self, arr: np.ndarray) -> np.ndarray:
        """
        L2-normalize a 2D array of embeddings row-wise.
        Handles zero vectors safely.
        """
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        # avoid division by zero
        norms[norms == 0] = 1.0
        return arr / norms

    def topics_exist_semantic(
        self,
        subject_id: str,
        input_topics: List[Dict[str, Any]],
        similarity_threshold: float = 0.8,
    ) -> List[bool]:
        """
        Check if each input topic already exists (semantically) in MongoDB
        for the given subject_id using embeddings + cosine similarity.
        """

        for topic in input_topics:
            topic["is_exists"] = False

        existing_docs = get_topics_from_subject_mongo(
            self.mongo, os.getenv("MONGO_TOPIC_COLLECTION"), subject_id
        )

        if not existing_docs:
            return input_topics

        # --- OPTIMIZATION START ---

        # 1. Exact Match Check (Fast Path)
        # Pre-process existing docs for faster lookup
        # Map normalized name -> doc
        existing_map = {}
        for doc in existing_docs:
            title = doc.get("title", "").strip().lower()
            if title:
                existing_map[title] = doc

        # Check inputs against map
        topics_to_check_semantic = []
        for i, topic in enumerate(input_topics):
            name = (topic.get("name") or topic.get("topic_name") or "").strip().lower()
            if name in existing_map:
                # Exact match found!
                matched_doc = existing_map[name]
                topic["is_exists"] = True
                topic["id"] = str(
                    matched_doc.get("_id", "")
                )  # _id might be missing in projection if 0
                # If projection excluded _id, we rely on topicId from loopup
                if not topic["id"] and "topicId" in matched_doc:
                    topic["id"] = str(matched_doc["topicId"])

                topic["name"] = matched_doc.get("title", "")
                topic["description"] = matched_doc.get("description", "")
                logger.debug(f"Exact match found for '{name}'")
            else:
                topics_to_check_semantic.append(i)

        if not topics_to_check_semantic:
            return input_topics

        # 2. Semantic Search with Caching

        # Filter existing docs to those that have titles
        valid_existing_docs = [doc for doc in existing_docs if doc.get("title")]
        if not valid_existing_docs:
            return input_topics

        # Identify which existing docs need embeddings
        docs_needing_embedding = []
        existing_embeddings = []

        for doc in valid_existing_docs:
            emb = doc.get("embedding")
            if emb and len(emb) > 0:
                existing_embeddings.append(emb)
            else:
                docs_needing_embedding.append(doc)

        # Bulk embed missing ones
        if docs_needing_embedding:
            logger.info(
                f"Computing embeddings for {len(docs_needing_embedding)} existing topics..."
            )
            texts_to_embed = [
                f"{doc.get('title', '')} {doc.get('description', '')}"
                for doc in docs_needing_embedding
            ]
            new_embeddings = self.embedding.embed(texts_to_embed)

            # Update MongoDB and local list
            from pymongo import UpdateOne

            bulk_ops = []

            for doc, emb in zip(docs_needing_embedding, new_embeddings):
                existing_embeddings.append(emb)
                # We need the original _id to update.
                # Our aggregation projection in available context excludes _id but includes topicId.
                # topicId is the _id in the 'topics' collection.
                topic_id = doc.get("topicId")
                if topic_id:
                    bulk_ops.append(
                        UpdateOne({"_id": topic_id}, {"$set": {"embedding": emb}})
                    )

            if bulk_ops:
                try:
                    self.actual_topics_collection.bulk_write(bulk_ops)
                    logger.info(f"Cached {len(bulk_ops)} embeddings to MongoDB")
                except Exception as e:
                    logger.error(f"Failed to cache embeddings: {e}")

        # Now we have existing_embeddings for ALL valid_existing_docs (aligned order? No, we just appended)
        # Wait, the order in 'existing_embeddings' must match 'valid_existing_docs' for us to know WHICH doc matched.
        # Let's rebuild the list carefully to ensure alignment.

        final_existing_docs = []  # Will contain docs with embeddings
        final_existing_matrix = []

        # We need a map of topicId -> embedding to reconstruct easily or just iterate carefully?
        # Simpler: We just computed them. Let's merge them back into the docs objects in memory.

        # Map newly computed embeddings back to docs
        if docs_needing_embedding:
            for doc, emb in zip(docs_needing_embedding, new_embeddings):
                doc["embedding"] = emb

        # Now all valid_existing_docs should have embeddings (if we didn't fail)
        for doc in valid_existing_docs:
            if doc.get("embedding"):
                final_existing_docs.append(doc)
                final_existing_matrix.append(doc["embedding"])

        if not final_existing_matrix:
            return input_topics

        # 3. Compute Embeddings for Inputs (only those needing check)
        input_idxs_to_process = topics_to_check_semantic
        input_names = []
        for i in input_idxs_to_process:
            t = input_topics[i]
            name = t.get("name") or t.get("topic_name") or ""
            description = t.get("description") or ""
            input_names.append(name + " " + description)

        input_embs = self.embedding.embed(input_names)

        # Convert to numpy
        existing_embs_np = np.array(final_existing_matrix, dtype=float)
        input_embs_np = np.array(input_embs, dtype=float)

        # Normalize
        existing_norm = self._normalize_embeddings(existing_embs_np)
        input_norm = self._normalize_embeddings(input_embs_np)

        # Similarity
        sim_matrix = np.dot(input_norm, existing_norm.T)

        # Check thresholds
        for idx_in_matrix, original_idx in enumerate(input_idxs_to_process):
            row = sim_matrix[idx_in_matrix]
            if row.size > 0:
                best_idx = int(row.argmax())
                best_sim = float(row[best_idx])
            else:
                best_idx = None
                best_sim = 0.0

            matched_doc = final_existing_docs[best_idx]
            logger.debug(
                f"Input topic '{input_names[idx_in_matrix]}' best match: '{matched_doc.get('title')}' sim: {best_sim}"
            )

            if best_sim >= similarity_threshold:
                input_topics[original_idx]["is_exists"] = True
                # Use topicId as id, similar to logic above
                input_topics[original_idx]["id"] = str(matched_doc.get("topicId", ""))
                input_topics[original_idx]["name"] = matched_doc.get("title", "")
                input_topics[original_idx]["description"] = matched_doc.get(
                    "description", ""
                )
            else:
                input_topics[original_idx]["is_exists"] = False

        return input_topics


if __name__ == "__main__":
    mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
    topics_collection = mongo.get_collection(os.getenv("MONGO_TOPIC_COLLECTION"))

    input_topics = [
        {"name": "Metals vs. Non-metals", "relevance": 0.95, "description": ""},
        {"name": "Kinetic Theory of Gases", "relevance": 0.88, "description": ""},
        {"name": "Random LLM Topic", "relevance": 0.5, "description": ""},
    ]

    logger.info("Input topics: %s", input_topics)

    topic_search = TopicSearch()
    present_flags = topic_search.topics_exist_semantic(
        subject_id="67fed6e64f4451c4718bd135",
        input_topics=input_topics,
        similarity_threshold=0.6,  # tune 0.75–0.85 based on experiments
    )

    logger.info(present_flags)  # e.g. [True, False, False]
