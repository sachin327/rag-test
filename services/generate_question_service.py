"""Question Generation Service Implements the core question generation logic
with Qdrant retrieval, clustering, LLM generation, and MongoDB caching."""

import json
import time
from typing import Dict, List, Optional

import numpy as np
from qdrant_client.http import models

from llm_gemini import LLMService
from logger import get_logger
from mongo_db import MongoDB
from qdrant_db import QdrantDB
from question_utils import (
    cluster_embeddings,
    compute_overlap_count,
    deduplicate_by_similarity,
    merge_chunks_by_budget,
    normalize_topics,
    select_diverse_questions,
)

logger = get_logger(__name__)


class GenerateQuestionService:
    def __init__(self):
        """Initialize the question generation service."""
        self.qdrant = QdrantDB()
        self.mongo = MongoDB()
        self.llm = LLMService()

        logger.info("Generate Question Service initialized")

        # Configuration
        self.candidate_pool_size = 80
        self.token_budget_per_cluster = 1200
        self.dedupe_threshold = 0.92
        self.questions_per_cluster = 1

        logger.info("Question Generation Service initialized")

    def generate_questions_for_topic_list(
        self,
        class_id: str,
        chapter_id: str,
        input_topics: List[str],
        n: int = 10,
        mode: str = "or",
        collection_name: str = "documents",
    ) -> List[Dict]:
        """Generates N questions for given topics using RAG approach.

        Args:
            class_id: Document identifier
            chapter_id: Chapter identifier
            input_topics: List of topic strings
            n: Number of questions to generate
            mode: 'or' (any topic) or 'and' (all topics)
            collection_name: Qdrant collection name

        Returns:
            List of question dictionaries
        """
        logger.info(
            f"Generating {n} questions for doc={class_id}, chapter={chapter_id}, topics={input_topics}, mode={mode}"
        )
        start_time = time.time()

        # Step 1: Normalize topics
        normalized_topics = normalize_topics(input_topics)
        logger.info(f"Normalized topics: {normalized_topics}")

        # Step 2: Check MongoDB cache
        cached_questions = self._check_cache(
            class_id, chapter_id, normalized_topics, n, mode
        )
        if len(cached_questions) >= n:
            logger.info(f"Returning {len(cached_questions)} cached questions")
            return cached_questions[:n]

        # Step 3: Retrieve candidates from Qdrant
        candidates = self._query_qdrant_for_topics(
            class_id, chapter_id, normalized_topics, collection_name
        )

        if not candidates:
            logger.warning("No candidates found in Qdrant")
            return cached_questions  # Return whatever we have from cache

        logger.info(f"Retrieved {len(candidates)} candidates from Qdrant")

        # Step 4: Cluster candidates
        clusters = self._cluster_candidates(candidates, n)

        # Step 5: Generate questions per cluster
        generated_questions = []
        for cluster_idx, cluster_indices in enumerate(clusters):
            cluster_chunks = [candidates[i] for i in cluster_indices]

            # Build context for this cluster
            context = merge_chunks_by_budget(
                cluster_chunks, self.token_budget_per_cluster
            )

            # Generate questions
            questions = self._llm_generate_questions(
                context, normalized_topics, self.questions_per_cluster
            )

            # Add cluster info
            for q in questions:
                q["cluster_id"] = cluster_idx

            generated_questions.extend(questions)

        logger.info(f"Generated {len(generated_questions)} questions from LLM")

        # Step 6: Compute embeddings for deduplication
        for q in generated_questions:
            q["embedding"] = self._compute_question_embedding(q["question_text"])

        # Step 7: Deduplicate
        unique_questions = deduplicate_by_similarity(
            generated_questions, self.dedupe_threshold
        )
        logger.info(f"After deduplication: {len(unique_questions)} questions")

        # Step 8: Select top N diverse questions
        selected_questions = select_diverse_questions(
            unique_questions, n, normalized_topics
        )

        # Step 9: Add metadata and persist to MongoDB
        questions_collection = self.mongo.get_questions_collection()
        for q in selected_questions:
            del q["embedding"]
            q["class_id"] = class_id
            q["chapter_id"] = chapter_id
            q["created_at"] = time.time()
            q["status"] = "draft"
            q["origin"] = "generated"
            q["overlap_count"] = compute_overlap_count(
                q.get("topic_keys", []), normalized_topics
            )

            # Insert into MongoDB
            try:
                result = questions_collection.insert_one(q.copy())
                q["_id"] = str(result.inserted_id)
            except Exception as e:
                logger.exception(f"Failed to insert question to MongoDB: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"Question generation completed in {elapsed:.2f}s, returning {len(selected_questions)} questions"
        )

        return selected_questions

    def _check_cache(
        self, class_id: str, chapter_id: str, topics: List[str], n: int, mode: str
    ) -> List[Dict]:
        """Checks MongoDB cache for existing questions.

        Args:
            class_id: Document ID
            chapter_id: Chapter ID
            topics: Normalized topics
            n: Number of questions needed
            mode: 'or' or 'and'

        Returns:
            List of cached questions
        """
        return self.mongo.search_questions(
            class_id=class_id,
            chapter_id=chapter_id,
            topic_keys=topics,
            difficulty=None,
            limit=n,
            sort_by="created_at",
        )

    def _query_qdrant_for_topics(
        self, class_id: str, chapter_id: str, topics: List[str], collection_name: str
    ) -> List[Dict]:
        """Queries Qdrant for chunks matching topics.

        Args:
            class_id: Document ID
            chapter_id: Chapter ID
            topics: Normalized topics
            collection_name: Qdrant collection

        Returns:
            List of chunk dictionaries
        """
        # Build filter
        must_conditions = [
            models.FieldCondition(
                key="class_id", match=models.MatchValue(value=class_id)
            ),
            models.FieldCondition(
                key="chapter_id", match=models.MatchValue(value=chapter_id)
            ),
        ]

        should_conditions = [
            models.FieldCondition(
                key="topic_keys", match=models.MatchValue(value=topic)
            )
            for topic in topics
        ]

        filter_conditions = models.Filter(
            must=must_conditions,
            should=should_conditions if should_conditions else None,
        )

        # Search with filter only (no query vector for now)
        try:
            # Use scroll to get all matching points
            results, _ = self.qdrant.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_conditions,
                limit=self.candidate_pool_size,
                with_payload=True,
                with_vectors=True,
            )

            # Convert to list of dicts
            candidates = []
            for point in results:
                payload = point.payload
                payload["embedding"] = point.vector
                candidates.append(payload)

            return candidates

        except Exception as e:
            logger.exception(f"Qdrant query failed: {e}")
            return []

    def _cluster_candidates(self, candidates: List[Dict], n: int) -> List[List[int]]:
        """Clusters candidates by embeddings.

        Args:
            candidates: List of candidate chunks
            n: Desired number of clusters (approximately)

        Returns:
            List of cluster indices
        """
        # Extract embeddings
        embeddings = np.array([c.get("embedding", []) for c in candidates])

        if len(embeddings) == 0:
            return []

        # Determine k
        k = min(len(candidates), max(n, 2))

        # Cluster
        clusters = cluster_embeddings(embeddings, k, method="kmeans")

        return clusters

    def _llm_generate_questions(
        self, context: str, topics: List[str], per_cluster: int = 2
    ) -> List[Dict]:
        """Generates questions using LLM for a given context.

        Args:
            context: Merged chunk context
            topics: Input topics for guidance
            per_cluster: Number of questions to generate

        Returns:
            List of question dictionaries
        """
        prompt = f"""You are a question generator. Using the text below (which comes from chunks of a chapter), produce up to {per_cluster} questions relevant to the content.

Return ONLY a JSON array where each item has:
- question_text: The question (string)
- answer: Answer in 1-2 sentences (string)
- difficulty: One of "easy", "medium", "hard"
- type: One of "fact", "conceptual", "mcq", "short_answer"
- topic_keys: Array of relevant topic labels (lowercase)
- source_chunks: Array of chunk indices mentioned in the text (integers)

For MCQ type, also include:
- options: Array of 4 option strings
- correct_option_index: Integer 0-3

Constraints:
- Avoid duplicating questions
- Use these topics as guidance: {", ".join(topics)}
- Ensure variety in difficulty and type

Text:
{context}

JSON array:"""

        try:
            response = self.llm.generate_response(prompt, [])

            # Parse JSON
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            questions = json.loads(response_text.strip())

            if not isinstance(questions, list):
                questions = [questions]

            # Normalize topic_keys in each question
            for q in questions:
                q["topic_keys"] = normalize_topics(q.get("topic_keys", []))

            return questions

        except Exception as e:
            logger.exception(f"LLM question generation failed: {e}")
            return []

    def _compute_question_embedding(self, question_text: str) -> List[float]:
        """Computes embedding for a question text.

        Args:
            question_text: Question string

        Returns:
            Embedding vector
        """
        try:
            # Use Qdrant's model to compute embedding
            model = self.qdrant.model

            if (
                hasattr(self.qdrant, "_model_type")
                and self.qdrant._model_type == "fastembed"
            ):
                embedding = list(model.embed([question_text]))[0]
            else:
                embedding = model.encode(question_text).tolist()

            return embedding

        except Exception as e:
            logger.exception(f"Failed to compute question embedding: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    try:
        service = QuestionGenerationService()

        questions = service.generate_questions_for_topic_list(
            class_id="9th ncert", chapter_id="chapter1", input_topics=[], n=1
        )

        logger.info(f"Generated {len(questions)} questions")
        for i, q in enumerate(questions):
            logger.info(f"Q{i + 1}: {q.get('question_text', 'N/A')}")

    except Exception as e:
        logger.exception(f"Question generation test failed: {e}")
