"""Question Generation Service Implements the core question generation logic
with Qdrant retrieval and LLM generation."""

import json
import time
from typing import Dict, List, Optional

from qdrant_client.http import models

from llm.llm_open_router import LLMService
from logger import get_logger
from db.mongo_db import MongoDB
from db.qdrant_db import QdrantDB
from utils.question_utils import normalize_topics

logger = get_logger(__name__)


class GenerateQuestionService:
    def __init__(self):
        """Initialize the question generation service."""
        self.qdrant = QdrantDB()
        self.mongo = MongoDB()
        self.llm = LLMService()

        logger.info("Generate Question Service initialized")

    def generate_questions_for_topic_list(
        self,
        class_id: str,
        subject_ids: List[str],
        chapter_ids: List[str],
        input_topics: List[str],
        n: int = 10,
        question_type: str = "mcq",
        collection_name: str = "documents",
    ) -> List[Dict]:
        """Generates N questions for given topics using RAG approach.

        Args:
            class_id: Document identifier
            subject_ids: List of subject identifiers
            chapter_ids: List of chapter identifiers
            input_topics: List of topic strings
            n: Number of questions to generate
            question_type: 'mcq' or 'subjective'
            collection_name: Qdrant collection name

        Returns:
            List of question dictionaries
        """
        logger.info(
            f"Generating {n} questions for class={class_id}, subjects={subject_ids}, chapters={chapter_ids}, topics={input_topics}, type={question_type}"
        )
        start_time = time.time()

        # Step 1: Normalize topics
        normalized_topics = normalize_topics(input_topics)
        topic_concat_str = " ".join(normalized_topics)

        # Step 2: Retrieve candidates from Qdrant
        # Search 10 results with embedding of topic list concatenated
        candidates = self._query_qdrant(
            class_id,
            subject_ids,
            chapter_ids,
            topic_concat_str,
            collection_name,
            limit=10,
        )

        if not candidates:
            logger.warning("No candidates found in Qdrant")
            return []

        logger.info(f"Retrieved {len(candidates)} candidates from Qdrant")

        # Step 3: Generate questions using LLM
        # Pass chunks + topic concat string to LLM
        generated_questions = self._llm_generate_questions(
            candidates,
            normalized_topics,
            n,
            question_type,
        )

        logger.info(f"Generated {len(generated_questions)} questions from LLM")

        # Step 4: Add metadata and persist to MongoDB
        questions_collection = self.mongo.get_questions_collection()
        final_questions = []

        for q in generated_questions:
            q["class_id"] = class_id
            # For simplicity, we might not know exactly which chapter/subject a generated question belongs to
            # if we fed multiple chapters/subjects. But usually the user filters by specific ones.
            # We'll store the request context.
            q["subject_ids"] = subject_ids
            q["chapter_ids"] = chapter_ids
            q["created_at"] = time.time()
            q["status"] = "draft"
            q["origin"] = "generated"
            q["type"] = question_type

            # Insert into MongoDB
            try:
                result = questions_collection.insert_one(q.copy())
                q["_id"] = str(result.inserted_id)
                final_questions.append(q)
            except Exception as e:
                logger.exception(f"Failed to insert question to MongoDB: {e}")

        elapsed = time.time() - start_time
        logger.info(
            f"Question generation completed in {elapsed:.2f}s, returning {len(final_questions)} questions"
        )

        return final_questions

    def _query_qdrant(
        self,
        class_id: str,
        subject_ids: List[str],
        chapter_ids: List[str],
        query_text: str,
        collection_name: str,
        limit: int = 10,
    ) -> List[Dict]:
        """Queries Qdrant for chunks matching topics and filters.

        Args:
            class_id: Document ID
            subject_ids: List of subject IDs
            chapter_ids: List of chapter IDs
            query_text: Concatenated topics string
            collection_name: Qdrant collection
            limit: Number of results

        Returns:
            List of chunk dictionaries
        """
        # Build filter
        must_conditions = [
            models.FieldCondition(
                key="class_id", match=models.MatchValue(value=class_id)
            )
        ]

        # Subject IDs (MatchAny)
        if subject_ids:
            must_conditions.append(
                models.FieldCondition(
                    key="subject_id", match=models.MatchAny(any=subject_ids)
                )
            )

        # Chapter IDs (MatchAny)
        if chapter_ids:
            must_conditions.append(
                models.FieldCondition(
                    key="chapter_id", match=models.MatchAny(any=chapter_ids)
                )
            )

        filter_conditions = models.Filter(must=must_conditions)

        # Search using text (which converts to vector internally in QdrantDB.search_by_text)
        try:
            results = self.qdrant.search_by_text(
                collection_name=collection_name,
                query_text=query_text,
                limit=limit,
                filter_conditions=filter_conditions,
            )

            # Extract payload and id
            candidates = []
            for res in results:
                payload = res.get("payload", {})
                payload["id"] = res.get("id")
                candidates.append(payload)

            return candidates

        except Exception as e:
            logger.exception(f"Qdrant query failed: {e}")
            return []

    def _llm_generate_questions(
        self,
        candidates: List[Dict],
        topics: List[str],
        n: int,
        question_type: str,
    ) -> List[Dict]:
        """Generates questions using LLM for a given context.

        Args:
            candidates: List of retrieved chunks
            topics: Input topics for guidance
            n: Number of questions to generate
            question_type: 'mcq' or 'subjective'

        Returns:
            List of question dictionaries
        """
        # Prepare context from chunks
        context_parts = []
        for i, c in enumerate(candidates):
            text = c.get("text", "")
            chunk_id = c.get("id", "unknown")
            context_parts.append(f"Chunk {i} (ID: {chunk_id}):\n{text}\n")

        context = "\n".join(context_parts)
        topics_str = ", ".join(topics)

        prompt = f"""You are an expert educational content generator. 
Using the provided text chunks, generate {n} {question_type.upper()} questions.

Context Text:
{context}

Target Topics: {topics_str}

Requirements:
1. Generate exactly {n} questions.
2. Question Type: {question_type}
   - If MCQ: Provide 4 options and the correct option index (0-3).
   - If Subjective: Provide a detailed answer.
3. Difficulty: Mix of easy, medium, hard.
4. Source Chunks: Identify which chunk IDs were used to answer the question.
5. Topic Keys: Tag each question with relevant topics from the provided list.

Output Format:
Return ONLY a JSON array of objects with this schema:
[
  {{
    "question_text": "string",
    "answer": "string (correct option text for MCQ, or full answer for subjective)",
    "difficulty": "easy|medium|hard",
    "type": "{question_type}",
    "topic_keys": ["topic1", "topic2"],
    "source_chunks": [chunk_id1, chunk_id2],
    "options": ["opt1", "opt2", "opt3", "opt4"], // Only for MCQ
    "correct_option_index": 0 // Only for MCQ
  }}
]
"""

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
                # Ensure source_chunks are present
                if "source_chunks" not in q:
                    q["source_chunks"] = []

            return questions

        except Exception as e:
            logger.exception(f"LLM question generation failed: {e}")
            return []
