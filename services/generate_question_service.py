"""Question Generation Service Implements the core question generation logic
with Qdrant retrieval and LLM generation."""

import os
import time
from typing import List, Dict, Optional
from qdrant_client import models
from bson.objectid import ObjectId
from dotenv import load_dotenv

from llm.llm_open_router import LLMService
from logger import get_logger
from db.mongo_db import MongoDB
from db.qdrant_db import QdrantDB
from utils.response_format import (
    ResponseSchema,
    JsonSchema,
    QuestionResponse,
)

logger = get_logger(__name__)
load_dotenv()


class GenerateQuestionService:
    def __init__(self):
        """Initialize the question generation service."""
        # Initialize Qdrant DB with gRPC enabled
        self.qdrant = QdrantDB(
            host=os.getenv("QDRANT_HOST"),
            port=os.getenv("QDRANT_PORT"),
            prefer_grpc=False,
            api_key=os.getenv("QDRANT_API_KEY"),
            vector_size=384,  # Fixed for all-MiniLM-L6-v2
        )

        # Create collection (512 is the vector size for all-MiniLM-L6-v2)
        self.qdrant.create_collection(os.getenv("QDRANT_COLLECTION_NAME"))
        self.mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
        self.llm_service = LLMService()

        logger.info("Generate Question Service initialized")

    def get_topics_mongo(self, topic_ids: List[str]):
        try:
            # 1. Get existing topics for this subject_id
            topics = self.mongo.get_collection("topics").find(
                {"_id": {"$in": [ObjectId(topic_id) for topic_id in topic_ids]}}
            )
            topics = [
                {"title": topic.get("title"), "description": topic.get("description")}
                for topic in topics
            ]
            return topics
        except Exception as e:
            logger.error(f"Error getting docs from mongo: {e}")
            return None

    def build_llm_context(self, candidates: List[Dict]):
        llm_context = ""
        # print("################")
        # print(candidates)
        # print("################")
        if candidates:
            for candidate in candidates:
                llm_context += f"""Text: {candidate["payload"]["text"]}
Relevant Topics: {candidate["payload"]["relevant_topic_keys"]}
"""
            llm_context += f"\nSummary: {candidates[0]['payload']['summary']}"
        return llm_context

    def generate_questions_for_topic_list(
        self,
        class_id: str,
        subject_id: str,
        chapter_id: str,
        input_topics: List[str],
        n: int = 10,
        question_type: str = "mcq",
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
            f"Generating {n} questions for class={class_id}, subject={subject_id}, chapter={chapter_id}, topics={input_topics}, type={question_type}"
        )

        topics = self.get_topics_mongo(input_topics)

        candidates = self.search_topics_qdrant(
            collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
            class_id=class_id,
            subject_id=subject_id,
            chapter_id=chapter_id,
            topics=topics,
            limit=10,
        )

        if not candidates:
            logger.warning("No candidates found in Qdrant")
            return []

        logger.info(f"Retrieved {len(candidates)} candidates from Qdrant")

        llm_context = self.build_llm_context(candidates)
        generated_questions = self._llm_generate_questions(
            llm_context,
            question_type,
            n,
        )

        return generated_questions

    def build_qdrant_filter(
        self,
        class_id: str,
        subject_id: str,
        chapter_id: str,
        topics: Optional[List[Dict]] = None,
    ) -> models.Filter:
        """
        Build a Qdrant models.Filter that:
        - MUST match exact class_id, subject_id, chapter_id
        - SHOULD match any topic title against payload path 'relevant_topic_keys.name'
            (if topics provided). At least one SHOULD must match (min_should=1).
        """
        must_conditions: List[models.FieldCondition] = [
            models.FieldCondition(
                key="class_id",
                match=models.MatchValue(value=class_id),
            ),
            models.FieldCondition(
                key="subject_id",
                match=models.MatchValue(value=subject_id),
            ),
            models.FieldCondition(
                key="chapter_id",
                match=models.MatchValue(value=chapter_id),
            ),
        ]

        should_conditions: List[models.FieldCondition] = []
        if topics:
            for t in topics:
                title = t.get("title")
                if title:
                    # dot-notation to access list-of-dicts field 'relevant_topic_keys.name'
                    should_conditions.append(
                        models.FieldCondition(
                            key="relevant_topic_keys.name",
                            match=models.MatchValue(value=title),
                        )
                    )

        if should_conditions:
            return models.Filter(
                must=must_conditions, should=should_conditions, min_should=1
            )
        else:
            return models.Filter(must=must_conditions)

    def search_topics_qdrant(
        self,
        collection_name: str,
        class_id: str,
        subject_id: str,
        chapter_id: str,
        topics: Optional[List[Dict]] = None,
        limit: int = 10,
    ):
        qdrant_filter = self.build_qdrant_filter(
            class_id, subject_id, chapter_id, topics
        )
        results = self.qdrant.search_by_filter(
            collection_name=collection_name,
            filter_conditions=qdrant_filter,
            limit=limit,
        )
        return results

    def _llm_generate_questions(
        self,
        context: str,
        question_type: str,
        n: int,
    ) -> List[Dict]:
        """Generates questions using LLM for a given context.

        Args:
            context: Context for question generation
            n: Number of questions to generate
            question_type: 'mcq' or 'subjective'

        Returns:
            List of question dictionaries
        """

        question_response_schema = ResponseSchema(
            json_schema=JsonSchema(
                name="question",
                schema=QuestionResponse.model_json_schema(),
            )
        )

        questions = []
        for event in self.llm_service.generate_questions(
            context, question_type, n, response_schema=question_response_schema
        ):
            # logger.debug(event)
            response = event.get("response", {})
            # logger.debug(response)
            questions = response.get("questions", [])
        logger.info(f"Generated {len(questions)} questions")
        return questions
