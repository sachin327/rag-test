"""Question Generation Service Implements the core question generation logic
with Rag retrieval and LLM generation."""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

from llm.llm_open_router import LLMService
from logger import get_logger
from db.mongo_db import MongoDB
from services.rag import RAGSystem
from utils.response_format import (
    ResponseSchema,
    JsonSchema,
    QuestionResponse,
)
from utils.mongo_util import (
    get_topics_mongo,
)

logger = get_logger(__name__)
load_dotenv()


class GenerateQuestionService:
    def __init__(self):
        """Initialize the question generation service."""
        self.rag_service = RAGSystem()
        self.rag_service.create_payload_index(
            fields=[
                "class_id",
                "chapter_id",
                "subject_id",
                "relevant_topic_keys[].name",
            ]
        )
        self.mongo = MongoDB(os.getenv("MONGO_URI"), os.getenv("MONGO_DB_NAME"))
        self.llm_service = LLMService()

        logger.info("Generate Question Service initialized")

    def build_llm_context(self, candidates: List[Dict]):
        llm_context = ""
        if candidates:
            for candidate in candidates:
                llm_context += f"""Text: {candidate["text"]}
Relevant Topics: {candidate["topics"]}
"""
            llm_context += f"\nSummary: {candidates[0]['summary']}"
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

        Returns:
            List of question dictionaries
        """
        logger.info(
            f"Generating {n} questions for class={class_id}, subject={subject_id}, chapter={chapter_id}, topics={input_topics}, type={question_type}"
        )

        topics = get_topics_mongo(self.mongo, "topics", input_topics)

        candidates = self.search_topics_rag(
            class_id=class_id,
            subject_id=subject_id,
            chapter_id=chapter_id,
            topics=topics,
            limit=10,
        )

        if not candidates:
            logger.warning("No candidates found in RAG")
            return []

        logger.info(f"Retrieved {len(candidates)} candidates from RAG")

        llm_context = self.build_llm_context(candidates)

        generated_questions = self._llm_generate_questions(
            llm_context,
            question_type,
            n,
        )

        return generated_questions

    def search_topics_rag(
        self,
        class_id: str,
        subject_id: str,
        chapter_id: str,
        topics: Optional[List[Dict]] = None,
        limit: int = 10,
    ):
        rag_filters = {
            "class_id": class_id,
            "subject_id": subject_id,
            "chapter_id": chapter_id,
            "relevant_topic_keys.name": [topic.get("title", "") for topic in topics],
        }
        search_results = self.rag_service.search_by_filter(
            filters=rag_filters,
            limit=limit,
            is_nested_filter=True,
        )

        topic_name_to_key = {topic["title"]: topic["id"] for topic in topics}
        results = []
        for search_result in search_results:
            for topic in search_result["payload"]["relevant_topic_keys"]:
                topic["id"] = topic_name_to_key.get(topic["name"], "")

            results.append(
                {
                    "text": search_result["payload"]["text"],
                    "summary": search_result["payload"]["summary"],
                    "topics": search_result["payload"]["relevant_topic_keys"],
                }
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
            logger.debug(event)
            response = event.get("response", {})
            # logger.debug(response)
            questions = response.get("questions", [])
        logger.info(f"Generated {len(questions)} questions")
        return questions
