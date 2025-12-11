from abc import ABC, abstractmethod
from typing import List, Generator, Any, Dict
from utils.prompt import PromptService
from logger import get_logger

logger = get_logger(__name__)


class LLMService(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
        tools: List[Any] = None,
        response_schema: Any = None,
        **kwargs,
    ) -> str | Generator:
        """
        Generates a response from the LLM.

        Args:
            system_prompt: The system prompt.
            user_prompt: The user's query.
            stream: Whether to stream the response.
            tools: List of tools (functions) available to the LLM.
            response_schema: Pydantic model or schema for structured output.
            **kwargs: Additional arguments for the specific LLM implementation.

        Returns:
            The generated response as a string, or a generator if stream=True.
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Lists available models for the service."""
        pass

    @abstractmethod
    def health(self) -> bool:
        """Checks the health/availability of the LLM service."""
        pass

    def generate_summary(
        self,
        text: str,
        want_topics: bool = False,
        is_final: bool = False,
        stream: bool = False,
        response_schema: Any = None,
    ) -> str:
        """Generates a short summary (around 200 words) for a large chunk using
        LLM.

        Args:
            text: Text chunk (up to 4000 chars)
            want_topics: Whether to generate topics
            is_final: Whether this is the final summary

        Returns:
            Summary text (around 200 words)
        """

        if want_topics:
            if is_final:
                system_prompt = PromptService.get_final_summary_topic_prompt()
            else:
                system_prompt = PromptService.get_summary_topic_prompt()
        else:
            if is_final:
                system_prompt = PromptService.get_final_summary_system_prompt()
            else:
                system_prompt = PromptService.get_summary_system_prompt()

        try:
            # logger.debug(f"Generating summary for text: {text}")
            # logger.debug(f"System prompt: {system_prompt}")
            for summary in self.get_response(
                system_prompt=system_prompt,
                user_prompt=text,
                stream=stream,
                response_schema=response_schema,
            ):
                # logger.debug(f"Generated summary: {summary}")
                yield summary

        except Exception as e:
            logger.warning(f"Failed to generate summary with LLM: {e}")
            # Fallback: return first 200 words
            words = text.split()[:200]
            return " ".join(words)

    def generate_questions(
        self,
        text: str,
        question_type: str,
        limit: int,
        stream: bool = False,
        response_schema: Any = None,
    ) -> str:
        """Generates a short summary (around 200 words) for a large chunk using
        LLM.

        Args:
            text: Text chunk (up to 4000 chars)
            want_topics: Whether to generate topics
            is_final: Whether this is the final summary

        Returns:
            Summary text (around 200 words)
        """

        system_prompt = PromptService.get_questions_generate_prompt(
            question_type, limit
        )

        try:
            logger.debug("Generating questions")
            for summary in self.get_response(
                system_prompt=system_prompt,
                user_prompt=text,
                stream=stream,
                response_schema=response_schema,
            ):
                # logger.debug(f"Generated summary: {summary}")
                yield summary

        except Exception as e:
            logger.warning(f"Failed to generate summary with LLM: {e}")
            # Fallback: return first 200 words
            words = text.split()[:200]
            return " ".join(words)

    def generate_rag_response(
        self,
        query: str,
        context: List[Dict[str, str]],
        stream: bool = False,
        response_schema: Any = None,
    ):
        system_prompt = PromptService.get_rag_system_prompt()
        user_prompt = PromptService.get_rag_user_prompt(query, context)

        try:
            logger.debug("Generating questions")
            for summary in self.get_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                stream=stream,
                response_schema=response_schema,
            ):
                # logger.debug(f"Generated summary: {summary}")
                yield summary

        except Exception as e:
            logger.warning(f"Failed to generate summary with LLM: {e}")
            # Fallback: return first 200 words
            raise e

