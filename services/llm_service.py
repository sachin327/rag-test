from abc import ABC, abstractmethod
from typing import List, Generator, Any


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
