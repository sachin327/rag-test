import os
from typing import List, Generator, Any
from dotenv import load_dotenv
from openai import OpenAI

from logger import get_logger
from llm_service import LLMService as BaseLLMService

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()


class LLMService(BaseLLMService):
    def __init__(self):
        """Initialize the LLM service with OpenAI client."""
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def list_models(self) -> List[str]:
        """Lists available OpenAI models."""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.exception(f"Failed to list models: {e}")
            return []

    def health(self) -> bool:
        """Checks if the OpenAI API is accessible."""
        try:
            # Simple check by listing models
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
        tools: List[Any] = None,
        response_schema: Any = None,
        **kwargs,
    ) -> str | Generator:
        """Generates a response using the LLM based on the query and retrieved context."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Prepare arguments
            api_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": float(os.getenv("OPENAI_TEMPERATURE", 0.7)),
                "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", 500)),
                "stream": stream,
            }

            if tools:
                api_kwargs["tools"] = [tool.model_json_schema() for tool in tools]
                api_kwargs["tool_choice"] = "auto"

            if response_schema:
                api_kwargs["response_format"] = response_schema.model_json_schema()

            for event in self.client.chat.completions.create(**api_kwargs):
                if stream:
                    if event.choices[0].delta.content is not None:
                        yield {
                            "response": event.choices[0].delta.content,
                            "finish_reason": event.choices[0].finish_reason,
                        }
                else:
                    return {
                        "response": event.choices[0].message.content,
                        "finish_reason": event.choices[0].finish_reason,
                    }

        except Exception as e:
            logger.exception(f"Error calling LLM: {e}")
            return "I encountered an error while generating the response."


if __name__ == "__main__":
    # Example Usage
    try:
        llm = LLMService()
        logger.info(f"Health: {llm.health()}")

        system_prompt = "You are a helpful assistant."
        user_prompt = "What is the operating voltage?"

        for response in llm.get_response(
            system_prompt=system_prompt, user_prompt=user_prompt
        ):
            logger.info(f"Response: {response}")

    except Exception as e:
        logger.exception(f"Setup failed: {e}")
