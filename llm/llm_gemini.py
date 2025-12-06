import os
from typing import Dict, List, Generator, Any
import google.generativeai as genai
from dotenv import load_dotenv

from logger import get_logger
from llm_service import LLMService as BaseLLMService

logger = get_logger(__name__)
load_dotenv()


class LLMService(BaseLLMService):
    def __init__(self):
        """Initialize the LLM service with Gemini API client."""
        super().__init__()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        # Default model
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        self.model = genai.GenerativeModel(self.model_name)

    def list_models(self) -> List[str]:
        """Lists available Gemini models."""
        try:
            models = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    models.append(m.name)
            return models
        except Exception as e:
            logger.exception(f"Failed to list models: {e}")
            return []

    def health(self) -> bool:
        """Checks if the Gemini API is accessible."""
        try:
            # Simple check by listing models or generating a small token
            self.list_models()
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
        """Generates a response using Gemini."""
        try:
            generation_config = {
                "temperature": float(os.getenv("GEMINI_TEMPERATURE", 0.7)),
                "max_output_tokens": int(os.getenv("GEMINI_MAX_TOKENS", 2048)),
            }

            if response_schema:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = (
                    response_schema.model_json_schema()
                )

            logger.debug(f"System Prompt: {system_prompt[:100]}...")
            logger.debug(f"User Prompt: {user_prompt[:100]}...")

            if tools:
                model = genai.GenerativeModel(
                    self.model_name, tools=[tool.model_json_schema() for tool in tools]
                )
            else:
                model = self.model

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = model.generate_content(
                full_prompt,
                stream=stream,
                generation_config=generation_config,
            )
            for chunk in response:
                if hasattr(chunk, "text"):
                    chunk_text = chunk.text
                    logger.info("Chunk Text: ", chunk_text)

                    # Determine finish reason
                    finish_reason = None
                    if hasattr(chunk, "candidates") and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, "finish_reason"):
                            finish_reason = str(candidate.finish_reason)
                    yield {
                        "response": chunk_text,
                        "finish_reason": finish_reason,
                    }

            return ""

        except Exception as e:
            logger.exception(f"Error calling Gemini: {e}")
            return "I encountered an error while generating the response."


if __name__ == "__main__":
    # Example Usage
    try:
        llm = LLMService()
        logger.info(f"Health: {llm.health()}")
        logger.info(f"Models: {llm.list_models()}")

        system_prompt = "You are a helpful assistant."
        user_prompt = "What is the operating voltage?"

        for response in llm.get_response(
            system_prompt=system_prompt, user_prompt=user_prompt
        ):
            logger.info(f"Response: {response}")
    except Exception as e:
        logger.exception(f"Setup failed: {e}")
