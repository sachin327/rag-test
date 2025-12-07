import os
from typing import List, Generator, Any, Dict
import requests
from dotenv import load_dotenv

from logger import get_logger
from services.llm_service import LLMService as BaseLLMService
from utils.common import safe_str_to_json

logger = get_logger(__name__)
load_dotenv()


class LLMService(BaseLLMService):
    def __init__(self):
        """Initialize the LLM service with OpenRouter."""
        super().__init__()
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = os.getenv("OPENROUTER_URL")
        self.model = os.getenv("OPEN_ROUTER_MODEL", "openai/gpt-3.5-turbo")

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    def list_models(self) -> List[str]:
        """Lists available OpenRouter models."""
        # OpenRouter has a separate endpoint for models, but for now we can just return the configured model
        # or implement a fetch if needed.
        return [self.model]

    def health(self) -> bool:
        """Checks if the OpenRouter API is accessible."""
        try:
            # Simple check by making a minimal request or checking auth
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            # Just checking if we can reach the endpoint
            response = requests.get(self.base_url, headers=headers, timeout=10)
            return response.ok
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
    ) -> Dict[str, Any] | Generator:
        """Generates a response using OpenRouter."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("GITHUB_REPO_URL"),  # Required by OpenRouter
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "stream": stream,
        }

        if tools:
            payload["tools"] = [tool.model_dump() for tool in tools]
            payload["tool_choice"] = "auto"

        if response_schema:
            payload["response_format"] = response_schema.model_dump()

        try:
            logger.debug("Generating response -- Open Router")
            # logger.debug(f"Payload: {payload}")
            response = requests.post(
                self.base_url,
                headers=headers,
                stream=stream,
                json=payload,
                timeout=30,
            )
            if not response.ok:
                raise RuntimeError(
                    f"LLM API Error {response.status_code}: {response.text}"
                )

            if not stream:
                event = response.json()
                # logger.debug(event)
                finish_reason = event["choices"][0]["finish_reason"]
                if finish_reason and finish_reason == "tool_calls":
                    yield {
                        "response": safe_str_to_json(
                            event["choices"][0]["message"]["content"]
                        ),
                        "tool_calls": safe_str_to_json(
                            event["choices"][0]["message"]["tool_calls"]
                        ),
                        "finish_reason": finish_reason,
                    }
                else:
                    yield {
                        "response": safe_str_to_json(
                            event["choices"][0]["message"]["content"]
                        ),
                        "finish_reason": finish_reason,
                    }
                return

            # Streaming mode:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break

                finish_reason = event["choices"][0]["finish_reason"]
                if finish_reason and finish_reason == "tool_calls":
                    yield {
                        "response": safe_str_to_json(
                            event["choices"][0]["delta"]["content"]
                        ),
                        "tool_calls": safe_str_to_json(
                            event["choices"][0]["delta"]["tool_calls"]
                        ),
                        "finish_reason": finish_reason,
                    }
                else:
                    yield {
                        "response": safe_str_to_json(
                            event["choices"][0]["delta"]["content"]
                        ),
                        "finish_reason": finish_reason,
                    }

        except Exception as e:
            logger.exception(f"Error calling LLM: {e}")
            yield "I encountered an error while generating the response."


if __name__ == "__main__":
    # Example Usage
    try:
        from utils.tool import Tool, ToolFunction, GetWeatherArgs
        from utils.response_format import ResponseSchema, JsonSchema, SummaryResponse

        llm = LLMService()
        logger.info(f"Health: {llm.health()}")

        system_prompt = "You are a helpful assistant."
        user_prompt = "Weather in New York?"

        test_tool = Tool(
            function=ToolFunction(
                name="get_weather",
                description="Get weather data for a city",
                parameters=GetWeatherArgs.model_json_schema(),
            )
        )

        test_response_schema = ResponseSchema(
            json_schema=JsonSchema(
                name="summary",
                description="Get summary and topics of input text",
                schema=SummaryResponse.model_json_schema(),
            )
        )

        for response in llm.get_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            stream=True,
            tools=[test_tool],
            response_schema=test_response_schema,
        ):
            logger.info(f"Response: {response}")
    except Exception as e:
        logger.exception(f"Setup failed: {e}")
