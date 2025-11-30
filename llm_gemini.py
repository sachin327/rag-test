import os
from typing import List, Dict
from dotenv import load_dotenv
from logger import get_logger
import google.generativeai as genai
from redis_db import RedisDB

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

class LLMService:
    def __init__(self):
        """
        Initialize the LLM service with Gemini API client.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        
        # List available models for debugging
        # self.list_available_models()

        # Default model - can be updated after checking available models
        self.model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")

        # Initialize Redis client
        try:
            self.redis_db = RedisDB()
            logger.info("Redis client initialized for LLM streaming")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client: {e}")
            self.redis_db = None

    def list_available_models(self):
        """
        Lists and logs available Gemini models.
        """
        try:
            logger.info("Available Gemini Models:")
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    logger.info(f" - {m.name}")
        except Exception as e:
            logger.error(f"Failed to list models: {e}")

    def _construct_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """Helper to construct the full prompt from query and context."""
        if not context_chunks:
            return query

        # Prepare context
        context_text = "\n\n---\n\n".join([
            f"Source: {chunk['payload'].get('source_file', 'Unknown')}\n"
            f"Content: {chunk['payload'].get('text', '')}"
            for chunk in context_chunks
        ])

        # Construct system prompt
        system_prompt = f"""You are a helpful and precise AI assistant for a RAG (Retrieval-Augmented Generation) system.
Your goal is to answer the user's question based ONLY on the provided context.

Instructions:
- Use the provided context to answer the question.
- If the answer is not in the context, politely state that you cannot answer based on the available information.
- Do not hallucinate or make up information.
- Cite the source file if possible.

Context Data:
{context_text}
"""
        user_prompt = f"Question: {query}"
        return system_prompt + "\n" + user_prompt

    def generate_response(self, query: str = None, context_chunks: List[Dict] = None, 
                         system_prompt: str = None, user_input: str = None) -> str:
        """
        Generates a response using Gemini.
        
        Args:
            query: (Optional) The user's query - deprecated, use user_input instead
            context_chunks: (Optional) Context chunks for RAG - deprecated, build prompt yourself
            system_prompt: (Optional) System prompt/instructions for the LLM
            user_input: (Optional) User input/query
            
        Returns:
            Generated response text
        """
        try:
            # Support both old and new API
            if system_prompt is not None and user_input is not None:
                # New API: use system_prompt and user_input directly
                prompt = f"{system_prompt}\n\n{user_input}"
            elif query is not None:
                # Old API: construct prompt from query and context
                prompt = self._construct_prompt(query, context_chunks or [])
            else:
                raise ValueError("Must provide either (system_prompt, user_input) or query")
            
            logger.debug(f"Prompt: {prompt[0:100]}...")
            
            response = self.model.generate_content(prompt, generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2048
            })
            logger.debug(f"Response: {response}")
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return "I encountered an error while generating the response."

    def generate_response_stream(self, query: str = None, context_chunks: List[Dict] = None,
                                system_prompt: str = None, user_input: str = None) -> str:
        """
        Generates a response using Gemini with streaming, printing chunks to console.
        Publishes each chunk to Redis channel.
        Returns the final complete response string.
        
        Args:
            query: (Optional) The user's query - deprecated, use user_input instead
            context_chunks: (Optional) Context chunks for RAG - deprecated, build prompt yourself
            system_prompt: (Optional) System prompt/instructions for the LLM
            user_input: (Optional) User input/query
            
        Returns:
            Complete response text
        """
        try:
            # Support both old and new API
            if system_prompt is not None and user_input is not None:
                # New API: use system_prompt and user_input directly
                prompt = f"{system_prompt}\n\n{user_input}"
            elif query is not None:
                # Old API: construct prompt from query and context
                prompt = self._construct_prompt(query, context_chunks or [])
            else:
                raise ValueError("Must provide either (system_prompt, user_input) or query")
            
            logger.debug(f"Prompt: {prompt[0:100]}...")

            response = self.model.generate_content(prompt, stream=True, generation_config={
                "temperature": 0.7,
                "max_output_tokens": 4096
            })
            
            full_response = ""
            print("="*20 + " Streaming Response " + "="*20)
            
            for chunk in response:
                if hasattr(chunk, 'text'):
                    chunk_text = chunk.text
                    print("Chunk Text: ", chunk_text)
                    full_response += chunk_text
                    
                    # Determine finish reason
                    finish_reason = None
                    if hasattr(chunk, 'candidates') and len(chunk.candidates) > 0:
                        candidate = chunk.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            finish_reason = str(candidate.finish_reason)
                    
                    # Publish to Redis
                    if self.redis_db:
                        try:
                            message = {
                                "response": chunk_text,
                                "finish_reason": finish_reason
                            }
                            self.redis_db.publish(message)
                            logger.debug(f"Published chunk to Redis: finish_reason={finish_reason}")
                        except Exception as e:
                            logger.error(f"Failed to publish to Redis: {e}")
            
            print("="*60)
            
            return full_response
        except Exception as e:
            logger.error(f"Error calling Gemini: {e}")
            return "I encountered an error while generating the response."

if __name__ == "__main__":
    # Example Usage
    try:
        llm = LLMService()
        # Mock context for testing
        mock_context = [
            {
                "payload": {
                    "source_file": "manual.pdf",
                    "text": "The device typically operates at 240V."
                }
            },
            {
                "payload": {
                    "source_file": "safety.txt",
                    "text": "Always wear protective gear when handling the device."
                }
            }
        ]
        query = "What is the operating voltage?"
        response = llm.generate_response(query, mock_context)
        logger.info(f"Query: {query}")
        logger.info(f"Response: {response}")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
