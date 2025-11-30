import os
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

class LLMService:
    def __init__(self):
        """
        Initialize the LLM service with OpenAI client.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"  # You can change this to gpt-4 if needed

    def generate_response(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generates a response using the LLM based on the query and retrieved context.

        Args:
            query: The user's question.
            context_chunks: List of relevant document chunks from Qdrant.

        Returns:
            The generated response string.
        """
        
        # 1. Prepare the Context
        # Extract text from the chunks and join them
        context_text = "\n\n---\n\n".join([
            f"Source: {chunk['payload'].get('source_file', 'Unknown')}\n"
            f"Content: {chunk['payload'].get('text', '')}"
            for chunk in context_chunks
        ])

        # 2. Construct System Prompt
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

        # 3. Construct User Prompt
        user_prompt = f"Question: {query}"

        try:
            # 4. Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7, # Adjust for creativity vs precision
                max_tokens=500
            )
            
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
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
