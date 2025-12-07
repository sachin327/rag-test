from typing import List, Dict


class PromptService:
    @staticmethod
    def get_rag_system_prompt() -> str:
        return """You are a helpful and precise AI assistant for a RAG (Retrieval-Augmented Generation) system.
Your goal is to answer the user's question based ONLY on the provided context.

Instructions:
- Use the provided context to answer the question.
- If the answer is not in the context, politely state that you cannot answer based on the available information.
- Do not hallucinate or make up information.
- Cite the source file if possible.
"""

    @staticmethod
    def get_rag_user_prompt(query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return f"Question: {query}\n\nNo context provided."

        context_text = "\n\n---\n\n".join(
            [
                f"Source: {chunk['payload'].get('source_file', 'Unknown')}\n"
                f"Content: {chunk['payload'].get('text', '')}"
                for chunk in context_chunks
            ]
        )

        return f"""Context Data:
{context_text}

Question: {query}
"""

    @staticmethod
    def get_summary_system_prompt() -> str:
        return """
Analyze the following text and provide a concise summary in approximately 200 words.
Focus on the main concepts, key points, and important information relevant to this subject and chapter.
"""

    @staticmethod
    def get_final_summary_system_prompt() -> str:
        return """Your task is to generate a final summary from the given chunks of summaries.
Try to preserrve the original meaning and context of the text.
Also the important points should be included in the final summary.
"""

    @staticmethod
    def get_summary_topic_prompt() -> str:
        return """Your task is to generate summary and a topic list from the given text.
Try to preserrve the original meaning and context of the text.
Output should be following the given response_schema.
Also the final topic list should be generated from the given topic list.
For topics use standard educational topic names. should not be very long.
"""

    @staticmethod
    def get_final_summary_topic_prompt() -> str:
        return """Your task is to generate a final summary from the given chunks of summaries.
Try to preserrve the original meaning and context of the text.
Output should be following the given response_schema.
Also the final topic list should be generated from the given topic list.
For topics use standard educational topic names. should not be very long.
"""
