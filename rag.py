import os
import json
from typing import List, Dict, Optional
from document_loader import DocumentLoader
from qdrant_db import QdrantDB
import time
from functools import wraps
from qdrant_client.http import models
from logger import get_logger
from call_llm_open_router import call_llm

# Initialize logger
logger = get_logger(__name__)

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"[TIME] {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class RAGSystem():
    """
    A Retrieval-Augmented Generation (RAG) system with Qdrant vector database.
    """
    
    def __init__(self, collection_name: str = "documents", model_name: str = "all-MiniLM-L6-v2", 
                 chunk_size: int = 512, chunk_overlap: int = 100, 
                 host: str = "localhost", port: int = 6333, grpc_port: int = 6334):
        """
        Initializes the RAG system with Qdrant DB.
        
        Args:
            collection_name: Name of the Qdrant collection.
            model_name: Name of the sentence transformer model.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
            host: Qdrant server host.
            port: Qdrant server HTTP port.
            grpc_port: Qdrant server gRPC port.
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize Qdrant DB with gRPC enabled
        self.db = QdrantDB(host=host, port=port, grpc_port=grpc_port, model_name=model_name, prefer_grpc=True)
        
        # Create collection (384 is the vector size for all-MiniLM-L6-v2)
        self.db.create_collection(self.collection_name, vector_size=384)
    
    def chunk_with_overlap(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Splits text into chunks with overlap.
        
        Args:
            text: Full text
            chunk_size: Target chunk size (default: uses self.chunk_size)
            overlap: Overlap size (default: uses self.chunk_overlap)
            
        Returns:
            List of text chunks with overlap
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.chunk_overlap
            
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Extend to sentence boundary if possible
            if end < text_length:
                next_period = text.find('.', end)
                next_space = text.find(' ', end)
                
                # Find the nearest boundary within reasonable distance
                if next_period != -1 and next_period - end < 100:
                    end = next_period + 1
                elif next_space != -1 and next_space - end < 50:
                    end = next_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position considering overlap
            if end < text_length:
                start = end - overlap
            else:
                start = text_length
        
        return chunks
        
    def _split_text_into_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Splits text into chunks (sentence-aware, no overlap).
        
        Args:
            text: Full text
            chunk_size: Target chunk size (default: uses self.chunk_size)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        if not text:
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Extend to sentence boundary
            if end < text_length and text[end-1] != '.':
                next_stop = text.find('.', end)
                if next_stop != -1:
                    end = next_stop + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def extract_topics_and_summary(self, text: str, max_topics: int = 3) -> Dict[str, any]:
        """
        Extracts topics and summary from text using LLM.
        
        Args:
            text: Chunk text
            max_topics: Maximum number of topics to extract
            
        Returns:
            Dictionary with 'topic_keys', 'summary', and 'importance_score'
        """
        system_prompt = """You are an AI assistant that analyzes educational content and extracts metadata."""
        
        user_prompt = f"""Analyze the following text and extract key topics and a summary.

Text:
{text[:800]}

Return ONLY a JSON object with:
- "topic_keys": array of {max_topics} short topic labels (lowercase, 1-3 words each)
- "summary": one-sentence summary of the main point
- "importance_score": float 0-1 indicating how important/informative this text is

Example:
{{"topic_keys": ["fluids", "pressure", "states_of_matter"], "summary": "Explains how pressure varies in fluids.", "importance_score": 0.75}}

JSON:"""

        try:
            response = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            
            # Try to parse JSON from response
            # Remove markdown code blocks if present
            response_text = response.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            result = json.loads(response_text.strip())
            
            # Normalize topics (import normalize_topics if needed, otherwise do basic normalization)
            if 'topic_keys' in result:
                result['topic_keys'] = [t.lower().strip().replace(' ', '_') for t in result.get('topic_keys', [])]
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to extract topics/summary with LLM: {e}")
            # Fallback: simple heuristics
            return {
                'topic_keys': [],
                'summary': text[:100] + '...' if len(text) > 100 else text,
                'importance_score': 0.5
            }
    
    def generate_chunk_summary(self, text: str, chapter_name: str = "", 
                              class_name: str = "", subject_name: str = "") -> str:
        """
        Generates a short summary (around 200 words) for a large chunk using LLM.
        
        Args:
            text: Text chunk (up to 4000 chars)
            chapter_name: Name of the chapter (optional context)
            class_name: Name of the class (optional context)
            subject_name: Name of the subject (optional context)
            
        Returns:
            Summary text (around 200 words)
        """
        # Build context string
        context_parts = []
        if subject_name:
            context_parts.append(f"Subject: {subject_name}")
        if class_name:
            context_parts.append(f"Class: {class_name}")
        if chapter_name:
            context_parts.append(f"Chapter: {chapter_name}")
        
        context_str = "\n".join(context_parts) if context_parts else "General educational content"
        
        system_prompt = f"""You are analyzing educational content with the following context:
{context_str}

Provide concise summaries focused on the main concepts relevant to this subject and chapter."""

        user_prompt = f"""Analyze the following text and provide a concise summary in approximately 200 words.
Focus on the main concepts, key points, and important information relevant to this subject and chapter.

Text:
{text}

Summary (around 200 words):"""

        try:
            summary = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            logger.debug(f"Generated summary: {len(summary)} chars")
            return summary.strip()
        except Exception as e:
            logger.warning(f"Failed to generate summary with LLM: {e}")
            # Fallback: return first 200 words
            words = text.split()[:200]
            return ' '.join(words)
    
    def generate_final_summary_and_topics(self, combined_summaries: str,
                                          chapter_name: str = "",
                                          class_name: str = "", 
                                          subject_name: str = "") -> Dict[str, any]:
        """
        Generates final summary and topic list from concatenated summaries.
        
        Args:
            combined_summaries: All chunk summaries concatenated
            chapter_name: Name of the chapter (optional context)
            class_name: Name of the class (optional context)
            subject_name: Name of the subject (optional context)
            
        Returns:
            Dictionary with 'final_summary' and 'topic_keys' (5-6 topics)
        """
        # Build context string
        context_parts = []
        if subject_name:
            context_parts.append(f"Subject: {subject_name}")
        if class_name:
            context_parts.append(f"Class: {class_name}")
        if chapter_name:
            context_parts.append(f"Chapter: {chapter_name}")
        
        context_str = "\n".join(context_parts) if context_parts else "General educational content"
        
        system_prompt = f"""You are analyzing educational content with the following context:
{context_str}

Generate comprehensive summaries and extract relevant topics for this subject."""

        user_prompt = f"""Analyze the following summaries from this document and provide:
1. A comprehensive final summary (around 300 words) that captures the essence of this chapter
2. A list of 5-6 key topics that cover the main themes relevant to this subject

Summaries:
{combined_summaries[:8000]}

Return ONLY a JSON object with:
- "final_summary": comprehensive summary of the entire document tailored to the {subject_name} subject and {chapter_name} chapter
- "topic_keys": array of exactly 5-6 topic labels (lowercase, 1-3 words each) relevant to {subject_name}

Example:
{{"final_summary": "This document covers...", "topic_keys": ["photosynthesis", "cell_structure", "plant_biology", "chloroplast_function", "energy_conversion"]}}

JSON:"""

        try:
            response = call_llm(system_prompt=system_prompt, user_prompt=user_prompt)
            
            # Parse JSON from response
            response_text = response.strip()
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
            
            result = json.loads(response_text.strip())
            
            # Normalize topics
            if 'topic_keys' in result:
                result['topic_keys'] = [t.lower().strip().replace(' ', '_') for t in result.get('topic_keys', [])]
            
            # Ensure we have 5-6 topics
            if len(result.get('topic_keys', [])) < 5:
                logger.warning(f"Only got {len(result.get('topic_keys', []))} topics, expected 5-6")
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to generate final summary/topics with LLM: {e}")
            return {
                'final_summary': combined_summaries[:500] + '...',
                'topic_keys': []
            }
    
    def ingest_document(self, file_path: str, class_id: str, chapter_id: str,
                       chapter_name: str, class_name: str, subject_name: str,
                       subject_id: str, collection_name: str = None) -> Dict:
        """
        Ingests a document with enhanced metadata extraction using multi-stage processing:
        1. Chunk into 4000 chars and generate summaries
        2. Generate final summary and topics from all summaries
        3. Re-chunk into 512 chars with 100 char overlap for storage

        Args:
            file_path: Path to document file
            class_id: Unique document identifier
            chapter_id: Chapter identifier
            chapter_name: Name of the chapter
            class_name: Name of the class
            subject_name: Name of the subject
            subject_id: ID of the subject
            collection_name: Qdrant collection name (optional, uses self.collection_name if None)

        Returns:
            Dictionary with ingestion statistics
        """
        if collection_name is None:
            collection_name = self.collection_name
            
        logger.info(f"Starting enhanced ingestion for {file_path}")
        start_time = time.time()
        
        # Step 1: Load document
        raw_text = DocumentLoader.load_document(file_path)
        
        if not raw_text:
            logger.error(f"No text loaded from {file_path}")
            return {"success": False, "error": "No text loaded"}
        
        logger.info(f"Loaded document: {len(raw_text)} chars")
        
        # Step 2: Chunk into 4000 char chunks for summary generation
        large_chunks = self._split_text_into_chunks(raw_text, chunk_size=4000)
        logger.info(f"Created {len(large_chunks)} large chunks (4000 chars each)")
        
        # Step 3: Generate summary for each large chunk
        chunk_summaries = []
        for i, chunk in enumerate(large_chunks):
            logger.info(f"Generating summary for chunk {i+1}/{len(large_chunks)}")
            summary = self.generate_chunk_summary(
                chunk,
                chapter_name=chapter_name,
                class_name=class_name,
                subject_name=subject_name
            )
            chunk_summaries.append(summary)
        
        # Step 4: Concatenate all summaries
        combined_summaries = "\n\n".join(chunk_summaries)
        logger.info(f"Combined summaries: {len(combined_summaries)} chars")
        
        # Step 5: Generate final summary and topics list (5-6 topics)
        logger.info("Generating final summary and topics list")
        final_metadata = self.generate_final_summary_and_topics(
            combined_summaries,
            chapter_name=chapter_name,
            class_name=class_name,
            subject_name=subject_name
        )
        final_summary = final_metadata.get('final_summary', '')
        topic_keys = final_metadata.get('topic_keys', [])
        
        logger.info(f"Generated {len(topic_keys)} topics: {topic_keys}")
        logger.info(f"Final summary: {len(final_summary)} chars")
        
        # Step 6: Re-chunk into 512 chars with 100 char overlap for Qdrant storage
        storage_chunks = self.chunk_with_overlap(raw_text, chunk_size=512, overlap=100)
        logger.info(f"Created {len(storage_chunks)} storage chunks (512 chars with 100 overlap)")
        
        # Step 7: Prepare chunks for storage with metadata
        for i, chunk_text in enumerate(storage_chunks):
            # Count tokens (simple approximation: split by whitespace)
            token_count = len(chunk_text.split())
            
            chunk_data = {
                'chapter_name': chapter_name,
                'class_name': class_name,
                'subject_name': subject_name,
                'index': i,
                'text': chunk_text,
                'topic_keys': topic_keys,  # Use the global topics from LLM
                'source_file': os.path.basename(file_path),
                'created_at': time.time(),
                # Additional metadata for backwards compatibility
                'subject_id': subject_id,
                'class_id': class_id,
                'chapter_id': chapter_id,
                'token_count': token_count,
                'sentence_count': chunk_text.count('.')
            }

            self.db.add_text(collection_name, chunk_text, payload=chunk_data)

        elapsed = time.time() - start_time
        logger.info(f"Ingestion completed in {elapsed:.2f}s")
        
        return {
            'success': True,
            'class_id': class_id,
            'chapter_id': chapter_id,
            'chapter_name': chapter_name,
            'class_name': class_name,
            'subject_id': subject_id,
            'subject_name': subject_name,
            'chunks_processed': len(storage_chunks),
            'topics_extracted': len(topic_keys),
            'topic_keys': topic_keys,
            'summary': final_summary,
            'summary_length': len(final_summary),
            'elapsed_time': elapsed
        }

    @timing_decorator
    def add_document(self, file_path: str, class_id: str, chapter_id: str):
        """
        Loads a document, chunks it (extend-to-sentence), and adds it to Qdrant.
        
        Args:
            file_path: Path to the document file.
            class_id: Unique ID for the document.
            chapter_id: ID for the specific chapter/section.
        """
        # Load document
        raw_text = DocumentLoader.load_document(file_path)
        
        if not raw_text:
            logger.warning(f"No text loaded from {file_path}")
            return 0
        
        # Split into chunks (Extend-to-Sentence Mode)
        text_chunks = self._split_text_into_chunks(raw_text)
        
        # Current timestamp for all chunks in this batch
        timestamp = time.time()
        
        # Add each chunk to Qdrant with rich metadata
        for i, chunk in enumerate(text_chunks):
            payload = {
                'source_file': os.path.basename(file_path),
                'class_id': class_id,
                'chapter_id': chapter_id,
                'chunk_index': i,
                'timestamp': timestamp,
                'total_chunks': len(text_chunks),
                'chunk_mode': 'extend_to_sentence'
            }
            self.db.add_text(self.collection_name, chunk, payload=payload)
        
        logger.info(f"Added {len(text_chunks)} chunks from {file_path} (Doc ID: {class_id}, Chapter: {chapter_id})")

        return len(text_chunks)
    
    @timing_decorator
    def search(self, query: str, limit: int = 5, class_id: Optional[str] = None, chapter_id: Optional[str] = None) -> List[Dict]:
        """
        Searches for relevant text chunks using a query and optional filters.
        
        Args:
            query: The search query text.
            limit: Number of results to return.
            class_id: Optional filter by document ID.
            chapter_id: Optional filter by chapter ID.
            
        Returns:
            List of search results with scores and metadata.
        """
        filter_conditions = None
        must_conditions = []
        
        if class_id:
            must_conditions.append(
                models.FieldCondition(
                    key="class_id",
                    match=models.MatchValue(value=class_id)
                )
            )
            
        if chapter_id:
            must_conditions.append(
                models.FieldCondition(
                    key="chapter_id",
                    match=models.MatchValue(value=chapter_id)
                )
            )
            
        if must_conditions:
            filter_conditions = models.Filter(must=must_conditions)
            
        return self.db.search_by_text(self.collection_name, query, limit=limit, filter_conditions=filter_conditions)

    def close(self):
        """Closes the RAG system connections."""
        if self.db:
            self.db.close()

if __name__ == "__main__":
    rag = RAGSystem()
    
    # Add a document (Example usage needs update for new signature)
    # rag.add_document("data/sample.txt", "doc_1", "chap_1")
    
    # Search
    results = rag.search("sample query", limit=3)
    logger.info("\n--- Search Results ---")
    for result in results:
        logger.info(f"Score: {result['score']:.4f}")
        logger.info(f"Text: {result['payload']['text'][:100]}...")
        logger.info(f"Metadata: {result['payload']}")
        logger.info("")