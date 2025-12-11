import re
from typing import List


class DocumentChunker:
    """
    Splits text into chunks, prioritizing full sentence coherence
    and aiming for a target token length.
    """

    # Define common sentence terminators for robust splitting
    SENTENCE_TERMINATORS = r"(?<=[.!?])\s+"

    def __init__(self, target_tokens: int = 300, overlap_tokens: int = 50):
        # We target 300 tokens (well within the 512-limit for 384-dim models)
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

        # --- APPROXIMATION HACK for Token Length ---
        # Since we avoid an external tokenizer, we approximate tokens
        # using the 1 token ~= 4 character rule for English text.
        self.target_chars = target_tokens * 4
        self.overlap_chars = overlap_tokens * 4

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences robustly. (Better to use NLTK/Spacy in production)"""
        if not text:
            return []

        # Use regex to split while keeping the terminator punctuation
        sentences = [
            s.strip() for s in re.split(self.SENTENCE_TERMINATORS, text) if s.strip()
        ]
        return sentences

    def split_chunks(self, text: str) -> List[str]:
        """
        Splits text into chunks using sentence boundaries,
        targeting the token limit.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_length = 0  # Length in approximated characters

        for sentence in sentences:
            sentence_len = len(sentence)

            # Predict length if we add this sentence
            # If current_chunk is not empty, we add a space (+1)
            additional_len = sentence_len + (1 if current_chunk_sentences else 0)
            next_len = current_length + additional_len

            # Check if adding the sentence exceeds the target size
            # We only split if we already have content (current_chunk_sentences)
            # This ensures we don't start an infinite loop if a single sentence is huge
            if next_len > self.target_chars and current_chunk_sentences:
                # 1. Save the current chunk
                chunks.append(" ".join(current_chunk_sentences))

                # 2. Prepare the overlapping section for the next chunk
                overlap_sentences = []
                overlap_length = 0

                # Iterate backward through the sentences just added
                for s in reversed(current_chunk_sentences):
                    s_len = len(s)
                    # Check if adding this sentence to the overlap buffer exceeds the limit
                    # We look ahead: current overlap + space (if not first) + sentence
                    cost = s_len + (1 if overlap_sentences else 0)

                    if overlap_length + cost > self.overlap_chars:
                        break

                    overlap_sentences.insert(0, s)
                    overlap_length += cost

                # The new chunk starts with the overlap
                current_chunk_sentences = overlap_sentences

                # Recalculate length strictly to avoid drift/errors
                if current_chunk_sentences:
                    # Sum of lengths + spaces (N-1)
                    current_length = sum(len(s) for s in current_chunk_sentences) + (
                        len(current_chunk_sentences) - 1
                    )
                else:
                    current_length = 0

                # Re-evaluate next_len for the NEW current_chunk state + current sentence
                # We need to add the current sentence to this new start
                additional_len = sentence_len + (1 if current_chunk_sentences else 0)
                # Note: We don't check limit again here, we force add it to progress

            # Add the current sentence
            current_chunk_sentences.append(sentence)
            current_length += additional_len

        # Add the last remaining chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks


# --- Example Usage ---
# chunker = DocumentChunker(target_tokens=300, overlap_tokens=50)
# chunks = chunker.split_chunks(large_text)
