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
            # Estimate sentence length in characters
            sentence_len = len(sentence)
            # +1 for the space that will join the sentences
            next_len = (
                current_length + sentence_len + (1 if current_chunk_sentences else 0)
            )

            # Check if adding the sentence exceeds the target size
            if next_len > self.target_chars and current_chunk_sentences:
                # 1. Save the current chunk
                chunks.append(" ".join(current_chunk_sentences))

                # 2. Prepare the overlapping section for the next chunk

                # Reset chunk sentences and length
                overlap_sentences = []
                overlap_length = 0

                # Iterate backward through the sentences just added
                # to find the sentences that constitute the desired overlap
                for s in reversed(current_chunk_sentences):
                    # Check if adding this sentence to the overlap buffer exceeds the limit
                    # We use <= to include the sentence that makes the buffer slightly larger
                    if overlap_length + len(s) + 1 > self.overlap_chars:
                        break

                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1

                # The new chunk starts with the overlap and then adds the current sentence
                current_chunk_sentences = overlap_sentences
                current_length = overlap_length

            # Add the current sentence to the new (or existing) chunk
            current_chunk_sentences.append(sentence)
            current_length += sentence_len + 1  # +1 for the space

        # Add the last remaining chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks


# --- Example Usage ---
# chunker = DocumentChunker(target_tokens=300, overlap_tokens=50)
# chunks = chunker.split_chunks(large_text)
