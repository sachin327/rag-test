import re
from typing import List


class DocumentChunker:
    """
    Splits text into chunks, prioritizing full sentence coherence
    and aiming for a target token length.
    """

    # Define common sentence terminators for robust splitting
    SENTENCE_TERMINATORS = r"(?<=[.!?])\s+"

    def __init__(self, target_words: int = 300, overlap_words: int = 50):
        # We target word count for better predictability
        self.target_words = target_words
        self.overlap_words = overlap_words

    def _split_into_sentences(self, text: str) -> List[str]:
        """Splits text into sentences robustly."""
        if not text:
            return []

        # Use regex to split while keeping the terminator punctuation
        sentences = [
            s.strip() for s in re.split(self.SENTENCE_TERMINATORS, text) if s.strip()
        ]
        return sentences

    def _count_words(self, text: str) -> int:
        """Helper to count words in a string."""
        return len(text.split())

    def split_chunks(self, text: str) -> List[str]:
        """
        Splits text into chunks using sentence boundaries,
        targeting the word limit.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = self._count_words(sentence)

            # If adding this sentence exceeds the target words
            if (
                current_word_count + sentence_words > self.target_words
                and current_chunk_sentences
            ):
                # 1. Save the current chunk (up to the last sentence only)
                chunks.append(" ".join(current_chunk_sentences))

                # 2. Prepare the overlap for the next chunk
                overlap_sentences = []
                overlap_word_count = 0

                # Iterate backward through the sentences in the current chunk to build overlap
                for s in reversed(current_chunk_sentences):
                    s_words = self._count_words(s)
                    if overlap_word_count + s_words > self.overlap_words:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_word_count += s_words

                # Start the next chunk with the overlap
                current_chunk_sentences = overlap_sentences
                current_word_count = overlap_word_count

            # Add the current sentence
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words

        # Add the last remaining chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks


if __name__ == "__main__":
    # --- Example Usage ---
    from document.document_loader import DocumentLoader

    large_text = DocumentLoader().load_document("data/iesc101.pdf")
    chunker = DocumentChunker(target_words=1000, overlap_words=0)
    chunks = chunker.split_chunks(large_text)
    print(f"Converted {len(large_text)} into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk{i}: {chunk[:20]}, Chars {len(chunk)}, Words {len(chunk.split())}")
