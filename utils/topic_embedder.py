from typing import List
import math
import numpy as np
from logger import get_logger

logger = get_logger(__name__)


class TopicEmbedder:
    """
    A class containing the logic for finding the most similar topic
    using cosine similarity between two embeddings.
    """

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates the cosine similarity between two vectors.
        Formula: (A . B) / (||A|| * ||B||)
        """
        if not vec1 or not vec2:
            return 0.0

        # 1. Calculate the dot product (A . B)
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # 2. Calculate the magnitude (Euclidean norm) of each vector (||A|| and ||B||)
        magnitude_a = math.sqrt(sum(a * a for a in vec1))
        magnitude_b = math.sqrt(sum(b * b for b in vec2))

        # 3. Calculate the cosine similarity
        denominator = magnitude_a * magnitude_b

        # Handle zero division if either magnitude is zero (e.g., zero vector)
        if denominator == 0:
            return 0.0

        return dot_product / denominator

    @staticmethod
    def get_relevant_topics(
        text_embedding: List[float], topic_keys_embeddings: List[List[float]]
    ) -> List[int]:
        """
        Returns the indexes of topic_keys_embeddings most similar to text_embedding
        using cosine similarity, sorted from most to least similar.
        """
        similarities = []

        # 1. Calculate similarity for every topic key
        for index, topic_vector in enumerate(topic_keys_embeddings):
            similarity_score = TopicEmbedder.cosine_similarity(
                text_embedding, topic_vector
            )

            # Store a tuple of (score, index)
            similarities.append((similarity_score, index))

        # 2. Sort the results in descending order by score (most similar first)
        # item[0] is the similarity score
        similarities.sort(key=lambda item: item[0], reverse=True)

        # 3. Extract and return only the sorted indexes
        # item[1] is the original index
        sorted_indexes = [item[1] for item in similarities if item[0] >= 0.3]

        return sorted_indexes

    @staticmethod
    def get_relevant_topics_batch(
        text_embeddings: List[List[float]],
        topic_keys_embeddings: List[List[float]],
        threshold: float = 0.3,
    ) -> List[List[int]]:
        """
        Returns the indexes of topic_keys_embeddings most similar to each text_embedding
        using cosine similarity, sorted from most to least similar.

        Args:
            text_embeddings: List of N embedding vectors (each vector is a list of floats)
            topic_keys_embeddings: List of M topic embedding vectors
            threshold: Minimum similarity score to include

        Returns:
            List of length N, where each element is a list of topic indices
        """
        if not text_embeddings or not topic_keys_embeddings:
            return [[] for _ in range(len(text_embeddings))]

        # Convert to numpy arrays
        # text_embeddings: (N, D)
        # topic_keys_embeddings: (M, D)
        text_arr = np.array(text_embeddings)
        topic_arr = np.array(topic_keys_embeddings)

        # Normalize (L2)
        # Avoid division by zero
        text_norm = np.linalg.norm(text_arr, axis=1, keepdims=True)
        text_norm[text_norm == 0] = 1.0
        text_arr_normalized = text_arr / text_norm

        topic_norm = np.linalg.norm(topic_arr, axis=1, keepdims=True)
        topic_norm[topic_norm == 0] = 1.0
        topic_arr_normalized = topic_arr / topic_norm

        # Cosine similarity matrix: (N, M)
        # sim[i, j] = user i vs topic j
        sim_matrix = np.dot(text_arr_normalized, topic_arr_normalized.T)

        results = []
        for row in sim_matrix:
            # Filter by threshold and sort
            # Enumerate to keep index: (score, index)
            scored_indices = [
                (score, idx) for idx, score in enumerate(row) if score >= threshold
            ]
            # Sort descending by score
            scored_indices.sort(key=lambda x: x[0], reverse=True)

            # Extract indices
            results.append([idx for _, idx in scored_indices])

        return results


if __name__ == "__main__":
    # --- Example Usage ---

    # Initialize the class
    embedder = TopicEmbedder()

    # Example Embeddings (simplified 3-dimensional vectors for demonstration)
    text_emb = [0.8, 0.2, 0.1]  # Vector for "AI and Machine Learning"

    topic_embs = [
        [0.9, 0.1, 0.05],  # Index 0: "Deep Learning" (High Similarity)
        [0.1, 0.9, 0.8],  # Index 1: "Database Management" (Low Similarity)
        [0.75, 0.25, 0.1],  # Index 2: "Natural Language Processing" (Medium Similarity)
        [0.0, 0.0, 1.0],  # Index 3: "Front-End Development" (Very Low Similarity)
    ]

    # Get the sorted relevant topics
    relevant_topics = embedder.get_relevant_topics(text_emb, topic_embs)

    logger.info(f"Text Embedding: {text_emb}")
    logger.info("-" * 30)
    logger.info(f"Topic Embeddings (Indices): {list(range(len(topic_embs)))}")
    logger.info("-" * 30)
    logger.info(f"Relevant Topics (Sorted Indices): {relevant_topics}")

    # You can also use the following formula visualization for better understanding:
    #
