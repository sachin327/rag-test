"""Utility functions for the question generation system."""

import re
from typing import Dict, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from logger import get_logger

logger = get_logger(__name__)

# Topic normalization mappings
TOPIC_ALIASES = {
    "states of matter": "states_of_matter",
    "fluid mechanics": "fluids",
    "thermodynamics": "thermo",
}


def normalize_topic(topic: str) -> str:
    """Normalizes a topic string to lowercase, removes punctuation, and applies
    aliases.

    Args:
        topic: Raw topic string

    Returns:
        Normalized topic string
    """
    # Convert to lowercase and strip
    normalized = topic.lower().strip()

    # Remove punctuation except underscores
    normalized = re.sub(r"[^\w\s_]", "", normalized)

    # Replace spaces with underscores
    normalized = normalized.replace(" ", "_")

    return normalized


def normalize_topics(topics: List[str]) -> List[str]:
    """Normalizes a list of topics.

    Args:
        topics: List of raw topic strings

    Returns:
        List of normalized topic strings
    """
    return [normalize_topic(t) for t in topics if t]


def cluster_embeddings(
    embeddings: np.ndarray, k: int, method: str = "kmeans"
) -> List[List[int]]:
    """Clusters embeddings and returns groups of indices.

    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        k: Number of clusters
        method: 'kmeans' or 'agglomerative'

    Returns:
        List of lists, where each inner list contains indices belonging to that cluster
    """
    if len(embeddings) <= k:
        # If we have fewer samples than clusters, each sample is its own cluster
        return [[i] for i in range(len(embeddings))]

    try:
        if method == "kmeans":
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            clusterer = AgglomerativeClustering(n_clusters=k)

        labels = clusterer.fit_predict(embeddings)

        # Group indices by cluster label
        clusters = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)

        # Filter out empty clusters
        clusters = [c for c in clusters if c]

        logger.debug(
            f"Clustered {len(embeddings)} embeddings into {len(clusters)} clusters"
        )
        return clusters

    except Exception as e:
        logger.exception(f"Clustering failed: {e}")
        # Fallback: return all indices in one cluster
        return [list(range(len(embeddings)))]


def deduplicate_by_similarity(
    items: List[Dict], threshold: float = 0.92, embedding_key: str = "embedding"
) -> List[Dict]:
    """Deduplicates items based on cosine similarity of their embeddings.

    Args:
        items: List of dictionaries, each containing an embedding
        threshold: Cosine similarity threshold above which items are considered duplicates
        embedding_key: Key in the dictionary where embedding is stored

    Returns:
        Deduplicated list of items
    """
    if not items:
        return []

    # Extract embeddings
    print("Check 0.1")
    embeddings = []
    for item in items:
        if embedding_key in item and item[embedding_key] is not None:
            embeddings.append(item[embedding_key])
        else:
            embeddings.append(None)

    # Track which items to keep
    keep = [True] * len(items)

    for i in range(len(items)):
        if not keep[i] or embeddings[i] is None:
            continue

        for j in range(i + 1, len(items)):
            if not keep[j] or embeddings[j] is None:
                continue

            # Compute cosine similarity
            sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if sim >= threshold:
                # Mark as duplicate
                keep[j] = False
                logger.debug(
                    f"Marked item {j} as duplicate of item {i} (similarity: {sim:.3f})"
                )

    result = [items[i] for i in range(len(items)) if keep[i]]
    logger.info(
        f"Deduplicated {len(items)} items to {len(result)} items (threshold: {threshold})"
    )

    return result


def count_tokens(text: str) -> int:
    """
    Estimates token count for a text string.
    Simple heuristic: ~4 characters per token.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def merge_chunks_by_budget(
    chunks: List[Dict], token_budget: int = 1200, sort_by: str = "chunk_index"
) -> str:
    """Merges chunks into a single context string within a token budget.

    Args:
        chunks: List of chunk dictionaries with 'text' and 'chunk_index' keys
        token_budget: Maximum tokens for the merged context
        sort_by: Key to sort chunks by (default: 'chunk_index')

    Returns:
        Merged context string with chunk markers
    """
    # Sort chunks
    sorted_chunks = sorted(chunks, key=lambda x: x.get(sort_by, 0))

    context_parts = []
    total_tokens = 0

    for chunk in sorted_chunks:
        chunk_text = chunk.get("text", "")
        chunk_index = chunk.get("chunk_index", 0)

        # Add chunk marker
        marker = f"\n[Chunk {chunk_index}]\n"
        chunk_with_marker = marker + chunk_text

        chunk_tokens = count_tokens(chunk_with_marker)

        if total_tokens + chunk_tokens > token_budget:
            logger.debug(f"Reached token budget at chunk {chunk_index}")
            break

        context_parts.append(chunk_with_marker)
        total_tokens += chunk_tokens

    return "".join(context_parts)


def compute_overlap_count(item_topics: List[str], input_topics: List[str]) -> int:
    """Computes the number of overlapping topics between two lists.

    Args:
        item_topics: Topics associated with an item
        input_topics: Input topics to compare against

    Returns:
        Count of overlapping topics
    """
    return len(set(item_topics) & set(input_topics))


def select_diverse_questions(
    questions: List[Dict], n: int, input_topics: List[str]
) -> List[Dict]:
    """Selects n diverse questions from a larger pool. Prioritizes topic
    coverage and diversity.

    Args:
        questions: List of question dictionaries
        n: Number of questions to select
        input_topics: Input topics for overlap scoring

    Returns:
        Selected questions
    """
    if len(questions) <= n:
        return questions

    # Score each question
    for q in questions:
        q["_score"] = 0

        # Score by topic overlap
        overlap = compute_overlap_count(q.get("topic_keys", []), input_topics)
        q["_score"] += overlap * 10

        # Bonus for difficulty diversity
        difficulty_bonus = {"easy": 1, "medium": 2, "hard": 3}
        q["_score"] += difficulty_bonus.get(q.get("difficulty", "medium"), 2)

        # Bonus for type diversity
        type_bonus = {"fact": 1, "conceptual": 3, "mcq": 2, "short_answer": 2}
        q["_score"] += type_bonus.get(q.get("type", "fact"), 1)

    # Sort by score and select top n
    sorted_questions = sorted(questions, key=lambda x: x["_score"], reverse=True)
    selected = sorted_questions[:n]

    # Remove temporary score field
    for q in selected:
        q.pop("_score", None)

    logger.info(
        f"Selected {len(selected)} diverse questions from pool of {len(questions)}"
    )
    return selected
