"""
Bonus - Semantic Search

Provides semantic search over contract text using sentence embeddings.

Workflow:
1. Split the contract into overlapping chunks.
2. Generate embeddings for each chunk.
3. Generate an embedding for the user's query.
4. Compute cosine similarity.
5. Return the most relevant chunks.

This module uses the embedding model exposed through LLMClient so the
rest of the project remains provider-agnostic.
"""

from __future__ import annotations

import logging

import numpy as np

import config
from src.data_loader import chunk_text
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def build_index(
    client: LLMClient,
    contract_text: str,
) -> tuple[list[str], list[np.ndarray]]:
    """
    Split the contract into chunks and generate embeddings.

    Returns:
        (chunks, embeddings)
    """

    logger.info("Building semantic search index...")

    chunks = chunk_text(
        contract_text,
        config.MAX_CHARS_PER_CHUNK,
        config.CHUNK_OVERLAP_CHARS,
    )

    embeddings = client.embed(chunks)

    embeddings = [np.array(e, dtype=np.float32) for e in embeddings]

    logger.info("Indexed %d chunks.", len(chunks))

    return chunks, embeddings


def search(
    client: LLMClient,
    query: str,
    chunks: list[str],
    embeddings: list[np.ndarray],
    top_k: int = 3,
) -> list[dict]:
    """
    Perform semantic search.

    Returns:
        [
            {
                "score": 0.91,
                "text": "..."
            }
        ]
    """

    logger.info("Searching for: %s", query)

    query_embedding = np.array(
        client.embed([query])[0],
        dtype=np.float32,
    )

    scores = []

    for chunk, embedding in zip(chunks, embeddings):
        similarity = cosine_similarity(query_embedding, embedding)

        scores.append(
            {
                "score": similarity,
                "text": chunk,
            }
        )

    scores.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    logger.info("Returning top %d matching chunks.", top_k)

    return scores[:top_k]


def search_contract(
    client: LLMClient,
    contract_text: str,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Convenience function that builds the index and performs search.

    Example:
        results = search_contract(
            client,
            contract_text,
            "termination clause"
        )
    """

    chunks, embeddings = build_index(
        client,
        contract_text,
    )

    return search(
        client,
        query,
        chunks,
        embeddings,
        top_k=top_k,
    )