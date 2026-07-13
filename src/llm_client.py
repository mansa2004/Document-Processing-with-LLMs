"""
LLM Client

Thin wrapper around the Groq API.

Responsibilities:
- Send prompts to Groq.
- Retry transient failures with exponential backoff.
- Keep the rest of the pipeline independent of the LLM SDK.
- Generate local embeddings for semantic search (bonus feature).
"""

from __future__ import annotations

import logging

from groq import Groq
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around the Groq API.

    All modules in the pipeline communicate with the LLM through
    this class so the implementation remains centralized and easy
    to maintain.
    """

    def __init__(self, provider: str | None = None):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found.\n"
                "Create a .env file and add your Groq API key."
            )

        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL

        # Embedding model (loaded only once)
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL
        )

        logger.info(
            "Initialized Groq client with model '%s'.",
            self.model,
        )

    @retry(
        stop=stop_after_attempt(config.REQUEST_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 1500,
        temperature: float = 0.0,
    ) -> str:
        """
        Send a prompt to Groq and return the generated text.

        Args:
            system:
                System instructions.

            user:
                User prompt.

            max_tokens:
                Maximum number of output tokens.

            temperature:
                Controls randomness.
                0.0 gives deterministic outputs, which is preferred
                for clause extraction.

        Returns:
            Model response as plain text.
        """

        logger.info("Sending request to Groq...")

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user,
                },
            ],
        )

        logger.info("Received response from Groq.")

        return response.choices[0].message.content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings locally using SentenceTransformers.

        This avoids API costs and is used by the semantic search
        bonus implementation.

        Args:
            texts:
                List of input texts.

        Returns:
            List of embedding vectors.
        """

        logger.info(
            "Generating embeddings for %d text(s).",
            len(texts),
        )

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()