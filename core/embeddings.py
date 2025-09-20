"""
Text embedding utilities using OpenAI API.
"""

import logging
from typing import List
import httpx

import openai

from config.settings import settings

logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = settings.openai_api_key
openai.base_url = settings.openai_base_url

if settings.openai_proxy:
    openai.http_client = httpx.Client(verify=False, base_url=settings.openai_proxy)


class EmbeddingManager:
    """Manages text embeddings using OpenAI API."""

    def __init__(self):
        """Initialize the embedding manager."""
        self.model = settings.embedding_model

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = openai.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = openai.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise


# Global embedding manager instance
embedding_manager = EmbeddingManager()
