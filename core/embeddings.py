"""
Text embedding utilities using OpenAI API.
"""

import logging
from typing import List
import httpx
import requests

import openai

from config.settings import settings

logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = settings.openai_api_key
openai.base_url = settings.openai_base_url

if settings.openai_proxy:
    openai.http_client = httpx.Client(verify=False, base_url=settings.openai_proxy)


class EmbeddingManager:
    """Manages text embeddings using OpenAI API or Ollama."""

    def __init__(self):
        """Initialize the embedding manager."""
        self.provider = getattr(settings, 'llm_provider').lower()
        
        if self.provider == 'openai':
            self.model = settings.embedding_model
        else:  # ollama
            self.model = getattr(settings, 'ollama_embedding_model')
            self.ollama_base_url = getattr(settings, 'ollama_base_url')

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            if self.provider == 'ollama':
                return self._get_ollama_embedding(text)
            else:
                response = openai.embeddings.create(input=text, model=self.model)
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama."""
        response = requests.post(
            f"{self.ollama_base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120
        )
        response.raise_for_status()
        return response.json().get('embedding', [])

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            if self.provider == 'ollama':
                return [self._get_ollama_embedding(text) for text in texts]
            else:
                response = openai.embeddings.create(input=texts, model=self.model)
                return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise


# Global embedding manager instance
embedding_manager = EmbeddingManager()
