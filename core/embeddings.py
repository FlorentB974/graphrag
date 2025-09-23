"""
Text embedding utilities using OpenAI API.
"""

import logging
from typing import List
import httpx
import requests
import asyncio

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

    async def aget_embedding(self, text: str) -> List[float]:
        """Asynchronously generate embedding for a single text using httpx.AsyncClient."""
        try:
            async with httpx.AsyncClient(verify=False if settings.openai_proxy else True) as client:
                if self.provider == 'ollama':
                    url = f"{self.ollama_base_url.rstrip('/')}/api/embeddings"
                    resp = await client.post(url, json={"model": self.model, "prompt": text}, timeout=120.0)
                    resp.raise_for_status()
                    return resp.json().get('embedding', [])
                else:
                    base = settings.openai_base_url.rstrip('/')
                    url = f"{base}/embeddings"
                    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
                    resp = await client.post(url, json={"input": text, "model": self.model}, headers=headers, timeout=120.0)
                    resp.raise_for_status()
                    data = resp.json()
                    return data.get('data', [])[0].get('embedding', [])
        except Exception as e:
            logger.error(f"Failed to generate async embedding: {e}")
            raise


# Global embedding manager instance
embedding_manager = EmbeddingManager()
