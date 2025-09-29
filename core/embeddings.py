"""
Text embedding utilities using OpenAI API.
"""

import asyncio
import logging
import random
import time
from typing import List

import httpx
import openai
import requests

from config.settings import settings

logger = logging.getLogger(__name__)

# Configure OpenAI client
openai.api_key = settings.openai_api_key
openai.base_url = settings.openai_base_url

if settings.openai_proxy:
    openai.http_client = httpx.Client(verify=False, base_url=settings.openai_proxy)


def retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """
    Decorator for retrying API calls with exponential backoff on rate limiting errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


def async_retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0):
    """
    Async decorator for retrying API calls with exponential backoff on rate limiting errors.
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    # Check for rate limiting error (429) or connection errors
                    if attempt == max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for {func.__name__}"
                        )
                        raise

                    # Check if this is a retryable error
                    is_retryable = False
                    if (
                        hasattr(e, "status_code")
                        and getattr(e, "status_code", None) == 429
                    ):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit hit in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Too Many Requests" in str(e) or "429" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Rate limit detected in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )
                    elif "Connection" in str(e) or "Timeout" in str(e):
                        is_retryable = True
                        logger.warning(
                            f"Connection error in {func.__name__}, attempt {attempt + 1}/{max_retries}"
                        )

                    if not is_retryable:
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2**attempt), max_delay)
                    jitter = random.uniform(0.1, 0.3) * delay  # Add 10-30% jitter
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    await asyncio.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


class EmbeddingManager:
    """Manages text embeddings using OpenAI API or Ollama."""

    def __init__(self):
        """Initialize the embedding manager."""
        self.provider = getattr(settings, "llm_provider").lower()

        if self.provider == "openai":
            self.model = settings.embedding_model
        else:  # ollama
            self.model = getattr(settings, "ollama_embedding_model")
            self.ollama_base_url = getattr(settings, "ollama_base_url")

    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic."""
        try:
            if self.provider == "ollama":
                return self._get_ollama_embedding(text)
            else:
                response = openai.embeddings.create(input=text, model=self.model)
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    @retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama with retry logic."""
        response = requests.post(
            f"{self.ollama_base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("embedding", [])

    async def aget_embedding(self, text: str) -> List[float]:
        """Asynchronously generate embedding for a single text using httpx.AsyncClient with retry logic."""
        # Reuse the synchronous get_embedding (which already has retry logic)
        # by running it in a thread so callers can `await` it without performing
        # manual HTTP calls here. This keeps all OpenAI interactions using the
        # `openai` client and preserves existing retry behavior.
        try:
            return await asyncio.to_thread(self.get_embedding, text)
        except Exception as e:
            logger.error(f"Failed to generate async embedding: {e}")
            raise


# Global embedding manager instance
embedding_manager = EmbeddingManager()
