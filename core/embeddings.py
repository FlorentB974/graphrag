"""
Text embedding utilities using OpenAI API.
"""

import asyncio
import logging
import random
import threading
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


def retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0):
    """
    Decorator for retrying API calls with exponential backoff on rate limiting errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (increased from 1.0 to 3.0)
        max_delay: Maximum delay in seconds (increased from 60.0 to 180.0)
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
                    jitter = random.uniform(0.2, 0.5) * delay  # Add 20-50% jitter (increased)
                    total_delay = delay + jitter

                    logger.info(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)

            return None  # Should never reach here

        return wrapper

    return decorator


def async_retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0):
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
                    jitter = random.uniform(0.2, 0.5) * delay  # Add 20-50% jitter (increased)
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
        # Use embeddings_provider if set, otherwise fall back to llm_provider
        embeddings_provider = getattr(settings, "embeddings_provider", None)
        if embeddings_provider:
            self.provider = embeddings_provider.lower()
        else:
            self.provider = getattr(settings, "llm_provider").lower()
        
        self._last_request_time = 0
        self._request_lock = threading.Lock()

        if self.provider == "openai":
            self.model = settings.embedding_model
        else:  # ollama
            self.model = getattr(settings, "ollama_embedding_model")
            self.ollama_base_url = getattr(settings, "ollama_base_url")

    def _wait_for_rate_limit(self):
        """Enforce minimum delay between requests to prevent rate limiting."""
        with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            min_delay = random.uniform(settings.embedding_delay_min, settings.embedding_delay_max)
            
            if time_since_last < min_delay:
                sleep_time = min_delay - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self._last_request_time = time.time()

    @retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text with retry logic."""
        # Enforce rate limiting before making request
        self._wait_for_rate_limit()
        
        try:
            if self.provider == "ollama":
                return self._get_ollama_embedding(text)
            else:
                response = openai.embeddings.create(input=text, model=self.model)
                return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    @retry_with_exponential_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama with retry logic."""
        # Rate limiting is already handled in get_embedding
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
