"""
Reranker module using Ollama with BGE Reranker model.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


class OllamaReranker:
    """Reranker using Ollama with BGE Reranker model.
    
    Uses the BGE reranker model to compute embeddings for query and documents,
    then calculates relevance scores using cosine similarity.
    """

    def __init__(
        self,
        model: str = "xitao/bge-reranker-v2-m3",
        base_url: Optional[str] = None,
    ):
        """Initialize the Ollama reranker.
        
        Args:
            model: The reranker model to use (default: xitao/bge-reranker-v2-m3)
            base_url: Ollama base URL (default: from settings)
        """
        self.model = model
        self.base_url = base_url or settings.ollama_base_url
        self._validate_connection()

    def _validate_connection(self) -> bool:
        """Validate connection to Ollama and check if model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if any(self.model in name for name in model_names):
                    logger.info(f"Reranker model '{self.model}' is available")
                    return True
                else:
                    logger.warning(
                        f"Reranker model '{self.model}' not found. Available models: {model_names}"
                    )
                    return False
            return False
        except Exception as e:
            logger.warning(f"Could not validate Ollama connection: {e}")
            return False

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text using Ollama.
        
        Args:
            text: The text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text,
                },
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            
            logger.warning(f"Embedding API call failed with status {response.status_code}")
            return None

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        content_key: str = "content",
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to the query.
        
        Uses BGE reranker embeddings and cosine similarity to score relevance.
        
        Args:
            query: The search query
            documents: List of documents with content to rerank
            top_k: Number of top documents to return (None = return all)
            content_key: Key to use for document content
            
        Returns:
            Reranked list of documents with added rerank_score
        """
        if not documents:
            return []

        if len(documents) == 1:
            documents[0]["rerank_score"] = 1.0
            return documents

        try:
            # Get query embedding using query prefix for better results
            query_text = f"query: {query}"
            query_embedding = self._get_embedding(query_text)
            
            if not query_embedding:
                logger.warning("Failed to get query embedding, returning original order")
                for doc in documents:
                    doc["rerank_score"] = doc.get("similarity", 0.5)
                return documents[:top_k] if top_k else documents

            # Score each document
            scored_documents = []
            for doc in documents:
                content = doc.get(content_key, "")
                if not content:
                    doc["rerank_score"] = 0.0
                    scored_documents.append(doc)
                    continue

                # Get document embedding using passage prefix
                doc_text = f"passage: {content}"
                doc_embedding = self._get_embedding(doc_text)
                
                if doc_embedding:
                    # Calculate cosine similarity
                    score = self._cosine_similarity(query_embedding, doc_embedding)
                    # Normalize from [-1, 1] to [0, 1]
                    normalized_score = (score + 1) / 2
                    doc["rerank_score"] = normalized_score
                else:
                    # Fall back to original similarity if available
                    doc["rerank_score"] = doc.get("similarity", 0.5)
                
                scored_documents.append(doc)

            # Sort by rerank score (descending)
            scored_documents.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

            # Apply top_k if specified
            if top_k is not None and top_k > 0:
                scored_documents = scored_documents[:top_k]

            logger.info(
                f"Reranked {len(documents)} documents, returning top {len(scored_documents)}"
            )
            return scored_documents

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original documents with default scores
            for doc in documents:
                doc["rerank_score"] = doc.get("similarity", 0.5)
            return documents[:top_k] if top_k else documents

    def rerank_batch(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        content_key: str = "content",
        batch_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """Rerank documents in batches for better performance.
        
        Args:
            query: The search query
            documents: List of documents with content to rerank
            top_k: Number of top documents to return
            content_key: Key to use for document content
            batch_size: Number of documents to process in each batch
            
        Returns:
            Reranked list of documents with added rerank_score
        """
        if not documents:
            return []

        # For small sets, use regular rerank
        if len(documents) <= batch_size:
            return self.rerank(query, documents, top_k, content_key)

        try:
            # Get query embedding once
            query_text = f"query: {query}"
            query_embedding = self._get_embedding(query_text)
            
            if not query_embedding:
                logger.warning("Failed to get query embedding for batch rerank")
                for doc in documents:
                    doc["rerank_score"] = doc.get("similarity", 0.5)
                return documents[:top_k] if top_k else documents

            all_scored = []
            
            # Process all documents and score them
            for doc in documents:
                content = doc.get(content_key, "")
                if not content:
                    doc["rerank_score"] = 0.0
                    all_scored.append(doc)
                    continue

                doc_text = f"passage: {content}"
                doc_embedding = self._get_embedding(doc_text)
                
                if doc_embedding:
                    score = self._cosine_similarity(query_embedding, doc_embedding)
                    normalized_score = (score + 1) / 2
                    doc["rerank_score"] = normalized_score
                else:
                    doc["rerank_score"] = doc.get("similarity", 0.5)
                
                all_scored.append(doc)

            # Sort all by rerank score
            all_scored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

            # Apply top_k
            if top_k is not None and top_k > 0:
                all_scored = all_scored[:top_k]

            return all_scored

        except Exception as e:
            logger.error(f"Batch reranking failed: {e}")
            return documents[:top_k] if top_k else documents


# Global reranker instance
_reranker_instance: Optional[OllamaReranker] = None


def get_reranker() -> OllamaReranker:
    """Get the global reranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = OllamaReranker(
            model=settings.reranker_model,
            base_url=settings.ollama_base_url,
        )
    return _reranker_instance


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convenience function to rerank chunks.
    
    Args:
        query: The search query
        chunks: List of chunks to rerank
        top_k: Number of top chunks to return
        
    Returns:
        Reranked list of chunks
    """
    if not settings.enable_reranking:
        return chunks[:top_k] if top_k else chunks
    
    reranker = get_reranker()
    return reranker.rerank(query, chunks, top_k=top_k, content_key="content")
