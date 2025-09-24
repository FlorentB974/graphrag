"""
Document retrieval node for LangGraph RAG pipeline.
"""

import asyncio
import logging
from typing import Any, Dict, List

from rag.enhanced_retriever import EnhancedDocumentRetriever, RetrievalMode
from config.settings import settings

logger = logging.getLogger(__name__)

# Initialize enhanced retriever
enhanced_retriever = EnhancedDocumentRetriever()


async def retrieve_documents_async(
    query: str,
    query_analysis: Dict[str, Any],
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    chunk_weight: float = 0.5,
    graph_expansion: bool = True,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents based on query and analysis using enhanced retriever.

    Args:
        query: User query string
        query_analysis: Query analysis results
        retrieval_mode: Retrieval strategy ("chunk_only", "entity_only", "hybrid", "auto")
        top_k: Number of chunks to retrieve

    Returns:
        List of relevant document chunks
    """
    try:
        # Determine retrieval strategy based on query analysis and mode
        complexity = query_analysis.get("complexity", "simple")
        requires_multiple = query_analysis.get("requires_multiple_sources", False)
        query_type = query_analysis.get("query_type", "factual")

        # Adjust top_k based on query complexity
        adjusted_top_k = top_k
        adjustment_reason = None
        if complexity == "complex" or requires_multiple:
            adjusted_top_k = min(top_k + 3, 10)
            adjustment_reason = "complexity or multiple sources required"
        elif query_type == "comparative":
            adjusted_top_k = min(top_k + 5, 12)
            adjustment_reason = "comparative query"

        # Log why top_k was adjusted (if it changed)
        if adjusted_top_k != top_k:
            logger.info(
                "Adjusted top_k from %d to %d (%s) — query_type=%s, complexity=%s, requires_multiple=%s",
                top_k,
                adjusted_top_k,
                adjustment_reason or "adjusted",
                query_type,
                complexity,
                requires_multiple,
            )

        # Map retrieval modes to enhanced retriever modes
        mode_mapping = {
            "simple": RetrievalMode.CHUNK_ONLY,
            "chunk_only": RetrievalMode.CHUNK_ONLY,
            "entity_only": RetrievalMode.ENTITY_ONLY,
            "hybrid": RetrievalMode.HYBRID,
            "graph_enhanced": RetrievalMode.HYBRID,  # Legacy compatibility
            "auto": RetrievalMode.HYBRID if settings.enable_entity_extraction else RetrievalMode.CHUNK_ONLY
        }
        
        # Get the appropriate retrieval mode
        enhanced_mode = mode_mapping.get(retrieval_mode, RetrievalMode.HYBRID)
        
        # Use enhanced retriever. Prefer graph expansion when configured
        if (complexity == "complex" or query_type == "comparative") and graph_expansion:
            chunks = await enhanced_retriever.retrieve_with_graph_expansion(
                query=query,
                mode=enhanced_mode,
                top_k=adjusted_top_k,
            )
        else:
            # Pass chunk_weight through to hybrid retriever if present
            chunks = await enhanced_retriever.retrieve(
                query=query,
                mode=enhanced_mode,
                top_k=adjusted_top_k,
                chunk_weight=chunk_weight,
            )

        logger.info(
            f"Retrieved {len(chunks)} chunks using {retrieval_mode} mode (enhanced_mode: {enhanced_mode.value}) with top_k={adjusted_top_k}"
        )
        return chunks

    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []


def retrieve_documents(
    query: str,
    query_analysis: Dict[str, Any],
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    chunk_weight: float = 0.5,
    graph_expansion: bool = True,
) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for document retrieval.
    
    Args:
        query: User query string
        query_analysis: Query analysis results
        retrieval_mode: Retrieval strategy
        top_k: Number of chunks to retrieve
        
    Returns:
        List of relevant document chunks
    """
    try:
        return asyncio.run(
            retrieve_documents_async(
                query, query_analysis, retrieval_mode, top_k, chunk_weight, graph_expansion
            )
        )
    except RuntimeError:
        # If event loop is already running, create new task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return loop.run_until_complete(
                retrieve_documents_async(
                    query, query_analysis, retrieval_mode, top_k, chunk_weight, graph_expansion
                )
            )
        else:
            return loop.run_until_complete(
                retrieve_documents_async(
                    query, query_analysis, retrieval_mode, top_k, chunk_weight, graph_expansion
                )
            )
    except Exception as e:
        logger.error(f"Error in synchronous retrieval wrapper: {e}")
        return []
