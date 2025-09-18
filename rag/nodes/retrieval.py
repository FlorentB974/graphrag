"""
Document retrieval node for LangGraph RAG pipeline.
"""

import logging
from typing import Any, Dict, List

from rag.retriever import document_retriever

logger = logging.getLogger(__name__)


def retrieve_documents(
    query: str,
    query_analysis: Dict[str, Any],
    retrieval_mode: str = "graph_enhanced",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents based on query and analysis.

    Args:
        query: User query string
        query_analysis: Query analysis results
        retrieval_mode: Retrieval strategy ("simple", "graph_enhanced", "hybrid")
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
        if complexity == "complex" or requires_multiple:
            adjusted_top_k = min(top_k + 3, 10)
        elif query_type == "comparative":
            adjusted_top_k = min(top_k + 5, 12)

        # Choose retrieval method based on mode
        if retrieval_mode == "simple":
            # Use simple vector similarity
            chunks = document_retriever.retrieve_similar_chunks(query, adjusted_top_k)
        elif retrieval_mode == "hybrid":
            # Use hybrid retrieval
            chunks = document_retriever.hybrid_retrieval(query, top_k=adjusted_top_k)
        else:  # graph_enhanced (default)
            if query_analysis.get("requires_reasoning", False):
                # Use graph expansion for reasoning queries
                chunks = document_retriever.retrieve_with_graph_expansion(
                    query,
                    top_k=min(adjusted_top_k, 5),  # Initial chunks
                    expand_depth=2,
                )
            else:
                # Use simple vector similarity for factual queries
                chunks = document_retriever.retrieve_similar_chunks(
                    query, adjusted_top_k
                )

        logger.info(
            f"Retrieved {len(chunks)} chunks using {retrieval_mode} mode with top_k={top_k}"
        )
        return chunks

    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return []
