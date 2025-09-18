"""
Response generation node for LangGraph RAG pipeline.
"""

import logging
from typing import Any, Dict, List

from core.llm import llm_manager

logger = logging.getLogger(__name__)


def generate_response(
    query: str,
    context_chunks: List[Dict[str, Any]],
    query_analysis: Dict[str, Any],
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generate response using retrieved context and query analysis.

    Args:
        query: User query string
        context_chunks: Retrieved document chunks
        query_analysis: Query analysis results
        temperature: LLM temperature for response generation

    Returns:
        Dictionary containing response and metadata
    """
    try:
        if not context_chunks:
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "metadata": {
                    "chunks_used": 0,
                    "query_type": query_analysis.get("query_type", "unknown"),
                },
            }

        # Generate response using LLM
        response_data = llm_manager.generate_rag_response(
            query=query,
            context_chunks=context_chunks,
            include_sources=True,
            temperature=temperature,
        )

        # Prepare sources information
        sources = []
        for i, chunk in enumerate(context_chunks):
            source_info = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "content": chunk.get("content", ""),
                "similarity": chunk.get("similarity", 0.0),
                "document_name": chunk.get("document_name", "Unknown Document"),
                "document_id": chunk.get("document_id", ""),
                "filename": chunk.get(
                    "filename", chunk.get("document_name", "Unknown Document")
                ),
                "metadata": chunk.get("metadata", {}),
            }
            sources.append(source_info)

        # Enhance response with analysis insights
        query_type = query_analysis.get("query_type", "factual")
        complexity = query_analysis.get("complexity", "simple")

        metadata = {
            "chunks_used": len(context_chunks),
            "query_type": query_type,
            "complexity": complexity,
            "requires_reasoning": query_analysis.get("requires_reasoning", False),
            "key_concepts": query_analysis.get("key_concepts", []),
        }

        logger.info(f"Generated response using {len(context_chunks)} chunks")

        return {
            "response": response_data.get("answer", ""),
            "sources": sources,
            "metadata": metadata,
        }

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return {
            "response": f"I apologize, but I encountered an error generating the response: {str(e)}",
            "sources": [],
            "metadata": {
                "error": str(e),
                "chunks_used": len(context_chunks) if context_chunks else 0,
            },
        }
