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
    chat_history: List[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Generate response using retrieved context and query analysis.

    Args:
        query: User query string
        context_chunks: Retrieved document chunks
        query_analysis: Query analysis results
        temperature: LLM temperature for response generation
        chat_history: Optional conversation history for follow-up questions

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
                "quality_score": None,
            }

        # Filter out chunks with 0.000 similarity before processing sources
        relevant_chunks = [
            chunk
            for chunk in context_chunks
            if chunk.get("similarity", chunk.get("hybrid_score", 0.0)) > 0.0
        ]

        # Generate response using LLM with only relevant chunks
        # Include chat history if this is a follow-up question
        response_data = llm_manager.generate_rag_response(
            query=query,
            context_chunks=relevant_chunks,
            include_sources=True,
            temperature=temperature,
            chat_history=chat_history if query_analysis.get("is_follow_up") else None,
        )

        # Prepare sources information with entity support
        sources = []
        for i, chunk in enumerate(relevant_chunks):
            source_info = {
                "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                "content": chunk.get("content", ""),
                "similarity": chunk.get("similarity", chunk.get("hybrid_score", 0.0)),
                "document_name": chunk.get("document_name", "Unknown Document"),
                "document_id": chunk.get("document_id", ""),
                "filename": chunk.get(
                    "filename", chunk.get("document_name", "Unknown Document")
                ),
                "metadata": chunk.get("metadata", {}),
                "chunk_index": chunk.get("chunk_index"),
            }

            # Add entity information if available
            retrieval_mode = chunk.get("retrieval_mode", "")
            retrieval_source = chunk.get("retrieval_source", "")

            # Check if chunk has entity information regardless of retrieval mode
            contained_entities = chunk.get("contained_entities", [])
            relevant_entities = chunk.get("relevant_entities", [])

            # Use the most relevant entities or contained entities
            entities = relevant_entities or contained_entities

            # For entity-based retrieval, create entity sources
            if retrieval_mode == "entity_based" or retrieval_source == "entity_based":
                if entities:
                    # Create separate entity sources for entity-based retrieval
                    for entity_name in entities[:3]:  # Limit to top 3 entities
                        entity_source = {
                            "entity_name": entity_name,
                            "entity_type": "Entity",  # Default type
                            "entity_id": f"entity_{hash(entity_name) % 10000}",
                            "relevance_score": source_info["similarity"],
                            "content": chunk.get("content", ""),
                            "related_chunks": [
                                {
                                    "chunk_id": chunk.get("chunk_id"),
                                    "content": chunk.get("content", "")[:200] + "...",
                                }
                            ],
                            "document_name": source_info["document_name"],
                            "filename": source_info["filename"],
                        }
                        sources.append(entity_source)
                else:
                    # No entities, add as regular chunk
                    sources.append(source_info)
            else:
                # For chunk-based or hybrid mode, add entity info to chunk source
                if entities:
                    source_info["contained_entities"] = entities
                    source_info["entity_enhanced"] = True

                sources.append(source_info)

        # Enhance response with analysis insights
        query_type = query_analysis.get("query_type", "factual")
        complexity = query_analysis.get("complexity", "simple")

        metadata = {
            "chunks_used": len(relevant_chunks),
            "chunks_filtered": len(context_chunks) - len(relevant_chunks),
            "query_type": query_type,
            "complexity": complexity,
            "requires_reasoning": query_analysis.get("requires_reasoning", False),
            "key_concepts": query_analysis.get("key_concepts", []),
        }

        if len(relevant_chunks) < len(context_chunks):
            logger.info(
                f"Filtered out {len(context_chunks) - len(relevant_chunks)} chunks with 0.000 similarity"
            )
        logger.info(f"Generated response using {len(relevant_chunks)} relevant chunks")

        # Don't calculate quality score here - let it be done asynchronously
        # during response streaming to reduce wait time for users
        return {
            "response": response_data.get("answer", ""),
            "sources": sources,
            "metadata": metadata,
            "quality_score": None,  # Will be calculated asynchronously
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
            "quality_score": None,
        }
