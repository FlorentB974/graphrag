"""
Graph reasoning node for LangGraph RAG pipeline.
"""

import logging
from typing import Any, Dict, List

from core.graph_db import graph_db

logger = logging.getLogger(__name__)


def reason_with_graph(
    query: str, retrieved_chunks: List[Dict[str, Any]], query_analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Perform graph-based reasoning to enhance context.

    Args:
        query: User query string
        retrieved_chunks: Initially retrieved chunks
        query_analysis: Query analysis results

    Returns:
        Enhanced list of chunks with graph context
    """
    try:
        if not retrieved_chunks:
            logger.warning("No retrieved chunks for graph reasoning")
            return []

        enhanced_chunks = list(retrieved_chunks)  # Start with original chunks

        # Only perform graph reasoning for complex or analytical queries
        requires_reasoning = query_analysis.get("requires_reasoning", False)
        complexity = query_analysis.get("complexity", "simple")

        if not requires_reasoning and complexity != "complex":
            logger.info("Simple query - skipping graph reasoning")
            return enhanced_chunks

        # Find related chunks through graph traversal
        seen_chunk_ids = {chunk.get("chunk_id") for chunk in retrieved_chunks}

        for chunk in retrieved_chunks[:3]:  # Only expand from top 3 chunks
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue

            try:
                # Get chunks related through graph relationships
                related_chunks = graph_db.get_related_chunks(
                    chunk_id=chunk_id,
                    relationship_types=["SIMILAR_TO", "HAS_CHUNK"],
                    max_depth=2,  # Keep it shallow for performance
                )

                # Add unique related chunks
                for related_chunk in related_chunks:
                    related_id = related_chunk.get("chunk_id")
                    if related_id and related_id not in seen_chunk_ids:
                        # Add relationship context to metadata
                        related_chunk["reasoning_context"] = {
                            "related_to": chunk_id,
                            "relationship_type": "graph_expansion",
                            "distance": related_chunk.get("distance", 1),
                        }
                        enhanced_chunks.append(related_chunk)
                        seen_chunk_ids.add(related_id)

                        # Limit total chunks to prevent overwhelming the LLM
                        if len(enhanced_chunks) >= 10:
                            break

            except Exception as e:
                logger.warning(f"Failed to get related chunks for {chunk_id}: {e}")
                continue

        logger.info(
            f"Graph reasoning: {len(retrieved_chunks)} -> {len(enhanced_chunks)} chunks"
        )
        return enhanced_chunks

    except Exception as e:
        logger.error(f"Graph reasoning failed: {e}")
        return retrieved_chunks  # Return original chunks on error
