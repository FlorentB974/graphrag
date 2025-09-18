"""
Document retrieval node for LangGraph RAG pipeline.
"""
import logging
from typing import Dict, List, Any
from rag.retriever import document_retriever

logger = logging.getLogger(__name__)


def retrieve_documents(query: str, query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Retrieve relevant documents based on query and analysis.
    
    Args:
        query: User query string
        query_analysis: Query analysis results
        
    Returns:
        List of relevant document chunks
    """
    try:
        # Determine retrieval strategy based on query analysis
        complexity = query_analysis.get("complexity", "simple")
        requires_multiple = query_analysis.get("requires_multiple_sources", False)
        query_type = query_analysis.get("query_type", "factual")
        
        # Adjust retrieval parameters
        top_k = 5  # Default
        
        if complexity == "complex" or requires_multiple:
            top_k = 8
        elif query_type == "comparative":
            top_k = 10
        
        # Choose retrieval method
        if query_analysis.get("requires_reasoning", False):
            # Use graph expansion for reasoning queries
            chunks = document_retriever.retrieve_with_graph_expansion(
                query, 
                top_k=min(top_k, 5),  # Initial chunks
                expand_depth=2
            )
        else:
            # Use simple vector similarity for factual queries
            chunks = document_retriever.retrieve_similar_chunks(query, top_k)
        
        logger.info(f"Retrieved {len(chunks)} chunks for query type: {query_type}")
        return chunks
        
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        return []