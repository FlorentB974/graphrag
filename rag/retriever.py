"""
Document retrieval logic for the RAG pipeline.
"""
import logging
from typing import Dict, List, Any
from core.graph_db import graph_db
from core.embeddings import embedding_manager

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """Handles document retrieval using vector similarity and graph traversal."""
    
    def __init__(self):
        """Initialize the document retriever."""
        pass
    
    def retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve chunks similar to the query using vector search.
        
        Args:
            query: User query
            top_k: Number of top similar chunks to retrieve
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = embedding_manager.get_embedding(query)
            
            # Perform vector similarity search
            similar_chunks = graph_db.vector_similarity_search(query_embedding, top_k)
            
            logger.info(f"Retrieved {len(similar_chunks)} similar chunks for query")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar chunks: {e}")
            return []
    
    def retrieve_with_graph_expansion(self, query: str, top_k: int = 3, 
                                      expand_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieve chunks and expand using graph relationships.
        
        Args:
            query: User query
            top_k: Number of initial chunks to retrieve
            expand_depth: Depth of graph expansion
            
        Returns:
            List of chunks including expanded context
        """
        try:
            # Get initial similar chunks
            initial_chunks = self.retrieve_similar_chunks(query, top_k)
            
            if not initial_chunks:
                return []
            
            expanded_chunks = []
            seen_chunk_ids = set()
            
            for chunk in initial_chunks:
                chunk_id = chunk['chunk_id']
                
                if chunk_id not in seen_chunk_ids:
                    expanded_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)
                
                # Get related chunks through graph traversal
                related_chunks = graph_db.get_related_chunks(
                    chunk_id, 
                    max_depth=expand_depth
                )
                
                for related_chunk in related_chunks:
                    related_id = related_chunk['chunk_id']
                    
                    if related_id not in seen_chunk_ids:
                        expanded_chunks.append(related_chunk)
                        seen_chunk_ids.add(related_id)
            
            logger.info(f"Expanded retrieval: {len(initial_chunks)} -> {len(expanded_chunks)} chunks")
            return expanded_chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve with graph expansion: {e}")
            return []
    
    def retrieve_by_document(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from a specific document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of chunks from the document
        """
        try:
            chunks = graph_db.get_document_chunks(doc_id)
            logger.info(f"Retrieved {len(chunks)} chunks from document {doc_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve document chunks: {e}")
            return []
    
    def hybrid_retrieval(self, query: str, doc_filter: List[str] = None,  # type: ignore
                         top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining vector search and keyword matching.
        
        Args:
            query: User query
            doc_filter: Optional list of document IDs to filter by
            top_k: Number of chunks to retrieve
            
        Returns:
            List of relevant chunks
        """
        try:
            # For now, implement simple vector search
            # Future enhancement: add keyword/BM25 scoring
            chunks = self.retrieve_similar_chunks(query, top_k * 2)
            
            # Filter by document IDs if provided
            if doc_filter:
                chunks = [
                    chunk for chunk in chunks 
                    if any(doc_id in chunk.get('content', '') for doc_id in doc_filter)
                ]
            
            # Return top_k results
            return chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid retrieval: {e}")
            return []


# Global retriever instance
document_retriever = DocumentRetriever()
