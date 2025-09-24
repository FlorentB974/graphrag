"""
Enhanced retrieval logic with support for chunk-based, entity-based, and hybrid modes.
"""

import logging
from enum import Enum
from typing import Any, Dict, List
import hashlib

from core.embeddings import embedding_manager
from core.graph_db import graph_db

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Different retrieval modes supported by the system."""
    CHUNK_ONLY = "chunk_only"
    ENTITY_ONLY = "entity_only"
    HYBRID = "hybrid"


class EnhancedDocumentRetriever:
    """Enhanced document retriever with multiple retrieval strategies."""

    def __init__(self):
        """Initialize the enhanced document retriever."""
        pass

    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate a consistent entity ID from entity name."""
        return hashlib.md5(entity_name.upper().strip().encode()).hexdigest()

    async def chunk_based_retrieval(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Traditional chunk-based retrieval using vector similarity.
        
        Args:
            query: User query
            top_k: Number of similar chunks to retrieve
            
        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = embedding_manager.get_embedding(query)

            # Perform vector similarity search
            similar_chunks = graph_db.vector_similarity_search(query_embedding, top_k)

            logger.info(f"Retrieved {len(similar_chunks)} chunks using chunk-based retrieval")
            return similar_chunks

        except Exception as e:
            logger.error(f"Chunk-based retrieval failed: {e}")
            return []

    async def entity_based_retrieval(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Entity-based retrieval using entity similarity and relationships.
        
        Args:
            query: User query
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            List of chunks related to relevant entities
        """
        try:
            # First, find relevant entities using full-text search
            relevant_entities = graph_db.entity_similarity_search(query, top_k)
            
            if not relevant_entities:
                logger.info("No relevant entities found for entity-based retrieval")
                return []

            # Get entity IDs
            entity_ids = [entity["entity_id"] for entity in relevant_entities]
            
            # Get chunks that contain these entities
            relevant_chunks = graph_db.get_chunks_for_entities(entity_ids)
            
            # Enhance chunks with entity information
            for chunk in relevant_chunks:
                chunk["retrieval_mode"] = "entity_based"
                chunk["relevant_entities"] = chunk.get("contained_entities", [])
            
            logger.info(f"Retrieved {len(relevant_chunks)} chunks using entity-based retrieval")
            return relevant_chunks[:top_k]

        except Exception as e:
            logger.error(f"Entity-based retrieval failed: {e}")
            return []

    async def entity_expansion_retrieval(
        self, initial_entities: List[str], expansion_depth: int = 1, max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Expand retrieval by following entity relationships.
        
        Args:
            initial_entities: List of initial entity IDs
            expansion_depth: How many relationship hops to follow
            max_chunks: Maximum chunks to retrieve
            
        Returns:
            List of chunks from expanded entity network
        """
        try:
            expanded_entities = set(initial_entities)
            
            # Expand entity network by following relationships
            for entity_id in initial_entities:
                relationships = graph_db.get_entity_relationships(entity_id)
                for rel in relationships:
                    expanded_entities.add(rel["related_entity_id"])
            
            # Get chunks for expanded entity set
            expanded_chunks = graph_db.get_chunks_for_entities(list(expanded_entities))
            
            # Add expansion metadata
            for chunk in expanded_chunks:
                chunk["retrieval_mode"] = "entity_expansion"
                chunk["expansion_depth"] = expansion_depth
            
            logger.info(f"Entity expansion retrieved {len(expanded_chunks)} chunks")
            return expanded_chunks[:max_chunks]

        except Exception as e:
            logger.error(f"Entity expansion retrieval failed: {e}")
            return []

    async def hybrid_retrieval(
        self, query: str, top_k: int = 5, chunk_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining chunk-based and entity-based approaches.
        
        Args:
            query: User query
            top_k: Total number of chunks to retrieve
            chunk_weight: Weight for chunk-based results (0.0-1.0)
            
        Returns:
            List of chunks from both approaches, de-duplicated and ranked
        """
        try:
            # Calculate split for each approach
            chunk_count = max(1, int(top_k * chunk_weight))
            entity_count = max(1, top_k - chunk_count)
            
            # Get results from both approaches
            chunk_results = await self.chunk_based_retrieval(query, chunk_count)
            entity_results = await self.entity_based_retrieval(query, entity_count)
            
            # Combine and deduplicate results by chunk_id
            combined_results = {}
            
            # Add chunk-based results
            for result in chunk_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    result["retrieval_source"] = "chunk_based"
                    result["chunk_score"] = result.get("similarity", 0.0)
                    combined_results[chunk_id] = result
            
            # Add entity-based results (merge if duplicate)
            for result in entity_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    if chunk_id in combined_results:
                        # Merge information from both sources
                        existing = combined_results[chunk_id]
                        existing["retrieval_source"] = "hybrid"
                        existing["relevant_entities"] = result.get("contained_entities", [])
                        # Boost score for chunks found by both methods
                        existing["hybrid_score"] = existing.get("chunk_score", 0.0) * 1.2
                    else:
                        result["retrieval_source"] = "entity_based"
                        result["hybrid_score"] = result.get("similarity", 0.5)
                        combined_results[chunk_id] = result
            
            # Sort by hybrid score and return top_k
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
            
            logger.info(f"Hybrid retrieval combined {len(chunk_results)} chunk results "
                        f"and {len(entity_results)} entity results into "
                        f"{len(final_results)} final results")
            
            return final_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method that dispatches to the appropriate strategy.
        
        Args:
            query: User query
            mode: Retrieval mode to use
            top_k: Number of results to return
            **kwargs: Additional parameters for specific retrieval modes
            
        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"Starting retrieval with mode: {mode.value}, top_k: {top_k}")
        
        if mode == RetrievalMode.CHUNK_ONLY:
            return await self.chunk_based_retrieval(query, top_k)
        elif mode == RetrievalMode.ENTITY_ONLY:
            return await self.entity_based_retrieval(query, top_k)
        elif mode == RetrievalMode.HYBRID:
            chunk_weight = kwargs.get("chunk_weight", 0.5)
            return await self.hybrid_retrieval(query, top_k, chunk_weight)
        else:
            logger.error(f"Unknown retrieval mode: {mode}")
            return []

    async def retrieve_with_graph_expansion(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 3,
        expand_depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks and expand using graph relationships.
        
        Args:
            query: User query
            mode: Initial retrieval mode
            top_k: Number of initial chunks to retrieve
            expand_depth: Depth of graph expansion
            
        Returns:
            List of chunks including expanded context
        """
        try:
            # Get initial results
            initial_chunks = await self.retrieve(query, mode, top_k)
            
            if not initial_chunks:
                return []

            expanded_chunks = []
            seen_chunk_ids = set()

            # Add initial chunks
            for chunk in initial_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    expanded_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

            # Expand based on mode
            if mode in [RetrievalMode.ENTITY_ONLY, RetrievalMode.HYBRID]:
                # Entity-based expansion
                for chunk in initial_chunks:
                    entities = chunk.get("contained_entities", [])
                    if entities:
                        entity_ids = [self._generate_entity_id(name) for name in entities]
                        expanded = await self.entity_expansion_retrieval(
                            entity_ids, expansion_depth=expand_depth
                        )
                        
                        for exp_chunk in expanded:
                            exp_chunk_id = exp_chunk.get("chunk_id")
                            if exp_chunk_id and exp_chunk_id not in seen_chunk_ids:
                                exp_chunk["expansion_context"] = {
                                    "source_chunk": chunk.get("chunk_id"),
                                    "expansion_type": "entity_relationship"
                                }
                                expanded_chunks.append(exp_chunk)
                                seen_chunk_ids.add(exp_chunk_id)

            if mode in [RetrievalMode.CHUNK_ONLY, RetrievalMode.HYBRID]:
                # Chunk similarity expansion
                for chunk in initial_chunks:
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id:
                        related_chunks = graph_db.get_related_chunks(
                            chunk_id, max_depth=expand_depth
                        )
                        
                        for rel_chunk in related_chunks:
                            rel_chunk_id = rel_chunk.get("chunk_id")
                            if rel_chunk_id and rel_chunk_id not in seen_chunk_ids:
                                rel_chunk["expansion_context"] = {
                                    "source_chunk": chunk_id,
                                    "expansion_type": "chunk_similarity"
                                }
                                expanded_chunks.append(rel_chunk)
                                seen_chunk_ids.add(rel_chunk_id)

            logger.info(f"Graph expansion: {len(initial_chunks)} -> {len(expanded_chunks)} chunks")
            return expanded_chunks

        except Exception as e:
            logger.error(f"Graph expansion retrieval failed: {e}")
            # Return empty list if we don't have initial_chunks
            return []


# Global enhanced retriever instance
enhanced_retriever = EnhancedDocumentRetriever()
