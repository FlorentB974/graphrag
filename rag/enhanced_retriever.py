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
    
    def _get_entity_ids_from_names(self, entity_names: List[str]) -> List[str]:
        """Get actual entity IDs from entity names by querying the database.
        
        Args:
            entity_names: List of entity names to look up
            
        Returns:
            List of actual entity IDs found in database
        """
        if not entity_names:
            return []
            
        with graph_db.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.name IN $entity_names
                RETURN e.id as entity_id, e.name as name
                """,
                entity_names=entity_names,
            )
            found_entities = [record.data() for record in result]
            
        entity_ids = [entity["entity_id"] for entity in found_entities]
        
        if len(entity_ids) < len(entity_names):
            missing_count = len(entity_names) - len(entity_ids)
            logger.debug(f"Found {len(entity_ids)}/{len(entity_names)} entities in database. {missing_count} not found.")
            
        return entity_ids

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
            from config.settings import settings
            
            # Generate query embedding
            query_embedding = embedding_manager.get_embedding(query)

            # Perform vector similarity search with larger top_k to allow filtering
            similar_chunks = graph_db.vector_similarity_search(query_embedding, top_k * 3)
            
            # Filter chunks by minimum similarity threshold
            filtered_chunks = [
                chunk for chunk in similar_chunks 
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]
            
            # Return only top_k after filtering
            final_chunks = filtered_chunks[:top_k]

            logger.info(f"Retrieved {len(similar_chunks)} chunks, filtered to {len(filtered_chunks)}, returning {len(final_chunks)} chunks using chunk-based retrieval")
            return final_chunks

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
            
            # Calculate similarity scores for chunks based on query
            query_embedding = embedding_manager.get_embedding(query)
            
            # Enhance chunks with entity information and similarity scores
            for chunk in relevant_chunks:
                chunk["retrieval_mode"] = "entity_based"
                chunk["relevant_entities"] = chunk.get("contained_entities", [])
                
                # Calculate similarity score if chunk has content
                if chunk.get("content"):
                    # Get or calculate chunk embedding
                    chunk_embedding = None
                    chunk_id = chunk.get("chunk_id")
                    if chunk_id:
                        # Get embedding from database
                        with graph_db.driver.session() as session:  # type: ignore
                            result = session.run(
                                "MATCH (c:Chunk {id: $chunk_id}) RETURN c.embedding as embedding",
                                chunk_id=chunk_id
                            )
                            record = result.single()
                            if record and record["embedding"]:
                                chunk_embedding = record["embedding"]
                    
                    if chunk_embedding:
                        # Calculate cosine similarity
                        similarity = graph_db._calculate_cosine_similarity(query_embedding, chunk_embedding)
                        chunk["similarity"] = similarity
                    else:
                        # No embedding available - this chunk should be filtered out
                        chunk["similarity"] = 0.0
                else:
                    # No content - this chunk should be filtered out
                    chunk["similarity"] = 0.0
            
            # Filter chunks by minimum similarity threshold
            from config.settings import settings
            filtered_chunks = [
                chunk for chunk in relevant_chunks
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]
            
            # Sort by similarity score
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            # Return top_k chunks
            final_chunks = filtered_chunks[:top_k]
            logger.info(f"Entity-based retrieval: found {len(relevant_chunks)} chunks, filtered to {len(filtered_chunks)}, returning {len(final_chunks)} with scores")
            
            return final_chunks

        except Exception as e:
            logger.error(f"Entity-based retrieval failed: {e}")
            return []

    async def entity_expansion_retrieval(
        self, initial_entities: List[str], expansion_depth: int = 1, max_chunks: int = 50
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
            total_relationships = 0
            
            # Track entity relationship strengths for scoring
            entity_scores = {}
            for entity_id in initial_entities:
                entity_scores[entity_id] = 1.0  # Initial entities get full score
            
            # Expand entity network by following relationships
            for entity_id in initial_entities:
                relationships = graph_db.get_entity_relationships(entity_id)
                total_relationships += len(relationships)
                for rel in relationships:
                    related_id = rel["related_entity_id"]
                    expanded_entities.add(related_id)
                    # Score related entities based on relationship strength
                    strength = rel.get("strength", 0.5)
                    entity_scores[related_id] = max(entity_scores.get(related_id, 0.0), strength * 0.7)
            
            # Get chunks for expanded entity set
            expanded_chunks = graph_db.get_chunks_for_entities(list(expanded_entities))
            
            # Add expansion metadata and similarity scores
            for chunk in expanded_chunks:
                chunk["retrieval_mode"] = "entity_expansion"
                chunk["expansion_depth"] = expansion_depth
                
                # Calculate similarity based on contained entities' scores
                contained_entities = chunk.get("contained_entities", [])
                if contained_entities:
                    # Get entity IDs for contained entities
                    contained_entity_ids = self._get_entity_ids_from_names(contained_entities)
                    if contained_entity_ids:
                        # Use the highest score among contained entities
                        max_entity_score = max(
                            entity_scores.get(entity_id, 0.0) for entity_id in contained_entity_ids
                            if entity_id in entity_scores
                        ) if any(eid in entity_scores for eid in contained_entity_ids) else 0.0
                        chunk["similarity"] = max_entity_score
                    else:
                        chunk["similarity"] = 0.0  # No entity match
                else:
                    chunk["similarity"] = 0.0  # No entities
            
            # Filter by minimum similarity threshold
            from config.settings import settings
            filtered_chunks = [
                chunk for chunk in expanded_chunks
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]
            
            # Sort by similarity score
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            final_chunks = filtered_chunks[:max_chunks]
            logger.info(f"Entity expansion: {len(initial_entities)} entities → {len(expanded_entities)} expanded entities "
                        f"({total_relationships} relationships) → {len(expanded_chunks)} chunks → {len(filtered_chunks)} filtered → {len(final_chunks)} returned")
            
            return final_chunks

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
                        existing["contained_entities"] = result.get("contained_entities", [])
                        # Boost score for chunks found by both methods
                        chunk_score = existing.get("chunk_score", 0.0)
                        entity_score = result.get("similarity", 0.3)
                        existing["hybrid_score"] = min(1.0, (chunk_score + entity_score) * 0.8)  # Combined score with cap
                    else:
                        result["retrieval_source"] = "entity_based"
                        # Use actual similarity score, with better fallback
                        result["hybrid_score"] = result.get("similarity", 0.3)
                        combined_results[chunk_id] = result
            
            # Sort by hybrid score and return top_k
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
            
            # Count overlaps for better reporting
            chunk_only_count = sum(1 for r in final_results if r.get("retrieval_source") == "chunk_based")
            entity_only_count = sum(1 for r in final_results if r.get("retrieval_source") == "entity_based")
            hybrid_count = sum(1 for r in final_results if r.get("retrieval_source") == "hybrid")
            
            logger.info(f"Hybrid retrieval: {len(chunk_results)} chunk + {len(entity_results)} entity → "
                        f"{len(final_results)} total ({chunk_only_count} chunk-only, {entity_only_count} entity-only, {hybrid_count} overlapping)")
            
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
                        entity_ids = self._get_entity_ids_from_names(entities)
                        if entity_ids:  # Only expand if we found valid entity IDs
                            expanded = await self.entity_expansion_retrieval(
                                entity_ids, expansion_depth=expand_depth
                            )
                            
                            source_similarity = chunk.get("similarity", chunk.get("hybrid_score", 0.3))
                            
                            for exp_chunk in expanded:
                                exp_chunk_id = exp_chunk.get("chunk_id")
                                if exp_chunk_id and exp_chunk_id not in seen_chunk_ids:
                                    exp_chunk["expansion_context"] = {
                                        "source_chunk": chunk.get("chunk_id"),
                                        "expansion_type": "entity_relationship"
                                    }
                                    # Inherit similarity with decay for expansion, but don't force minimum
                                    if not exp_chunk.get("similarity") or exp_chunk.get("similarity", 0.0) == 0.0:
                                        exp_chunk["similarity"] = source_similarity * 0.6
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
                        
                        source_similarity = chunk.get("similarity", chunk.get("hybrid_score", 0.3))
                        
                        for rel_chunk in related_chunks:
                            rel_chunk_id = rel_chunk.get("chunk_id")
                            if rel_chunk_id and rel_chunk_id not in seen_chunk_ids:
                                rel_chunk["expansion_context"] = {
                                    "source_chunk": chunk_id,
                                    "expansion_type": "chunk_similarity"
                                }
                                # Calculate similarity based on distance and source similarity
                                distance = rel_chunk.get("distance", 1)
                                # Decay factor based on graph distance
                                decay_factor = 1.0 / (distance + 1)
                                rel_chunk["similarity"] = source_similarity * decay_factor
                                expanded_chunks.append(rel_chunk)
                                seen_chunk_ids.add(rel_chunk_id)

            # Filter out chunks with similarity below threshold
            from config.settings import settings
            filtered_chunks = [
                chunk for chunk in expanded_chunks
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]
            
            # Count expansion types
            original_count = sum(1 for chunk in filtered_chunks if not chunk.get("expansion_context"))
            entity_expansion_count = sum(1 for chunk in filtered_chunks
                                         if chunk.get("expansion_context", {}).get("expansion_type") == "entity_relationship")
            chunk_expansion_count = sum(1 for chunk in filtered_chunks
                                        if chunk.get("expansion_context", {}).get("expansion_type") == "chunk_similarity")
            
            logger.info(f"Graph expansion ({mode.value}): {len(initial_chunks)} initial → {len(expanded_chunks)} total → {len(filtered_chunks)} filtered "
                        f"({original_count} original + {entity_expansion_count} entity-expanded + {chunk_expansion_count} similarity-expanded)")
            return filtered_chunks

        except Exception as e:
            logger.error(f"Graph expansion retrieval failed: {e}")
            # Return empty list if we don't have initial_chunks
            return []


# Global enhanced retriever instance
enhanced_retriever = EnhancedDocumentRetriever()
