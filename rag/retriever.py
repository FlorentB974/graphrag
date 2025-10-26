"""
Enhanced retrieval logic with support for chunk-based, entity-based, hybrid modes, and multi-hop reasoning.
"""

import hashlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.embeddings import embedding_manager
from core.graph_db import PathResult, graph_db
from rag.nodes.query_analysis import analyze_query

logger = logging.getLogger(__name__)


class RetrievalMode(Enum):
    """Different retrieval modes supported by the system."""

    CHUNK_ONLY = "chunk_only"
    ENTITY_ONLY = "entity_only"
    HYBRID = "hybrid"


class DocumentRetriever:
    """Document retriever with multiple retrieval strategies."""

    def __init__(self):
        """Initialize the enhanced document retriever."""
        pass

    @staticmethod
    def _extract_hashtags_from_query(query: str) -> List[str]:
        """Extract hashtags from query (words starting with #)."""
        import re
        hashtags = re.findall(r'#\w+', query)
        return hashtags

    @staticmethod
    def _filter_chunks_by_documents(
        chunks: List[Dict[str, Any]],
        allowed_document_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Restrict chunks to a set of allowed document IDs."""
        if not allowed_document_ids:
            return chunks

        allowed_set = set(allowed_document_ids)
        return [chunk for chunk in chunks if chunk.get("document_id") in allowed_set]

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
            logger.debug(
                f"Found {len(entity_ids)}/{len(entity_names)} entities in database. {missing_count} not found."
            )

        return entity_ids

    async def chunk_based_retrieval(
        self,
        query: str,
        top_k: int = 5,
        allowed_document_ids: Optional[List[str]] = None,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Traditional chunk-based retrieval using vector similarity.

        Args:
            query: User query
            top_k: Number of similar chunks to retrieve
            allowed_document_ids: Optional list of document IDs to restrict retrieval
            query_embedding: Pre-computed query embedding (to avoid recomputation)

        Returns:
            List of similar chunks with metadata
        """
        try:
            # Generate query embedding if not provided
            if query_embedding is None:
                query_embedding = embedding_manager.get_embedding(query)
            
            # Perform vector similarity search with larger top_k to allow filtering
            search_limit = top_k * 3
            if allowed_document_ids:
                search_limit = max(search_limit, top_k * 5)

            similar_chunks = graph_db.vector_similarity_search(
                query_embedding, search_limit
            )

            # Enforce document restriction if provided
            similar_chunks = self._filter_chunks_by_documents(
                similar_chunks, allowed_document_ids
            )

            # Filter chunks by minimum similarity threshold
            filtered_chunks = [
                chunk
                for chunk in similar_chunks
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]

            # Return only top_k after filtering
            final_chunks = filtered_chunks[:top_k]

            logger.info(
                "Retrieved %d chunks, filtered to %d, returning %d chunks using chunk-based retrieval (restricted=%s)",
                len(similar_chunks),
                len(filtered_chunks),
                len(final_chunks),
                bool(allowed_document_ids),
            )
            return final_chunks

        except Exception as e:
            logger.error(f"Chunk-based retrieval failed: {e}")
            return []

    async def entity_based_retrieval(
        self,
        query: str,
        top_k: int = 5,
        allowed_document_ids: Optional[List[str]] = None,
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

            # Respect document restriction if provided
            relevant_chunks = self._filter_chunks_by_documents(
                relevant_chunks, allowed_document_ids
            )

            if not relevant_chunks:
                logger.info(
                    "Entity-based retrieval filtered out all chunks due to document restriction"
                )
                return []

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
                                chunk_id=chunk_id,
                            )
                            record = result.single()
                            if record and record["embedding"]:
                                chunk_embedding = record["embedding"]

                    if chunk_embedding:
                        # Calculate cosine similarity
                        similarity = graph_db._calculate_cosine_similarity(
                            query_embedding, chunk_embedding
                        )
                        chunk["similarity"] = similarity
                    else:
                        # No embedding available - this chunk should be filtered out
                        chunk["similarity"] = 0.0
                else:
                    # No content - this chunk should be filtered out
                    chunk["similarity"] = 0.0

            # Filter chunks by minimum similarity threshold
            filtered_chunks = [
                chunk
                for chunk in relevant_chunks
                if chunk.get("similarity", 0.0) >= settings.min_retrieval_similarity
            ]

            # Sort by similarity score
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

            # Return top_k chunks
            final_chunks = filtered_chunks[:top_k]
            logger.info(
                "Entity-based retrieval: found %d chunks, filtered to %d, returning %d with scores (restricted=%s)",
                len(relevant_chunks),
                len(filtered_chunks),
                len(final_chunks),
                bool(allowed_document_ids),
            )

            return final_chunks

        except Exception as e:
            logger.error(f"Entity-based retrieval failed: {e}")
            return []

    async def entity_expansion_retrieval(
        self,
        initial_entities: List[str],
        expansion_depth: int = 1,
        max_chunks: Optional[int] = None,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Expand retrieval by following entity relationships.

        Args:
            initial_entities: List of initial entity IDs
            expansion_depth: How many relationship hops to follow
            max_chunks: Maximum chunks to retrieve (defaults to settings.max_expanded_chunks)

        Returns:
            List of chunks from expanded entity network
        """
        if max_chunks is None:
            max_chunks = settings.max_expanded_chunks

        try:
            expanded_entities = set(initial_entities)
            total_relationships = 0

            # Track entity relationship strengths for scoring
            entity_scores = {}
            for entity_id in initial_entities:
                entity_scores[entity_id] = 1.0  # Initial entities get full score

            # Expand entity network by following relationships (with limits)
            entities_to_process = list(initial_entities)
            current_depth = 0

            while entities_to_process and current_depth < min(
                expansion_depth, settings.max_expansion_depth
            ):
                next_entities = []
                entities_processed_this_depth = 0

                for entity_id in entities_to_process:
                    if entities_processed_this_depth >= settings.max_entity_connections:
                        break  # Limit entities processed per depth

                    relationships = graph_db.get_entity_relationships(entity_id)
                    total_relationships += len(relationships)

                    # Limit and prioritize relationships by strength
                    relationships.sort(
                        key=lambda x: x.get("strength", 0.0), reverse=True
                    )
                    relationships = relationships[: settings.max_entity_connections]

                    for rel in relationships:
                        related_id = rel["related_entity_id"]
                        strength = rel.get("strength", 0.5)

                        # Only follow high-quality relationships
                        if strength >= settings.expansion_similarity_threshold:
                            if related_id not in expanded_entities:
                                expanded_entities.add(related_id)
                                next_entities.append(related_id)

                            # Score related entities based on relationship strength with depth decay
                            decay_factor = 0.7 ** (current_depth + 1)
                            entity_scores[related_id] = max(
                                entity_scores.get(related_id, 0.0),
                                strength * decay_factor,
                            )

                    entities_processed_this_depth += 1

                entities_to_process = next_entities
                current_depth += 1

                # Stop if we have too many entities
                if len(expanded_entities) > settings.max_entity_connections * 3:
                    break

            # Get chunks for expanded entity set (limit the entity set if too large)
            if len(expanded_entities) > settings.max_entity_connections * 2:
                # Keep only the highest scoring entities
                sorted_entities = sorted(
                    expanded_entities,
                    key=lambda eid: entity_scores.get(eid, 0.0),
                    reverse=True,
                )
                expanded_entities = set(
                    sorted_entities[: settings.max_entity_connections * 2]
                )

            expanded_chunks = graph_db.get_chunks_for_entities(list(expanded_entities))

            expanded_chunks = self._filter_chunks_by_documents(
                expanded_chunks, allowed_document_ids
            )

            if not expanded_chunks:
                return []

            # Add expansion metadata and similarity scores
            for chunk in expanded_chunks:
                chunk["retrieval_mode"] = "entity_expansion"
                chunk["expansion_depth"] = current_depth

                # Calculate similarity based on contained entities' scores
                contained_entities = chunk.get("contained_entities", [])
                if contained_entities:
                    # Get entity IDs for contained entities
                    contained_entity_ids = self._get_entity_ids_from_names(
                        contained_entities
                    )
                    if contained_entity_ids:
                        # Use the highest score among contained entities
                        max_entity_score = (
                            max(
                                entity_scores.get(entity_id, 0.0)
                                for entity_id in contained_entity_ids
                                if entity_id in entity_scores
                            )
                            if any(eid in entity_scores for eid in contained_entity_ids)
                            else 0.0
                        )
                        chunk["similarity"] = max_entity_score
                    else:
                        chunk["similarity"] = 0.0  # No entity match
                else:
                    chunk["similarity"] = 0.0  # No entities

            # Filter by enhanced similarity threshold
            filtered_chunks = [
                chunk
                for chunk in expanded_chunks
                if chunk.get("similarity", 0.0)
                >= settings.expansion_similarity_threshold
            ]

            # Sort by similarity score
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

            # Apply final chunk limit
            final_chunks = filtered_chunks[:max_chunks]
            logger.info(
                f"Entity expansion: {len(initial_entities)} entities → {len(expanded_entities)} expanded entities "
                f"({total_relationships} relationships, depth {current_depth}) → {len(expanded_chunks)} chunks → {len(filtered_chunks)} filtered → {len(final_chunks)} returned"
            )

            return final_chunks

        except Exception as e:
            logger.error(f"Entity expansion retrieval failed: {e}")
            return []

    async def multi_hop_reasoning_retrieval(
        self,
        query: str,
        seed_top_k: int = 5,
        max_hops: int = 2,
        beam_size: int = 8,
        use_hybrid_seeding: bool = True,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Multi-hop reasoning retrieval using path traversal.

        Args:
            query: User query
            seed_top_k: Number of seed entities to start from
            max_hops: Maximum number of hops to traverse
            beam_size: Beam size for path search
            use_hybrid_seeding: Whether to use hybrid seeding (chunks + entities)

        Returns:
            List of chunks with path-based scoring and provenance
        """
        try:
            if allowed_document_ids:
                logger.info(
                    "Skipping multi-hop reasoning because retrieval is restricted to specific documents"
                )
                return []
            # Step 1: Seed entity selection
            seed_entity_ids = []

            if use_hybrid_seeding:
                # Hybrid seeding: get chunks first, then extract entities
                query_embedding = embedding_manager.get_embedding(query)
                similar_chunks = graph_db.vector_similarity_search(
                    query_embedding, seed_top_k * 2
                )

                # Extract entities from these chunks
                chunk_ids = [chunk["chunk_id"] for chunk in similar_chunks]
                if chunk_ids:
                    entities_in_chunks = graph_db.get_entities_for_chunks(chunk_ids)
                    # Sort by importance and take top seed_top_k
                    entities_in_chunks.sort(
                        key=lambda e: e.get("importance_score", 0.0), reverse=True
                    )
                    seed_entity_ids = [
                        e["entity_id"] for e in entities_in_chunks[:seed_top_k]
                    ]
            else:
                # Entity-only seeding: find entities by similarity
                relevant_entities = graph_db.entity_similarity_search(query, seed_top_k)
                seed_entity_ids = [e["entity_id"] for e in relevant_entities]

            if not seed_entity_ids:
                logger.info("No seed entities found for multi-hop reasoning")
                return []

            logger.info(
                f"Multi-hop reasoning: starting with {len(seed_entity_ids)} seed entities"
            )

            # Step 2: Find scored paths using beam search
            paths = graph_db.find_scored_paths(
                seed_entity_ids=seed_entity_ids,
                max_hops=max_hops,
                beam_size=beam_size,
                min_edge_strength=settings.multi_hop_min_edge_strength,
            )

            if not paths:
                logger.info("No paths found in multi-hop reasoning")
                return []

            # Step 3: Compose path contexts and score
            query_embedding = embedding_manager.get_embedding(query)
            path_results = []

            for path in paths:
                # Gather supporting chunks for all hops
                all_chunk_ids = []
                for hop_chunks in path.supporting_chunk_ids:
                    all_chunk_ids.extend(hop_chunks)

                # Remove duplicates while preserving order
                unique_chunk_ids = list(dict.fromkeys(all_chunk_ids))

                if not unique_chunk_ids:
                    # Path has no supporting chunks, skip
                    continue

                # Get chunk data
                with graph_db.driver.session() as session:  # type: ignore
                    chunks_data = session.run(
                        """
                        MATCH (c:Chunk)
                        WHERE c.id IN $chunk_ids
                        OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
                        RETURN c.id as chunk_id, c.content as content,
                               c.embedding as embedding,
                               d.filename as document_name, d.id as document_id
                        """,
                        chunk_ids=unique_chunk_ids,
                    ).data()

                # Calculate path embedding (average of entity embeddings)
                entity_embeddings = [
                    e.embedding for e in path.entities if e.embedding is not None
                ]

                if entity_embeddings:
                    # Average entity embeddings for path representation
                    path_embedding = [
                        sum(emb[i] for emb in entity_embeddings)
                        / len(entity_embeddings)
                        for i in range(len(entity_embeddings[0]))
                    ]

                    # Calculate query similarity to path
                    path_query_similarity = graph_db._calculate_cosine_similarity(
                        query_embedding, path_embedding
                    )
                else:
                    path_query_similarity = 0.0

                # Calculate max chunk similarity
                max_chunk_similarity = 0.0
                for chunk_data in chunks_data:
                    if chunk_data.get("embedding"):
                        chunk_sim = graph_db._calculate_cosine_similarity(
                            query_embedding, chunk_data["embedding"]
                        )
                        max_chunk_similarity = max(max_chunk_similarity, chunk_sim)

                # Compute final path score with configurable weights
                # alpha * path_score_from_edges + beta * query_similarity + gamma * max_chunk_sim
                alpha, beta, gamma = 0.6, 0.3, 0.1
                final_score = (
                    alpha * path.score
                    + beta * path_query_similarity
                    + gamma * max_chunk_similarity
                )

                # Create result entries for each chunk in the path
                for chunk_data in chunks_data:
                    path_result = {
                        "chunk_id": chunk_data["chunk_id"],
                        "content": chunk_data["content"],
                        "document_name": chunk_data.get("document_name", "Unknown"),
                        "document_id": chunk_data.get("document_id", ""),
                        "similarity": final_score,
                        "retrieval_mode": "multi_hop_reasoning",
                        "path_score": path.score,
                        "path_query_similarity": path_query_similarity,
                        "max_chunk_similarity": max_chunk_similarity,
                        "path_entities": [e.name for e in path.entities],
                        "path_relationships": [
                            {
                                "type": r.type,
                                "description": r.description,
                                "strength": r.strength,
                            }
                            for r in path.relationships
                        ],
                        "path_length": len(path.entities),
                    }
                    path_results.append(path_result)

            # Sort by similarity and deduplicate by chunk_id
            seen_chunks = {}
            for result in path_results:
                chunk_id = result["chunk_id"]
                if (
                    chunk_id not in seen_chunks
                    or result["similarity"] > seen_chunks[chunk_id]["similarity"]
                ):
                    seen_chunks[chunk_id] = result

            final_results = list(seen_chunks.values())
            final_results.sort(key=lambda x: x["similarity"], reverse=True)

            logger.info(
                f"Multi-hop reasoning: found {len(paths)} paths, "
                f"generated {len(final_results)} unique chunks"
            )
            return final_results

        except Exception as e:
            logger.error(f"Multi-hop reasoning retrieval failed: {e}")
            return []

    async def hybrid_retrieval(
        self,
        query: str,
        top_k: int = 5,
        chunk_weight: float = 0.5,
        use_multi_hop: bool = False,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining chunk-based, entity-based, and optionally multi-hop approaches.

        Args:
            query: User query
            top_k: Total number of chunks to retrieve
            chunk_weight: Weight for chunk-based results (0.0-1.0)
            use_multi_hop: Whether to include multi-hop reasoning

        Returns:
            List of chunks from all approaches, de-duplicated and ranked
        """
        try:
            # Analyze query to determine if multi-hop would be beneficial
            query_analysis = analyze_query(query)
            multi_hop_recommended = query_analysis.get("multi_hop_recommended", True)
            query_type = query_analysis.get("query_type", "factual")

            allowed_set = set(allowed_document_ids) if allowed_document_ids else None

            # Override multi-hop decision based on query analysis
            effective_use_multi_hop = (
                use_multi_hop
                and multi_hop_recommended
                and not (allowed_document_ids and len(allowed_document_ids) > 0)
            )

            # Adjust path weight based on query type
            base_path_weight = settings.hybrid_path_weight
            if query_type == "comparative":
                # Comparative queries benefit more from multi-hop
                path_weight = min(0.8, base_path_weight * 1.3)
            elif query_type == "analytical":
                # Analytical queries benefit from multi-hop
                path_weight = min(0.7, base_path_weight * 1.1)
            else:
                # Factual queries - reduce multi-hop influence
                path_weight = max(0.2, base_path_weight * 0.7)

            logger.info(
                "Multi-hop decision: requested=%s, recommended=%s, effective=%s, query_type=%s, path_weight=%.2f, restricted=%s",
                use_multi_hop,
                multi_hop_recommended,
                effective_use_multi_hop,
                query_type,
                path_weight,
                bool(allowed_document_ids),
            )

            # Calculate split for each approach
            if effective_use_multi_hop:
                # With multi-hop: chunk, entity, and path results
                remaining_weight = 1.0 - path_weight
                chunk_count = max(1, int(top_k * chunk_weight * remaining_weight))
                entity_count = max(
                    1, int(top_k * (1 - chunk_weight) * remaining_weight)
                )
                # Allocate path slots based on adjusted weight and query type
                if query_type == "comparative":
                    path_count = max(
                        int(top_k * path_weight), top_k // 2
                    )  # More paths for comparisons
                elif query_type == "analytical":
                    path_count = max(
                        int(top_k * path_weight), top_k // 3
                    )  # Moderate paths for analysis
                else:
                    path_count = max(
                        1, int(top_k * path_weight)
                    )  # Fewer paths for factual queries
            else:
                # Without multi-hop: just chunk and entity
                chunk_count = max(1, int(top_k * chunk_weight))
                entity_count = max(1, top_k - chunk_count)
                path_count = 0

            # Get results from different approaches
            chunk_results = await self.chunk_based_retrieval(
                query, chunk_count, allowed_document_ids=allowed_document_ids
            )
            entity_results = await self.entity_based_retrieval(
                query, entity_count, allowed_document_ids=allowed_document_ids
            )

            path_results = []
            if effective_use_multi_hop:
                path_results = await self.multi_hop_reasoning_retrieval(
                    query,
                    seed_top_k=5,
                    max_hops=settings.multi_hop_max_hops,
                    beam_size=settings.multi_hop_beam_size,
                    use_hybrid_seeding=True,
                    allowed_document_ids=allowed_document_ids,
                )
                # Limit path results, but be more generous if we have high-quality results
                if len(path_results) > path_count:
                    # Sort by similarity to keep the best ones
                    path_results.sort(
                        key=lambda x: x.get("similarity", 0.0), reverse=True
                    )
                    # Keep top path_count, but allow up to 2x if quality is high
                    high_quality_count = sum(
                        1 for r in path_results if r.get("similarity", 0.0) > 0.5
                    )
                    keep_count = min(
                        len(path_results),
                        max(path_count, min(high_quality_count, path_count * 2)),
                    )
                    path_results = path_results[:keep_count]
                    logger.info(
                        f"Multi-hop: keeping {keep_count}/{len(path_results)} path results (path_count={path_count}, high_quality={high_quality_count})"
                    )

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
                        existing["relevant_entities"] = result.get(
                            "contained_entities", []
                        )
                        existing["contained_entities"] = result.get(
                            "contained_entities", []
                        )
                        # Boost score for chunks found by both methods
                        chunk_score = existing.get("chunk_score", 0.0)
                        entity_score = result.get("similarity", 0.3)
                        existing["hybrid_score"] = min(
                            1.0, (chunk_score + entity_score) * 0.8
                        )  # Combined score with cap
                    else:
                        result["retrieval_source"] = "entity_based"
                        # Use actual similarity score, with better fallback
                        if allowed_set and result.get("document_id") not in allowed_set:
                            continue
                        result["hybrid_score"] = result.get("similarity", 0.3)
                        combined_results[chunk_id] = result

            # Add path-based results (merge if duplicate)
            for result in path_results:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    if chunk_id in combined_results:
                        # Merge with existing
                        existing = combined_results[chunk_id]
                        existing["retrieval_source"] = "hybrid_with_paths"
                        existing["path_entities"] = result.get("path_entities", [])
                        existing["path_relationships"] = result.get(
                            "path_relationships", []
                        )
                        existing["path_length"] = result.get("path_length", 0)
                        # Boost score for chunks found by multiple methods
                        current_score = existing.get(
                            "hybrid_score", existing.get("chunk_score", 0.0)
                        )
                        path_score = result.get("similarity", 0.3)
                        existing["hybrid_score"] = min(
                            1.0, (current_score + path_score) * 0.7
                        )
                    else:
                        result["retrieval_source"] = "path_based"
                        result["hybrid_score"] = result.get("similarity", 0.3)
                        combined_results[chunk_id] = result

            # Sort by hybrid score and return top_k
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

            # Count overlaps for better reporting
            chunk_only_count = sum(
                1 for r in final_results if r.get("retrieval_source") == "chunk_based"
            )
            entity_only_count = sum(
                1 for r in final_results if r.get("retrieval_source") == "entity_based"
            )
            path_only_count = sum(
                1 for r in final_results if r.get("retrieval_source") == "path_based"
            )
            hybrid_count = sum(
                1
                for r in final_results
                if r.get("retrieval_source") in ["hybrid", "hybrid_with_paths"]
            )

            logger.info(
                f"Hybrid retrieval: {len(chunk_results)} chunk + {len(entity_results)} entity"
                + (f" + {len(path_results)} path" if effective_use_multi_hop else "")
                + f" → {len(final_results)} total "
                f"({chunk_only_count} chunk-only, {entity_only_count} entity-only"
                + (f", {path_only_count} path-only" if effective_use_multi_hop else "")
                + f", {hybrid_count} overlapping)"
            )

            return final_results[:top_k]

        except Exception as e:
            logger.error(f"Hybrid retrieval failed: {e}")
            return []

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
        use_multi_hop: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method that dispatches to the appropriate strategy.

        Args:
            query: User query
            mode: Retrieval mode to use
            top_k: Number of results to return
            use_multi_hop: Whether to use multi-hop reasoning (for hybrid mode)
            **kwargs: Additional parameters for specific retrieval modes

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(
            f"Starting retrieval with mode: {mode.value}, top_k: {top_k}, multi_hop: {use_multi_hop}"
        )

        allowed_document_ids = kwargs.pop("allowed_document_ids", None)

        if mode == RetrievalMode.CHUNK_ONLY:
            return await self.chunk_based_retrieval(
                query, top_k, allowed_document_ids=allowed_document_ids
            )
        elif mode == RetrievalMode.ENTITY_ONLY:
            return await self.entity_based_retrieval(
                query, top_k, allowed_document_ids=allowed_document_ids
            )
        elif mode == RetrievalMode.HYBRID:
            chunk_weight = kwargs.get("chunk_weight", 0.5)
            return await self.hybrid_retrieval(
                query,
                top_k,
                chunk_weight,
                use_multi_hop,
                allowed_document_ids=allowed_document_ids,
            )
        else:
            logger.error(f"Unknown retrieval mode: {mode}")
            return []

    async def retrieve_with_graph_expansion(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 3,
        expand_depth: int = 2,
        use_multi_hop: bool = False,
        allowed_document_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks and expand using graph relationships.

        Args:
            query: User query
            mode: Initial retrieval mode
            top_k: Number of initial chunks to retrieve
            expand_depth: Depth of graph expansion
            use_multi_hop: Whether to use multi-hop reasoning

        Returns:
            List of chunks including expanded context
        """
        try:
            # Get initial results
            initial_chunks = await self.retrieve(
                query,
                mode,
                top_k,
                use_multi_hop=use_multi_hop,
                allowed_document_ids=allowed_document_ids,
            )

            if not initial_chunks:
                return []

            allowed_set = set(allowed_document_ids) if allowed_document_ids else None

            if allowed_set:
                initial_chunks = [
                    chunk
                    for chunk in initial_chunks
                    if chunk.get("document_id") in allowed_set
                ]

                if not initial_chunks:
                    logger.info(
                        "Initial retrieval yielded no chunks after applying document restriction"
                    )
                    return []

            expanded_chunks = []
            seen_chunk_ids = set()

            # Add initial chunks
            for chunk in initial_chunks:
                chunk_id = chunk.get("chunk_id")
                if chunk_id and chunk_id not in seen_chunk_ids:
                    if allowed_set and chunk.get("document_id") not in allowed_set:
                        continue
                    expanded_chunks.append(chunk)
                    seen_chunk_ids.add(chunk_id)

            # Expand based on mode
            if mode in [RetrievalMode.ENTITY_ONLY, RetrievalMode.HYBRID]:
                # Entity-based expansion (with limits)
                entity_chunks_added = 0
                for chunk in initial_chunks:
                    if entity_chunks_added >= settings.max_expanded_chunks // 2:
                        break  # Reserve half the limit for entity expansion

                    entities = chunk.get("contained_entities", [])
                    if entities:
                        entity_ids = self._get_entity_ids_from_names(entities)
                        if entity_ids:  # Only expand if we found valid entity IDs
                            # Limit entities to process
                            entity_ids = entity_ids[: settings.max_entity_connections]

                            expanded = await self.entity_expansion_retrieval(
                                entity_ids,
                                expansion_depth=expand_depth,
                                max_chunks=settings.max_expanded_chunks
                                // len(initial_chunks)
                                + 1,
                                allowed_document_ids=allowed_document_ids,
                            )

                            source_similarity = chunk.get(
                                "similarity", chunk.get("hybrid_score", 0.3)
                            )

                            for exp_chunk in expanded:
                                exp_chunk_id = exp_chunk.get("chunk_id")
                                if exp_chunk_id and exp_chunk_id not in seen_chunk_ids:
                                    if allowed_set and exp_chunk.get("document_id") not in allowed_set:
                                        continue
                                    chunk_similarity = exp_chunk.get("similarity", 0.0)

                                    # Only add high-quality expansions
                                    if (
                                        chunk_similarity
                                        >= settings.expansion_similarity_threshold
                                    ):
                                        exp_chunk["expansion_context"] = {
                                            "source_chunk": chunk.get("chunk_id"),
                                            "expansion_type": "entity_relationship",
                                        }
                                        # Use the calculated similarity from entity expansion
                                        expanded_chunks.append(exp_chunk)
                                        seen_chunk_ids.add(exp_chunk_id)
                                        entity_chunks_added += 1

                                        if (
                                            entity_chunks_added
                                            >= settings.max_expanded_chunks // 2
                                        ):
                                            break

                        if entity_chunks_added >= settings.max_expanded_chunks // 2:
                            break

            if mode in [RetrievalMode.CHUNK_ONLY, RetrievalMode.HYBRID]:
                # Chunk similarity expansion (with limits)
                chunks_added = 0
                for chunk in initial_chunks:
                    if chunks_added >= settings.max_chunk_connections * len(
                        initial_chunks
                    ):
                        break  # Limit total chunk expansion

                    chunk_id = chunk.get("chunk_id")
                    if chunk_id:
                        # Limit depth and get related chunks
                        effective_depth = min(
                            expand_depth, settings.max_expansion_depth
                        )
                        related_chunks = graph_db.get_related_chunks(
                            chunk_id, max_depth=effective_depth
                        )

                        # Limit and prioritize by similarity
                        related_chunks = related_chunks[
                            : settings.max_chunk_connections
                        ]

                        source_similarity = chunk.get(
                            "similarity", chunk.get("hybrid_score", 0.3)
                        )

                        for rel_chunk in related_chunks:
                            rel_chunk_id = rel_chunk.get("chunk_id")
                            if rel_chunk_id and rel_chunk_id not in seen_chunk_ids:
                                if allowed_set and rel_chunk.get("document_id") not in allowed_set:
                                    continue
                                # Calculate similarity based on distance and source similarity
                                distance = rel_chunk.get("distance", 1)
                                # Decay factor based on graph distance
                                decay_factor = 1.0 / (distance + 1)
                                calculated_similarity = source_similarity * decay_factor

                                # Only add if similarity meets threshold
                                if (
                                    calculated_similarity
                                    >= settings.expansion_similarity_threshold
                                ):
                                    rel_chunk["expansion_context"] = {
                                        "source_chunk": chunk_id,
                                        "expansion_type": "chunk_similarity",
                                    }
                                    rel_chunk["similarity"] = calculated_similarity
                                    expanded_chunks.append(rel_chunk)
                                    seen_chunk_ids.add(rel_chunk_id)
                                    chunks_added += 1

                                    # Stop if we've added too many
                                    if chunks_added >= settings.max_expanded_chunks:
                                        break

                        if chunks_added >= settings.max_expanded_chunks:
                            break

            # Filter out chunks with similarity below threshold
            filtered_chunks = [
                chunk
                for chunk in expanded_chunks
                if chunk.get("similarity", 0.0)
                >= settings.expansion_similarity_threshold
            ]

            # Sort by similarity and apply final limit
            filtered_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)

            # Apply absolute limit to prevent resource overload
            if len(filtered_chunks) > settings.max_expanded_chunks:
                filtered_chunks = filtered_chunks[: settings.max_expanded_chunks]
                logger.warning(
                    f"Truncated expansion results to {settings.max_expanded_chunks} chunks"
                )

            # Count expansion types
            original_count = sum(
                1 for chunk in filtered_chunks if not chunk.get("expansion_context")
            )
            entity_expansion_count = sum(
                1
                for chunk in filtered_chunks
                if chunk.get("expansion_context", {}).get("expansion_type")
                == "entity_relationship"
            )
            chunk_expansion_count = sum(
                1
                for chunk in filtered_chunks
                if chunk.get("expansion_context", {}).get("expansion_type")
                == "chunk_similarity"
            )

            logger.info(
                f"Graph expansion ({mode.value}): {len(initial_chunks)} initial → {len(expanded_chunks)} total → {len(filtered_chunks)} final "
                f"({original_count} original + {entity_expansion_count} entity-expanded + {chunk_expansion_count} similarity-expanded)"
            )
            return filtered_chunks

        except Exception as e:
            logger.error(f"Graph expansion retrieval failed: {e}")
            # Return empty list if we don't have initial_chunks
            return []


# Global instance of the document retriever
document_retriever = DocumentRetriever()
