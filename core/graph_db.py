"""
Neo4j graph database operations for the RAG pipeline.
"""

import asyncio
import logging
import math
import mimetypes
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from config.settings import settings
from core.embeddings import embedding_manager

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Entity node in the graph."""

    id: str
    name: str
    type: str
    description: str = ""
    importance_score: float = 0.5
    embedding: Optional[List[float]] = None


@dataclass
class Relationship:
    """Relationship between entities."""

    source_entity_id: str
    target_entity_id: str
    type: str
    description: str = ""
    strength: float = 0.5
    source_chunks: List[str] = field(default_factory=list)


@dataclass
class PathResult:
    """Result of a multi-hop path traversal."""

    entities: List[Entity]
    relationships: List[Relationship]
    score: float
    supporting_chunk_ids: List[List[str]]  # List of chunk ids per hop


class GraphDB:
    """Neo4j database manager for document storage and retrieval."""

    def __init__(self):
        """Initialize Neo4j connection."""
        self.driver: Optional[Driver] = None
        self.connect()

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
            # Test the connection
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def create_document_node(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Create a document node in the graph."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d += $metadata
                """,
                doc_id=doc_id,
                metadata=metadata,
            )

    def update_document_summary(
        self,
        doc_id: str,
        summary: str,
        document_type: str,
        hashtags: List[str]
    ) -> None:
        """Update a document node with summary, document type, and hashtags."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                SET d.summary = $summary,
                    d.document_type = $document_type,
                    d.hashtags = $hashtags
                """,
                doc_id=doc_id,
                summary=summary,
                document_type=document_type,
                hashtags=hashtags,
            )

    def get_documents_with_summaries(self) -> List[Dict[str, Any]]:
        """Get all documents that have summaries."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.summary IS NOT NULL AND d.summary <> ''
                RETURN d.id as document_id,
                       d.summary as summary,
                       d.document_type as document_type,
                       d.hashtags as hashtags,
                       d.filename as filename
                """
            )
            return [record.data() for record in result]

    def create_chunk_node(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Create a chunk node and link it to its document."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.content = $content,
                    c.embedding = $embedding,
                    c.chunk_index = $chunk_index,
                    c.offset = $offset,
                    c += $metadata
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                chunk_index=metadata.get("chunk_index", 0),
                offset=metadata.get("offset", 0),
                metadata=metadata,
            )

    def create_similarity_relationship(
        self, chunk_id1: str, chunk_id2: str, similarity_score: float
    ) -> None:
        """Create similarity relationship between chunks."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MATCH (c1:Chunk {id: $chunk_id1})
                MATCH (c2:Chunk {id: $chunk_id2})
                MERGE (c1)-[r:SIMILAR_TO]-(c2)
                SET r.score = $similarity_score
                """,
                chunk_id1=chunk_id1,
                chunk_id2=chunk_id2,
                similarity_score=similarity_score,
            )

    @staticmethod
    def _calculate_cosine_similarity(
        embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same length")

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = math.sqrt(sum(a * a for a in embedding1))
        magnitude2 = math.sqrt(sum(a * a for a in embedding2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def create_chunk_similarities(self, doc_id: str, threshold: float = None) -> int:  # type: ignore
        """Create similarity relationships between chunks of a document."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.driver.session() as session:  # type: ignore
            # Get all chunks for the document with their embeddings
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id as chunk_id, c.embedding as embedding
                ORDER BY c.chunk_index ASC
                """,
                doc_id=doc_id,
            )

            chunks_data = [
                (record["chunk_id"], record["embedding"]) for record in result
            ]

            if len(chunks_data) < 2:
                logger.info(
                    f"Skipping similarity creation for document {doc_id}: less than 2 chunks"
                )
                return 0

            relationships_created = 0
            max_connections = settings.max_similarity_connections

            # Calculate similarities between all pairs of chunks
            for i in range(len(chunks_data)):
                chunk_id1, embedding1 = chunks_data[i]
                similarities = []

                for j in range(len(chunks_data)):
                    if i != j:
                        chunk_id2, embedding2 = chunks_data[j]
                        similarity = self._calculate_cosine_similarity(
                            embedding1, embedding2
                        )

                        if similarity >= threshold:
                            similarities.append((chunk_id2, similarity))

                # Sort by similarity and take top connections
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:max_connections]

                # Create relationships
                for chunk_id2, similarity in top_similarities:
                    self.create_similarity_relationship(
                        chunk_id1, chunk_id2, similarity
                    )
                    relationships_created += 1

            logger.info(
                f"Created {relationships_created} similarity relationships for document {doc_id}"
            )
            return relationships_created

    def create_all_chunk_similarities(self, threshold: float = None, batch_size: int = 10) -> Dict[str, int]:  # type: ignore
        """Create similarity relationships for all documents in the database."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.driver.session() as session:  # type: ignore
            # Get all document IDs
            result = session.run("MATCH (d:Document) RETURN d.id as doc_id")
            doc_ids = [record["doc_id"] for record in result]

        total_relationships = 0
        processed_docs = 0
        results = {}

        for doc_id in doc_ids:
            try:
                relationships_created = self.create_chunk_similarities(
                    doc_id, threshold
                )
                results[doc_id] = relationships_created
                total_relationships += relationships_created
                processed_docs += 1

                logger.info(
                    f"Processed document {doc_id}: {relationships_created} relationships"
                )

                # Process in batches to avoid memory issues
                if processed_docs % batch_size == 0:
                    logger.info(
                        f"Processed {processed_docs}/{len(doc_ids)} documents so far"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to create similarities for document {doc_id}: {e}"
                )
                results[doc_id] = 0

        logger.info(
            f"Batch processing complete: {total_relationships} total relationships created for {processed_docs} documents"
        )
        return results

    def create_entity_similarities(self, doc_id: str = None, threshold: float = None) -> int:  # type: ignore
        """Create similarity relationships between entities based on their embeddings."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.driver.session() as session:  # type: ignore
            # Build query based on whether we're processing specific doc or all entities
            if doc_id:
                # Get entities for specific document
                result = session.run(
                    """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                    WHERE e.embedding IS NOT NULL
                    RETURN DISTINCT e.id as entity_id, e.embedding as embedding, e.name as name, e.type as type
                    """,
                    doc_id=doc_id,
                )
            else:
                # Get all entities with embeddings
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.embedding IS NOT NULL
                    RETURN e.id as entity_id, e.embedding as embedding, e.name as name, e.type as type
                    """
                )

            entities_data = [
                (
                    record["entity_id"],
                    record["embedding"],
                    record["name"],
                    record["type"],
                )
                for record in result
            ]

            if len(entities_data) < 2:
                scope = f"document {doc_id}" if doc_id else "database"
                logger.info(
                    f"Skipping entity similarity creation for {scope}: less than 2 entities with embeddings"
                )
                return 0

            relationships_created = 0
            max_connections = settings.max_similarity_connections

            # Calculate similarities between all pairs of entities
            for i in range(len(entities_data)):
                entity_id1, embedding1, name1, type1 = entities_data[i]
                similarities = []

                for j in range(len(entities_data)):
                    if i != j:
                        entity_id2, embedding2, name2, type2 = entities_data[j]

                        # Skip if same entity type and name (likely duplicate)
                        if type1 == type2 and name1 == name2:
                            continue

                        similarity = self._calculate_cosine_similarity(
                            embedding1, embedding2
                        )

                        if similarity >= threshold:
                            similarities.append((entity_id2, similarity))

                # Sort by similarity and take top connections
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:max_connections]

                # Create relationships
                for entity_id2, similarity in top_similarities:
                    self._create_entity_similarity_relationship(
                        entity_id1, entity_id2, similarity
                    )
                    relationships_created += 1

            scope = f"document {doc_id}" if doc_id else "all entities"
            logger.info(
                f"Created {relationships_created} entity similarity relationships for {scope}"
            )
            return relationships_created

    def create_all_entity_similarities(self, threshold: float = None, batch_size: int = 10) -> Dict[str, int]:  # type: ignore
        """Create entity similarity relationships for all documents in the database."""
        if threshold is None:
            threshold = settings.similarity_threshold

        with self.driver.session() as session:  # type: ignore
            # Get all document IDs that have entities
            result = session.run(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN DISTINCT d.id as doc_id
                """
            )
            doc_ids = [record["doc_id"] for record in result]

        if not doc_ids:
            logger.info("No documents with entities found")
            return {}

        total_relationships = 0
        processed_docs = 0
        results = {}

        for doc_id in doc_ids:
            try:
                relationships_created = self.create_entity_similarities(
                    doc_id, threshold
                )
                results[doc_id] = relationships_created
                total_relationships += relationships_created
                processed_docs += 1

                logger.info(
                    f"Processed document {doc_id}: {relationships_created} entity relationships"
                )

                # Process in batches to avoid memory issues
                if processed_docs % batch_size == 0:
                    logger.info(
                        f"Processed {processed_docs}/{len(doc_ids)} documents so far"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to create entity similarities for document {doc_id}: {e}"
                )
                results[doc_id] = 0

        logger.info(
            f"Entity similarity batch processing complete: {total_relationships} total relationships created for {processed_docs} documents"
        )
        return results

    def _create_entity_similarity_relationship(
        self, entity_id1: str, entity_id2: str, similarity: float
    ) -> None:
        """Create a similarity relationship between two entities."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MATCH (e1:Entity {id: $entity_id1})
                MATCH (e2:Entity {id: $entity_id2})
                MERGE (e1)-[r:SIMILAR_TO]-(e2)
                SET r.similarity = $similarity, r.created_at = datetime()
                """,
                entity_id1=entity_id1,
                entity_id2=entity_id2,
                similarity=similarity,
            )

    def vector_similarity_search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using cosine similarity."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                WITH c, d, gds.similarity.cosine(c.embedding, $query_embedding) AS similarity
                RETURN c.id as chunk_id, c.content as content, similarity,
                       d.filename as document_name, d.id as document_id
                ORDER BY similarity DESC
                LIMIT $top_k
                """,
                query_embedding=query_embedding,
                top_k=top_k,
            )
            return [record.data() for record in result]

    def get_related_chunks(
        self,
        chunk_id: str,
        relationship_types: List[str] = None,  # type: ignore
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get chunks related to a given chunk through various relationships."""
        if relationship_types is None:
            relationship_types = ["SIMILAR_TO", "HAS_CHUNK"]

        with self.driver.session() as session:  # type: ignore
            # Build the query dynamically since Neo4j doesn't allow parameters in pattern ranges
            query = f"""
                MATCH (start:Chunk {{id: $chunk_id}})
                MATCH path = (start)-[*1..{max_depth}]-(related:Chunk)
                WHERE ALL(r in relationships(path) WHERE type(r) IN $relationship_types)
                OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(related)
                WITH related, d, length(path) as distance,
                     [r in relationships(path) WHERE type(r) = 'SIMILAR_TO' | r.score] as similarity_scores
                WITH related, d, distance,
                     CASE
                         WHEN size(similarity_scores) > 0 THEN
                             reduce(avg = 0.0, s in similarity_scores | avg + s) / size(similarity_scores)
                         ELSE
                             CASE distance
                                 WHEN 1 THEN 0.3
                                 WHEN 2 THEN 0.2
                                 ELSE 0.15
                             END
                     END as calculated_similarity
                RETURN DISTINCT related.id as chunk_id, related.content as content,
                       distance, d.filename as document_name, d.id as document_id,
                       calculated_similarity as similarity
                ORDER BY distance ASC, calculated_similarity DESC
                """

            result = session.run(
                query,  # type: ignore
                chunk_id=chunk_id,
                relationship_types=relationship_types,
            )
            return [record.data() for record in result]

    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id as chunk_id, c.content as content
                ORDER BY c.chunk_index ASC
                """,
                doc_id=doc_id,
            )
            return [record.data() for record in result]

    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks."""
        with self.driver.session() as session:  # type: ignore
            # 1. Collect chunk ids for the document
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN collect(c.id) as chunk_ids
                """,
                doc_id=doc_id,
            )

            record = result.single()
            chunk_ids = (
                record["chunk_ids"]
                if record and record["chunk_ids"] is not None
                else []
            )

            if chunk_ids:
                # 2. Remove references to these chunks from Entity.source_chunks lists
                session.run(
                    """
                    UNWIND $chunk_ids AS cid
                    MATCH (e:Entity)
                    WHERE cid IN coalesce(e.source_chunks, [])
                    SET e.source_chunks = [s IN coalesce(e.source_chunks, []) WHERE s <> cid]
                    """,
                    chunk_ids=chunk_ids,
                )

                # 3. Delete CONTAINS_ENTITY relationships from the chunks (so entities lose relationships to these chunks)
                session.run(
                    """
                    MATCH (c:Chunk)-[r:CONTAINS_ENTITY]->(e:Entity)
                    WHERE c.id IN $chunk_ids
                    DELETE r
                    """,
                    chunk_ids=chunk_ids,
                )

                # 4. Delete entities that are now orphaned: no source_chunks and no incoming CONTAINS_ENTITY relationships
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE (coalesce(e.source_chunks, []) = [] OR e.source_chunks IS NULL)
                    AND NOT ( ()-[:CONTAINS_ENTITY]->(e) )
                    DETACH DELETE e
                    """,
                )

            # 5. Finally delete chunks and the document
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c, d
                """,
                doc_id=doc_id,
            )

            logger.info(
                f"Deleted document {doc_id} and cleaned up {len(chunk_ids)} chunks and related entities"
            )

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their metadata, chunk counts, and OCR information."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                RETURN d.id as document_id,
                       d.filename as filename,
                       d.file_size as file_size,
                       d.file_extension as file_extension,
                       d.created_at as created_at,
                       d.modified_at as modified_at,
                       COALESCE(d.processing_method, '') as processing_method,
                       COALESCE(d.ocr_applied_pages, 0) as ocr_applied_pages,
                       COALESCE(d.readable_text_pages, 0) as readable_text_pages,
                       COALESCE(d.total_pages, 0) as total_pages,
                       COALESCE(d.ocr_items_count, 0) as ocr_items_count,
                       COALESCE(d.summary_total_pages, 0) as summary_total_pages,
                       COALESCE(d.summary_readable_pages, 0) as summary_readable_pages,
                       COALESCE(d.summary_ocr_pages, 0) as summary_ocr_pages,
                       COALESCE(d.summary_image_pages, 0) as summary_image_pages,
                       COALESCE(d.summary_mixed_pages, 0) as summary_mixed_pages,
                       COALESCE(d.content_primary_type, '') as content_primary_type,
                       chunk_count
                ORDER BY d.filename ASC
                """
            )
            return [record.data() for record in result]

    def get_graph_stats(self) -> Dict[str, int]:
        """Get basic statistics about the graph database."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                OPTIONAL MATCH (d:Document)
                WITH count(d) AS documents
                OPTIONAL MATCH (c:Chunk)
                WITH documents, count(c) AS chunks
                OPTIONAL MATCH (e:Entity)
                WITH documents, chunks, count(e) AS entities
                OPTIONAL MATCH ()-[r]-()
                WITH documents, chunks, entities,
                     sum(CASE WHEN type(r) = 'HAS_CHUNK' THEN 1 ELSE 0 END) AS has_chunk_relations,
                     sum(CASE WHEN type(r) = 'SIMILAR_TO' AND (startNode(r):Chunk OR endNode(r):Chunk) THEN 1 ELSE 0 END) AS similarity_relations,
                     sum(CASE WHEN type(r) = 'RELATED_TO' OR (type(r) = 'SIMILAR_TO' AND startNode(r):Entity AND endNode(r):Entity) THEN 1 ELSE 0 END) AS entity_relations,
                     sum(CASE WHEN type(r) = 'CONTAINS_ENTITY' THEN 1 ELSE 0 END) AS chunk_entity_relations
                RETURN documents, chunks, entities, has_chunk_relations,
                       similarity_relations, entity_relations, chunk_entity_relations
                """
            )
            record = result.single()
            if record is not None:
                return record.data()
            else:
                return {
                    "documents": 0,
                    "chunks": 0,
                    "entities": 0,
                    "has_chunk_relations": 0,
                    "similarity_relations": 0,
                    "entity_relations": 0,
                    "chunk_entity_relations": 0,
                }

    def get_entity_extraction_status(self) -> Dict[str, Any]:
        """Get entity extraction status for all documents."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:CONTAINS_ENTITY]->(e:Entity)
                WITH d, count(DISTINCT c) as total_chunks, count(DISTINCT e) as total_entities,
                     count(DISTINCT CASE WHEN e IS NOT NULL THEN c END) as chunks_with_entities
                RETURN d.id as document_id,
                       d.filename as filename,
                       total_chunks,
                       total_entities,
                       chunks_with_entities,
                       CASE
                           WHEN total_chunks = 0 THEN true
                           WHEN total_entities > 0 AND chunks_with_entities >= (total_chunks * 0.7) THEN true
                           ELSE false
                       END as entities_extracted
                ORDER BY d.filename ASC
                """
            )

            documents = [record.data() for record in result]

            # Calculate overall stats
            total_docs = len(documents)
            docs_with_entities = len([d for d in documents if d["entities_extracted"]])
            docs_without_entities = total_docs - docs_with_entities

            return {
                "documents": documents,
                "total_documents": total_docs,
                "documents_with_entities": docs_with_entities,
                "documents_without_entities": docs_without_entities,
                "all_extracted": docs_without_entities == 0,
            }

    def get_document_entities(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all entities extracted from a specific document."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN DISTINCT e.id as entity_id, e.name as name, e.type as type,
                       e.description as description, e.importance_score as importance_score,
                       count(DISTINCT c) as chunk_count
                ORDER BY e.importance_score DESC, e.name ASC
                """,
                doc_id=doc_id,
            )
            return [record.data() for record in result]

    def setup_indexes(self) -> None:
        """Create necessary indexes for performance."""
        with self.driver.session() as session:  # type: ignore
            # Create indexes for faster lookups
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            logger.info("Database indexes created successfully")

    # Entity-related methods

    async def acreate_entity_node(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        importance_score: float = 0.5,
        source_chunks: Optional[List[str]] = None,
    ) -> None:
        """Create an entity node in the graph with embedding (async version)."""
        if source_chunks is None:
            source_chunks = []

        # Generate embedding for the entity using name and description
        entity_text = f"{name}: {description}"
        embedding = await embedding_manager.aget_embedding(entity_text)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._create_entity_node_sync,
            entity_id,
            name,
            entity_type,
            description,
            importance_score,
            source_chunks,
            embedding,
        )

    def _create_entity_node_sync(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        importance_score: float,
        source_chunks: List[str],
        embedding: List[float],
    ) -> None:
        """Synchronous helper for creating entity node in database."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MERGE (e:Entity {id: $entity_id})
                SET e.name = $name,
                    e.type = $entity_type,
                    e.description = $description,
                    e.importance_score = $importance_score,
                    e.source_chunks = $source_chunks,
                    e.embedding = $embedding,
                    e.updated_at = timestamp()
                """,
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                description=description,
                importance_score=importance_score,
                source_chunks=source_chunks,
                embedding=embedding,
            )

    def create_entity_node(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        description: str,
        importance_score: float = 0.5,
        source_chunks: Optional[List[str]] = None,
    ) -> None:
        """Create an entity node in the graph with embedding (sync version kept for compatibility)."""
        if source_chunks is None:
            source_chunks = []

        # Generate embedding for the entity using name and description
        entity_text = f"{name}: {description}"
        embedding = embedding_manager.get_embedding(entity_text)

        self._create_entity_node_sync(
            entity_id,
            name,
            entity_type,
            description,
            importance_score,
            source_chunks,
            embedding,
        )

    async def aupdate_entities_with_embeddings(self) -> int:
        """Update existing entities that don't have embeddings (async version)."""
        with self.driver.session() as session:  # type: ignore
            # Get entities without embeddings
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                RETURN e.id as entity_id, e.name as name, e.description as description
                """
            )

            entities_to_update = [
                (record["entity_id"], record["name"], record["description"])
                for record in result
            ]

            logger.info(f"Found {len(entities_to_update)} entities without embeddings")

            if not entities_to_update:
                return 0

            # Process entities with parallel embedding generation
            updated_count = 0
            concurrency = getattr(settings, "embedding_concurrency")
            sem = asyncio.Semaphore(concurrency)

            async def _embed_and_update_entity(entity_data):
                nonlocal updated_count
                entity_id, name, description = entity_data

                async with sem:
                    try:
                        # Add small delay to prevent API flooding
                        await asyncio.sleep(0.3)
                        entity_text = f"{name}: {description}" if description else name
                        embedding = await embedding_manager.aget_embedding(entity_text)
                    except Exception as e:
                        logger.error(
                            f"Async embedding failed for entity {entity_id}: {e}"
                        )
                        return None

                # Persist to DB in a thread to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                try:
                    await loop.run_in_executor(
                        None,
                        self._update_entity_embedding_sync,
                        entity_id,
                        embedding,
                    )
                    updated_count += 1

                    if updated_count % 100 == 0:
                        logger.info(
                            f"Updated {updated_count}/{len(entities_to_update)} entities with embeddings"
                        )
                    return entity_id
                except Exception as e:
                    logger.error(
                        f"Failed to update entity {entity_id} with embedding: {e}"
                    )
                    return None

            tasks = [
                asyncio.create_task(_embed_and_update_entity(entity))
                for entity in entities_to_update
            ]

            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as e:
                    logger.error(f"Error in entity update task: {e}")

            logger.info(
                f"Successfully updated {updated_count} entities with embeddings"
            )
            return updated_count

    def _update_entity_embedding_sync(
        self, entity_id: str, embedding: List[float]
    ) -> None:
        """Synchronous helper for updating entity embedding in database."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MATCH (e:Entity {id: $entity_id})
                SET e.embedding = $embedding
                """,
                entity_id=entity_id,
                embedding=embedding,
            )

    def update_entities_with_embeddings(self) -> int:
        """Update existing entities that don't have embeddings (sync version kept for compatibility)."""
        updated_count = 0

        with self.driver.session() as session:  # type: ignore
            # Get entities without embeddings
            result = session.run(
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                RETURN e.id as entity_id, e.name as name, e.description as description
                """
            )

            entities_to_update = [
                (record["entity_id"], record["name"], record["description"])
                for record in result
            ]

            logger.info(f"Found {len(entities_to_update)} entities without embeddings")

            # Update entities with embeddings
            for entity_id, name, description in entities_to_update:
                try:
                    entity_text = f"{name}: {description}" if description else name
                    embedding = embedding_manager.get_embedding(entity_text)

                    session.run(
                        """
                        MATCH (e:Entity {id: $entity_id})
                        SET e.embedding = $embedding
                        """,
                        entity_id=entity_id,
                        embedding=embedding,
                    )
                    updated_count += 1

                    if updated_count % 100 == 0:
                        logger.info(
                            f"Updated {updated_count}/{len(entities_to_update)} entities with embeddings"
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to update entity {entity_id} with embedding: {e}"
                    )

            logger.info(
                f"Successfully updated {updated_count} entities with embeddings"
            )
            return updated_count

    def create_entity_relationship(
        self,
        entity_id1: str,
        entity_id2: str,
        relationship_type: str,
        description: str,
        strength: float = 0.5,
        source_chunks: Optional[List[str]] = None,
    ) -> None:
        """Create a relationship between two entities."""
        if source_chunks is None:
            source_chunks = []

        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MATCH (e1:Entity {id: $entity_id1})
                MATCH (e2:Entity {id: $entity_id2})
                MERGE (e1)-[r:RELATED_TO]-(e2)
                SET r.type = $relationship_type,
                    r.description = $description,
                    r.strength = $strength,
                    r.source_chunks = $source_chunks,
                    r.updated_at = timestamp()
                """,
                entity_id1=entity_id1,
                entity_id2=entity_id2,
                relationship_type=relationship_type,
                description=description,
                strength=strength,
                source_chunks=source_chunks,
            )

    def create_chunk_entity_relationship(self, chunk_id: str, entity_id: str) -> None:
        """Create a relationship between a chunk and an entity it contains."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MATCH (c:Chunk {id: $chunk_id})
                MATCH (e:Entity {id: $entity_id})
                MERGE (c)-[:CONTAINS_ENTITY]->(e)
                """,
                chunk_id=chunk_id,
                entity_id=entity_id,
            )

    def get_entities_by_type(
        self, entity_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get entities of a specific type."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (e:Entity {type: $entity_type})
                RETURN e.id as entity_id, e.name as name, e.description as description,
                       e.importance_score as importance_score, e.source_chunks as source_chunks
                ORDER BY e.importance_score DESC
                LIMIT $limit
                """,
                entity_type=entity_type,
                limit=limit,
            )
            return [record.data() for record in result]

    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a specific entity."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (e1:Entity {id: $entity_id})-[r:RELATED_TO]-(e2:Entity)
                RETURN e2.id as related_entity_id, e2.name as related_entity_name,
                       e2.type as related_entity_type, r.type as relationship_type,
                       r.description as relationship_description, r.strength as strength
                ORDER BY r.strength DESC
                """,
                entity_id=entity_id,
            )
            return [record.data() for record in result]

    def entity_similarity_search(
        self, query_text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search entities by text similarity using full-text search."""
        with self.driver.session() as session:  # type: ignore
            # Create full-text index if it doesn't exist
            try:
                session.run(
                    "CREATE FULLTEXT INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.description]"
                )
            except Exception:
                pass  # Index might already exist

            result = session.run(
                """
                CALL db.index.fulltext.queryNodes('entity_text', $query_text)
                YIELD node, score
                RETURN node.id as entity_id, node.name as name, node.type as type,
                       node.description as description, node.importance_score as importance_score,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                query_text=query_text,
                top_k=top_k,
            )
            return [record.data() for record in result]

    def get_entities_for_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get all entities contained in the specified chunks."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE c.id IN $chunk_ids
                RETURN DISTINCT e.id as entity_id, e.name as name, e.type as type,
                       e.description as description, e.importance_score as importance_score,
                       collect(c.id) as source_chunks
                """,
                chunk_ids=chunk_ids,
            )
            return [record.data() for record in result]

    def get_chunks_for_entities(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Get all chunks that contain the specified entities."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                WHERE e.id IN $entity_ids
                OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c)
                RETURN DISTINCT c.id as chunk_id, c.content as content,
                       d.filename as document_name, d.id as document_id,
                       collect(e.name) as contained_entities
                """,
                entity_ids=entity_ids,
            )
            return [record.data() for record in result]

    def get_entity_graph_neighborhood(
        self, entity_id: str, max_depth: int = 2, max_entities: int = 50
    ) -> Dict[str, Any]:
        """Get a subgraph around a specific entity."""
        with self.driver.session() as session:  # type: ignore
            if max_depth == 1:
                query = """
                    MATCH (start:Entity {id: $entity_id})-[r:RELATED_TO]-(related:Entity)
                    RETURN collect(DISTINCT start) + collect(DISTINCT related) as entities,
                           collect({
                               start: startNode(r).id,
                               end: endNode(r).id,
                               type: r.type,
                               description: r.description,
                               strength: r.strength
                           }) as relationships
                    """
            else:
                query = """
                    MATCH (start:Entity {id: $entity_id})-[*1..2]-(related:Entity)
                    WITH start, related
                    MATCH (e1:Entity)-[r:RELATED_TO]-(e2:Entity)
                    WHERE (e1.id = start.id OR e1.id = related.id)
                      AND (e2.id = start.id OR e2.id = related.id)
                    RETURN collect(DISTINCT start) + collect(DISTINCT related) as entities,
                           collect(DISTINCT {
                               start: startNode(r).id,
                               end: endNode(r).id,
                               type: r.type,
                               description: r.description,
                               strength: r.strength
                           }) as relationships
                    """

            result = session.run(query, entity_id=entity_id)
            record = result.single()
            if record:
                entities = [
                    {
                        "id": node["id"],
                        "name": node["name"],
                        "type": node["type"],
                        "description": node["description"],
                        "importance_score": node["importance_score"],
                    }
                    for node in record["entities"]
                ]
                return {"entities": entities, "relationships": record["relationships"]}
            return {"entities": [], "relationships": []}

    def validate_chunk_embeddings(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate embeddings for chunks, checking for empty/invalid embeddings.

        Args:
            doc_id: Optional document ID to validate only specific document chunks

        Returns:
            Dictionary with validation results
        """
        with self.driver.session() as session:  # type: ignore
            if doc_id:
                query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                    RETURN c.id as chunk_id, c.embedding as embedding, c.content as content
                """
                result = session.run(query, doc_id=doc_id)
            else:
                query = """
                    MATCH (c:Chunk)
                    RETURN c.id as chunk_id, c.embedding as embedding, c.content as content
                """
                result = session.run(query)

            total_chunks = 0
            invalid_chunks = []
            empty_embeddings = 0
            wrong_size_embeddings = 0

            # Detect embedding size from existing embeddings instead of hardcoding
            expected_embedding_size = None

            for record in result:
                total_chunks += 1
                chunk_id = record["chunk_id"]
                embedding = record["embedding"]
                content = record["content"]

                # Check for empty or None embeddings
                if not embedding:
                    empty_embeddings += 1
                    invalid_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "issue": "empty_embedding",
                            "content_preview": (
                                content[:100] + "..." if len(content) > 100 else content
                            ),
                        }
                    )
                    continue

                # Detect expected embedding size from first valid embedding
                if expected_embedding_size is None and embedding:
                    expected_embedding_size = len(embedding)
                    logger.info(f"Detected embedding size: {expected_embedding_size}")

                # Check embedding size consistency (only flag if significantly different)
                if (
                    expected_embedding_size
                    and embedding
                    and len(embedding) != expected_embedding_size
                ):
                    wrong_size_embeddings += 1
                    invalid_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "issue": f"wrong_size_{len(embedding)}_expected_{expected_embedding_size}",
                            "content_preview": (
                                content[:100] + "..." if len(content) > 100 else content
                            ),
                        }
                    )

            validation_results = {
                "total_chunks": total_chunks,
                "valid_chunks": total_chunks - len(invalid_chunks),
                "invalid_chunks": len(invalid_chunks),
                "empty_embeddings": empty_embeddings,
                "wrong_size_embeddings": wrong_size_embeddings,
                "invalid_chunk_details": invalid_chunks,
                "validation_passed": len(invalid_chunks) == 0,
            }

            logger.info(
                f"Chunk embedding validation: {validation_results['valid_chunks']}/{total_chunks} valid"
            )
            if invalid_chunks:
                logger.warning(f"Found {len(invalid_chunks)} invalid chunk embeddings")

            return validation_results

    def validate_entity_embeddings(
        self, doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate embeddings for entities, checking for empty/invalid embeddings.

        Args:
            doc_id: Optional document ID to validate only entities from specific document

        Returns:
            Dictionary with validation results
        """
        with self.driver.session() as session:  # type: ignore
            if doc_id:
                query = """
                    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                    RETURN DISTINCT e.id as entity_id, e.embedding as embedding, e.name as entity_name
                """
                result = session.run(query, doc_id=doc_id)
            else:
                query = """
                    MATCH (e:Entity)
                    RETURN e.id as entity_id, e.embedding as embedding, e.name as entity_name
                """
                result = session.run(query)

            total_entities = 0
            invalid_entities = []
            empty_embeddings = 0
            wrong_size_embeddings = 0
            no_embeddings = 0  # Entities may not have embeddings by design

            # Detect embedding size from existing embeddings instead of hardcoding
            expected_embedding_size = None

            for record in result:
                total_entities += 1
                entity_id = record["entity_id"]
                embedding = record["embedding"]
                entity_name = record["entity_name"]

                # Skip entities without embeddings (they may be designed this way)
                if embedding is None:
                    no_embeddings += 1
                    continue

                # Check for empty embeddings
                if not embedding:
                    empty_embeddings += 1
                    invalid_entities.append(
                        {
                            "entity_id": entity_id,
                            "issue": "empty_embedding",
                            "entity_name": entity_name,
                        }
                    )
                    continue

                # Detect expected embedding size from first valid embedding
                if expected_embedding_size is None and embedding:
                    expected_embedding_size = len(embedding)
                    logger.info(
                        f"Detected entity embedding size: {expected_embedding_size}"
                    )

                # Check embedding size consistency (only flag if significantly different)
                if (
                    expected_embedding_size
                    and embedding
                    and len(embedding) != expected_embedding_size
                ):
                    wrong_size_embeddings += 1
                    invalid_entities.append(
                        {
                            "entity_id": entity_id,
                            "issue": f"wrong_size_{len(embedding)}_expected_{expected_embedding_size}",
                            "entity_name": entity_name,
                        }
                    )

            validation_results = {
                "total_entities": total_entities,
                "entities_with_embeddings": total_entities - no_embeddings,
                "valid_embeddings": (total_entities - no_embeddings)
                - len(invalid_entities),
                "invalid_embeddings": len(invalid_entities),
                "empty_embeddings": empty_embeddings,
                "wrong_size_embeddings": wrong_size_embeddings,
                "no_embeddings": no_embeddings,
                "invalid_entity_details": invalid_entities,
                "validation_passed": len(invalid_entities) == 0,
            }

            logger.info(
                f"Entity embedding validation: {validation_results['valid_embeddings']}/{validation_results['entities_with_embeddings']} valid"
            )
            if invalid_entities:
                logger.warning(
                    f"Found {len(invalid_entities)} invalid entity embeddings"
                )

            return validation_results

    async def afix_invalid_embeddings(
        self,
        chunk_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fix invalid embeddings by regenerating them (async version).

        Args:
            chunk_ids: List of chunk IDs to fix (if None, fixes all invalid chunk embeddings)
            entity_ids: List of entity IDs to fix (if None, fixes all invalid entity embeddings)

        Returns:
            Dictionary with fix results
        """
        results = {"chunks_fixed": 0, "entities_fixed": 0, "errors": []}

        # Fix chunk embeddings in parallel
        if chunk_ids is not None:
            concurrency = getattr(settings, "embedding_concurrency")
            sem = asyncio.Semaphore(concurrency)

            async def _fix_chunk_embedding(chunk_id):
                async with sem:
                    try:
                        # Get chunk content in executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        content = await loop.run_in_executor(
                            None,
                            self._get_chunk_content_sync,
                            chunk_id,
                        )

                        if content:
                            # Add small delay to prevent API flooding
                            await asyncio.sleep(0.3)
                            # Generate new embedding
                            embedding = await embedding_manager.aget_embedding(content)

                            # Update chunk with new embedding in executor
                            await loop.run_in_executor(
                                None,
                                self._update_chunk_embedding_sync,
                                chunk_id,
                                embedding,
                            )
                            results["chunks_fixed"] += 1
                            logger.info(f"Fixed embedding for chunk {chunk_id}")
                            return True
                    except Exception as e:
                        error_msg = f"Failed to fix embedding for chunk {chunk_id}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
                        return False
                return False

            if chunk_ids:
                tasks = [
                    asyncio.create_task(_fix_chunk_embedding(chunk_id))
                    for chunk_id in chunk_ids
                ]

                for coro in asyncio.as_completed(tasks):
                    try:
                        await coro
                    except Exception as e:
                        logger.error(f"Error in chunk fix task: {e}")

        # Fix entity embeddings in parallel
        if entity_ids is not None:
            concurrency = getattr(settings, "embedding_concurrency")
            sem = asyncio.Semaphore(concurrency)

            async def _fix_entity_embedding(entity_id):
                async with sem:
                    try:
                        # Get entity data in executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        entity_data = await loop.run_in_executor(
                            None,
                            self._get_entity_data_sync,
                            entity_id,
                        )

                        if entity_data:
                            # Add small delay to prevent API flooding
                            await asyncio.sleep(0.3)
                            # Use entity name + description for embedding
                            text = (
                                f"{entity_data['name']}: {entity_data['description']}"
                            )
                            # Generate new embedding
                            embedding = await embedding_manager.aget_embedding(text)

                            # Update entity with new embedding in executor
                            await loop.run_in_executor(
                                None,
                                self._update_entity_embedding_sync,
                                entity_id,
                                embedding,
                            )
                            results["entities_fixed"] += 1
                            logger.info(f"Fixed embedding for entity {entity_id}")
                            return True
                    except Exception as e:
                        error_msg = (
                            f"Failed to fix embedding for entity {entity_id}: {e}"
                        )
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
                        return False
                return False

            if entity_ids:
                tasks = [
                    asyncio.create_task(_fix_entity_embedding(entity_id))
                    for entity_id in entity_ids
                ]

                for coro in asyncio.as_completed(tasks):
                    try:
                        await coro
                    except Exception as e:
                        logger.error(f"Error in entity fix task: {e}")

        return results

    def _get_chunk_content_sync(self, chunk_id: str) -> Optional[str]:
        """Synchronous helper for getting chunk content."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                "MATCH (c:Chunk {id: $chunk_id}) RETURN c.content as content",
                chunk_id=chunk_id,
            )
            record = result.single()
            return record["content"] if record else None

    def _update_chunk_embedding_sync(
        self, chunk_id: str, embedding: List[float]
    ) -> None:
        """Synchronous helper for updating chunk embedding."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                "MATCH (c:Chunk {id: $chunk_id}) SET c.embedding = $embedding",
                chunk_id=chunk_id,
                embedding=embedding,
            )

    def _get_entity_data_sync(self, entity_id: str) -> Optional[Dict[str, str]]:
        """Synchronous helper for getting entity data."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                "MATCH (e:Entity {id: $entity_id}) RETURN e.name as name, e.description as description",
                entity_id=entity_id,
            )
            record = result.single()
            return (
                {"name": record["name"], "description": record["description"]}
                if record
                else None
            )

    def fix_invalid_embeddings(
        self,
        chunk_ids: Optional[List[str]] = None,
        entity_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Fix invalid embeddings by regenerating them (sync version kept for compatibility).

        Args:
            chunk_ids: List of chunk IDs to fix (if None, fixes all invalid chunk embeddings)
            entity_ids: List of entity IDs to fix (if None, fixes all invalid entity embeddings)

        Returns:
            Dictionary with fix results
        """
        results = {"chunks_fixed": 0, "entities_fixed": 0, "errors": []}

        # Fix chunk embeddings
        if chunk_ids is not None:
            with self.driver.session() as session:  # type: ignore
                for chunk_id in chunk_ids:
                    try:
                        # Get chunk content
                        result = session.run(
                            "MATCH (c:Chunk {id: $chunk_id}) RETURN c.content as content",
                            chunk_id=chunk_id,
                        )
                        record = result.single()
                        if record:
                            content = record["content"]
                            # Generate new embedding
                            embedding = embedding_manager.get_embedding(content)
                            # Update chunk with new embedding
                            session.run(
                                "MATCH (c:Chunk {id: $chunk_id}) SET c.embedding = $embedding",
                                chunk_id=chunk_id,
                                embedding=embedding,
                            )
                            results["chunks_fixed"] += 1
                            logger.info(f"Fixed embedding for chunk {chunk_id}")
                    except Exception as e:
                        error_msg = f"Failed to fix embedding for chunk {chunk_id}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)

        # Fix entity embeddings (if they're supposed to have embeddings)
        if entity_ids is not None:
            with self.driver.session() as session:  # type: ignore
                for entity_id in entity_ids:
                    try:
                        # Get entity name/description for embedding
                        result = session.run(
                            "MATCH (e:Entity {id: $entity_id}) RETURN e.name as name, e.description as description",
                            entity_id=entity_id,
                        )
                        record = result.single()
                        if record:
                            # Use entity name + description for embedding
                            text = f"{record['name']}: {record['description']}"
                            # Generate new embedding
                            embedding = embedding_manager.get_embedding(text)
                            # Update entity with new embedding
                            session.run(
                                "MATCH (e:Entity {id: $entity_id}) SET e.embedding = $embedding",
                                entity_id=entity_id,
                                embedding=embedding,
                            )
                            results["entities_fixed"] += 1
                            logger.info(f"Fixed embedding for entity {entity_id}")
                    except Exception as e:
                        error_msg = (
                            f"Failed to fix embedding for entity {entity_id}: {e}"
                        )
                        results["errors"].append(error_msg)
                        logger.error(error_msg)

        return results

    def find_scored_paths(
        self,
        seed_entity_ids: List[str],
        max_hops: int = 2,
        beam_size: int = 8,
        min_edge_strength: float = 0.0,
        node_filter: Optional[Callable[[Entity], bool]] = None,
    ) -> List[PathResult]:
        """
        Find scored paths from seed entities using beam search.

        Args:
            seed_entity_ids: Starting entity IDs for path traversal
            max_hops: Maximum number of hops to traverse
            beam_size: Number of best paths to keep at each depth
            min_edge_strength: Minimum relationship strength to follow
            node_filter: Optional filter function for entities

        Returns:
            List of PathResult objects sorted by score
        """
        if not seed_entity_ids:
            logger.warning("No seed entities provided for path search")
            return []

        try:
            with self.driver.session() as session:  # type: ignore
                # Get seed entities with their data
                seed_entities_data = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.id IN $entity_ids
                    RETURN e.id as id, e.name as name, e.type as type,
                           e.description as description, e.importance_score as importance_score,
                           e.embedding as embedding
                    """,
                    entity_ids=seed_entity_ids,
                ).data()

                if not seed_entities_data:
                    logger.warning(f"No entities found for IDs: {seed_entity_ids}")
                    return []

                # Initialize paths with seed entities
                current_paths = []
                for entity_data in seed_entities_data:
                    entity = Entity(
                        id=entity_data["id"],
                        name=entity_data["name"],
                        type=entity_data["type"],
                        description=entity_data.get("description", ""),
                        importance_score=entity_data.get("importance_score", 0.5),
                        embedding=entity_data.get("embedding"),
                    )

                    # Apply node filter if provided
                    if node_filter and not node_filter(entity):
                        continue

                    # Start with single-entity paths
                    path = PathResult(
                        entities=[entity],
                        relationships=[],
                        score=entity.importance_score,
                        supporting_chunk_ids=[],
                    )
                    current_paths.append(path)

                # Perform beam search up to max_hops
                for hop in range(max_hops):
                    next_paths = []
                    visited_in_hop = set()  # Track what we've expanded this hop

                    for path in current_paths:
                        # Get the last entity in the path
                        last_entity = path.entities[-1]

                        # Skip if we've already expanded from this entity in this hop
                        path_key = (tuple(e.id for e in path.entities), hop)
                        if path_key in visited_in_hop:
                            continue
                        visited_in_hop.add(path_key)

                        # Get relationships from last entity
                        relationships_data = session.run(
                            """
                            MATCH (e1:Entity {id: $entity_id})-[r:RELATED_TO]-(e2:Entity)
                            WHERE r.strength >= $min_strength
                            AND NOT e2.id IN $visited_ids
                            RETURN e2.id as target_id, e2.name as target_name,
                                   e2.type as target_type, e2.description as target_description,
                                   e2.importance_score as target_importance,
                                   e2.embedding as target_embedding,
                                   r.type as rel_type, r.description as rel_description,
                                   r.strength as rel_strength,
                                   coalesce(r.source_chunks, []) as source_chunks,
                                   startNode(r).id as source_id
                            ORDER BY r.strength DESC
                            LIMIT $limit
                            """,
                            entity_id=last_entity.id,
                            min_strength=min_edge_strength,
                            visited_ids=[e.id for e in path.entities],
                            limit=beam_size * 2,  # Get more candidates than beam size
                        ).data()

                        # Expand path with each relationship
                        for rel_data in relationships_data:
                            target_entity = Entity(
                                id=rel_data["target_id"],
                                name=rel_data["target_name"],
                                type=rel_data["target_type"],
                                description=rel_data.get("target_description", ""),
                                importance_score=rel_data.get("target_importance", 0.5),
                                embedding=rel_data.get("target_embedding"),
                            )

                            # Apply node filter if provided
                            if node_filter and not node_filter(target_entity):
                                continue

                            # Determine direction of relationship
                            if rel_data["source_id"] == last_entity.id:
                                source_id = last_entity.id
                                target_id = target_entity.id
                            else:
                                source_id = target_entity.id
                                target_id = last_entity.id

                            relationship = Relationship(
                                source_entity_id=source_id,
                                target_entity_id=target_id,
                                type=rel_data["rel_type"],
                                description=rel_data.get("rel_description", ""),
                                strength=rel_data.get("rel_strength", 0.5),
                                source_chunks=rel_data.get("source_chunks", []),
                            )

                            # Calculate new path score
                            # Score = average of: path score, relationship strength, target importance
                            new_score = (
                                path.score * 0.5
                                + relationship.strength * 0.3
                                + target_entity.importance_score * 0.2
                            )

                            # Create new path
                            new_path = PathResult(
                                entities=path.entities + [target_entity],
                                relationships=path.relationships + [relationship],
                                score=new_score,
                                supporting_chunk_ids=path.supporting_chunk_ids
                                + [relationship.source_chunks],
                            )
                            next_paths.append(new_path)

                    # Apply beam search: keep only top beam_size paths
                    next_paths.sort(key=lambda p: p.score, reverse=True)
                    current_paths = next_paths[:beam_size]

                    # Stop if no more paths to expand
                    if not current_paths:
                        break

                # Return all paths sorted by score
                current_paths.sort(key=lambda p: p.score, reverse=True)
                logger.info(
                    f"Found {len(current_paths)} paths from {len(seed_entity_ids)} seed entities "
                    f"with max_hops={max_hops}, beam_size={beam_size}"
                )
                return current_paths

        except Exception as e:
            logger.error(f"Path finding failed: {e}")
            return []

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics for the API."""
        with self.driver.session() as session:  # type: ignore
            # Get basic stats
            result = session.run(
                """
                MATCH (d:Document)
                WITH count(d) AS total_documents
                OPTIONAL MATCH (c:Chunk)
                WITH total_documents, count(c) AS total_chunks
                OPTIONAL MATCH (e:Entity)
                WITH total_documents, total_chunks, count(e) AS total_entities
                OPTIONAL MATCH ()-[r]->()
                RETURN total_documents, total_chunks, total_entities, count(r) AS total_relationships
                """
            )
            record = result.single()
            stats = record.data() if record else {}

            # Get document list
            doc_result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                RETURN d.id as document_id,
                       d.filename as filename,
                       d.created_at as created_at,
              coalesce(d.processing_status, 'idle') as processing_status,
              coalesce(d.processing_stage, 'idle') as processing_stage,
              coalesce(d.processing_progress, 0.0) as processing_progress,
              chunk_count,
              d.document_type as document_type
                ORDER BY d.created_at DESC
                """
            )
            documents = [record.data() for record in doc_result]

            return {
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "total_entities": stats.get("total_entities", 0),
                "total_relationships": stats.get("total_relationships", 0),
                "documents": documents,
            }

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the database."""
        with self.driver.session() as session:  # type: ignore
            result = session.run(
                """
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                WITH d, count(c) as chunk_count
                RETURN d.id as document_id,
                       d.filename as filename,
                       d.created_at as created_at,
                       chunk_count
                ORDER BY d.created_at DESC
                """
            )
            return [record.data() for record in result]

    def get_document_details(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve detailed metadata for a document."""

        def _timestamp_to_iso(value: Any) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
            if isinstance(value, str):
                try:
                    numeric = float(value)
                    return datetime.fromtimestamp(numeric, tz=timezone.utc).isoformat()
                except ValueError:
                    return value
            return str(value)

        with self.driver.session() as session:  # type: ignore
            doc_record = session.run(
                """
                MATCH (d:Document {id: $doc_id})
                RETURN d
                """,
                doc_id=doc_id,
            ).single()

            if doc_record is None:
                raise ValueError("Document not found")

            doc_node = doc_record["d"]
            doc_data = dict(doc_node)

            chunk_records = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id as id,
                       c.content as text,
                       c.chunk_index as index,
                       coalesce(c.offset, 0) as offset,
                       c.score as score
                ORDER BY coalesce(c.chunk_index, 0) ASC, c.id ASC
                """,
                doc_id=doc_id,
            )
            chunks = [
                {
                    "id": record["id"],
                    "text": record["text"] or "",
                    "index": record["index"],
                    "offset": record["offset"],
                    "score": record["score"],
                }
                for record in chunk_records
            ]

            entity_records = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)-[:CONTAINS_ENTITY]->(e:Entity)
                RETURN e.type as type,
                       e.name as text,
                       count(*) as count,
                       collect(DISTINCT c.chunk_index) as positions
                ORDER BY type ASC, text ASC
                """,
                doc_id=doc_id,
            )
            entities = [
                {
                    "type": record["type"],
                    "text": record["text"],
                    "count": record["count"],
                    "positions": [pos for pos in (record["positions"] or []) if pos is not None],
                }
                for record in entity_records
            ]

            related_records = session.run(
                """
                MATCH (d:Document {id: $doc_id})-[r:RELATED_TO|SIMILAR_TO]-(other:Document)
                RETURN DISTINCT other.id as id,
                                coalesce(other.title, other.filename) as title,
                                coalesce(other.link, '') as link,
                                other.filename as filename
                ORDER BY coalesce(other.title, other.filename) ASC
                """,
                doc_id=doc_id,
            )
            related_documents = [
                {
                    "id": record["id"],
                    "title": record["title"] or record["filename"],
                    "link": record["link"] if record["link"] else None,
                }
                for record in related_records
            ]

            uploader_info: Optional[Dict[str, Any]] = None
            uploader_value = doc_data.get("uploader")
            if isinstance(uploader_value, dict):
                uploader_info = {
                    "id": uploader_value.get("id"),
                    "name": uploader_value.get("name"),
                }
            else:
                uploader_id = doc_data.get("uploader_id")
                uploader_name = doc_data.get("uploader_name")
                if uploader_id or uploader_name:
                    uploader_info = {"id": uploader_id, "name": uploader_name}

            uploaded_at = (
                _timestamp_to_iso(doc_data.get("uploaded_at"))
                or _timestamp_to_iso(doc_data.get("created_at"))
            )

            known_keys = {
                "id",
                "title",
                "filename",
                "mime_type",
                "preview_url",
                "uploaded_at",
                "created_at",
                "uploader",
                "uploader_id",
                "uploader_name",
                "quality_scores",
                "summary",
                "document_type",
                "hashtags",
            }

            metadata = {
                key: value
                for key, value in doc_data.items()
                if key not in known_keys
            }

            return {
                "id": doc_data.get("id", doc_id),
                "title": doc_data.get("title"),
                "file_name": doc_data.get("filename"),
                "mime_type": doc_data.get("mime_type"),
                "preview_url": doc_data.get("preview_url"),
                "uploaded_at": uploaded_at,
                "uploader": uploader_info,
                "summary": doc_data.get("summary"),
                "document_type": doc_data.get("document_type"),
                "hashtags": doc_data.get("hashtags", []),
                "chunks": chunks,
                "entities": entities,
                "quality_scores": doc_data.get("quality_scores"),
                "related_documents": related_documents or None,
                "metadata": metadata or None,
            }

    def get_document_file_info(self, doc_id: str) -> Dict[str, Any]:
        """Return file metadata for previewing a document."""

        with self.driver.session() as session:  # type: ignore
            record = session.run(
                """
                MATCH (d:Document {id: $doc_id})
                RETURN d.filename as file_name,
                       d.file_path as file_path,
                       d.mime_type as mime_type,
                       d.preview_url as preview_url
                """,
                doc_id=doc_id,
            ).single()

            if record is None:
                raise ValueError("Document not found")

            file_path = record["file_path"]
            mime_type = record["mime_type"]
            if not mime_type and file_path:
                mime_type = mimetypes.guess_type(file_path)[0]

            return {
                "file_name": record["file_name"],
                "file_path": file_path,
                "mime_type": mime_type,
                "preview_url": record["preview_url"],
            }

    def clear_database(self) -> None:
        """Clear all data from the database."""
        with self.driver.session() as session:  # type: ignore
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database cleared")


# Global database instance
graph_db = GraphDB()
