"""
Neo4j graph database operations for the RAG pipeline.
"""
import logging
import math
from typing import Dict, List, Optional, Any
from neo4j import GraphDatabase, Driver
from config.settings import settings

logger = logging.getLogger(__name__)


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
                auth=(settings.neo4j_username, settings.neo4j_password)
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
                metadata=metadata
            )
    
    def create_chunk_node(self, chunk_id: str, doc_id: str, content: str,
                          embedding: List[float], metadata: Dict[str, Any]) -> None:
        """Create a chunk node and link it to its document."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.content = $content,
                    c.embedding = $embedding,
                    c += $metadata
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                metadata=metadata
            )
    
    def create_similarity_relationship(self, chunk_id1: str, chunk_id2: str,
                                       similarity_score: float) -> None:
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
                similarity_score=similarity_score
            )
    
    @staticmethod
    def _calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
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
                doc_id=doc_id
            )
            
            chunks_data = [(record["chunk_id"], record["embedding"]) for record in result]
            
            if len(chunks_data) < 2:
                logger.info(f"Skipping similarity creation for document {doc_id}: less than 2 chunks")
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
                        similarity = self._calculate_cosine_similarity(embedding1, embedding2)
                        
                        if similarity >= threshold:
                            similarities.append((chunk_id2, similarity))
                
                # Sort by similarity and take top connections
                similarities.sort(key=lambda x: x[1], reverse=True)
                top_similarities = similarities[:max_connections]
                
                # Create relationships
                for chunk_id2, similarity in top_similarities:
                    self.create_similarity_relationship(chunk_id1, chunk_id2, similarity)
                    relationships_created += 1
            
            logger.info(f"Created {relationships_created} similarity relationships for document {doc_id}")
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
                relationships_created = self.create_chunk_similarities(doc_id, threshold)
                results[doc_id] = relationships_created
                total_relationships += relationships_created
                processed_docs += 1
                
                logger.info(f"Processed document {doc_id}: {relationships_created} relationships")
                
                # Process in batches to avoid memory issues
                if processed_docs % batch_size == 0:
                    logger.info(f"Processed {processed_docs}/{len(doc_ids)} documents so far")
                    
            except Exception as e:
                logger.error(f"Failed to create similarities for document {doc_id}: {e}")
                results[doc_id] = 0
        
        logger.info(f"Batch processing complete: {total_relationships} total relationships created for {processed_docs} documents")
        return results
    
    def vector_similarity_search(self, query_embedding: List[float],
                                 top_k: int = 5) -> List[Dict[str, Any]]:
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
                top_k=top_k
            )
            return [record.data() for record in result]
    
    def get_related_chunks(self, chunk_id: str, relationship_types: List[str] = None,   # type: ignore
                           max_depth: int = 2) -> List[Dict[str, Any]]:
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
                RETURN DISTINCT related.id as chunk_id, related.content as content,
                       length(path) as distance, d.filename as document_name, d.id as document_id
                ORDER BY distance ASC
                """
            
            result = session.run(
                query,  # type: ignore
                chunk_id=chunk_id,
                relationship_types=relationship_types
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
                doc_id=doc_id
            )
            return [record.data() for record in result]
    
    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks."""
        with self.driver.session() as session:  # type: ignore
            session.run(
                """
                MATCH (d:Document {id: $doc_id})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE c, d
                """,
                doc_id=doc_id
            )
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents with their metadata and chunk counts."""
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
                CALL {
                    MATCH (d:Document)
                    RETURN count(d) AS documents
                }
                CALL {
                    MATCH (c:Chunk)
                    RETURN count(c) AS chunks
                }
                CALL {
                    MATCH ()-[r]-()
                    WHERE type(r) = 'HAS_CHUNK'
                    RETURN count(r) AS has_chunk_relations
                }
                CALL {
                    MATCH ()-[r]-()
                    WHERE type(r) = 'SIMILAR_TO'
                    RETURN count(r) AS similarity_relations
                }
                RETURN documents, chunks, has_chunk_relations, similarity_relations
                """
            )
            record = result.single()
            if record is not None:
                return record.data()
            else:
                return {"documents": 0, "chunks": 0, "has_chunk_relations": 0, "similarity_relations": 0}
    
    def setup_indexes(self) -> None:
        """Create necessary indexes for performance."""
        with self.driver.session() as session:  # type: ignore
            # Create indexes for faster lookups
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            logger.info("Database indexes created successfully")


# Global database instance
graph_db = GraphDB()
