"""
Configuration management for the GraphRAG pipeline.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider Configuration
    llm_provider: str = Field(default="openai", description="LLM provider to use")

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI base URL")
    openai_model: Optional[str] = Field(
        default="gpt-3.5-turbo", description="OpenAI model name"
    )
    openai_proxy: Optional[str] = Field(default=None, description="OpenAI proxy URL")

    # Ollama Configuration
    ollama_base_url: Optional[str] = Field(
        default="http://localhost:11434", description="Ollama base URL"
    )
    ollama_model: Optional[str] = Field(
        default="llama2", description="Ollama model name"
    )
    ollama_embedding_model: Optional[str] = Field(
        default="nomic-embed-text", description="Ollama embedding model"
    )

    # Neo4j Configuration
    neo4j_uri: str = Field(
        default="bolt://localhost:7687", description="Neo4j connection URI"
    )
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="neo4j", description="Neo4j password")

    # Embedding Configuration
    embedding_model: str = Field(
        default="text-embedding-ada-002", description="Embedding model"
    )
    # Number of concurrent embedding requests
    embedding_concurrency: int = Field(default=3, description="Embedding concurrency")
    llm_concurrency: int = Field(default=2, description="LLM concurrency")
    # Rate limiting delays (in seconds)
    embedding_delay_min: float = Field(default=0.5, description="Minimum delay between embedding requests")
    embedding_delay_max: float = Field(default=1.0, description="Maximum delay between embedding requests")
    llm_delay_min: float = Field(default=0.5, description="Minimum delay between LLM requests")
    llm_delay_max: float = Field(default=1.0, description="Maximum delay between LLM requests")

    # Document Processing Configuration
    chunk_size: int = Field(default=1000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, description="Document chunk overlap")

    # Similarity Configuration
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    max_similarity_connections: int = Field(
        default=5, description="Max similarity connections"
    )

    # Entity Extraction Configuration
    enable_entity_extraction: bool = Field(
        default=True, description="Enable entity extraction"
    )

    # OCR Configuration
    enable_ocr: bool = Field(
        default=True, description="Enable OCR processing for scanned documents"
    )
    enable_quality_filtering: bool = Field(
        default=True, description="Enable chunk quality filtering"
    )
    ocr_quality_threshold: float = Field(
        default=0.6, description="Quality threshold for OCR processing"
    )

    # Retrieval Configuration
    min_retrieval_similarity: float = Field(
        default=0.1, description="Minimum similarity for chunk retrieval"
    )
    hybrid_chunk_weight: float = Field(
        default=0.6, description="Weight for chunk-based results"
    )
    enable_graph_expansion: bool = Field(
        default=True, description="Enable graph expansion"
    )

    # Graph Expansion Limits
    max_expanded_chunks: int = Field(
        default=500, description="Maximum number of chunks after expansion"
    )
    max_entity_connections: int = Field(
        default=20, description="Maximum entity connections to follow"
    )
    max_chunk_connections: int = Field(
        default=10, description="Maximum chunk similarity connections to follow"
    )
    expansion_similarity_threshold: float = Field(
        default=0.1, description="Minimum similarity for expansion"
    )
    max_expansion_depth: int = Field(
        default=2, description="Maximum depth for graph traversal"
    )

    # Multi-hop Reasoning Configuration
    multi_hop_max_hops: int = Field(
        default=2, description="Maximum number of hops for multi-hop reasoning"
    )
    multi_hop_beam_size: int = Field(
        default=8, description="Beam size for multi-hop path search"
    )
    multi_hop_min_edge_strength: float = Field(
        default=0.0, description="Minimum edge strength for multi-hop traversal"
    )
    hybrid_path_weight: float = Field(
        default=0.6, description="Weight for path-based results in hybrid mode"
    )

    # Application Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    max_upload_size: int = Field(
        default=100 * 1024 * 1024, description="Max upload size"
    )  # 100MB

    # Quality Scoring Configuration
    enable_quality_scoring: bool = Field(
        default=True, description="Enable quality scoring for LLM answers"
    )
    quality_score_weights: dict = Field(
        default={
            "context_relevance": 0.30,
            "answer_completeness": 0.25,
            "factual_grounding": 0.25,
            "coherence": 0.10,
            "citation_quality": 0.10,
        },
        description="Weights for different quality score components",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Global settings instance - will read from environment or use defaults
settings = Settings()
