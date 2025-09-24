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
    openai_model: Optional[str] = Field(default="gpt-3.5-turbo", description="OpenAI model name")
    openai_proxy: Optional[str] = Field(default=None, description="OpenAI proxy URL")

    # Ollama Configuration
    ollama_base_url: Optional[str] = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: Optional[str] = Field(default="llama2", description="Ollama model name")
    ollama_embedding_model: Optional[str] = Field(default="nomic-embed-text", description="Ollama embedding model")

    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_username: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="neo4j", description="Neo4j password")

    # Embedding Configuration
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    # Number of concurrent embedding requests
    embedding_concurrency: int = Field(default=3, description="Embedding concurrency")

    # Document Processing Configuration
    chunk_size: int = Field(default=1000, description="Document chunk size")
    chunk_overlap: int = Field(default=200, description="Document chunk overlap")

    # Similarity Configuration
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    max_similarity_connections: int = Field(default=5, description="Max similarity connections")

    # Entity Extraction Configuration
    enable_entity_extraction: bool = Field(default=True, description="Enable entity extraction")
    entity_extraction_model: str = Field(default="gpt-oss-120b", description="Model for entity extraction")
    entity_extraction_concurrency: int = Field(default=2, description="Entity extraction concurrency")
    entity_importance_threshold: float = Field(default=0.3, description="Entity importance threshold")
    relationship_strength_threshold: float = Field(default=0.4, description="Relationship strength threshold")
    max_entities_per_chunk: int = Field(default=20, description="Max entities per chunk")

    # Retrieval Configuration
    default_retrieval_mode: str = Field(default="hybrid", description="Default retrieval mode")
    hybrid_chunk_weight: float = Field(default=0.6, description="Weight for chunk-based results")
    enable_graph_expansion: bool = Field(default=True, description="Enable graph expansion")
    max_expansion_depth: int = Field(default=2, description="Max expansion depth")

    # Application Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    max_upload_size: int = Field(default=100 * 1024 * 1024, description="Max upload size")  # 100MB

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Global settings instance - will read from environment or use defaults
settings = Settings()
