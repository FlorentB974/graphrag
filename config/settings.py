"""
Configuration management for the GraphRAG pipeline.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Provider Configuration
    llm_provider: str = Field(..., env="LLM_PROVIDER")  # "openai" or "ollama"

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: str = Field(..., env="OPENAI_BASE_URL")
    openai_model: str = Field(..., env="OPENAI_MODEL")
    openai_proxy: Optional[str] = Field(None, env="OPENAI_PROXY")

    # Ollama Configuration
    ollama_base_url: str = Field(..., env="OLLAMA_BASE_URL")
    ollama_model: str = Field(..., env="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(..., env="OLLAMA_EMBEDDING_MODEL")

    # Neo4j Configuration
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j", env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")

    # Embedding Configuration
    embedding_model: str = Field("text-embedding-ada-002", env="EMBEDDING_MODEL")

    # Document Processing Configuration
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")

    # Similarity Configuration
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")
    max_similarity_connections: int = Field(5, env="MAX_SIMILARITY_CONNECTIONS")

    # Application Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    max_upload_size: int = Field(100 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 100MB

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
