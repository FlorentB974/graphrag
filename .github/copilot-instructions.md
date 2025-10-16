## Project Overview
This is a comprehensive RAG (Retrieval-Augmented Generation) pipeline using LangGraph and FastAPI/Uvicorn with Neo4j graph database integration.

## Architecture Components
- **LangGraph**: Orchestrates the RAG workflow with graph-based reasoning
- **FastAPI/Uvicorn**: Provides the web interface for user interaction and document ingestion
- **Neo4j**: Graph database for storing document chunks and relationships
- **OpenAI API**: Configurable LLM integration with custom endpoints
- **Multi-format Document Ingestion**: Supports various file types (PDF, DOCX, TXT, etc.)

## Key Features
- Document upload and processing through web interface
- Graph visualization of document relationships
- Configurable OpenAI API settings (base URL, API key, model, proxy)
- Real-time display of relevant chunks and graph traversal
- Multi-format document support with intelligent chunking
- Neo4j integration for persistent graph storage

## Development Guidelines
- Follow modular architecture with separate components for ingestion, graph operations, and frontend
- Use environment variables for all configuration
- Implement proper error handling and logging
- Maintain clear separation between data processing and UI layers
- Document all API endpoints and configuration options
- When running a python command, always source the virtual environment first:
  ```bash
  source .venv/bin/activate
  ```
- Do not write MD files documentation unless specified
- Do not write tests unless specified