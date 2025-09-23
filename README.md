# GraphRAG Pipeline with LangGraph and Streamlit

A comprehensive RAG (Retrieval-Augmented Generation) pipeline built with LangGraph for workflow orchestration, Streamlit for the web interface, and Neo4j for graph-based document storage and retrieval.

## Features

- 🔄 **LangGraph Orchestration**: Graph-based RAG workflow with intelligent reasoning
- 🌐 **Streamlit Web Interface**: Interactive document management, chat interface, and graph visualization
- 🗂️ **Multi-format Document Support**: PDF, DOCX, TXT, and more
- 📊 **Neo4j Graph Database**: Persistent storage with relationship mapping
- 🔧 **Configurable OpenAI and Ollama API**: Custom base URL, API key, model, and proxy settings
- 📈 **Interactive Graph Visualization**: Real-time view of document relationships and retrieval paths
- 🔍 **Streaming Responses**: Progressive answer display for better user experience
- 🎯 **Background File Processing**: Upload documents with progress indicators

See below a (very) short demo:

[demo.webm](https://github.com/user-attachments/assets/be528543-84c1-42e6-a979-2b9cd9f177f2)

## Run it (docker compose)

```yaml
version: '3.8'

services:
  neo4j:
    image: neo4j:latest
    container_name: neo4j
    environment:
      - NEO4J_AUTH=neo4j/graphrag_password
      - NEO4J_PLUGINS=["graph-data-science"]
    ports:
      - "7474:7474"  # HTTP 
      - "7687:7687"  # Bolt 
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_plugins:/plugins
    networks:
      - graphrag_network
    restart: unless-stopped

  graphrag-app:
    image: ghcr.io/florentb974/graphrag:latest
    container_name: graphrag-app
    environment:
      # Create a .env based on .env.example
      # LLM provider selection: 'openai' or 'ollama'
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      # OpenAI configuration (used when LLM_PROVIDER=openai)
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL}
      - OPENAI_MODEL=${OPENAI_MODEL}
      - OPENAI_PROXY=${OPENAI_PROXY}
      - EMBEDDING_MODEL=${EMBEDDING_MODEL:-text-embedding-ada-002}
      - EMBEDDING_CONCURRENCY=${EMBEDDING_CONCURRENCY:-3}  # Number of parallel embedding requests (default: 3)
      # Ollama configuration (used when LLM_PROVIDER=ollama)
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://localhost:11434}
      - OLLAMA_MODEL=${OLLAMA_MODEL:-llama3}
      - OLLAMA_EMBEDDING_MODEL=${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text}
      - OLLAMA_API_KEY=${OLLAMA_API_KEY:-}
      # Neo4j Configuration - connecting to the neo4j service
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USERNAME=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD:-graphrag_password}
      # Document Processing Configuration
      - CHUNK_SIZE=${CHUNK_SIZE:-1000}
      - CHUNK_OVERLAP=${CHUNK_OVERLAP:-200}
      # Similarity Configuration
      - SIMILARITY_THRESHOLD=${SIMILARITY_THRESHOLD:-0.1}
      - MAX_SIMILARITY_CONNECTIONS=${MAX_SIMILARITY_CONNECTIONS:-5}
      # Application Configuration
      - LOG_LEVEL=${LOG_LEVEL:-INFO}
      - MAX_UPLOAD_SIZE=${MAX_UPLOAD_SIZE:-104857600}
    ports:
      - "8501:8501"  # Streamlit port
    volumes:
      - ./data:/app/data  # Mount data directory for file uploads
    networks:
      - graphrag_network
    depends_on:
      - neo4j
    restart: unless-stopped

networks:
  graphrag_network:
    driver: bridge

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_plugins:
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│   LangGraph      │───▶│   Neo4j Graph   │
│   (Frontend)    │    │   (RAG Pipeline) │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Document Upload │    │  OpenAI API      │    │ Vector Storage  │
│ & Ingestion     │    │  Integration     │    │ & Retrieval     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API access
- Docker

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd graphrag
```

2. Create virtual environment (optional if using docker):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies (optional if using docker):

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your settings
```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: custom endpoint
OPENAI_MODEL=gpt-4  # Model to use
OPENAI_PROXY=  # Optional: proxy URL

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Application Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
EMBEDDING_CONCURRENCY=3
```

### Docker Image (Dockerfile)

This repository includes a `Dockerfile` that can be used to build a container image for running the application. Below are the common commands to build and run the image locally, and how to use the provided `docker-compose.yml` for running Neo4j alongside the app.

- Run with Docker Compose (recommended for development with Neo4j):

```bash
docker compose up -d --build
```

Notes:
- When running the app container and Neo4j on the same machine, use `host.docker.internal` for the `NEO4J_URI` so the container can reach the host Neo4j instance.
- Provide required environment variables (`OPENAI_API_KEY`, `NEO4J_PASSWORD`, etc.) either via the shell or a `.env` file consumed by `docker compose`.
- The `Dockerfile` is configured to install dependencies from `requirements-docker.txt` for a smaller image; if you modify dependencies, update that file accordingly.

### Setup Neo4j database

```bash
python scripts/setup_neo4j.py
```

## Usage

### Data Ingestion

Data ingestion can be achieved in two ways: Web interface or using the ingest_documents.py script.

Ingest documents using the CLI script:

```bash
python scripts/ingest_documents.py --input-dir ./documents --recursive
```

### Web Interface

Access the interface at `http://localhost:8501`

If not using docker, start the Streamlit web interface:

```bash
streamlit run app.py
```

### 3. API Server (Optional)

Start the FastAPI server for programmatic access:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

## Project Structure

```
graphrag/
├── .env                        # Local environment
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── requirements-docker.txt     # Python dependencies (lighter for docker image)
├── README.md                   # This file
├── app.py                      # Streamlit main application
├── docker-compose.yml          # Docker services (Neo4 + rag app)
├── Dockerfile                  
├── config/
│   └── settings.py             # Configuration management
├── core/
│   ├── __init__.py
│   ├── graph_db.py             # Neo4j database operations
│   ├── embeddings.py           # Text embedding utilities
│   ├── chunking.py             # Document chunking logic
│   ├── graph_viz.py            # Graph visualization utilities
│   └── llm.py                  # OpenAI API integration
├── ingestion/
│   ├── __init__.py
│   ├── document_processor.py   # Multi-format document processing
│   └── loaders/                # Document loaders by type
│       ├── __init__.py
│       ├── pdf_loader.py
│       ├── docx_loader.py
│       └── text_loader.py
├── rag/
│   ├── __init__.py
│   ├── graph_rag.py            # LangGraph RAG implementation
│   ├── retriever.py            # Document retrieval logic
│   └── nodes/                  # LangGraph node definitions
│       ├── __init__.py
│       ├── query_analysis.py
│       ├── retrieval.py
│       ├── generation.py
│       └── graph_reasoning.py
├── scripts/
│   ├── __init__.py
│   ├── create_similarities.py  # Utility: create similarities (exists in repo)
│   ├── ingest_documents.py     # CLI document ingestion
│   └── setup_neo4j.py          # Neo4j database setup
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
