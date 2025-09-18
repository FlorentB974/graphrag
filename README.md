# GraphRAG Pipeline with LangGraph and Streamlit

A comprehensive RAG (Retrieval-Augmented Generation) pipeline built with LangGraph for workflow orchestration, Streamlit for the web interface, and Neo4j for graph-based document storage and retrieval.

## Features

- 🔄 **LangGraph Orchestration**: Graph-based RAG workflow with intelligent reasoning
- 🌐 **Streamlit Web Interface**: Interactive document upload, chat interface, and graph visualization
- 🗂️ **Multi-format Document Support**: PDF, DOCX, TXT, and more
- 📊 **Neo4j Graph Database**: Persistent storage with relationship mapping
- 🔧 **Configurable OpenAI API**: Custom base URL, API key, model, and proxy settings
- 📈 **Interactive Graph Visualization**: Real-time view of document relationships and retrieval paths
- 🔍 **Streaming Responses**: Progressive answer display for better user experience
- 🎯 **Background File Processing**: Upload documents with progress indicators

## Architecture

```mermaid
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

### Installation

1. Clone the repository:

```bash
git clone <your-repo-url>
cd graphrag
```

2. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your settings
```

5. Start and setup Neo4j database:

```bash
docker compose up -d
python scripts/setup_neo4j.py
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

## Usage

### 1. Data Ingestion (CLI)

Ingest documents using the CLI script:

```bash
python scripts/ingest_documents.py --input-dir ./documents --recursive
```

### 2. Web Interface

Start the Streamlit web interface:

```bash
streamlit run app.py
```

Access the interface at `http://localhost:8501`

#### Key Features

**Interactive Chat Interface**

- Ask questions about your uploaded documents
- Stream responses for better user experience
- View query analysis and source chunks
- Real-time feedback and progress indicators

**Document Upload & Processing**

- Upload multiple files simultaneously (PDF, DOCX, TXT, MD)
- Background processing with progress tracking
- Automatic chunking and graph integration
- Error handling and status reporting

**Knowledge Graph Visualization**

- Interactive graph visualization using Plotly
- Multiple layout algorithms (spring, circular, kamada-kawai)
- Adjustable node limits for performance
- Query-specific subgraph visualization

**Advanced RAG Features**

- Graph-enhanced retrieval for better context
- Configurable retrieval modes and parameters
- Multi-step reasoning through graph traversal
- Real-time database statistics

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
├── README.md                   # This file
├── app.py                      # Streamlit main application
├── docker-compose.yml          # Docker services (Neo4j)
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

## Features Overview

### Document Ingestion

- Support for PDF, DOCX, TXT, MD, and other text files
- Intelligent chunking with configurable size and overlap
- Automatic metadata extraction
- Batch processing capabilities

### Graph Database Integration

- Neo4j for storing document chunks and relationships
- Vector similarity search
- Relationship mapping between documents and concepts
- Query result visualization

### RAG Pipeline

- LangGraph-based orchestration
- Multi-step reasoning with graph traversal
- Configurable retrieval strategies
- Context-aware response generation

### Web Interface

- Document upload and management
- Interactive Streamlit interface with streaming responses
- Real-time graph visualization with Plotly
- Relevant chunk display and query analysis
- Background file processing with progress indicators
- Configuration management UI

## Development

### Code Formatting

```bash
black .
isort .
```

### Database Setup

Initialize Neo4j database:

```bash
python scripts/setup_neo4j.py
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
