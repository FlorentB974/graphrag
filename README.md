# GraphRAG Pipeline with LangGraph and Chainlit

A comprehensive RAG (Retrieval-Augmented Generation) pipeline using LangGraph for orchestration and Chainlit for the web interface, with Neo4j graph database integration.

## Features

- 🔄 **LangGraph Orchestration**: Graph-based RAG workflow with intelligent reasoning
- 🌐 **Chainlit Web Interface**: Interactive document upload and chat interface
- 🗂️ **Multi-format Document Support**: PDF, DOCX, TXT, and more
- 📊 **Neo4j Graph Database**: Persistent storage with relationship mapping
- 🔧 **Configurable OpenAI API**: Custom base URL, API key, model, and proxy settings
- 📈 **Graph Visualization**: Real-time view of document relationships and retrieval paths
- 🔍 **Chunk Visualization**: Display relevant document chunks for each query

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Chainlit UI   │───▶│   LangGraph      │───▶│   Neo4j Graph   │
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
- Neo4j Database (running on default port 7687)
- OpenAI API access

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd graphrag4
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

## Usage

### 1. Data Ingestion (CLI)

Ingest documents using the CLI script:

```bash
python scripts/ingest_documents.py --input-dir ./documents --recursive
```

### 2. Web Interface

Start the Chainlit web interface:

```bash
chainlit run app.py --port 8000
```

Access the interface at `http://localhost:8000`

### 3. API Server (Optional)

Start the FastAPI server for programmatic access:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

## Project Structure

```
graphrag4/
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── app.py                       # Chainlit main application
├── config/
│   └── settings.py             # Configuration management
├── core/
│   ├── __init__.py
│   ├── graph_db.py             # Neo4j database operations
│   ├── embeddings.py           # Text embedding utilities
│   ├── chunking.py             # Document chunking logic
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
│   ├── graph_rag.py           # LangGraph RAG implementation
│   ├── retriever.py           # Document retrieval logic
│   └── nodes/                 # LangGraph node definitions
│       ├── __init__.py
│       ├── query_analysis.py
│       ├── retrieval.py
│       ├── generation.py
│       └── graph_reasoning.py
├── api/                       # FastAPI endpoints (optional)
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       ├── documents.py
│       └── query.py
├── scripts/
│   ├── __init__.py
│   ├── ingest_documents.py    # CLI document ingestion
│   └── setup_neo4j.py        # Neo4j database setup
├── static/                    # Static assets for web interface
│   └── css/
│       └── custom.css
└── tests/                     # Test suite
    ├── __init__.py
    ├── test_ingestion.py
    ├── test_rag.py
    └── conftest.py
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
- Interactive chat interface
- Real-time graph visualization
- Relevant chunk display
- Configuration management UI

## Development

### Running Tests

```bash
pytest tests/ -v
```

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