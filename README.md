# GraphRAG Pipeline with LangGraph and Streamlit

A comprehensive RAG (Retrieval-Augmented Generation) pipeline built with LangGraph for workflow orchestration, Streamlit for the web interface, and Neo4j for graph-based document storage and retrieval.

## Features

- 🔄 **LangGraph Orchestration**: Graph-based RAG workflow with intelligent reasoning
- 🌐 **Streamlit Web Interface**: Interactive document management, chat interface, and graph visualization
- 🗂️ **Multi-format Document Support**: PDF, DOCX, TXT, CSV, PPTX, XLSX with intelligent processing
- 📊 **Structured Data Processing**: Smart analysis of spreadsheets and presentations with business context detection
- 📊 **Neo4j Graph Database**: Persistent storage with relationship mapping
- 🔧 **Configurable OpenAI and Ollama API**: Custom base URL, API key, model, and proxy settings
- 📈 **Interactive Graph Visualization**: Real-time view of document relationships and retrieval paths
- 🔍 **Streaming Responses**: Progressive answer display for better user experience
- 🎯 **Background File Processing**: Upload documents with progress indicators
- 🧮 **Token-aware Request Management**: Avoid overwhelming LLM with intelligent token management and request splitting
- 🖨️ **OCR / Smart OCR Support**: Robust OCR pipeline for scanned documents and images (see `docs/OCR_IMPLEMENTATION.md`)
- 🔗 **Multi-hop Search / Graph Expansion**: Deep graph traversal for multi-step reasoning and investigative queries (see `docs/MULTI_HOP_IMPLEMENTATION.md`)
 - 🤖 **Follow-up Questions Support**: Conversation-aware follow-up detection and contextualized query rewriting for multi-turn chats
 
### 🆕 **Hybrid Entity-Chunk Retrieval**

- 🧠 **Entity Extraction**: LLM-powered extraction of entities and relationships from documents
- 🔀 **Multiple Retrieval Modes**: Choose between chunk-only, entity-only, or hybrid retrieval strategies
- 🚀 **Smart Search Modes**: Pre-configured Quick, Normal, and Deep search modes for different use cases
- 🌐 **Graph Expansion**: Uses entity relationships to expand context and find related information
- ⚙️ **Configurable Parameters**: Fine-tune retrieval behavior with advanced settings
- 🎛️ **Cost Control**: Configurable concurrency and model selection for budget management
- 📊 **Rich Visualization**: View both chunks and entities in interactive graph displays

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
│   • Search Modes│    │                  │    │   • Chunks      │
│   • Settings    │    │                  │    │   • Entities    │
└─────────────────┘    └──────────────────┘    │   • Relations   │
         │                       │              └─────────────────┘
         ▼                       ▼                       │
┌─────────────────┐    ┌──────────────────┐             ▼
│ Document Upload │    │  Enhanced        │    ┌─────────────────┐
│ & Ingestion     │    │  Retriever       │    │ Vector Storage  │
│   • Multi-format│    │   • Chunk-only   │    │ & Similarity    │
│   • Entity Extr.│    │   • Entity-only  │    │ Search          │
└─────────────────┘    │   • Hybrid       │    └─────────────────┘
         │              │   • Graph Exp.   │             │
         ▼              └──────────────────┘             ▼
┌─────────────────┐             │              ┌─────────────────┐
│  LLM APIs       │◀────────────┘              │ Knowledge Graph │
│  • OpenAI       │                            │ Traversal &     │
│  • Ollama       │                            │ Reasoning       │
└─────────────────┘                            └─────────────────┘
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
# LLM Provider Configuration
LLM_PROVIDER=openai  # or 'ollama'

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: custom endpoint
OPENAI_MODEL=gpt-4  # Model to use
OPENAI_PROXY=  # Optional: proxy URL

# Ollama Configuration (if using Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Document Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Entity Extraction Configuration (NEW)
ENABLE_ENTITY_EXTRACTION=true  # Enable/disable entity extraction

---

Notes on dependencies

The `requirements-docker.txt` file has been trimmed to include only packages that are directly imported by the codebase and are required at runtime. A few transitive or unused packages (for example `python-dotenv` and `python-multipart`) were removed to reduce Docker image size and maintenance burden. If you rely on functionality provided by those packages in your deployment, re-add them to `requirements-docker.txt`.

If you add new imports to the codebase, please update `requirements-docker.txt` before building the Docker image.
LLM_CONCURRENCY=2  # Concurrent LLM requests for entity extraction
EMBEDDING_CONCURRENCY=3  # Concurrent embedding requests

# Retrieval Configuration (NEW)
MIN_RETRIEVAL_SIMILARITY=0.1  # Minimum similarity threshold
HYBRID_CHUNK_WEIGHT=0.6  # Weight for chunk-based results in hybrid mode
ENABLE_GRAPH_EXPANSION=true  # Enable graph traversal for context expansion

# Graph Expansion Limits (NEW)
MAX_EXPANDED_CHUNKS=500  # Maximum chunks after graph expansion
MAX_ENTITY_CONNECTIONS=20  # Maximum entity connections to follow
MAX_CHUNK_CONNECTIONS=10  # Maximum chunk similarity connections
EXPANSION_SIMILARITY_THRESHOLD=0.1  # Minimum similarity for expansion
MAX_EXPANSION_DEPTH=2  # Maximum depth for graph traversal

# Application Configuration
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

GraphRAG4 supports intelligent processing of multiple document formats with specialized loaders:

#### Supported Document Types:
- **PDF** (`.pdf`) - Text extraction with page structure
- **Word** (`.docx`) - Document text, tables, and formatting
- **Text** (`.txt`, `.md`) - Plain text and markdown files
- **CSV** (`.csv`) - Intelligent data analysis with business context detection
- **PowerPoint** (`.pptx`) - Slide content with structure and visual element analysis
- **Excel** (`.xlsx`, `.xls`) - Multi-sheet processing with data type recognition

#### Processing Features:
- **Intelligent Analysis**: Automatic detection of data types, business contexts, and relationships
- **Structure Preservation**: Maintains document hierarchy and formatting context
- **Progress Tracking**: Real-time feedback during file processing
- **Batch Processing**: Upload and process multiple files simultaneously
- **Smart OCR**: Applies OCR to PDF scanned pages, diagrams or images

### Hybrid Entity-Chunk Approach

This implementation combines the reliability of traditional chunk-based retrieval with the semantic richness of entity-based graphs. Key features include:

- **Backward Compatibility**: Existing chunk-only retrieval continues to work unchanged
- **Progressive Enhancement**: Enable entity extraction when ready for enhanced capabilities  
- **Cost-Aware Processing**: Configurable concurrency and model selection for budget management
- **Rich Graph Visualization**: View both chunks and entities in interactive Neo4j browser

### OCR (Scanned Documents & Images)

- The project includes a robust OCR pipeline for handling scanned PDFs and images. See `core/ocr.py` for the core OCR utilities and `ingestion/document_processor.py` for how OCR is integrated into the ingestion flow.
- A "Smart OCR" mode applies image pre-processing and layout-aware text reconstruction to improve extraction quality for noisy scans and multi-column layouts. Detailed implementation notes and tuning tips are in `docs/OCR_IMPLEMENTATION.md`.
- When OCR is enabled (via the `.env` or application settings), extracted text is chunked, embedded, and stored alongside born-digital documents so retrieval and graph construction are uniform across sources.

### Multi-hop Reasoning & Graph Expansion

- The system supports multi-hop reasoning across the knowledge graph to gather broader context for complex queries. This is implemented in `rag/graph_rag.py` and the LangGraph nodes under `rag/nodes/graph_reasoning.py`.
- Multi-hop expansion follows entity relationships and chunk similarity links up to configurable depths (see the `MAX_EXPANSION_DEPTH` and related environment variables in this README and `config/settings.py`). Details, trade-offs, and examples are documented in `docs/MULTI_HOP_IMPLEMENTATION.md`.
- Use the "Deep Search" mode in the Streamlit UI to enable deeper graph traversal for investigative queries that need multi-step reasoning across documents and entities.

For detailed information about the hybrid approach, configuration options, and usage patterns, see [HYBRID_APPROACH.md](HYBRID_APPROACH.md).

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

### 3. Search Modes & Retrieval Strategies

The application now supports three pre-configured search modes and multiple retrieval strategies:

#### Search Modes

- **🚀 Quick Search**: Fast results with minimal graph traversal (3 chunks, shallow expansion)
- **⚖️ Normal Search**: Balanced performance and comprehensiveness (5 chunks, moderate expansion)
- **🔍 Deep Search**: Maximum comprehensiveness with extensive graph exploration (10 chunks, deep expansion)

#### Retrieval Strategies

- **Chunk Only**: Traditional vector similarity search (fastest, backward compatible)
- **Entity Only**: GraphRAG-style entity relationship search (comprehensive, slower)
- **Hybrid**: Combines chunk similarity and entity relationships (recommended)

#### Entity Extraction

When entity extraction is enabled, the system:

- Extracts entities and relationships from documents using configurable LLM models
- Builds a rich knowledge graph with both chunks and entities
- Enables advanced retrieval through entity relationships
- Provides graph expansion for broader context discovery

### 4. API Server (Optional)

Start the FastAPI server for programmatic access:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

## Token Management & Request Splitting

This project includes a dedicated token management utility (`core/token_manager.py`) which helps the application safely interact with LLMs of different providers and context lengths. Key features and behaviors:

- **Automatic context-size detection**: the TokenManager contains a mapping of common models (OpenAI, Ollama, etc.) to conservative context size estimates. For unknown models it can query the model with a safe low-cost prompt to get the model's reported maximum context length and adjust behavior accordingly.
- **Precise token counting when available**: if the `tiktoken` package is installed the manager uses model encodings to count tokens accurately. When `tiktoken` is not available it falls back to a robust character-based approximation (~4 characters per token).
- **Reserved tokens and safety margins**: the manager reserves tokens for system messages, expected model output and an additional safety buffer to avoid truncation or prompt rejection.
- **Automatic request splitting and batching**: given a user query and a list of context chunks the TokenManager can split chunks into one or more batches that fit within the model's context window. Very large individual chunks are truncated (and marked as truncated) to fit a single batch when necessary.
- **Response merging**: when multiple model responses are produced from split batches the TokenManager offers two merge strategies: a simple deduplicating concatenation, or an LLM-based merge that asks the model to intelligently consolidate and deduplicate the partial responses into a single coherent answer.

### Where to look and how to tune

- **Implementation**: `core/token_manager.py` (global instance `token_manager`) — other components call into it to count tokens, check `needs_splitting`, create safe batches with `split_context_chunks`, and estimate output budgets using `available_output_tokens_for_messages` / `available_output_tokens_for_prompt`.
- **Configuration**: model names are read from the application settings (`config/settings.py`) and matched against the internal `MODEL_CONTEXT_SIZES` mapping. If you add a custom model, add an entry there or allow the TokenManager to detect context size at runtime.
- **Recommended**: install `tiktoken` in production environments to get accurate token accounting. Without it, the system still works using approximations but results can be conservative.

### Practical tips

- For large documents rely on chunking (see `CHUNK_SIZE` and `CHUNK_OVERLAP` environment variables) and let the TokenManager batch chunks automatically to avoid overfull prompts.
- If you enable LLM-based merging for higher-quality consolidated answers, ensure your model configuration has sufficient output token budget — the TokenManager reserves a default safety margin but you can tune `reserved_tokens` in `core/token_manager.py` if required.
- Monitor logs for warnings about unknown models, truncated chunks, or token-count fallbacks — these messages indicate when manual tuning or `MODEL_CONTEXT_SIZES` updates are useful.

This token management layer makes the RAG pipeline robust across different models, deployments and input sizes by preventing overfull prompts and by providing deterministic, explainable splitting/merging behavior.

## Project Structure

```
graphrag/
├── .env                        # Local environment
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
├── requirements-docker.txt     # Python dependencies (lighter for docker image)
├── README.md                   # This file
├── docs/                       # Design & implementation docs
│   ├── OCR_IMPLEMENTATION.md
│   ├── HYBRID_APPROACH.md
│   └── MULTI_HOP_IMPLEMENTATION.md
├── app.py                      # Streamlit main application
├── docker-compose.yml          # Docker services (Neo4 + rag app)
├── Dockerfile                  
├── config/
│   └── settings.py             # Configuration management with hybrid settings
├── core/
│   ├── __init__.py
│   ├── graph_db.py             # Neo4j database operations with entity support
│   ├── ocr.py                  # OCR and Smart OCR utilities
│   ├── embeddings.py           # Text embedding utilities
│   ├── chunking.py             # Document chunking logic
│   ├── entity_extraction.py    # 🆕 LLM-powered entity extraction
│   ├── graph_viz.py            # Graph visualization utilities
│   ├── llm.py                  # OpenAI/Ollama API integration
│   └── token_manager.py        # Token counting and management
├── ingestion/
│   ├── __init__.py
│   ├── document_processor.py   # Multi-format document processing with entities
│   └── loaders/                # Document loaders by type
│       ├── __init__.py
│       ├── pdf_loader.py       # PDF document processing
│       ├── docx_loader.py      # Word document processing
│       ├── text_loader.py      # Plain text files
│       ├── csv_loader.py       # CSV with intelligent data analysis
│       ├── pptx_loader.py      # PowerPoint with slide structure analysis
│       └── xlsx_loader.py      # Excel with multi-sheet processing
├── rag/
│   ├── __init__.py
│   ├── graph_rag.py            # LangGraph RAG implementation
│   ├── retriever.py            # Legacy document retrieval logic
│   ├── retriever.py            # 🆕 Multi-mode retrieval (chunk/entity/hybrid)
│   └── nodes/                  # LangGraph node definitions
│       ├── __init__.py
│       ├── query_analysis.py   # Query analysis and classification
│       ├── retrieval.py        # Enhanced retrieval node
│       ├── generation.py       # Response generation
│       └── graph_reasoning.py  # Graph-based reasoning and expansion
├── scripts/
│   ├── __init__.py
│   ├── create_similarities.py  # Utility: create chunk similarities
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
