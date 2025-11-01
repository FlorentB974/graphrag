# GraphRAG v2.0 🚀

<!-- markdownlint-disable -->

A state-of-the-art document intelligence system powered by graph-based RAG (Retrieval-Augmented Generation). Built with Next.js, FastAPI, and Neo4j.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Node 18+](https://img.shields.io/badge/node-18+-green.svg)](https://nodejs.org/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

## ✨ Features

### 🎨 ALL NEW Modern UI
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Dark Mode Ready**: Clean, modern interface
- **Smooth Animations**: Polished user experience
- **Accessibility**: Built with accessibility in mind

### 📊 ALL NEW Conversation History
- **Persistent Sessions**: Store and retrieve past conversations
- **Session Management**: View, search, and delete conversations
- **Context Preservation**: Maintain conversation context across sessions

### 💬 Intelligent Chat
- **NEW Follow-up Questions**: AI-generated suggestions to continue the conversation
- **Real-time Streaming**: Token-by-token response generation with SSE
- **Context-Aware**: Leverages graph relationships for accurate answers
- **Quality Scoring**: Real-time assessment of answer quality

### 📚 Document Management
- **NEW Summary extraction**: Automatic summary extraction during ingestion
- **NEW In-app Document View**: Inspect metadata, chunks, entities, and live previews
- **NEW Tags extraction**: Automatic tags extraction during ingestion (editable)
- **Multi-format Support**: PDF, DOCX, TXT, MD, PPT, XLS
- **Smart Chunking**: Intelligent document segmentation
- **Entity Extraction**: Automatic identification of key entities
- **Graph Relationships**: Connects related concepts across documents

### 🔍 Advanced Retrieval
- **NEW Context Restriction**: Rectrict context by specifying documents or tags in chat
- **Hybrid Search**: Combines vector similarity and graph traversal
- **Multi-hop Reasoning**: Connects information across multiple documents
- **Relevance Scoring**: Transparent source ranking
- **Entity-Enhanced**: Leverages extracted entities for better context

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│                   (Next.js 14 + React)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Chat Interface  │  History  │  Upload  │  Database  │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────┬────────────────────────────────────────────┘
                 │ REST API + SSE
                 │
┌────────────────▼────────────────────────────────────────────┐
│                      Backend API                             │
│                    (FastAPI + Python)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Chat   │   History   │   Database   │   Upload   │   │
│  └─────────────────────────────────────────────────────┘   │
└────────┬───────────────────────┬────────────────────────────┘
         │                       │
         │ LangGraph             │ Neo4j Driver
         │ Pipeline              │
         │                       │
┌────────▼───────────┐  ┌────────▼────────────┐
│                    │  │                     │
│  LangChain/OpenAI  │  │      Neo4j          │
│  (LLM & Embeddings)│  │   (Graph Database)  │
│                    │  │                     │
└────────────────────┘  └─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** and pip
- **Node.js 18+** and npm
- **Neo4j 5.0+** database
- **OpenAI API key** (or compatible endpoint)

### One-Command Setup

```bash
git clone https://github.com/FlorentB974/graphrag4.git
cd graphrag4
./setup.sh
```

This will:
1. Create a Python virtual environment
2. Install all Python dependencies
3. Install all Node.js dependencies
4. Create configuration templates

### Manual Setup

#### 1. Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys and Neo4j credentials
```

#### 2. Start Neo4j

**Option A: Docker**
```bash
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest
```

**Option B: Local Installation**
```bash
# Start your locally installed Neo4j instance
neo4j start
```

#### 3. Start Backend API

```bash
source .venv/bin/activate
python api/main.py
```

API will be available at `http://localhost:8000`  
API docs at `http://localhost:8000/docs`

#### 4. Frontend Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Frontend will be available at `http://localhost:3000`

## 📖 Usage

### Uploading Documents

1. Click the **Upload** tab in the sidebar
2. Drag and drop files or click to select
3. Wait for processing to complete
4. Documents will appear in the Database tab

### Asking Questions

1. Type your question in the chat input
2. Press Enter or click the send button
3. Watch as the AI streams the response
4. View sources by expanding the Sources section
5. Click follow-up questions to continue the conversation

### Managing History

1. Click the **History** tab
2. View all past conversations
3. Click on a conversation to load it
4. Delete conversations individually or clear all

### Database Management

1. Click the **Database** tab
2. View statistics (documents, chunks, entities, relationships)
3. Click a document row to open the full Document View with preview
4. Manage documents (delete, clear database) without leaving the chat context

### Viewing Documents

1. Select a document from the **Database** tab
2. Review metadata, chunk text, extracted entities, and related documents
3. Open the preview to stream PDFs, images, or download other formats
4. Use the back button to return to the chat without losing conversation state

## 🔧 Configuration

### Environment Variables

See `.env.example` for all available options. Key variables:

```bash
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Features
ENABLE_ENTITY_EXTRACTION=true
ENABLE_QUALITY_SCORING=true
```

### Advanced Configuration

Edit `config/settings.py` for fine-tuning:
- Chunk sizes and overlap
- Similarity thresholds
- Graph expansion parameters
- Multi-hop reasoning settings

## 📚 Documentation

- **[Setup Guide](SETUP_V2.md)**: Comprehensive setup instructions
- **[Migration Guide](MIGRATION_V2.md)**: Migrating from v1.x
- **[Frontend README](frontend/README.md)**: Frontend-specific documentation
- **[API Documentation](http://localhost:8000/docs)**: Interactive API docs (when running)

## 🏗️ Project Structure

```
graphrag4/
├── api/                    # FastAPI backend
│   ├── main.py             # API application
│   ├── models.py           # Pydantic models
│   ├── routers/            # API endpoints
│   └── services/           # Business logic
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/            # Next.js pages
│   │   ├── components/     # React components
│   │   ├── lib/            # Utilities
│   │   └── types/          # TypeScript types
│   └── package.json
├── core/                   # Core functionality
│   ├── graph_db.py         # Neo4j integration
│   ├── llm.py              # LLM management
│   ├── embeddings.py       # Vector embeddings
│   └── entity_extraction.py
├── rag/                    # RAG pipeline
│   ├── graph_rag.py        # Main RAG orchestrator
│   └── nodes/              # LangGraph nodes
├── ingestion/              # Document processing
│   ├── document_processor.py
│   └── loaders/            # File format loaders
├── scripts/                # Utility scripts
│   └── ingest_documents.py
├── config/                 # Configuration
│   └── settings.py
├── requirements.txt        # Python dependencies
└── setup.sh                # Quick setup script
```

## 🧪 Development

### Running Tests

```bash
source .venv/bin/activate
pytest api/tests/

cd frontend
npm run test
```

### Code Quality

```bash
# Format code
black .
isort .

# Lint
flake8 .

# Type checking
mypy .
```

### Frontend Development

```bash
cd frontend

# Lint
npm run lint

# Type checking
npm run type-check

# Build
npm run build
```

## 🐳 Docker Deployment

### Docker Compose

```bash
docker-compose up -d --build
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/) and [LangGraph](https://www.langchain.com/langgraph)
- Powered by [OpenAI](https://openai.com/) GPT models
- Graph database by [Neo4j](https://neo4j.com/)
- Frontend framework by [Next.js](https://nextjs.org/)
- Styled with [Tailwind CSS](https://tailwindcss.com/)
- Tested with [Akash Chat API](https://chatapi.akash.network/documentation) 

## 📞 Support

- **Documentation**: Check the `/docs` directory
- **Issues**: [GitHub Issues](https://github.com/FlorentB974/graphrag4/issues)
- **Discussions**: [GitHub Discussions](https://github.com/FlorentB974/graphrag4/discussions)

## 🗺️ Roadmap

- [ ] Multi-language support
- [ ] Voice input/output
- [x] Document preview
- [ ] Advanced search
- [ ] Export conversations
- [ ] Dark mode
- [ ] Mobile apps
- [ ] Plugin system
- [ ] Analytics dashboard