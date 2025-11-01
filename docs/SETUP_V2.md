# GraphRAG v2.0 - Complete Setup Guide

<!-- markdownlint-disable -->

This guide will help you set up and run the new GraphRAG v2.0 with the modern Next.js frontend and FastAPI backend.

## Quick Start

### 1. Backend Setup (Python/FastAPI)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install backend dependencies (if needed)
pip install -r requirements.txt

# Start Neo4j (if not running)
# Option 1: Docker
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:latest

# Option 2: Local installation
# Start your local Neo4j instance

# Configure environment
cp .env.example .env
# Edit .env and set your Neo4j credentials and API keys

# Run the FastAPI backend
python api/main.py
```

The API will be available at `http://localhost:8000`. You can view the API docs at `http://localhost:8000/docs`.

### 2. Frontend Setup (Next.js)

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local
# Edit .env.local if needed (default API URL is http://localhost:8000)

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## What's New in v2.0

### 🎨 Modern Frontend

- **Next.js 14**: Latest App Router with React Server Components
- **TypeScript**: Full type safety throughout the application
- **Tailwind CSS**: Modern, responsive design
- **Real-time Streaming**: Token-by-token response streaming with SSE

### 💡 New Features

- **Follow-up Questions**: AI-generated suggestions after each response
- **Chat History**: Persistent conversation management with Neo4j
- **Improved Sources**: Inline citations with expandable content
- **Quality Scoring**: Real-time answer quality assessment
- **File Upload**: Drag-and-drop document upload
- **Database Management**: View stats and manage documents
- **Document View**: Inspect document metadata, chunk text, entities, and previews without leaving the app

### ⚡ Performance Improvements

- **Optimized Frontend**: Faster load times and smooth animations
- **Streaming Responses**: Immediate feedback with progressive rendering
- **Async Operations**: Non-blocking quality scoring and follow-up generation

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────┐
│                 │         │                  │         │             │
│  Next.js        │ ◄─────► │  FastAPI         │ ◄─────► │   Neo4j     │
│  Frontend       │   REST  │  Backend         │  Cypher │  Database   │
│  (Port 3000)    │   SSE   │  (Port 8000)     │         │  (Port 7687)│
│                 │         │                  │         │             │
└─────────────────┘         └──────────────────┘         └─────────────┘
        │                           │
        │                           │
        └───────────────┬───────────┘
                        │
                ┌───────▼────────┐
                │                │
                │  OpenAI API    │
                │  (LLM & Embed) │
                │                │
                └────────────────┘
```

## API Endpoints

### Chat

- `POST /api/chat/query` - Send chat message (supports streaming)
- `POST /api/chat/follow-ups` - Generate follow-up questions

### History

- `GET /api/history/sessions` - List all conversation sessions
- `GET /api/history/{session_id}` - Get conversation details
- `DELETE /api/history/{session_id}` - Delete a conversation
- `POST /api/history/clear` - Clear all history

### Database

- `GET /api/database/stats` - Get database statistics
- `POST /api/database/upload` - Upload a document
- `DELETE /api/database/documents/{id}` - Delete a document
- `POST /api/database/clear` - Clear entire database
- `GET /api/database/documents` - List all documents

### Documents

- `GET /api/documents/{id}` - Retrieve full document metadata (chunks, entities, related docs)
- `GET /api/documents/{id}/preview` - Stream or redirect to a previewable version of the file

### Health

- `GET /api/health` - Health check endpoint

## Environment Variables

### Backend (.env)

```bash
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002

# Application Configuration
LOG_LEVEL=INFO
```

### Frontend (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development Workflow

### Starting Development

1. **Start Neo4j**: Ensure your Neo4j database is running
2. **Start Backend**: `python api/main.py` (from project root with venv activated)
3. **Start Frontend**: `npm run dev` (from frontend directory)

### Making Changes

- **Backend**: Changes to Python files will auto-reload (uvicorn reload)
- **Frontend**: Next.js fast refresh will hot-reload React components

### Testing

```bash
# Backend tests (from project root)
source .venv/bin/activate
pytest api/tests/

# Frontend linting (from frontend directory)
npm run lint

# Frontend unit tests (from frontend directory)
npm run test
```

## Document Ingestion

You can ingest documents through the UI or CLI:

### Via UI

1. Click on the "Upload" tab in the sidebar
2. Drag and drop or click to select a file
3. Wait for processing to complete

### Via CLI

```bash
# Single file
python scripts/ingest_documents.py --file path/to/document.pdf

# Directory
python scripts/ingest_documents.py --input-dir path/to/documents --recursive
```

## Troubleshooting

### Backend Issues

**Problem**: API won't start

```bash
# Check if port 8000 is in use
lsof -i :8000

# Try a different port
uvicorn api.main:app --port 8001
```

**Problem**: Neo4j connection errors

```bash
# Verify Neo4j is running
docker ps | grep neo4j

# Test connection
python -c "from core.graph_db import graph_db; print(graph_db.driver)"
```

### Frontend Issues

**Problem**: Can't connect to backend

- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Ensure backend is running: `curl http://localhost:8000/api/health`
- Check browser console for CORS errors

**Problem**: Build errors

```bash
# Clear Next.js cache
rm -rf frontend/.next

# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Common Errors

**"Module not found" errors in Python**
```bash
# Ensure venv is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**"Cannot find module" errors in Node.js**
```bash
cd frontend
npm install
```

## Production Deployment

### Backend (Docker)

```bash
# Build Docker image
docker build -t graphrag-backend -f Dockerfile .

# Run container
docker run -d \
    -p 8000:8000 \
    --env-file .env \
    graphrag-backend
```

### Frontend (Vercel)

```bash
cd frontend
vercel --prod
```

Or use the Vercel dashboard to import your GitHub repository.

### Frontend (Docker)

```bash
cd frontend

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
EXPOSE 3000
CMD ["node", "server.js"]
EOF

# Build and run
docker build -t graphrag-frontend .
docker run -d -p 3000:3000 --env-file .env.local graphrag-frontend
```

## Migration from v1.x

If you're migrating from the Streamlit version:

1. **Data is compatible**: Your existing Neo4j database works with v2.0
2. **No data migration needed**: All document chunks and entities are preserved
3. **New features**: Chat history will start fresh (old Streamlit sessions aren't migrated)

To run both versions:

```bash
# Old Streamlit version (different port)
streamlit run app.py --server.port 8501

# New version
# Backend: python api/main.py (port 8000)
# Frontend: npm run dev (port 3000)
```

## Support

For issues or questions:

1. Check the documentation in `/docs`
2. Review API docs at `http://localhost:8000/docs`
3. Check existing GitHub issues
4. Create a new issue with details

## Next Steps

- Upload some documents via the UI
- Try asking questions in the chat
- Explore follow-up questions
- Check chat history
- View database statistics

Enjoy GraphRAG v2.0! 🚀
