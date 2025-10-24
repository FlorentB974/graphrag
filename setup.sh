#!/bin/bash

# GraphRAG v2.0 Quick Start Script

set -e

echo "🚀 GraphRAG v2.0 Setup"
echo "====================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv .venv
fi

# Activate virtual environment
echo "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install Python dependencies
echo "${GREEN}Installing Python dependencies...${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo "${YELLOW}Creating .env file from example...${NC}"
    cat > .env << EOF
# LLM Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4jpassword

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002

# Application Configuration
LOG_LEVEL=INFO
EOF
    echo "${YELLOW}⚠️  Please edit .env and add your API keys${NC}"
fi

# Setup frontend
echo "${GREEN}Setting up frontend...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    echo "${GREEN}Installing Node.js dependencies...${NC}"
    npm install
fi

if [ ! -f ".env.local" ]; then
    echo "${GREEN}Creating frontend .env.local...${NC}"
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
fi

cd ..

echo ""
echo "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OpenAI API key and Neo4j credentials"
echo "2. Make sure Neo4j is running"
echo "3. Start the backend: python api/main.py"
echo "4. In another terminal, start the frontend: cd frontend && npm run dev"
echo "5. Open http://localhost:3000 in your browser"
echo ""
echo "For detailed instructions, see SETUP_V2.md"
