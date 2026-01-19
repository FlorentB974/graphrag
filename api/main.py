"""
FastAPI backend for GraphRAG chat application.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, database, documents, history
from config.settings import settings

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application."""
    logger.info("Starting GraphRAG API...")
    yield
    logger.info("Shutting down GraphRAG API...")


app = FastAPI(
    title="GraphRAG API",
    description="Backend API for GraphRAG chat application",
    version="2.0.0",
    lifespan=lifespan,
)


# Configure CORS - origins loaded from ALLOWED_ORIGINS environment variable
# Supports comma-separated list or '*' for development
def _parse_cors_origins(origins_str: str) -> list[str]:
    """Parse CORS origins from comma-separated string."""
    if origins_str.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(database.router, prefix="/api/database", tags=["database"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(history.router, prefix="/api/history", tags=["history"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "llm_provider": settings.llm_provider,
        "enable_entity_extraction": settings.enable_entity_extraction,
        "enable_quality_scoring": settings.enable_quality_scoring,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "GraphRAG API",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
