"""
Pydantic models for API requests and responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""

    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    quality_score: Optional[Dict[str, Any]] = None
    follow_up_questions: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for chat history")
    retrieval_mode: str = Field("hybrid", description="Retrieval mode")
    top_k: int = Field(5, description="Number of chunks to retrieve")
    temperature: float = Field(0.7, description="LLM temperature")
    use_multi_hop: bool = Field(False, description="Enable multi-hop reasoning")
    stream: bool = Field(True, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="Assistant response")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    quality_score: Optional[Dict[str, Any]] = None
    follow_up_questions: List[str] = Field(default_factory=list)
    session_id: str = Field(..., description="Session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FollowUpRequest(BaseModel):
    """Request model for follow-up question generation."""

    query: str = Field(..., description="User query")
    response: str = Field(..., description="Assistant response")
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


class FollowUpResponse(BaseModel):
    """Response model for follow-up questions."""

    questions: List[str] = Field(..., description="Generated follow-up questions")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    filename: str
    status: str
    chunks_created: int
    document_id: Optional[str] = None
    error: Optional[str] = None


class StagedDocument(BaseModel):
    """Model for a staged document waiting to be processed."""

    file_id: str
    filename: str
    file_size: int
    file_path: str
    timestamp: float


class StageDocumentResponse(BaseModel):
    """Response model for staging a document."""

    file_id: str
    filename: str
    status: str
    error: Optional[str] = None


class ProcessProgress(BaseModel):
    """Model for document processing progress."""

    file_id: str
    filename: str
    status: str  # 'processing', 'completed', 'error'
    chunks_processed: int
    total_chunks: int
    progress_percentage: float
    error: Optional[str] = None


class ProcessDocumentsRequest(BaseModel):
    """Request model for processing staged documents."""

    file_ids: List[str] = Field(..., description="List of file IDs to process")


class DatabaseStats(BaseModel):
    """Database statistics model."""

    total_documents: int
    total_chunks: int
    total_entities: int
    total_relationships: int
    documents: List[Dict[str, Any]] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """Chunk information associated with a document."""

    id: str | int
    text: str
    index: int | None = None
    offset: int | None = None
    score: float | None = None


class DocumentEntity(BaseModel):
    """Entity extracted from a document."""

    type: str
    text: str
    count: int | None = None
    positions: List[int] | None = None


class RelatedDocument(BaseModel):
    """Related document link."""

    id: str
    title: str | None = None
    link: str | None = None


class UploaderInfo(BaseModel):
    """Information about the document uploader."""

    id: str | None = None
    name: str | None = None


class DocumentMetadataResponse(BaseModel):
    """Response model for document metadata."""

    id: str
    title: str | None = None
    file_name: str | None = None
    mime_type: str | None = None
    preview_url: str | None = None
    uploaded_at: str | None = None
    uploader: UploaderInfo | None = None
    chunks: List[DocumentChunk] = Field(default_factory=list)
    entities: List[DocumentEntity] = Field(default_factory=list)
    quality_scores: Dict[str, Any] | None = None
    related_documents: List[RelatedDocument] | None = None
    metadata: Dict[str, Any] | None = None


class ConversationSession(BaseModel):
    """Conversation session model."""

    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    preview: Optional[str] = None


class ConversationHistory(BaseModel):
    """Conversation history model."""

    session_id: str
    messages: List[ChatMessage]
    created_at: str
    updated_at: str
