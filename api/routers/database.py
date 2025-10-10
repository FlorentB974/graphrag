"""
Database router for managing documents and database operations.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks

from api.models import (
    DatabaseStats,
    DocumentUploadResponse,
    StagedDocument,
    StageDocumentResponse,
    ProcessProgress,
    ProcessDocumentsRequest,
)
from core.graph_db import graph_db
from ingestion.document_processor import document_processor

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for staged documents and processing progress
# In production, you'd use Redis or a database
_staged_documents: Dict[str, StagedDocument] = {}
_processing_progress: Dict[str, ProcessProgress] = {}


@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """
    Get database statistics including document and chunk counts.

    Returns:
        Database statistics
    """
    try:
        stats = graph_db.get_database_stats()

        return DatabaseStats(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            total_entities=stats.get("total_entities", 0),
            total_relationships=stats.get("total_relationships", 0),
            documents=stats.get("documents", []),
        )

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.

    Args:
        file: Uploaded file

    Returns:
        Upload response with processing results
    """
    filename = file.filename or "unknown"
    
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"/tmp/{filename}")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the file in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            document_processor.process_file,
            temp_path
        )

        # Clean up temp file
        try:
            temp_path.unlink()
        except Exception:
            pass  # Ignore cleanup errors

        if result and result.get("status") == "success":
            return DocumentUploadResponse(
                filename=filename,
                status="success",
                chunks_created=result.get("chunks_created", 0),
                document_id=result.get("document_id"),
            )
        else:
            error_msg = result.get("error", "Unknown error") if result else "Processing failed"
            return DocumentUploadResponse(
                filename=filename,
                status="error",
                chunks_created=0,
                error=error_msg,
            )

    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        return DocumentUploadResponse(
            filename=filename,
            status="error",
            chunks_created=0,
            error=str(e),
        )


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks.

    Args:
        document_id: Document ID to delete

    Returns:
        Deletion result
    """
    try:
        graph_db.delete_document(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted"}

    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_database():
    """
    Clear all data from the database.

    Returns:
        Clear operation result
    """
    try:
        graph_db.clear_database()
        return {"status": "success", "message": "Database cleared"}

    except Exception as e:
        logger.error(f"Failed to clear database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents():
    """
    List all documents in the database.

    Returns:
        List of documents with metadata
    """
    try:
        documents = graph_db.list_documents()
        return {"documents": documents}

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage", response_model=StageDocumentResponse)
async def stage_document(file: UploadFile = File(...)):
    """
    Stage a document for later processing.

    Args:
        file: Uploaded file

    Returns:
        Staged document information
    """
    filename = file.filename or "unknown"

    try:
        # Generate unique file ID
        file_id = hashlib.md5(
            f"{filename}_{time.time()}".encode()
        ).hexdigest()[:16]

        # Save file to staging area
        staging_dir = Path("data/staged_uploads")
        staging_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = staging_dir / f"{file_id}_{filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Store staged document info
        staged_doc = StagedDocument(
            file_id=file_id,
            filename=filename,
            file_size=len(content),
            file_path=str(file_path),
            timestamp=time.time()
        )
        _staged_documents[file_id] = staged_doc

        return StageDocumentResponse(
            file_id=file_id,
            filename=filename,
            status="staged"
        )

    except Exception as e:
        logger.error(f"Failed to stage document: {e}")
        return StageDocumentResponse(
            file_id="",
            filename=filename,
            status="error",
            error=str(e)
        )


@router.get("/staged")
async def list_staged_documents():
    """
    List all staged documents.

    Returns:
        List of staged documents
    """
    return {"documents": list(_staged_documents.values())}


@router.delete("/staged/{file_id}")
async def delete_staged_document(file_id: str):
    """
    Delete a staged document.

    Args:
        file_id: File ID to delete

    Returns:
        Deletion result
    """
    try:
        if file_id not in _staged_documents:
            raise HTTPException(status_code=404, detail="Document not found")

        staged_doc = _staged_documents[file_id]
        
        # Delete file
        file_path = Path(staged_doc.file_path)
        if file_path.exists():
            file_path.unlink()

        # Remove from staged list
        del _staged_documents[file_id]

        # Remove from processing progress if exists
        if file_id in _processing_progress:
            del _processing_progress[file_id]

        return {"status": "success", "message": f"Staged document {file_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete staged document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process")
async def process_documents(request: ProcessDocumentsRequest, background_tasks: BackgroundTasks):
    """
    Process staged documents in the background.

    Args:
        request: List of file IDs to process
        background_tasks: FastAPI background tasks

    Returns:
        Processing status
    """
    try:
        # Validate file IDs
        invalid_ids = [fid for fid in request.file_ids if fid not in _staged_documents]
        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file IDs: {invalid_ids}"
            )

        # Initialize progress tracking
        for file_id in request.file_ids:
            staged_doc = _staged_documents[file_id]
            _processing_progress[file_id] = ProcessProgress(
                file_id=file_id,
                filename=staged_doc.filename,
                status="processing",
                chunks_processed=0,
                total_chunks=0,
                progress_percentage=0.0
            )

        # Start processing in background
        background_tasks.add_task(_process_documents_task, request.file_ids)

        return {
            "status": "processing_started",
            "message": f"Processing {len(request.file_ids)} documents",
            "file_ids": request.file_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/progress/{file_id}")
async def get_processing_progress(file_id: str):
    """
    Get processing progress for a specific file.

    Args:
        file_id: File ID

    Returns:
        Processing progress
    """
    if file_id not in _processing_progress:
        raise HTTPException(status_code=404, detail="Progress not found")

    return _processing_progress[file_id]


@router.get("/progress")
async def get_all_processing_progress():
    """
    Get processing progress for all files.

    Returns:
        List of processing progress
    """
    return {"progress": list(_processing_progress.values())}


async def _process_documents_task(file_ids: List[str]):
    """
    Background task to process documents.

    Args:
        file_ids: List of file IDs to process
    """
    for file_id in file_ids:
        try:
            if file_id not in _staged_documents:
                logger.error(f"File ID {file_id} not found in staged documents")
                continue

            staged_doc = _staged_documents[file_id]
            file_path = Path(staged_doc.file_path)

            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                _processing_progress[file_id].status = "error"
                _processing_progress[file_id].error = "File not found"
                continue

            # Estimate total chunks first
            logger.info(f"Processing file: {file_path}")
            
            # Create a progress callback
            def progress_callback(chunks_processed: int):
                if file_id in _processing_progress:
                    progress = _processing_progress[file_id]
                    progress.chunks_processed = chunks_processed
                    if progress.total_chunks > 0:
                        progress.progress_percentage = (
                            chunks_processed / progress.total_chunks
                        ) * 100

            # Get file extension and load content to estimate chunks
            from core.chunking import document_chunker
            file_ext = file_path.suffix.lower()
            loader = document_processor.loaders.get(file_ext)
            
            if loader:
                # Load content to estimate chunks
                from ingestion.loaders.image_loader import ImageLoader
                from ingestion.loaders.pdf_loader import PDFLoader
                
                if isinstance(loader, ImageLoader):
                    result = loader.load_with_metadata(file_path)
                    content = result["content"] if result else ""
                elif isinstance(loader, PDFLoader):
                    result = loader.load_with_metadata(file_path)
                    content = result["content"] if result else ""
                else:
                    content = loader.load(file_path)
                
                if content:
                    # Estimate total chunks
                    temp_chunks = document_chunker.chunk_text(
                        content, f"temp_{file_id}"
                    )
                    _processing_progress[file_id].total_chunks = len(temp_chunks)
                    logger.info(
                        f"Estimated {len(temp_chunks)} chunks for {staged_doc.filename}"
                    )

            # Process the file
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                document_processor.process_file,
                file_path,
                staged_doc.filename,
                progress_callback
            )

            if result and result.get("status") == "success":
                _processing_progress[file_id].status = "completed"
                _processing_progress[file_id].chunks_processed = result.get(
                    "chunks_created", 0
                )
                _processing_progress[file_id].total_chunks = result.get(
                    "chunks_created", 0
                )
                _processing_progress[file_id].progress_percentage = 100.0

                # Remove from staged documents
                del _staged_documents[file_id]

                # Clean up file
                try:
                    file_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file: {cleanup_error}")

                logger.info(
                    f"Successfully processed {staged_doc.filename}: "
                    f"{result.get('chunks_created', 0)} chunks"
                )
            else:
                error_msg = result.get("error", "Processing failed") if result else "Processing failed"
                _processing_progress[file_id].status = "error"
                _processing_progress[file_id].error = error_msg
                logger.error(f"Failed to process {staged_doc.filename}: {error_msg}")

        except Exception as e:
            logger.error(f"Error processing file {file_id}: {e}")
            if file_id in _processing_progress:
                _processing_progress[file_id].status = "error"
                _processing_progress[file_id].error = str(e)

    # Clean up completed/error progress after a delay
    await asyncio.sleep(5)
    for file_id in list(_processing_progress.keys()):
        if _processing_progress[file_id].status in ["completed", "error"]:
            del _processing_progress[file_id]
