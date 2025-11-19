"""
Database router for managing documents and database operations.
"""

import asyncio
import hashlib
import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.models import (
    DatabaseStats,
    DocumentUploadResponse,
    StagedDocument,
    StageDocumentResponse,
    ProcessProgress,
    ProcessDocumentsRequest,
    ProcessingSummary,
)
from config.settings import settings
from core.graph_db import graph_db
from ingestion.document_processor import EntityExtractionState, document_processor

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for staged documents and processing progress
# In production, you'd use Redis or a database
_staged_documents: Dict[str, StagedDocument] = {}
_processing_progress: Dict[str, ProcessProgress] = {}

_processing_queue: List[str] = []
_queue_lock = asyncio.Lock()
_processing_lock = asyncio.Lock()
_queue_event = asyncio.Event()
_processing_worker: Optional[asyncio.Task] = None  # type: ignore[var-annotated]
_global_processing_state: Dict[str, Any] = {
    "is_processing": False,
    "current_file_id": None,
    "current_document_id": None,
    "current_filename": None,
    "current_stage": None,
    "progress_percentage": 0.0,
    "queue_length": 0,
}

_ENTITY_STAGE_FRACTIONS = {
    EntityExtractionState.STARTING: 0.05,
    EntityExtractionState.LLM_EXTRACTION: 0.25,
    EntityExtractionState.EMBEDDING_GENERATION: 0.55,
    EntityExtractionState.DATABASE_OPERATIONS: 0.8,
    EntityExtractionState.VALIDATION: 0.95,
    EntityExtractionState.COMPLETED: 1.0,
    EntityExtractionState.ERROR: 1.0,
}


def _calculate_overall_progress(progress: ProcessProgress) -> float:
    """Compute overall progress percentage blending chunk/entity phases."""

    mode = progress.mode or "full"
    chunk_weight = 0.7 if mode == "full" else (1.0 if mode == "chunks_only" else 0.0)
    entity_weight = 0.3 if mode == "full" else (1.0 if mode == "entities_only" else 0.0)

    chunk_fraction = progress.chunk_progress or 0.0
    entity_fraction = progress.entity_progress or 0.0

    total = 0.0
    if chunk_weight:
        total += chunk_fraction * chunk_weight
    if entity_weight:
        total += entity_fraction * entity_weight

    return max(0.0, min(100.0, total * 100.0))


def _update_queue_positions() -> None:
    """Update queue position hints for pending progress entries."""

    for idx, file_id in enumerate(_processing_queue):
        progress = _processing_progress.get(file_id)
        if progress:
            progress.queue_position = idx

    _global_processing_state["queue_length"] = len(_processing_queue) + (
        1 if _global_processing_state.get("is_processing") else 0
    )


def _mark_document_status(doc_id: str, status: str, stage: str, progress_value: float) -> None:
    """Persist processing status directly on the document node."""

    try:
        graph_db.create_document_node(
            doc_id,
            {
                "processing_status": status,
                "processing_stage": stage,
                "processing_progress": progress_value,
                "processing_updated_at": time.time(),
            },
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("Failed to update processing metadata for %s: %s", doc_id, exc)


async def _ensure_worker_running() -> None:
    """Guarantee that the background worker is alive to process the queue."""

    global _processing_worker

    if _processing_worker is None or _processing_worker.done():
        loop = asyncio.get_running_loop()
        _processing_worker = loop.create_task(_processing_worker_loop())


async def _processing_worker_loop() -> None:
    """Continuously drain the processing queue in FIFO order."""

    while True:
        await _queue_event.wait()

        while True:
            async with _queue_lock:
                if not _processing_queue:
                    _queue_event.clear()
                    break
                file_id = _processing_queue[0]

            await _process_single_job(file_id)

        if not _queue_event.is_set():
            break


def _estimate_total_chunks_for_path(file_path: Path) -> int:
    """Best-effort chunk count estimation to drive progress UI."""

    try:
        if not file_path.exists():
            return 0

        from core.chunking import document_chunker  # Local import to avoid cycles
        from ingestion.loaders.image_loader import ImageLoader
        from ingestion.loaders.marker_pdf_loader import MarkerPdfLoader

        loader = document_processor.loaders.get(file_path.suffix.lower())
        if not loader:
            return 0

        if isinstance(loader, ImageLoader):
            result = loader.load_with_metadata(file_path)
            content = result.get("content") if result else ""
        elif isinstance(loader, MarkerPdfLoader):
            result = loader.load_with_metadata(file_path)
            content = result.get("content") if result else ""
        else:
            content = loader.load(file_path)

        if not content:
            return 0

        temp_chunks = document_chunker.chunk_text(content, f"preview_{file_path.stem}")
        return len(temp_chunks)
    except Exception as exc:  # pragma: no cover - advisory logging
        logger.debug("Failed to estimate chunks for %s: %s", file_path, exc)
        return 0


async def _process_single_job(file_id: str) -> None:
    """Process a single staged job while enforcing sequential execution."""

    async with _processing_lock:
        staged_doc = _staged_documents.get(file_id)
        progress = _processing_progress.get(file_id)

        if staged_doc is None:
            async with _queue_lock:
                if file_id in _processing_queue:
                    _processing_queue.remove(file_id)
                    _update_queue_positions()
            if progress:
                progress.status = "error"
                progress.error = "Staged document not found"
                progress.progress_percentage = 0.0
            return

        if progress is None:
            progress = ProcessProgress(
                file_id=file_id,
                document_id=staged_doc.document_id,
                filename=staged_doc.filename,
                status="processing",
                stage="chunking" if staged_doc.mode != "entities_only" else "entity_extraction",
                mode=staged_doc.mode,
                queue_position=0,
                chunks_processed=0,
                total_chunks=0,
                chunk_progress=0.0,
                entity_progress=0.0,
                progress_percentage=0.0,
                entity_state=None,
            )
            _processing_progress[file_id] = progress

        async with _queue_lock:
            if _processing_queue and _processing_queue[0] == file_id:
                _processing_queue.pop(0)
            elif file_id in _processing_queue:
                _processing_queue.remove(file_id)
            _update_queue_positions()

        file_path = Path(staged_doc.file_path)
        doc_id = staged_doc.document_id
        if not doc_id and file_path:
            doc_id = document_processor.compute_document_id(file_path)
            staged_doc.document_id = doc_id
        progress.document_id = doc_id

        stage_name = "entity_extraction" if staged_doc.mode == "entities_only" else "chunking"
        progress.stage = stage_name
        progress.status = "processing"
        progress.error = None
        progress.queue_position = 0

        _global_processing_state.update(
            {
                "is_processing": True,
                "current_file_id": file_id,
                "current_document_id": doc_id,
                "current_filename": staged_doc.filename,
                "current_stage": stage_name,
                "progress_percentage": progress.progress_percentage,
                "queue_length": len(_processing_queue) + 1,
            }
        )

        if doc_id:
            _mark_document_status(doc_id, "processing", stage_name, progress.progress_percentage)

        success = False
        try:
            if staged_doc.mode == "entities_only":
                success = await _run_entity_job(file_id, staged_doc, progress)
            else:
                success = await _run_chunk_job(file_id, staged_doc, progress)

                if success and staged_doc.mode != "chunks_only":
                    progress.stage = "entity_extraction"
                    _global_processing_state["current_stage"] = "entity_extraction"
                    if doc_id:
                        _mark_document_status(doc_id, "processing", "entity_extraction", progress.progress_percentage)
                    success = await _run_entity_job(file_id, staged_doc, progress)

        finally:
            if success:
                progress.status = "completed"
                progress.stage = "completed"
                if staged_doc.mode in {"full", "chunks_only"}:
                    progress.chunk_progress = 1.0
                if staged_doc.mode in {"full", "entities_only"}:
                    progress.entity_progress = 1.0 if progress.entity_progress not in (None, 0.0) else progress.entity_progress
                progress.progress_percentage = 100.0
                if doc_id:
                    _mark_document_status(doc_id, "completed", "completed", 100.0)
            else:
                if progress.status != "error":
                    progress.status = "error"
                    if not progress.error:
                        progress.error = "Processing failed"
                progress.progress_percentage = min(progress.progress_percentage, 99.0)
                if doc_id:
                    _mark_document_status(doc_id, "error", progress.stage or "error", progress.progress_percentage)

            if staged_doc.mode in {"full", "chunks_only", "entities_only"}:
                _staged_documents.pop(file_id, None)

            _global_processing_state.update(
                {
                    "is_processing": False,
                    "current_file_id": None,
                    "current_document_id": None,
                    "current_filename": None,
                    "current_stage": None,
                    "progress_percentage": 0.0,
                    "queue_length": len(_processing_queue),
                }
            )

            if progress.status in {"completed", "error"}:
                asyncio.create_task(_cleanup_progress_entry(file_id))


async def _run_chunk_job(
    file_id: str, staged_doc: StagedDocument, progress: ProcessProgress
) -> bool:
    """Process chunk extraction/embedding for a staged document."""

    file_path = Path(staged_doc.file_path)
    if not file_path.exists():
        progress.status = "error"
        progress.error = "File not found"
        progress.stage = "error"
        return False

    doc_id = staged_doc.document_id or document_processor.compute_document_id(file_path)
    staged_doc.document_id = doc_id
    progress.document_id = doc_id

    loop = asyncio.get_running_loop()
    estimated_chunks = await loop.run_in_executor(
        None, partial(_estimate_total_chunks_for_path, file_path)
    )
    if estimated_chunks:
        progress.total_chunks = estimated_chunks

    def _chunk_progress_callback(chunks_processed: int) -> None:
        progress.chunks_processed = chunks_processed
        if progress.total_chunks:
            progress.chunk_progress = min(1.0, chunks_processed / progress.total_chunks)
        else:
            progress.chunk_progress = 0.0 if chunks_processed == 0 else 0.9
        progress.progress_percentage = _calculate_overall_progress(progress)
        _global_processing_state["progress_percentage"] = progress.progress_percentage

    process_fn = partial(
        document_processor.process_file_chunks_only,
        file_path,
        staged_doc.filename,
        _chunk_progress_callback,
        None,
        doc_id,
    )

    result = await loop.run_in_executor(None, process_fn)

    if not result or result.get("status") != "success":
        progress.error = (
            result.get("error") if isinstance(result, dict) else "Failed to process document"
        )
        progress.status = "error"
        return False

    created_chunks = result.get("chunks_created", 0)
    progress.chunks_processed = created_chunks
    progress.total_chunks = created_chunks or progress.total_chunks
    progress.chunk_progress = 1.0 if created_chunks else progress.chunk_progress
    progress.progress_percentage = _calculate_overall_progress(progress)
    _global_processing_state["progress_percentage"] = progress.progress_percentage

    return True


async def _run_entity_job(
    file_id: str, staged_doc: StagedDocument, progress: ProcessProgress
) -> bool:
    """Run entity extraction for a document and update progress accordingly."""

    doc_id = progress.document_id
    if not doc_id:
        progress.error = "Unknown document identifier for entity extraction"
        progress.status = "error"
        progress.stage = "error"
        return False

    loop = asyncio.get_running_loop()

    def _entity_progress_callback(
        state: EntityExtractionState, fraction: float, info: Optional[str]
    ) -> None:
        progress.entity_state = state.value
        progress.entity_progress = _ENTITY_STAGE_FRACTIONS.get(state, fraction)
        progress.progress_percentage = _calculate_overall_progress(progress)
        _global_processing_state["progress_percentage"] = progress.progress_percentage

    entity_fn = partial(
        document_processor.extract_entities_for_document,
        doc_id,
        staged_doc.filename,
        _entity_progress_callback,
    )

    result = await loop.run_in_executor(None, entity_fn)

    status = result.get("status") if isinstance(result, dict) else "error"

    if status not in {"success", "skipped"}:
        progress.status = "error"
        progress.error = result.get("error") if isinstance(result, dict) else "Entity extraction failed"
        return False

    if status in {"success", "skipped"}:
        progress.entity_progress = 1.0
    progress.progress_percentage = _calculate_overall_progress(progress)
    _global_processing_state["progress_percentage"] = progress.progress_percentage

    return True


async def _cleanup_progress_entry(file_id: str, delay: float = 5.0) -> None:
    """Remove progress records after a cooldown to keep UI responsive."""

    try:
        await asyncio.sleep(delay)
        _processing_progress.pop(file_id, None)
    except Exception:  # pragma: no cover - defensive
        pass


async def _enqueue_processing_jobs(file_ids: List[str]) -> List[str]:
    """Normalize progress entries and add jobs to the processing queue."""

    enqueued: List[str] = []

    async with _queue_lock:
        for file_id in file_ids:
            staged_doc = _staged_documents[file_id]

            if file_id not in _processing_progress:
                _processing_progress[file_id] = ProcessProgress(
                    file_id=file_id,
                    document_id=staged_doc.document_id,
                    filename=staged_doc.filename,
                    status="queued",
                    stage="queued",
                    mode=staged_doc.mode,
                    queue_position=None,
                    chunks_processed=0,
                    total_chunks=0,
                    chunk_progress=0.0,
                    entity_progress=0.0,
                    progress_percentage=0.0,
                    entity_state=None,
                )
            else:
                progress = _processing_progress[file_id]
                progress.status = "queued"
                progress.stage = "queued"
                progress.mode = staged_doc.mode
                progress.chunks_processed = 0
                progress.total_chunks = 0
                progress.chunk_progress = 0.0
                progress.entity_progress = 0.0
                progress.progress_percentage = 0.0
                progress.error = None
                progress.entity_state = None

            if (
                file_id not in _processing_queue
                and _global_processing_state.get("current_file_id") != file_id
            ):
                _processing_queue.append(file_id)
                enqueued.append(file_id)

            if staged_doc.document_id:
                _mark_document_status(
                    staged_doc.document_id,
                    "queued",
                    "queued",
                    0.0,
                )

        _update_queue_positions()
        if _processing_queue:
            _queue_event.set()

    if enqueued:
        await _ensure_worker_running()

    return enqueued


def _is_document_processing(document_id: Optional[str]) -> bool:
    """Check if a document already has an active or pending processing job."""

    if not document_id:
        return False

    for progress in _processing_progress.values():
        if (
            progress.document_id == document_id
            and progress.status in {"queued", "processing"}
        ):
            return True

    return _global_processing_state.get("current_document_id") == document_id


@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """
    Get database statistics including document and chunk counts.

    Returns:
        Database statistics
    """
    try:
        stats = graph_db.get_database_stats()

        documents = stats.get("documents", [])

        # Enrich documents with live queue state
        for doc in documents:
            doc_id = doc.get("document_id")
            progress_match = None

            for progress in _processing_progress.values():
                if progress.document_id == doc_id:
                    progress_match = progress
                    break

            if progress_match:
                doc["processing_status"] = progress_match.status
                doc["processing_stage"] = progress_match.stage
                doc["processing_progress"] = progress_match.progress_percentage
                doc["queue_position"] = progress_match.queue_position
            else:
                doc.setdefault("processing_status", doc.get("processing_status", "idle"))
                doc.setdefault("processing_stage", doc.get("processing_stage", "idle"))
                doc.setdefault("processing_progress", doc.get("processing_progress", 0.0))

        active_progress = [
            progress
            for progress in _processing_progress.values()
            if progress.status in {"queued", "processing"}
        ]

        processing_summary = ProcessingSummary(
            is_processing=_global_processing_state.get("is_processing", False),
            current_file_id=_global_processing_state.get("current_file_id"),
            current_document_id=_global_processing_state.get("current_document_id"),
            current_filename=_global_processing_state.get("current_filename"),
            current_stage=_global_processing_state.get("current_stage"),
            progress_percentage=_global_processing_state.get("progress_percentage"),
            queue_length=_global_processing_state.get("queue_length", 0),
            pending_documents=active_progress,
        )

        return DatabaseStats(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            total_entities=stats.get("total_entities", 0),
            total_relationships=stats.get("total_relationships", 0),
            documents=documents,
            processing=processing_summary,
            enable_delete_operations=settings.enable_delete_operations,
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
        # Generate a unique file ID to avoid conflicts
        file_id = hashlib.md5(f"{filename}_{time.time()}".encode()).hexdigest()[:16]
        
        # Save uploaded file to staged_uploads directory (not /tmp) so it persists for preview
        staged_dir = Path("data/staged_uploads")
        staged_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the same naming pattern as stage endpoint for consistency
        safe_filename = filename.replace(" ", " ")  # Keep spaces for now
        file_path = staged_dir / f"{file_id}_{safe_filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Process the file in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            document_processor.process_file,
            file_path,
            filename  # Pass original filename
        )

        # DO NOT delete the file - we need it for previews
        # The file path is stored in the database and used by the preview endpoint

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
    if not settings.enable_delete_operations:
        raise HTTPException(
            status_code=403,
            detail="Delete operations are disabled in this configuration"
        )
    
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
    if not settings.enable_delete_operations:
        raise HTTPException(
            status_code=403,
            detail="Delete operations are disabled in this configuration"
        )
    
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


@router.get("/hashtags")
async def list_hashtags():
    """
    List all unique hashtags from all documents.

    Returns:
        List of hashtags
    """
    try:
        hashtags = graph_db.get_all_hashtags()
        return {"hashtags": hashtags}

    except Exception as e:
        logger.error(f"Failed to list hashtags: {e}")
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

        document_id = document_processor.compute_document_id(file_path)

        metadata = document_processor.build_metadata(file_path, filename)
        metadata.setdefault("uploaded_at", time.time())
        metadata["original_filename"] = filename
        metadata["processing_status"] = "staged"
        metadata["processing_stage"] = "staged"
        metadata["processing_progress"] = 0.0
        metadata["processing_updated_at"] = time.time()

        try:
            graph_db.create_document_node(document_id, metadata)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning(
                "Failed to create placeholder document node for %s: %s",
                document_id,
                exc,
            )

        # Store staged document info
        staged_doc = StagedDocument(
            file_id=file_id,
            filename=filename,
            file_size=len(content),
            file_path=str(file_path),
            timestamp=time.time(),
            document_id=document_id,
            mode="full",
        )
        _staged_documents[file_id] = staged_doc

        # Automatically enqueue for processing
        enqueued = await _enqueue_processing_jobs([file_id])
        
        return StageDocumentResponse(
            file_id=file_id,
            filename=filename,
            document_id=document_id,
            status="queued" if enqueued else "staged"
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

        # Remove from processing queue if exists
        async with _queue_lock:
            if file_id in _processing_queue:
                _processing_queue.remove(file_id)
                _update_queue_positions()

        return {"status": "success", "message": f"Staged document {file_id} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete staged document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process")
async def process_documents(request: ProcessDocumentsRequest):
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

        enqueued = await _enqueue_processing_jobs(request.file_ids)

        return {
            "status": "queued" if enqueued else "noop",
            "message": f"Queued {len(enqueued)} documents for processing",
            "file_ids": enqueued,
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
    progress_list = list(_processing_progress.values())
    pending_progress = [
        progress
        for progress in progress_list
        if progress.status in {"queued", "processing"}
    ]

    return {
        "progress": progress_list,
        "global": {
            **_global_processing_state,
            "pending_documents": pending_progress,
        },
    }


@router.post("/documents/{document_id}/process/chunks")
async def reprocess_document_chunks(document_id: str):
    """Queue full document processing for an existing document lacking chunks."""

    if _is_document_processing(document_id):
        raise HTTPException(status_code=409, detail="Document is already processing")

    try:
        file_info = graph_db.get_document_file_info(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc

    file_path = file_info.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Document file path missing")

    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Document file not available on server")

    filename = file_info.get("file_name") or path.name
    file_size = path.stat().st_size if path.exists() else 0

    file_id = hashlib.md5(f"{document_id}_{time.time()}".encode()).hexdigest()[:16]

    staged_doc = StagedDocument(
        file_id=file_id,
        filename=filename,
        file_size=file_size,
        file_path=str(path),
        timestamp=time.time(),
        document_id=document_id,
        mode="full",
    )
    _staged_documents[file_id] = staged_doc

    _mark_document_status(document_id, "queued", "queued", 0.0)

    enqueued = await _enqueue_processing_jobs([file_id])

    return {
        "status": "queued" if enqueued else "noop",
        "file_id": file_id,
        "document_id": document_id,
    }


@router.post("/documents/{document_id}/process/entities")
async def reprocess_document_entities(document_id: str):
    """Queue entity extraction for an existing document that lacks entities."""

    if _is_document_processing(document_id):
        raise HTTPException(status_code=409, detail="Document is already processing")

    try:
        details = graph_db.get_document_details(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc

    if not details.get("chunks"):
        raise HTTPException(
            status_code=400,
            detail="Document has no chunks. Re-run chunk processing first.",
        )

    file_info = graph_db.get_document_file_info(document_id)
    file_path = file_info.get("file_path")
    path = Path(file_path) if file_path else None

    filename = details.get("file_name") or file_info.get("file_name") or (
        path.name if path else document_id
    )
    file_size = path.stat().st_size if path and path.exists() else 0

    file_id = hashlib.md5(f"entities_{document_id}_{time.time()}".encode()).hexdigest()[:16]

    staged_doc = StagedDocument(
        file_id=file_id,
        filename=filename,
        file_size=file_size,
        file_path=str(path) if path else "",
        timestamp=time.time(),
        document_id=document_id,
        mode="entities_only",
    )
    _staged_documents[file_id] = staged_doc

    _mark_document_status(document_id, "queued", "entity_extraction", 0.0)

    enqueued = await _enqueue_processing_jobs([file_id])

    return {
        "status": "queued" if enqueued else "noop",
        "file_id": file_id,
        "document_id": document_id,
    }


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
                from ingestion.loaders.marker_pdf_loader import MarkerPdfLoader
                
                if isinstance(loader, ImageLoader):
                    result = loader.load_with_metadata(file_path)
                    content = result["content"] if result else ""
                elif isinstance(loader, MarkerPdfLoader):
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

                # DO NOT delete the file - we need it for previews
                # The file path is stored in the database and used by the preview endpoint
                # If disk space becomes an issue, implement a proper file retention policy
                logger.info(
                    f"Successfully processed {staged_doc.filename}: "
                    f"{result.get('chunks_created', 0)} chunks, file retained for preview"
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
