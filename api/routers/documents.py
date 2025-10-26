"""Document metadata and preview routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

from api.models import DocumentMetadataResponse
from core.document_summarizer import document_summarizer
from core.graph_db import graph_db

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{document_id}", response_model=DocumentMetadataResponse)
async def get_document_metadata(document_id: str) -> DocumentMetadataResponse:
    """Return document metadata and related analytics."""
    try:
        details = graph_db.get_document_details(document_id)
        return DocumentMetadataResponse(**details)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load document %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to retrieve document") from exc


@router.post("/{document_id}/generate-summary")
async def generate_document_summary(document_id: str):
    """Generate or regenerate summary for a document."""
    try:
        # Get document chunks
        chunks = graph_db.get_document_chunks(document_id)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Document has no chunks. Please process chunks first."
            )

        # Extract summary
        summary_data = document_summarizer.extract_summary(chunks)

        # Update document with summary
        graph_db.update_document_summary(
            doc_id=document_id,
            summary=summary_data.get("summary", ""),
            document_type=summary_data.get("document_type", "other"),
            hashtags=summary_data.get("hashtags", [])
        )

        return {
            "document_id": document_id,
            "summary": summary_data.get("summary", ""),
            "document_type": summary_data.get("document_type", "other"),
            "hashtags": summary_data.get("hashtags", []),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate summary for %s: %s", document_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary"
        ) from exc


@router.get("/{document_id}/preview")
async def get_document_preview(document_id: str):
    """Stream the document file or redirect to an existing preview URL."""
    try:
        info = graph_db.get_document_file_info(document_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Document not found") from None
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load preview info for %s: %s", document_id, exc)
        raise HTTPException(status_code=500, detail="Failed to prepare preview") from exc

    preview_url = info.get("preview_url")
    if preview_url:
        return RedirectResponse(preview_url)

    file_path = info.get("file_path")
    if not file_path:
        raise HTTPException(status_code=404, detail="Preview not available")

    path = Path(file_path)
    # If path is relative, make it absolute relative to the project root
    if not path.is_absolute():
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # Go up from api/routers/ to project root
        path = project_root / path

    if not path.exists() or not path.is_file():
        logger.error(f"File not found at path: {path}")
        raise HTTPException(status_code=404, detail="Preview not available")

    media_type = info.get("mime_type") or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=info.get("file_name"))
