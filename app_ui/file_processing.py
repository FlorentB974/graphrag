"""Document ingestion helpers used by the Streamlit UI."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from config.settings import settings
from ingestion.document_processor import document_processor

logger = logging.getLogger(__name__)


def process_files_background(
    uploaded_files: List[Any], progress_container, extract_entities: bool = True
) -> Dict[str, Any]:
    """Process uploaded files with chunk-level progress tracking."""
    results = {
        "processed_files": [],
        "total_chunks": 0,
        "total_entities": 0,
        "total_entity_relationships": 0,
        "errors": [],
    }

    # First, estimate total chunks for progress tracking
    status_text = progress_container.empty()
    status_text.text("Estimating total processing work...")
    total_estimated_chunks = document_processor.estimate_chunks_from_files(uploaded_files)

    # Initialize progress tracking
    progress_bar = progress_container.progress(0)
    total_processed_chunks = 0
    current_file_name = ""

    def chunk_progress_callback(new_chunk_processed):
        """Callback to update progress as chunks are processed."""
        nonlocal total_processed_chunks
        total_processed_chunks += 1  # Increment by 1 for each completed chunk
        progress = min(total_processed_chunks / total_estimated_chunks, 1.0)
        progress_bar.progress(progress)
        # Update status text in real-time with current file and chunk progress
        status_text.text(
            f"Processing {current_file_name}... ({total_processed_chunks}/{total_estimated_chunks} chunks completed)"
        )

    # Phase 1: Process ALL files for chunks only (no entity extraction)
    processed_documents = []  # Store document IDs for batch entity extraction

    for uploaded_file in uploaded_files:
        try:
            current_file_name = uploaded_file.name
            status_text.text(
                f"Processing {uploaded_file.name} chunks... ({total_processed_chunks}/{total_estimated_chunks} chunks completed)"
            )

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                # Process the file with chunks only (disable entity extraction for this phase)
                result = document_processor.process_file_chunks_only(
                    tmp_path, uploaded_file.name, chunk_progress_callback
                )

                if result and result.get("status") == "success":
                    file_info = {
                        "name": uploaded_file.name,
                        "chunks": result.get("chunks_created", 0),
                        "document_id": result.get("document_id"),
                    }

                    results["processed_files"].append(file_info)
                    results["total_chunks"] += result.get("chunks_created", 0)

                    # Store document info for batch entity extraction
                    processed_documents.append(
                        {
                            "document_id": result.get("document_id"),
                            "file_name": uploaded_file.name,
                        }
                    )
                else:
                    results["errors"].append(
                        {
                            "name": uploaded_file.name,
                            "error": (
                                result.get("error", "Unknown error")
                                if result
                                else "Processing failed"
                            ),
                        }
                    )

            finally:
                # Clean up temporary file
                if tmp_path.exists():
                    tmp_path.unlink()

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error processing file %s: %s", uploaded_file.name, exc)
            results["errors"].append({"name": uploaded_file.name, "error": str(exc)})

    # Final progress update for chunk phase
    progress_bar.progress(1.0)
    status_text.text(
        f"Chunk processing complete! Processed {results['total_chunks']} chunks from {len(uploaded_files)} files."
    )

    # Phase 2: Start batch entity extraction for ALL processed documents (if enabled)
    if processed_documents and extract_entities and settings.enable_entity_extraction:
        status_text.text("Starting entity extraction for all files in background...")
        try:
            # Start batch entity extraction in background thread
            entity_stats = document_processor.process_batch_entities(processed_documents)

            # Update results with entity statistics when available
            if entity_stats:
                results["total_entities"] = entity_stats.get("total_entities", 0)
                results["total_entity_relationships"] = entity_stats.get(
                    "total_relationships", 0
                )

                # Update individual file info with entity stats
                for file_info in results["processed_files"]:
                    doc_id = file_info.get("document_id")
                    if doc_id in entity_stats.get("by_document", {}):
                        doc_stats = entity_stats["by_document"][doc_id]
                        file_info["entities"] = doc_stats.get("entities", 0)
                        file_info["entity_relationships"] = doc_stats.get(
                            "relationships", 0
                        )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error in batch entity extraction: %s", exc)
    elif not extract_entities:
        status_text.text("Entity extraction skipped per user preference.")

    return results
