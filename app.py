"""
Streamlit web interface for the GraphRAG pipeline.
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Generator, List

import streamlit as st

from config.settings import settings
from core.graph_db import graph_db
from ingestion.document_processor import document_processor
from rag.graph_rag import graph_rag

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="GraphRAG Pipeline",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing_files" not in st.session_state:
    st.session_state.processing_files = False
if "show_graph" not in st.session_state:
    st.session_state.show_graph = False
if "file_uploader_key" not in st.session_state:
    st.session_state.file_uploader_key = 0


def stream_response(text: str, delay: float = 0.02) -> Generator[str, None, None]:
    """
    Stream text response word by word.

    Args:
        text: The text to stream
        delay: Delay between words in seconds

    Yields:
        Progressive text content
    """
    for words in text.split(" "):
        yield words + " "
        time.sleep(delay)


def process_files_background(
    uploaded_files: List[Any], progress_container, extract_entities: bool = True
) -> Dict[str, Any]:
    """
    Process uploaded files in background with chunk-level progress tracking.

    First processes ALL files for chunks, then does entity extraction for ALL files together.
    This prevents rate limiting issues when uploading multiple files.

    Args:
        uploaded_files: List of uploaded file objects
        progress_container: Streamlit container for progress updates
        extract_entities: Whether to run entity extraction in background

    Returns:
        Dictionary with processing results
    """
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
    total_estimated_chunks = document_processor.estimate_chunks_from_files(
        uploaded_files
    )

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

    for i, uploaded_file in enumerate(uploaded_files):
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

        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            results["errors"].append({"name": uploaded_file.name, "error": str(e)})

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
            entity_stats = document_processor.process_batch_entities(
                processed_documents
            )

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

        except Exception as e:
            logger.error(f"Error in batch entity extraction: {e}")
    elif not extract_entities:
        status_text.text("Entity extraction skipped per user preference.")

    return results


def display_stats():
    """Display database statistics in sidebar."""
    try:
        stats = graph_db.get_graph_stats()

        st.markdown("### 📊 Database Stats")

        # Main metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get("documents", 0))
            st.metric("Chunks", stats.get("chunks", 0))
            st.metric("Entities", stats.get("entities", 0))

        with col2:
            st.metric("Chunk Relations", stats.get("similarity_relations", 0))
            st.metric("Entity Relations", stats.get("entity_relations", 0))
            st.metric("Chunk-Entity Links", stats.get("chunk_entity_relations", 0))

        # Show entity extraction status - check if running first (global indicator)
        try:
            is_extraction_running = document_processor.is_entity_extraction_running()
            if is_extraction_running:
                # Show running indicator regardless of current entity count
                if stats.get("entities", 0) > 0:
                    entity_coverage = (
                        stats.get("chunk_entity_relations", 0)
                        / max(stats.get("chunks", 1), 1)
                    ) * 100
                    st.caption(
                        f"🔄 Entity extraction running in background — updating database ({entity_coverage:.1f}% chunk coverage)"
                    )
                else:
                    st.caption(
                        "🔄 Entity extraction running in background — processing documents..."
                    )
            elif stats.get("entities", 0) > 0:
                entity_coverage = (
                    stats.get("chunk_entity_relations", 0)
                    / max(stats.get("chunks", 1), 1)
                ) * 100
                st.caption(
                    f"✅ Entities extracted ({entity_coverage:.1f}% chunk coverage)"
                )
            else:
                st.caption("⚠️ No entities extracted yet")
        except Exception:
            # Fallback to default caption if detection fails
            if stats.get("entities", 0) > 0:
                entity_coverage = (
                    stats.get("chunk_entity_relations", 0)
                    / max(stats.get("chunks", 1), 1)
                ) * 100
                st.caption(
                    f"✅ Entity extraction active ({entity_coverage:.1f}% chunk coverage)"
                )
            else:
                st.caption("⚠️ No entities extracted yet")

    except Exception as e:
        st.error(f"Could not fetch database stats: {e}")


def display_document_list():
    """Display list of documents in the database with delete options and entity extraction status."""
    try:
        documents = graph_db.get_all_documents()

        # Get entity extraction status for all documents
        extraction_status = graph_db.get_entity_extraction_status()
        extraction_by_doc = {
            doc["document_id"]: doc for doc in extraction_status["documents"]
        }

        if not documents:
            st.info("No documents in the database yet.")
            return

        st.markdown("### 📂 Documents in Database")

        # Show overall entity extraction status
        try:
            is_extraction_running = document_processor.is_entity_extraction_running()
            if is_extraction_running:
                # Show global running indicator
                st.caption(
                    "🔄 Entity extraction running in background for multiple documents..."
                )
            elif extraction_status["documents_without_entities"] > 0:
                st.caption(
                    f"⚠️ {extraction_status['documents_without_entities']} documents missing entity extraction"
                )

                # Global entity extraction button
                if settings.enable_entity_extraction:
                    if st.button(
                        "🧠 Extract Entities for All Documents",
                        key="extract_entities_global_db",
                        type="primary",
                        help=f"Extract entities for {extraction_status['documents_without_entities']} documents that are missing entity extraction",
                    ):
                        result = document_processor.extract_entities_for_all_documents()
                        if result:
                            if result["status"] == "started":
                                st.success(f"✅ {result['message']}")
                                st.rerun()
                            elif result["status"] == "no_action_needed":
                                st.info(f"ℹ️ {result['message']}")
                            else:
                                st.error(f"❌ {result['message']}")
            else:
                st.success("✅ All documents have entities extracted")
        except Exception:
            # Fallback if entity extraction status check fails
            if extraction_status["documents_without_entities"] > 0:
                st.caption(
                    f"⚠️ {extraction_status['documents_without_entities']} documents missing entity extraction"
                )
            else:
                st.success("✅ All documents have entities extracted")

        # Add a session state for delete confirmations
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = {}

        for doc in documents:
            doc_id = doc["document_id"]
            filename = doc.get("filename", "Unknown")
            chunk_count = doc.get("chunk_count", 0)
            file_size = doc.get("file_size", 0)

            # Get entity status for this document
            entity_status = extraction_by_doc.get(doc_id, {})
            entities_extracted = entity_status.get("entities_extracted", False)
            total_entities = entity_status.get("total_entities", 0)

            # Format file size
            if file_size:
                if file_size > 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                elif file_size > 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size} B"
            else:
                size_str = "Unknown size"

            # Add entity status indicator to filename
            entity_indicator = "✅" if entities_extracted else "⚠️"

            with st.expander(f"{entity_indicator} {filename}", expanded=False):
                st.write(f"**Chunks:** {chunk_count}")
                st.write(f"**Size:** {size_str}")

                # OCR processing status
                processing_method = doc.get("processing_method", "")
                ocr_applied_pages = doc.get("ocr_applied_pages", 0)
                readable_text_pages = doc.get("readable_text_pages", 0)
                total_pages = doc.get("total_pages", 0)
                ocr_items_count = doc.get("ocr_items_count", 0)
                content_primary_type = doc.get("content_primary_type", "")
                summary_ocr_pages = doc.get("summary_ocr_pages", 0)
                summary_total_pages = doc.get("summary_total_pages", 0)

                if processing_method == "ocr" or processing_method == "image_ocr":
                    if processing_method == "image_ocr" and ocr_applied_pages > 0:
                        type_display = f" ({content_primary_type})" if content_primary_type else ""
                        st.write(f"**OCR Processing:** 🔍 Smart OCR applied (Image{type_display})")
                    elif summary_total_pages and summary_ocr_pages > 0:
                        st.write(f"**OCR Processing:** 🔍 Smart OCR applied ({summary_ocr_pages}/{summary_total_pages} pages)")
                        
                        # Show OCR details
                        if ocr_items_count > 0:
                            st.caption(f"OCR items processed: {ocr_items_count}")
                    elif summary_total_pages and (summary_total_pages - summary_ocr_pages) > 0:
                        readable_pages = summary_total_pages - summary_ocr_pages
                        st.write(f"**OCR Processing:** ✅ Readable text used ({readable_pages}/{summary_total_pages} pages)")
                    elif total_pages and readable_text_pages > 0:
                        st.write(f"**OCR Processing:** ✅ Readable text used ({readable_text_pages}/{total_pages} pages)")
                    else:
                        st.write("**OCR Processing:** ✅ Smart processing applied")

                # Entity extraction status
                if entities_extracted and total_entities > 0:
                    st.write(f"**Entities:** {total_entities} ✅")

                    # Show extracted entities for this document
                    try:
                        doc_entities = graph_db.get_document_entities(doc_id)
                        if doc_entities:
                            with st.expander("View Extracted Entities", expanded=False):
                                for entity in doc_entities[
                                    :10
                                ]:  # Limit to first 10 entities
                                    entity_name = entity.get("name", "Unknown")
                                    entity_type = entity.get("type", "Unknown")
                                    importance = entity.get("importance_score", 0)
                                    chunk_count_entity = entity.get("chunk_count", 0)

                                    st.write(f"**{entity_name}** ({entity_type})")
                                    st.caption(
                                        f"Importance: {importance:.2f} | Found in {chunk_count_entity} chunks"
                                    )

                                if len(doc_entities) > 10:
                                    st.caption(
                                        f"... and {len(doc_entities) - 10} more entities"
                                    )
                    except Exception as e:
                        st.error(f"Could not load entities: {e}")

                elif chunk_count > 0:
                    st.write("**Entities:** ⚠️ Not extracted")
                else:
                    st.write("**Entities:** N/A (no chunks)")

                col1, col2 = st.columns(2)

                with col1:
                    # Check if confirmation is pending
                    if st.session_state.confirm_delete.get(doc_id, False):
                        if st.button(
                            "✅ Confirm Delete", key=f"confirm_{doc_id}", type="primary"
                        ):
                            try:
                                graph_db.delete_document(doc_id)
                                st.success(f"Deleted {filename}")
                                st.session_state.confirm_delete[doc_id] = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {e}")
                    else:
                        if st.button(
                            "🗑️ Delete", key=f"delete_{doc_id}", type="secondary"
                        ):
                            st.session_state.confirm_delete[doc_id] = True
                            st.rerun()

                with col2:
                    if st.session_state.confirm_delete.get(doc_id, False):
                        if st.button("❌ Cancel", key=f"cancel_{doc_id}"):
                            st.session_state.confirm_delete[doc_id] = False
                            st.rerun()

        # Summary at the bottom
        st.markdown(f"**Total:** {len(documents)} documents")

    except Exception as e:
        st.error(f"Could not fetch document list: {e}")


def _display_content_with_truncation(
    content: str, key_prefix: str, index: int, max_length: int = 300
) -> None:
    """
    Helper function to display content with truncation and expansion option.

    Args:
        content: Content to display
        key_prefix: Prefix for Streamlit widget keys
        index: Index for unique key generation
        max_length: Maximum length before truncation
    """
    if len(content) > max_length:
        st.text_area(
            "Content Preview:",
            content[:max_length] + "...",
            height=100,
            key=f"{key_prefix}_preview_{index}",
            disabled=True,
        )
        with st.expander("Show Full Content"):
            st.text_area(
                "Full Content:",
                content,
                height=200,
                key=f"{key_prefix}_full_{index}",
                disabled=True,
            )
    else:
        st.text_area(
            "Content:",
            content,
            height=max(60, min(len(content.split("\n")) * 20, 150)),
            key=f"{key_prefix}_content_{index}",
            disabled=True,
        )


def _group_sources_by_document(
    sources: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Group sources by document, collecting all relevant chunks for each document.

    Args:
        sources: List of source chunks/entities with metadata

    Returns:
        Dict mapping document names to document info with grouped chunks
    """
    document_groups = {}

    for source in sources:
        # Handle entity sources differently - they don't belong to a specific document chunk
        if source.get("entity_name") or source.get("entity_id"):
            # For entities, use the document they were found in or create a special entity group
            doc_name = source.get("document_name", source.get("filename", "Entities"))
        else:
            # Regular chunk sources
            doc_name = source.get("document_name") or source.get(
                "filename", "Unknown Document"
            )

        if doc_name not in document_groups:
            document_groups[doc_name] = {
                "document_name": doc_name,
                "document_id": source.get("document_id", ""),
                "filename": source.get("filename", doc_name),
                "chunks": [],
                "entities": [],
            }

        # Add source to appropriate list
        if source.get("entity_name") or source.get("entity_id"):
            document_groups[doc_name]["entities"].append(source)
        else:
            document_groups[doc_name]["chunks"].append(source)

    return document_groups


def display_sources_detailed(sources: List[Dict[str, Any]]):
    """
    Display detailed source chunks and entities grouped by document.

    Args:
        sources: List of source chunks/entities with metadata
    """
    if not sources:
        st.write("No sources used in this response.")
        return

    # Group sources by document
    document_groups = _group_sources_by_document(sources)

    # Sort documents by the number of relevant chunks (descending)
    sorted_documents = sorted(
        document_groups.items(),
        key=lambda x: len(x[1]["chunks"]) + len(x[1]["entities"]),
        reverse=True,
    )

    # Display each document with its chunks
    for i, (doc_name, doc_info) in enumerate(sorted_documents, 1):
        chunks = doc_info["chunks"]
        entities = doc_info["entities"]
        total_sources = len(chunks) + len(entities)

        # Create expander title with chunk count
        if total_sources == 1:
            title = f"📚 {doc_name}"
        else:
            title = f"📚 {doc_name} ({total_sources} relevant sections)"

        with st.expander(title, expanded=False):
            # Display entities if present
            if entities:
                st.write("**🏷️ Relevant Entities:**")
                # Only show up to 10 entities per document to keep UI concise
                visible_entities = entities[:10]
                hidden_entities_count = max(0, len(entities) - len(visible_entities))
                for j, entity in enumerate(visible_entities):
                    with st.container():
                        st.write(f"• **{entity.get('entity_name', 'Unknown Entity')}**")
                        if entity.get("entity_type", "").lower() != "entity":
                            st.caption(f"Type: {entity.get('entity_type', 'Unknown')}")

                        if entity.get("entity_description"):
                            st.caption(
                                f"Description: {entity.get('entity_description')}"
                            )

                        # Show related content for entities
                        if entity.get("related_chunks"):
                            for chunk_info in entity.get("related_chunks", [])[
                                :1
                            ]:  # Show only first one
                                content = (
                                    chunk_info.get("content", "No content")[:200]
                                    + "..."
                                )
                                st.text_area(
                                    "Context:",
                                    content,
                                    height=60,
                                    key=f"entity_context_{i}_{j}_{chunk_info.get('chunk_id', 'unknown')}",
                                    disabled=True,
                                )
                        elif entity.get("content"):
                            content = (
                                entity.get("content", "No content available")[:200]
                                + "..."
                            )
                            st.text_area(
                                "Context:",
                                content,
                                height=60,
                                key=f"entity_content_{i}_{j}",
                                disabled=True,
                            )

                if chunks:  # Add separator if we have both entities and chunks
                    st.markdown("---")

            # Display chunks if present
            if chunks:
                # Only show up to 10 chunks per document for readability
                visible_chunks = chunks[:10]
                hidden_chunks_count = max(0, len(chunks) - len(visible_chunks))

                if len(chunks) == 1:
                    st.write("**📄 Relevant Content:**")
                else:
                    st.write(f"**📄 Relevant Content ({len(chunks)} sections):**")

                for j, chunk in enumerate(visible_chunks):
                    with st.container():
                        # Show chunk identifier if multiple chunks
                        if len(chunks) > 1:
                            chunk_idx = chunk.get("chunk_index")
                            if chunk_idx is not None:
                                st.write(f"**Section {chunk_idx + 1}:**")
                            else:
                                st.write(f"**Section {j + 1}:**")

                        # Display chunk content
                        content = chunk.get("content", "No content available")
                        _display_content_with_truncation(content, f"doc_{i}_chunk", j)

                        # Show contained entities if any
                        entities_in_chunk = chunk.get("contained_entities", [])
                        if entities_in_chunk:
                            st.caption(
                                f"**Contains:** {', '.join(entities_in_chunk[:5])}"
                            )
                            if len(entities_in_chunk) > 5:
                                st.caption(
                                    f"... and {len(entities_in_chunk) - 5} more entities"
                                )
                # If there were more chunks than displayed, indicate how many were hidden
                if hidden_chunks_count:
                    st.caption(
                        f"... and {hidden_chunks_count} more sections hidden (showing top 10)"
                    )
                # If there were more entities than displayed, indicate how many were hidden
                if entities and len(entities) > 10:
                    hidden_entities_count = len(entities) - 10
                    st.caption(
                        f"... and {hidden_entities_count} more entities hidden (showing top 10)"
                    )

    # Show total count
    total_documents = len(document_groups)
    total_chunks = sum(len(doc["chunks"]) for doc in document_groups.values())
    total_entities = sum(len(doc["entities"]) for doc in document_groups.values())

    if total_documents > 0:
        count_text = f"**Sources:** {total_documents} document{'s' if total_documents != 1 else ''}"
        if total_chunks > 0:
            count_text += f", {total_chunks} chunk{'s' if total_chunks != 1 else ''}"
        if total_entities > 0:
            count_text += (
                f", {total_entities} entit{'ies' if total_entities != 1 else 'y'}"
            )
        st.caption(count_text)


def display_query_analysis_detailed(analysis: Dict[str, Any]):
    """
    Display detailed query analysis in sidebar.

    Args:
        analysis: Query analysis results
    """
    if not analysis:
        return

    with st.expander("Analysis Details", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Type:** {analysis.get('query_type', 'Unknown')}")
            st.write(f"**Complexity:** {analysis.get('complexity', 'Unknown')}")

        with col2:
            key_concepts = analysis.get("key_concepts", [])
            if key_concepts:
                st.write("**Key Concepts:**")
                for concept in key_concepts[:5]:  # Limit to top 5
                    st.write(f"• {concept}")


def display_document_upload():
    """Encapsulated document upload UI and processing logic."""
    st.markdown("### 📁 Document Upload")

    # Add checkbox for entity extraction option
    extract_entities = st.checkbox(
        "Extract entities during upload",
        value=True,
        help="If checked, entities will be extracted in background after chunk creation. If unchecked, only chunks will be created (faster upload).",
        key="extract_entities_checkbox",
    )

    # Information message about smart processing
    if extract_entities:
        st.info(
            "🧠 Smart processing enabled: OCR automatically applied to images, diagrams, and scanned content only. Chunks created immediately, entity extraction runs in background."
        )
    else:
        st.info(
            "🧠 Smart processing enabled: OCR automatically applied to images, diagrams, and scanned content only. Entity extraction can be run manually later."
        )

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt", "md", "csv", "pptx", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Upload documents to expand the knowledge base. Supports PDF, Word, Text, Markdown, CSV, PowerPoint, and Excel files.",
        key=f"file_uploader_{st.session_state.file_uploader_key}",
    )

    # Process uploaded files
    if uploaded_files and not st.session_state.processing_files:
        if st.button("🚀 Process Files"):
            st.session_state.processing_files = True

            with st.container():
                st.markdown("### 📝 Processing Progress")
                progress_container = st.container()

                st.info("Processing uploaded files (chunk extraction).")

                # Process files (chunks only). Entity extraction runs in background if enabled.
                extract_entities = st.session_state.get(
                    "extract_entities_checkbox", True
                )
                results = process_files_background(
                    uploaded_files, progress_container, extract_entities
                )

                # Display results
                if results["processed_files"]:
                    # Create comprehensive success message
                    success_msg = f"Successfully processed {len(results['processed_files'])} files ({results['total_chunks']} chunks created"
                    if results.get("total_entities", 0) > 0:
                        success_msg += f", {results['total_entities']} entities, {results['total_entity_relationships']} relationships"
                    success_msg += ")"

                    st.toast(icon="✅", body=success_msg)

                if results["errors"]:
                    st.toast(
                        icon="❌",
                        body=f"Failed to process {len(results['errors'])} files",
                    )
                    for error_info in results["errors"]:
                        st.write(f"- 📄 {error_info['name']}: {error_info['error']}")

                # Always reset the file uploader after processing (success or failure)
                st.session_state.file_uploader_key += 1
                st.session_state.processing_files = False

                # Force a rerun to refresh the UI and clear the uploader
                time.sleep(3)
                st.rerun()

    return


def get_search_mode_config(search_mode: str):
    """
    Get configuration parameters for different search modes.

    Args:
        search_mode: One of 'quick', 'normal', or 'deep'

    Returns:
        Dictionary with all search parameters
    """
    configs = {
        "quick": {
            "min_retrieval_similarity": 0.3,
            "hybrid_chunk_weight": 0.8,
            "enable_graph_expansion": False,
            "max_expanded_chunks": 50,
            "max_entity_connections": 5,
            "max_chunk_connections": 3,
            "expansion_similarity_threshold": 0.4,
            "max_expansion_depth": 1,
            "top_k": 3,
            "temperature": 0.1,
            "retrieval_mode": (
                "hybrid" if settings.enable_entity_extraction else "chunk_only"
            ),
        },
        "normal": {
            "min_retrieval_similarity": 0.1,
            "hybrid_chunk_weight": 0.6,
            "enable_graph_expansion": True,
            "max_expanded_chunks": 500,
            "max_entity_connections": 20,
            "max_chunk_connections": 10,
            "expansion_similarity_threshold": 0.1,
            "max_expansion_depth": 2,
            "top_k": 5,
            "temperature": 0.1,
            "retrieval_mode": (
                "hybrid" if settings.enable_entity_extraction else "chunk_only"
            ),
        },
        "deep": {
            "min_retrieval_similarity": 0.05,
            "hybrid_chunk_weight": 0.4,
            "enable_graph_expansion": True,
            "max_expanded_chunks": 1000,
            "max_entity_connections": 50,
            "max_chunk_connections": 20,
            "expansion_similarity_threshold": 0.05,
            "max_expansion_depth": 3,
            "top_k": 10,
            "temperature": 0.1,
            "retrieval_mode": (
                "hybrid" if settings.enable_entity_extraction else "chunk_only"
            ),
        },
    }
    return configs.get(search_mode, configs["normal"])


def get_rag_settings(key_suffix: str = ""):
    """
    Render RAG settings controls in the sidebar and return their values.

    Args:
        key_suffix: Suffix to append to widget keys to keep them unique when
            the settings are rendered from multiple places in the UI.

    Returns:
        Dictionary with all search parameters
    """
    st.markdown("### 🧠 Search Settings")

    # Search Mode Selection
    search_modes = ["quick", "normal", "deep"]
    search_mode_labels = ["🚀 Quick Search", "⚖️ Normal Search", "🔍 Deep Search"]

    search_mode = st.selectbox(
        "Search Mode",
        search_modes,
        format_func=lambda x: search_mode_labels[search_modes.index(x)],
        index=1,  # Default to 'normal'
        key=f"search_mode{key_suffix}",
        help="""
        Choose your search strategy:
        • **Quick**: Fast results, fewer chunks, minimal graph traversal
        • **Normal**: Balanced performance and comprehensiveness (recommended)
        • **Deep**: Comprehensive search, more context, extensive graph exploration
        """,
    )

    # Get base configuration for selected mode
    config = get_search_mode_config(search_mode)

    # Brief explanation of current mode
    mode_explanations = {
        "quick": "🚀 **Quick mode**: Optimized for speed with focused results. Uses fewer chunks and minimal graph expansion.",
        "normal": "⚖️ **Normal mode**: Balanced approach providing good coverage without overwhelming context. Best for most queries.",
        "deep": "🔍 **Deep mode**: Maximum comprehensiveness. Explores more connections and relationships for complex queries.",
    }

    st.info(mode_explanations[search_mode])

    # Show entity extraction status if not using quick mode
    if search_mode != "quick":
        if settings.enable_entity_extraction:
            st.success("✅ Entity extraction enabled - Enhanced search available")
        else:
            st.warning("⚠️ Entity extraction disabled - Using chunk-only search")
            config["retrieval_mode"] = "chunk_only"

    # Advanced Settings Expander
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.markdown("**Current Configuration:**")

        # Display current configuration in a more readable format
        col1, col2 = st.columns(2)

        with col1:
            st.write(
                f"**Retrieval Mode:** {config['retrieval_mode'].replace('_', ' ').title()}"
            )
            st.write(f"**Chunks Retrieved:** {config['top_k']}")
            st.write(f"**Temperature:** {config['temperature']}")
            st.write(f"**Min Similarity:** {config['min_retrieval_similarity']}")

        with col2:
            st.write(
                f"**Graph Expansion:** {'✅' if config['enable_graph_expansion'] else '❌'}"
            )
            st.write(f"**Max Expanded Chunks:** {config['max_expanded_chunks']}")
            st.write(f"**Chunk Weight:** {config['hybrid_chunk_weight']}")
            st.write(f"**Expansion Depth:** {config['max_expansion_depth']}")

        st.markdown("---")
        st.markdown("**Override Settings** (optional):")

        # Allow users to override specific settings
        use_custom = st.checkbox(
            "Customize parameters",
            key=f"use_custom{key_suffix}",
            help="Enable to modify individual parameters",
        )

        if use_custom:
            # Basic settings
            st.markdown("**Basic Settings:**")

            # Update retrieval modes to match hybrid approach
            mode_options = ["chunk_only", "entity_only", "hybrid"]
            mode_labels = [
                "Chunk Only (Traditional)",
                "Entity Only (GraphRAG)",
                "Hybrid (Best of Both)",
            ]

            if not settings.enable_entity_extraction:
                mode_options = ["chunk_only"]
                mode_labels = ["Chunk Only (Traditional)"]

            current_mode_index = 0
            if config["retrieval_mode"] in mode_options:
                current_mode_index = mode_options.index(config["retrieval_mode"])

            config["retrieval_mode"] = st.selectbox(
                "Retrieval Strategy",
                mode_options,
                format_func=lambda x: (
                    mode_labels[mode_options.index(x)] if x in mode_options else x
                ),
                index=current_mode_index,
                key=f"retrieval_mode_custom{key_suffix}",
            )

            config["top_k"] = st.slider(
                "Number of chunks to retrieve",
                min_value=1,
                max_value=20,
                value=config["top_k"],
                key=f"top_k_custom{key_suffix}",
            )

            config["temperature"] = st.slider(
                "Response creativity (temperature)",
                min_value=0.0,
                max_value=1.0,
                value=config["temperature"],
                step=0.1,
                key=f"temperature_custom{key_suffix}",
            )

            # Advanced retrieval settings
            st.markdown("**Advanced Retrieval:**")

            config["min_retrieval_similarity"] = st.slider(
                "Minimum similarity threshold",
                min_value=0.0,
                max_value=0.5,
                value=config["min_retrieval_similarity"],
                step=0.05,
                key=f"min_similarity_custom{key_suffix}",
                help="Lower values retrieve more chunks, higher values are more selective",
            )

            if config["retrieval_mode"] == "hybrid":
                config["hybrid_chunk_weight"] = st.slider(
                    "Chunk weight (vs Entity weight)",
                    min_value=0.0,
                    max_value=1.0,
                    value=config["hybrid_chunk_weight"],
                    step=0.1,
                    key=f"chunk_weight_custom{key_suffix}",
                    help="Higher values favor chunk-based results, lower values favor entity-based results",
                )

            # Graph expansion settings
            st.markdown("**Graph Expansion:**")

            config["enable_graph_expansion"] = st.checkbox(
                "Enable graph expansion",
                value=config["enable_graph_expansion"],
                key=f"graph_expansion_custom{key_suffix}",
                help="Use entity relationships to expand context",
            )

            if config["enable_graph_expansion"]:
                config["max_expanded_chunks"] = st.number_input(
                    "Max expanded chunks",
                    min_value=50,
                    max_value=2000,
                    value=config["max_expanded_chunks"],
                    step=50,
                    key=f"max_chunks_custom{key_suffix}",
                )

                config["max_expansion_depth"] = st.slider(
                    "Max expansion depth",
                    min_value=1,
                    max_value=5,
                    value=config["max_expansion_depth"],
                    key=f"max_depth_custom{key_suffix}",
                    help="How many hops to follow in the graph",
                )

                config["expansion_similarity_threshold"] = st.slider(
                    "Expansion similarity threshold",
                    min_value=0.0,
                    max_value=0.5,
                    value=config["expansion_similarity_threshold"],
                    step=0.05,
                    key=f"expansion_threshold_custom{key_suffix}",
                    help="Minimum similarity for expanding through relationships",
                )

    return config


def main():
    """Main Streamlit application."""
    # Title and description
    html_style = """
    <style>
    /* Make the right-side floating container fixed and scrollable when content overflows */
    div:has( >.element-container div.floating) {
        display: flex;
        flex-direction: column;
        position: fixed;
        right: 1rem;
        top: 0rem; /* leave space for header */
        width: 33%;
        max-height: calc(100vh - 8rem);
        overflow-y: auto;
        padding-right: 0.5rem; /* avoid clipping scrollbars */
        box-sizing: border-box;
        z-index: 2000;
    }

    /* Hide deploy button introduced by Streamlit cloud UI if present */
    .stAppDeployButton {
        display: none;
    }

    /* Small top padding for the main app container */
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 2rem;}

    /* Ensure the inner floating wrapper doesn't collapse and allows scrolling */
    div.floating {
        height: auto;
        min-height: 4rem;
    }
    </style>
    """
    st.markdown(html_style, unsafe_allow_html=True)

    # Create main layout with columns
    main_col, sidebar_col = st.columns([2, 1])  # 2:1 ratio for main content vs sidebar

    with main_col:
        # Display chat title and clear chat button side-by-side
        st.title("💬 Chat with your documents")

        # Render chat messages below the title/actions
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Sidebar for additional information (sources and graphs)
    with sidebar_col:
        container = st.container(width=10)

        with container:
            st.markdown('<div class="floating">', unsafe_allow_html=True)
            # st.markdown("### 📊 Context Information")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📚 Sources", "🕸️ Context Graph", "📊 Database", "📁 Upload File", "⚙️"]
            )

            # Display information for the latest assistant message if available
            if st.session_state.messages:
                latest_message = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "assistant":
                        latest_message = msg
                        break

                if latest_message:
                    with tab1:
                        # Display sources in sidebar
                        if "sources" in latest_message:
                            display_sources_detailed(latest_message["sources"])

                            # Small clear chat button placed next to the title
                            if st.button(
                                "🧹 Clear chat",
                                key="clear_chat_top",
                                help="Clear conversation, graph and sources",
                            ):
                                st.session_state.messages = []
                                for k in [
                                    "latest_message",
                                    "latest_graph",
                                    "latest_sources",
                                ]:
                                    if k in st.session_state:
                                        del st.session_state[k]
                                # Also clear any latest widget-bound settings
                                for sfx in ["_latest", "_default"]:
                                    k = f"search_config{sfx}"
                                    if k in st.session_state:
                                        del st.session_state[k]
                                st.rerun()

                    with tab2:
                        # Display graph in sidebar
                        if "graph_fig" in latest_message:
                            st.plotly_chart(
                                latest_message["graph_fig"], use_container_width=True
                            )

                    with tab3:
                        display_stats()
                        display_document_list()

                    with tab4:
                        display_document_upload()

                    with tab5:
                        search_config = get_rag_settings(key_suffix="_latest")
                        st.session_state["search_config_latest"] = search_config

            else:
                # tab1, tab2, tab3 = st.tabs(["📊 Database", "📂 Upload File", "⚙️"])

                with tab1:
                    st.info("💡 Start a conversation to see context information here!")

                with tab2:
                    st.info("💡 Start a conversation to see context information here!")

                with tab3:
                    display_stats()
                    display_document_list()

                with tab4:
                    display_document_upload()

                with tab5:
                    search_config = get_rag_settings(key_suffix="_default")
                    st.session_state["search_config_default"] = search_config

            st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Rerun to display the user message immediately
        st.rerun()

    # Process the latest user query if there's one that needs processing
    if (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and len(st.session_state.messages) > 0
    ):

        # Check if we already processed this message
        user_message = st.session_state.messages[-1]
        needs_processing = True

        # Check if there's already a response to this user message
        if len(st.session_state.messages) > 1:
            for i in range(len(st.session_state.messages) - 1, 0, -1):
                if st.session_state.messages[i]["role"] == "assistant":
                    # Found an assistant message, check if it's newer than the user message
                    needs_processing = True
                    break
                elif (
                    st.session_state.messages[i]["role"] == "user"
                    and i < len(st.session_state.messages) - 1
                ):
                    # Found an older user message, so current one needs processing
                    needs_processing = True
                    break

        if needs_processing:
            user_query = user_message["content"]

            with main_col:
                with st.chat_message("assistant"):
                    try:
                        with st.spinner("🔍 Generating response..."):
                            # Get search configuration (from new unified settings)
                            search_config = st.session_state.get(
                                "search_config_latest",
                                st.session_state.get(
                                    "search_config_default",
                                    get_search_mode_config("normal"),
                                ),
                            )

                            result = graph_rag.query(
                                user_query,
                                retrieval_mode=search_config.get(
                                    "retrieval_mode", "hybrid"
                                ),
                                top_k=search_config.get("top_k", 5),
                                temperature=search_config.get("temperature", 0.1),
                                chunk_weight=search_config.get(
                                    "hybrid_chunk_weight", 0.6
                                ),
                                graph_expansion=search_config.get(
                                    "enable_graph_expansion", True
                                ),
                            )
                        full_response = result["response"]
                        st.write_stream(
                            stream_response(full_response, 0.02)
                        )  # Stream the response

                        # Add assistant message to session state
                        message_data = {
                            "role": "assistant",
                            "content": full_response,
                            "query_analysis": result.get("query_analysis"),
                            "sources": result.get("sources"),
                        }

                        # Always add contextual graph visualization for the answer
                        try:
                            # Import here to avoid issues if dependencies aren't installed
                            from core.graph_viz import \
                                create_query_result_graph

                            if result.get("sources"):
                                query_fig = create_query_result_graph(
                                    result["sources"], user_query
                                )
                                message_data["graph_fig"] = query_fig
                        except ImportError:
                            st.warning(
                                "Graph visualization dependencies not installed. Run: pip install plotly networkx"
                            )
                        except Exception as e:
                            st.warning(f"Could not create query graph: {e}")

                        st.session_state.messages.append(message_data)

                        # Force refresh to show sidebar content
                        st.rerun()

                    except Exception as e:
                        error_msg = f"❌ I encountered an error processing your request: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )

    # Optional: Show expanded knowledge graph if requested
    # with sidebar_col:
    #     if st.checkbox("Show Full Knowledge Graph", key="show_full_graph"):
    #         try:
    #             from core.graph_viz import get_graph_data, create_plotly_graph

    #             with st.spinner("Loading full knowledge graph..."):
    #                 graph_data = get_graph_data(limit=50)

    #                 if graph_data['nodes']:
    #                     st.markdown("### 🌐 Full Knowledge Graph")
    #                     full_fig = create_plotly_graph(graph_data, layout_algorithm="spring")
    #                     st.plotly_chart(full_fig, use_container_width=True)
    #                     st.info(f"📊 Graph contains {graph_data['total_nodes']} nodes and {graph_data['total_edges']} edges")
    #                 else:
    #                     st.warning("No full graph data available.")

    #         except ImportError:
    #             st.error("Graph visualization dependencies not installed. Run: pip install plotly networkx")
    #         except Exception as e:
    #             st.error(f"Error creating full graph visualization: {e}")

    # Knowledge Graph Visualization Section
    # st.markdown("---")

    # with st.expander("🔍 Advanced Graph Options", expanded=False):
    #     st.markdown("### 📊 Knowledge Graph Visualization")

    #     col1, col2, col3 = st.columns([2, 1, 1])

    #     with col1:
    #         if st.button("🔍 Show Knowledge Graph", key="show_advanced_graph"):
    #             st.session_state.show_graph = True

    #     with col2:
    #         layout_algo = st.selectbox(
    #             "Layout Algorithm",
    #             ["spring", "circular", "kamada_kawai"],
    #             key="advanced_layout_select"
    #         )

    #     with col3:
    #         node_limit = st.number_input(
    #             "Max Nodes",
    #             min_value=10,
    #             max_value=500,
    #             value=100,
    #             key="advanced_node_limit"
    #         )

    #     if st.session_state.show_graph:
    #         try:
    #             # Import here to avoid issues if dependencies aren't installed
    #             from core.graph_viz import get_graph_data, create_plotly_graph

    #             # Get current retrieval mode for graph visualization
    #             current_retrieval_mode = st.session_state.get(
    #                 "retrieval_mode_latest",
    #                 st.session_state.get("retrieval_mode_default", "chunk_only"),
    #             )

    #             with st.spinner("Loading knowledge graph..."):
    #                 graph_data = get_graph_data(limit=node_limit, retrieval_mode=current_retrieval_mode)

    #                 if graph_data['nodes']:
    #                     fig = create_plotly_graph(graph_data, layout_algorithm=layout_algo)
    #                     st.plotly_chart(fig, use_container_width=True)

    #                     # Show enhanced graph statistics
    #                     stats_text = f"📊 Graph contains {graph_data['total_nodes']} nodes and {graph_data['total_edges']} edges"
    #                     if current_retrieval_mode != "chunk_only":
    #                         stats_text += f" (Mode: {current_retrieval_mode.replace('_', ' ').title()})"
    #                     st.info(stats_text)

    #                     # Show node type breakdown if available
    #                     node_types = {}
    #                     for node in graph_data['nodes']:
    #                         node_type = node.get('label', 'Unknown')
    #                         node_types[node_type] = node_types.get(node_type, 0) + 1

    #                     if len(node_types) > 1:
    #                         breakdown = " | ".join([f"{node_type}: {count}" for node_type, count in node_types.items()])
    #                         st.caption(f"Node breakdown: {breakdown}")
    #                 else:
    #                     st.warning("No graph data available. Upload some documents first!")

    #         except ImportError:
    #             st.error("Graph visualization dependencies not installed. Run: pip install plotly networkx")
    #         except Exception as e:
    #             st.error(f"Error creating graph visualization: {e}")


if __name__ == "__main__":
    main()
