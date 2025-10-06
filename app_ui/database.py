"""Sidebar components for database statistics and document management."""

from __future__ import annotations

import logging
import streamlit as st

from config.settings import settings
from core.graph_db import graph_db
from ingestion.document_processor import document_processor

logger = logging.getLogger(__name__)


def display_stats() -> None:
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
                st.caption(f"✅ Entities extracted ({entity_coverage:.1f}% chunk coverage)")
            else:
                st.caption("⚠️ No entities extracted yet")
        except Exception:  # pylint: disable=broad-except
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

    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Could not fetch database stats: {exc}")
        logger.exception("Could not fetch database stats")


def _format_size(file_size: int) -> str:
    if not file_size:
        return "Unknown size"
    if file_size > 1024 * 1024:
        return f"{file_size / (1024 * 1024):.1f} MB"
    if file_size > 1024:
        return f"{file_size / 1024:.1f} KB"
    return f"{file_size} B"


def display_document_list() -> None:
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
        except Exception:  # pylint: disable=broad-except
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

            size_str = _format_size(file_size)

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
                        st.write(
                            f"**OCR Processing:** 🔍 Smart OCR applied (Image{type_display})"
                        )
                    elif summary_total_pages and summary_ocr_pages > 0:
                        st.write(
                            f"**OCR Processing:** 🔍 Smart OCR applied ({summary_ocr_pages}/{summary_total_pages} pages)"
                        )

                        # Show OCR details
                        if ocr_items_count > 0:
                            st.caption(f"OCR items processed: {ocr_items_count}")
                    elif summary_total_pages and (summary_total_pages - summary_ocr_pages) > 0:
                        readable_pages = summary_total_pages - summary_ocr_pages
                        st.write(
                            f"**OCR Processing:** ✅ Readable text used ({readable_pages}/{summary_total_pages} pages)"
                        )
                    elif total_pages and readable_text_pages > 0:
                        st.write(
                            f"**OCR Processing:** ✅ Readable text used ({readable_text_pages}/{total_pages} pages)"
                        )
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
                                for entity in doc_entities[:10]:  # Limit to first 10 entities
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
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Could not load entities: {exc}")
                        logger.exception("Could not load entities for document %s", doc_id)

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
                            except Exception as exc:  # pylint: disable=broad-except
                                st.error(f"Failed to delete: {exc}")
                                logger.exception("Failed to delete doc %s", doc_id)
                    else:
                        if st.button("🗑️ Delete", key=f"delete_{doc_id}", type="secondary"):
                            st.session_state.confirm_delete[doc_id] = True
                            st.rerun()

                with col2:
                    if st.session_state.confirm_delete.get(doc_id, False):
                        if st.button("❌ Cancel", key=f"cancel_{doc_id}"):
                            st.session_state.confirm_delete[doc_id] = False
                            st.rerun()

        # Summary at the bottom
        st.markdown(f"**Total:** {len(documents)} documents")

    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Could not fetch document list: {exc}")
        logger.exception("Could not fetch document list")
