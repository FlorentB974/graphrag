"""Upload widget and handlers for Streamlit UI."""

from __future__ import annotations

import time

import streamlit as st

from .file_processing import process_files_background


def display_document_upload() -> None:
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
