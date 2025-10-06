"""Components for rendering sources and analysis details in the sidebar."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def _display_content_with_truncation(
    content: str, key_prefix: str, index: int, max_length: int = 300
) -> None:
    """Helper to display content with truncation and expansion option."""
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
    """Group sources by document, collecting relevant chunks for each document."""
    document_groups: Dict[str, Dict[str, Any]] = {}

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


def display_sources_detailed(sources: List[Dict[str, Any]]) -> None:
    """Render detailed source chunks and entities grouped by document."""
    if not sources:
        st.write("No sources used in this response.")
        return

    # Group sources by document
    document_groups = _group_sources_by_document(sources)

    # Sort documents by the number of relevant chunks (descending)
    sorted_documents = sorted(
        document_groups.items(),
        key=lambda item: len(item[1]["chunks"]) + len(item[1]["entities"]),
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
                        related_chunks = entity.get("related_chunks")
                        if related_chunks:
                            for chunk_info in related_chunks[:1]:  # Show only first one
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
                hidden_chunks_count = max(0, len(chunks) - len(visible_chunks))
                if hidden_chunks_count:
                    st.caption(
                        f"... and {hidden_chunks_count} more sections hidden (showing top 10)"
                    )
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

