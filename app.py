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
    uploaded_files: List[Any], progress_container, processing_mode: str = "chunk_only"
) -> Dict[str, Any]:
    """
    Process uploaded files in background with chunk-level progress tracking.

    Args:
        uploaded_files: List of uploaded file objects
        progress_container: Streamlit container for progress updates

    Returns:
        Dictionary with processing results
    """
    results = {"processed_files": [], "total_chunks": 0, "total_entities": 0, "total_entity_relationships": 0, "errors": []}

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
        status_text.text(f"Processing {current_file_name}... ({total_processed_chunks}/{total_estimated_chunks} chunks completed)")

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            current_file_name = uploaded_file.name
            status_text.text(f"Processing {uploaded_file.name}... ({total_processed_chunks}/{total_estimated_chunks} chunks completed)")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                # Process the file: always perform chunk extraction immediately.
                # Entity extraction (if enabled) will run in background from the processor.
                result = document_processor.process_file(tmp_path, uploaded_file.name, chunk_progress_callback)

                if result and result.get("status") == "success":
                    file_info = {
                        "name": uploaded_file.name,
                        "chunks": result.get("chunks_created", 0),
                        "document_id": result.get("document_id"),
                    }
                    
                    # Add entity information if available
                    if result.get("entities_created", 0) > 0:
                        file_info["entities"] = result.get("entities_created", 0)
                        file_info["entity_relationships"] = result.get("entity_relationships_created", 0)
                    
                    results["processed_files"].append(file_info)
                    results["total_chunks"] += result.get("chunks_created", 0)
                    
                    # Track entity statistics
                    if "total_entities" not in results:
                        results["total_entities"] = 0
                        results["total_entity_relationships"] = 0
                    results["total_entities"] += result.get("entities_created", 0)
                    results["total_entity_relationships"] += result.get("entity_relationships_created", 0)
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

    # Final progress update
    progress_bar.progress(1.0)
    status_text.text(f"File processing complete! Processed {results['total_chunks']} chunks from {len(uploaded_files)} files.")
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

        # Show entity extraction status
        if stats.get("entities", 0) > 0:
            entity_coverage = (stats.get("chunk_entity_relations", 0) / max(stats.get("chunks", 1), 1)) * 100
            # If background entity extraction is running, show an updating caption
            try:
                if document_processor.is_entity_extraction_running():
                    st.caption(f"🔄 Entity extraction running in background — updating database ({entity_coverage:.1f}% chunk coverage)")
                else:
                    st.caption(f"✅ Entities extracted ({entity_coverage:.1f}% chunk coverage)")
            except Exception:
                # Fallback to default caption if detection fails
                st.caption(f"✅ Entity extraction active ({entity_coverage:.1f}% chunk coverage)")
        else:
            st.caption("⚠️ No entities extracted yet")

    except Exception as e:
        st.error(f"Could not fetch database stats: {e}")


def display_document_list():
    """Display list of documents in the database with delete options."""
    try:
        documents = graph_db.get_all_documents()

        if not documents:
            st.info("No documents in the database yet.")
            return

        st.markdown("### 📂 Documents in Database")

        # Add a session state for delete confirmations
        if "confirm_delete" not in st.session_state:
            st.session_state.confirm_delete = {}

        for doc in documents:
            doc_id = doc["document_id"]
            filename = doc.get("filename", "Unknown")
            chunk_count = doc.get("chunk_count", 0)
            file_size = doc.get("file_size", 0)

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

            with st.expander(f"📄 {filename}", expanded=False):
                st.write(f"**Chunks:** {chunk_count}")
                st.write(f"**Size:** {size_str}")

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


def display_sources_detailed(sources: List[Dict[str, Any]]):
    """
    Display detailed source chunks and entities in a formatted sidebar.

    Args:
        sources: List of source chunks/entities with metadata
    """
    if not sources:
        st.write("No sources used in this response.")
        return

    # Filter and sort sources by relevance
    relevant_sources = []
    for source in sources:
        score = source.get("similarity", source.get("relevance_score", 0))
        # Include all sources - don't filter out zero scores (entities might have 0.0 scores)
        relevant_sources.append(source)
    
    # Sort by relevance score, prioritizing entity sources
    def get_sort_score(source):
        # Get base score
        score = source.get("similarity", source.get("relevance_score", 0))
        # If it's an entity source with 0 score, give it a reasonable default
        if (source.get("entity_name") or source.get("entity_id")) and score == 0:
            return 0.5  # Default entity relevance
        return score
    
    relevant_sources.sort(key=get_sort_score, reverse=True)

    for i, source in enumerate(relevant_sources[:5], 1):  # Limit to top 5 sources
        score = source.get("similarity", source.get("relevance_score", 0))
        
        # Determine source type and icon
        if source.get("entity_name") or source.get("entity_id"):
            # This is an entity source
            icon = "🏷️"
            source_type = "Entity"
            title = source.get("entity_name", "Unknown Entity")
        elif source.get("entity_enhanced") or source.get("contained_entities"):
            # This is an entity-enhanced chunk source (hybrid mode)
            icon = "🔗"
            source_type = "Hybrid Chunk"
            title = source.get("filename", source.get("document_name", "Unknown Document"))
        else:
            # This is a regular chunk source
            icon = "📄"
            source_type = "Chunk"
            title = source.get("filename", source.get("document_name", "Unknown Document"))

        with st.expander(
            f"{icon} {source_type} {i}: {title}",
            expanded=False,
        ):            
            if source_type == "Entity":
                # Display entity information
                st.write(f"**Entity Name:** {source.get('entity_name', 'Unknown')}")
                st.write(f"**Type:** {source.get('entity_type', 'Unknown')}")
                
                if source.get("entity_description"):
                    st.write(f"**Description:** {source.get('entity_description')}")
                
                # Show related chunks
                if source.get("related_chunks"):
                    st.write("**Found in chunks:**")
                    for chunk_info in source.get("related_chunks", [])[:2]:  # Limit to 2
                        chunk_content = chunk_info.get("content", "No content")[:150] + "..."
                        st.text_area(
                            "Related content:",
                            chunk_content,
                            height=80,
                            key=f"entity_chunk_{i}_{chunk_info.get('chunk_id', 'unknown')}",
                            disabled=True,
                        )
                elif source.get("content"):
                    # Fallback to content if available
                    content = source.get("content", "No content available")
                    if len(content) > 200:
                        st.text_area(
                            "Context:",
                            content[:200] + "...",
                            height=80,
                            key=f"entity_content_{i}",
                            disabled=True,
                        )
                    else:
                        st.text_area(
                            "Context:",
                            content,
                            height=60,
                            key=f"entity_content_full_{i}",
                            disabled=True,
                        )
            elif source_type == "Hybrid Chunk":
                # Display hybrid chunk information (chunk + entities)
                content = source.get("content", "No content available")
                if len(content) > 300:
                    st.text_area(
                        "Content Preview:",
                        content[:300] + "...",
                        height=100,
                        key=f"hybrid_preview_{i}",
                        disabled=True,
                    )
                    with st.expander("Show Full Content"):
                        st.text_area(
                            "Full Content:",
                            content,
                            height=200,
                            key=f"hybrid_full_{i}",
                            disabled=True,
                        )
                else:
                    st.text_area(
                        "Content:",
                        content,
                        height=max(60, min(len(content.split("\n")) * 20, 150)),
                        key=f"hybrid_content_{i}",
                        disabled=True,
                    )
                
                # Show contained entities
                entities = source.get("contained_entities", [])
                if entities:
                    st.write(f"**Contains Entities:** {', '.join(entities[:5])}")
                    if len(entities) > 5:
                        st.caption(f"... and {len(entities) - 5} more entities")
                
                # Show additional metadata for hybrid chunks
                if source.get("document_name") and source.get("document_name") != source.get("filename"):
                    st.write(f"**Document:** {source.get('document_name')}")
                
                if source.get("chunk_index") is not None:
                    st.write(f"**Chunk:** {source.get('chunk_index') + 1}")
                    
            else:
                st.write(f"**Relevance Score:** {score:.4f}")

                # Display chunk information
                content = source.get("content", "No content available")
                if len(content) > 300:
                    st.text_area(
                        "Content Preview:",
                        content[:300] + "...",
                        height=100,
                        key=f"chunk_preview_{i}",
                        disabled=True,
                    )

                    with st.expander("Show Full Content"):
                        st.text_area(
                            "Full Content:",
                            content,
                            height=200,
                            key=f"chunk_full_{i}",
                            disabled=True,
                        )
                else:
                    st.text_area(
                        "Content:",
                        content,
                        height=max(60, min(len(content.split("\n")) * 20, 150)),
                        key=f"chunk_content_{i}",
                        disabled=True,
                    )
                
                # Show additional metadata for chunks
                if source.get("document_name") and source.get("document_name") != source.get("filename"):
                    st.write(f"**Document:** {source.get('document_name')}")
                
                if source.get("chunk_index") is not None:
                    st.write(f"**Chunk:** {source.get('chunk_index') + 1}")
                    
                # Show entities if this chunk contains any (even in non-hybrid mode)
                entities = source.get("contained_entities", [])
                if entities:
                    st.caption(f"**Contains:** {', '.join(entities[:3])}")
                    if len(entities) > 3:
                        st.caption(f"... +{len(entities) - 3} more")
    
    if len(relevant_sources) > 5:
        st.caption(f"Showing top 5 of {len(relevant_sources)} sources")


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

    st.info("Chunks are created and usable immediately. Entity extraction runs in background.")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload documents to expand the knowledge base",
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

                # Process files (chunks only). Entity extraction runs in background.
                results = process_files_background(uploaded_files, progress_container)

                # Display results
                if results["processed_files"]:
                    # Create comprehensive success message
                    success_msg = f"Successfully processed {len(results['processed_files'])} files ({results['total_chunks']} chunks created"
                    if results.get("total_entities", 0) > 0:
                        success_msg += f", {results['total_entities']} entities, {results['total_entity_relationships']} relationships"
                    success_msg += ")"
                    
                    st.toast(icon="✅", body=success_msg)

                if results["errors"]:
                    st.toast(icon="❌", body=f"Failed to process {len(results['errors'])} files")
                    for error_info in results["errors"]:
                        st.write(f"- 📄 {error_info['name']}: {error_info['error']}")

                # Always reset the file uploader after processing (success or failure)
                st.session_state.file_uploader_key += 1
                st.session_state.processing_files = False

                # Force a rerun to refresh the UI and clear the uploader
                time.sleep(3)
                st.rerun()

    return


def get_rag_settings(key_suffix: str = ""):
    """
    Render RAG settings controls in the sidebar and return their values.

    Args:
        key_suffix: Suffix to append to widget keys to keep them unique when
            the settings are rendered from multiple places in the UI.

    Returns:
        Tuple of (retrieval_mode, top_k, temperature, chunk_weight, graph_expansion)
    """
    st.markdown("### 🧠 RAG Settings")

    # Update retrieval modes to match hybrid approach
    mode_options = ["chunk_only", "entity_only", "hybrid"]
    mode_labels = [
        "Chunk Only (Traditional)",
        "Entity Only (GraphRAG)",
        "Hybrid (Best of Both)"
    ]
    
    # Show entity extraction status
    if settings.enable_entity_extraction:
        st.info("✅ Entity extraction enabled - All modes available")
        default_mode = 2  # hybrid
    else:
        st.warning("⚠️ Entity extraction disabled - Only chunk mode available")
        mode_options = ["chunk_only"]
        mode_labels = ["Chunk Only (Traditional)"]
        default_mode = 0

    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        mode_options,
        format_func=lambda x: mode_labels[mode_options.index(x)] if x in mode_options else x,
        index=min(default_mode, len(mode_options) - 1),
        key=f"retrieval_mode{key_suffix}",
        help="""
        • **Chunk Only**: Traditional vector similarity search (fastest)
        • **Entity Only**: GraphRAG-style entity relationship search (slowest, most comprehensive)
        • **Hybrid**: Combines chunk similarity + entity relationships (recommended)
        """
    )

    top_k = st.slider(
        "Number of chunks to retrieve (value can be adjusted after query analysis)",
        min_value=1,
        max_value=15,
        value=5,
        key=f"top_k{key_suffix}",
        help="More chunks provide better context but may include less relevant information"
    )

    temperature = st.slider(
        "Response creativity (temperature)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        key=f"temperature{key_suffix}",
        help="Lower values produce more focused responses, higher values more creative"
    )

    # Show additional hybrid settings if hybrid mode is selected
    chunk_weight = settings.hybrid_chunk_weight  # Default values
    graph_expansion = settings.enable_graph_expansion
    
    if retrieval_mode == "hybrid" and settings.enable_entity_extraction:
        with st.expander("🔧 Advanced Hybrid Settings", expanded=False):
            chunk_weight = st.slider(
                "Chunk Weight (vs Entity Weight)",
                min_value=0.0,
                max_value=1.0,
                value=settings.hybrid_chunk_weight,
                step=0.1,
                key=f"chunk_weight{key_suffix}",
                help="Higher values favor chunk-based results, lower values favor entity-based results"
            )
            
            graph_expansion = st.checkbox(
                "Enable Graph Expansion",
                value=settings.enable_graph_expansion,
                key=f"graph_expansion{key_suffix}",
                help="Use entity relationships to expand context (slower but more comprehensive)"
            )

    return retrieval_mode, top_k, temperature, chunk_weight, graph_expansion


def main():
    """Main Streamlit application."""
    # Title and description
    html_style = '''
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
    '''
    st.markdown(html_style, unsafe_allow_html=True)

    # Create main layout with columns
    main_col, sidebar_col = st.columns([2, 1])  # 2:1 ratio for main content vs sidebar
    
    with main_col:
        # Display chat messages in main column
        st.title("💬 Chat with your documents")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Sidebar for additional information (sources and graphs)
    with sidebar_col:
        container = st.container(width=10)
        
        with container:
            st.markdown('<div class="floating">', unsafe_allow_html=True)
            # st.markdown("### 📊 Context Information")
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🕸️ Context Graph",
                "📚 Sources",
                "📊 Database",
                "📁 Upload File",
                "⚙️"
            ])
            
            # Display information for the latest assistant message if available
            if st.session_state.messages:
                latest_message = None
                for msg in reversed(st.session_state.messages):
                    if msg["role"] == "assistant":
                        latest_message = msg
                        break

                if latest_message:
                    with tab1:
                        # Display graph in sidebar
                        if "graph_fig" in latest_message:
                            st.plotly_chart(
                                latest_message["graph_fig"], use_container_width=True
                            )
                            
                    with tab2:
                        # Display sources in sidebar
                        if "sources" in latest_message:
                            display_sources_detailed(latest_message["sources"])
                            
                    with tab3:
                        display_stats()
                        display_document_list()
                            
                    with tab4:
                        display_document_upload()
                        
                    with tab5:
                        retrieval_mode, top_k, temperature, chunk_weight, graph_expansion = get_rag_settings(key_suffix="_latest")

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
                    retrieval_mode, top_k, temperature, chunk_weight, graph_expansion = get_rag_settings(key_suffix="_default")
                                
            st.markdown('</div>', unsafe_allow_html=True)

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
                        # Ensure RAG settings are defined (read from session state set by widgets)
                        retrieval_mode = st.session_state.get(
                            "retrieval_mode_latest",
                            st.session_state.get("retrieval_mode_default", "graph_enhanced"),
                        )

                        top_k = st.session_state.get(
                            "top_k_latest", st.session_state.get("top_k_default", 5)
                        )

                        temperature = st.session_state.get(
                            "temperature_latest", st.session_state.get("temperature_default", 0.1)
                        )

                        with st.spinner("🔍 Generating response..."):
                            # Process query through RAG pipeline
                            # Ensure hybrid tuning options are read from session state
                            chunk_weight = st.session_state.get(
                                "chunk_weight_latest", st.session_state.get("chunk_weight_default", 0.5)
                            )
                            graph_expansion = st.session_state.get(
                                "graph_expansion_latest", st.session_state.get("graph_expansion_default", True)
                            )

                            result = graph_rag.query(
                                user_query,
                                retrieval_mode=retrieval_mode,
                                top_k=top_k,
                                temperature=temperature,
                                chunk_weight=chunk_weight,
                                graph_expansion=graph_expansion,
                            )

                        # Stream the response
                        full_response = result["response"]
                        st.write_stream(stream_response(full_response, 0.02))

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
