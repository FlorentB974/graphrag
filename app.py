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
    words = text.split()
    for i in range(len(words)):
        partial_text = " ".join(words[: i + 1])
        yield partial_text
        time.sleep(delay)


def process_files_background(
    uploaded_files: List[Any], progress_container
) -> Dict[str, Any]:
    """
    Process uploaded files in background.

    Args:
        uploaded_files: List of uploaded file objects
        progress_container: Streamlit container for progress updates

    Returns:
        Dictionary with processing results
    """
    results = {"processed_files": [], "total_chunks": 0, "errors": []}

    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)

            try:
                # Process the file with original filename
                result = document_processor.process_file(tmp_path, uploaded_file.name)

                if result and result.get("status") == "success":
                    results["processed_files"].append(
                        {
                            "name": uploaded_file.name,
                            "chunks": result.get("chunks_created", 0),
                            "document_id": result.get("document_id"),
                        }
                    )
                    results["total_chunks"] += result.get("chunks_created", 0)
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

            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)

        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            results["errors"].append({"name": uploaded_file.name, "error": str(e)})

    status_text.text("File processing complete!")
    return results


def display_stats():
    """Display database statistics in sidebar."""
    try:
        stats = graph_db.get_graph_stats()

        st.sidebar.markdown("### 📊 Database Stats")
        col1, col2 = st.sidebar.columns(2)

        with col1:
            st.metric("Documents", stats.get("documents", 0))
            st.metric("Relationships", stats.get("similarity_relations", 0))

        with col2:
            st.metric("Chunks", stats.get("chunks", 0))

    except Exception as e:
        st.sidebar.error(f"Could not fetch database stats: {e}")


def display_document_list():
    """Display list of documents in the database with delete options."""
    try:
        documents = graph_db.get_all_documents()

        if not documents:
            st.sidebar.info("No documents in the database yet.")
            return

        st.sidebar.markdown("### 📂 Documents in Database")

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

            with st.sidebar.expander(f"📄 {filename}", expanded=False):
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
        st.sidebar.markdown(f"**Total:** {len(documents)} documents")

    except Exception as e:
        st.sidebar.error(f"Could not fetch document list: {e}")


def display_sources_detailed(sources: List[Dict[str, Any]]):
    """
    Display detailed source chunks in a formatted sidebar.

    Args:
        sources: List of source chunks with metadata
    """
    if not sources:
        st.sidebar.write("No sources used in this response.")
        return

    st.markdown(f"### 📚 Sources ({len(sources)})")

    for i, source in enumerate(sources, 1):
        if source.get("similarity"):
            with st.expander(
                f"📄 Source {i} (Relevance: {source.get('similarity', 0.0):.3f})",
                expanded=False,
            ):
                # Display document information
                doc_name = source.get(
                    "document_name", source.get("filename", "Unknown Document")
                )
                st.write(f"**Document:** {doc_name}")

                st.write(f"**Relevance Score:** {source['similarity']:.4f}")

                # Display chunk content with proper formatting
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


def display_query_analysis_detailed(analysis: Dict[str, Any]):
    """
    Display detailed query analysis in sidebar.

    Args:
        analysis: Query analysis results
    """
    if not analysis:
        return

    st.markdown("### 🔍 Query Analysis")

    with st.expander("Analysis Details", expanded=False):
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


def main():
    """Main Streamlit application."""
    # Title and description
    st.title("🚀 GraphRAG Pipeline")

    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")

    # RAG settings
    with st.sidebar.expander("🧠 RAG Settings", expanded=True):
        retrieval_mode = st.selectbox(
            "Retrieval Mode", ["simple", "graph_enhanced", "hybrid"], index=1
        )

        top_k = st.slider(
            "Number of chunks to retrieve", min_value=1, max_value=10, value=5
        )

        temperature = st.slider(
            "Response creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
        )

    # Display database stats
    display_stats()

    # File upload section
    st.sidebar.markdown("### 📁 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose files to upload",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload documents to expand the knowledge base",
        key=f"file_uploader_{st.session_state.file_uploader_key}",
    )

    # Process uploaded files
    if uploaded_files and not st.session_state.processing_files:
        if st.sidebar.button("🚀 Process Files"):
            st.session_state.processing_files = True

            with st.sidebar.container():
                st.markdown("### 📝 Processing Progress")
                progress_container = st.container()

                # Process files
                results = process_files_background(uploaded_files, progress_container)

                # Display results
                if results["processed_files"]:
                    st.success(
                        f"✅ Successfully processed {len(results['processed_files'])} files ({results['total_chunks']} chunks created)"
                    )
                    for file_info in results["processed_files"]:
                        st.write(
                            f"- 📄 {file_info['name']}: {file_info['chunks']} chunks"
                        )

                if results["errors"]:
                    st.error(f"❌ Failed to process {len(results['errors'])} files")
                    for error_info in results["errors"]:
                        st.write(f"- 📄 {error_info['name']}: {error_info['error']}")

                # Always reset the file uploader after processing (success or failure)
                st.session_state.file_uploader_key += 1
                st.session_state.processing_files = False

                # Force a rerun to refresh the UI and clear the uploader
                time.sleep(3)  # Small delay to show results
                st.rerun()

    # Display document list
    display_document_list()

    # Create main layout with columns
    main_col, sidebar_col = st.columns([2, 1])  # 2:1 ratio for main content vs sidebar

    with main_col:
        # Display chat messages in main column
        st.markdown("### 💬 Chat with your documents")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Sidebar for additional information (sources and graphs)
    with sidebar_col:
        st.markdown("### 📊 Context Information")

        # Display information for the latest assistant message if available
        if st.session_state.messages:
            latest_message = None
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    latest_message = msg
                    break

            if latest_message:
                # Display graph in sidebar
                if "graph_fig" in latest_message:
                    st.markdown("### 🕸️ Context Graph")
                    st.plotly_chart(
                        latest_message["graph_fig"], use_container_width=True
                    )

                # Display query analysis in sidebar
                if "query_analysis" in latest_message:
                    display_query_analysis_detailed(latest_message["query_analysis"])

                # Display sources in sidebar
                if "sources" in latest_message:
                    display_sources_detailed(latest_message["sources"])
        else:
            st.info("💡 Start a conversation to see context information here!")

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
                    # Show processing indicator
                    processing_placeholder = st.empty()
                    processing_placeholder.markdown("🔍 Processing your query...")

                    try:
                        # Process query through RAG pipeline
                        result = graph_rag.query(
                            user_query,
                            retrieval_mode=retrieval_mode,
                            top_k=top_k,
                            temperature=temperature,
                        )

                        # Clear processing indicator
                        processing_placeholder.empty()

                        # Stream the response
                        response_placeholder = st.empty()
                        full_response = result["response"]

                        # Stream response word by word with markdown formatting
                        for partial_response in stream_response(full_response):
                            response_placeholder.markdown(partial_response)

                        response_placeholder.markdown(
                            full_response
                        )  # Ensure full response at the end

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
                        processing_placeholder.empty()
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

    # Knowledge Graph Visualization Section (moved to bottom)
    # st.markdown("---")

    # with st.expander("🔍 Advanced Graph Options", expanded=False):
    #     st.markdown("### 📊 Advanced Knowledge Graph Visualization")

    #     col1, col2, col3 = st.columns([2, 1, 1])

    #     with col1:
    #         if st.button("🔍 Show Advanced Graph", key="show_advanced_graph"):
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

    #             with st.spinner("Loading advanced knowledge graph..."):
    #                 graph_data = get_graph_data(limit=node_limit)

    #                 if graph_data['nodes']:
    #                     fig = create_plotly_graph(graph_data, layout_algorithm=layout_algo)
    #                     st.plotly_chart(fig, use_container_width=True)

    #                     # Show graph statistics
    #                     st.info(f"📊 Graph contains {graph_data['total_nodes']} nodes and {graph_data['total_edges']} edges")
    #                 else:
    #                     st.warning("No graph data available. Upload some documents first!")

    #         except ImportError:
    #             st.error("Graph visualization dependencies not installed. Run: pip install plotly networkx")
    #         except Exception as e:
    #             st.error(f"Error creating graph visualization: {e}")


if __name__ == "__main__":
    main()
