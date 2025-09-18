"""
Streamlit web interface for the GraphRAG pipeline.
"""
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Generator
import streamlit as st
import time
from rag.graph_rag import graph_rag
from ingestion.document_processor import document_processor
from core.graph_db import graph_db
from config.settings import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="GraphRAG Pipeline",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'processing_files' not in st.session_state:
    st.session_state.processing_files = False
if 'show_graph' not in st.session_state:
    st.session_state.show_graph = False


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
        partial_text = " ".join(words[:i + 1])
        yield partial_text
        time.sleep(delay)


def process_files_background(uploaded_files: List[Any], progress_container) -> Dict[str, Any]:
    """
    Process uploaded files in background.
    
    Args:
        uploaded_files: List of uploaded file objects
        progress_container: Streamlit container for progress updates
        
    Returns:
        Dictionary with processing results
    """
    results = {
        'processed_files': [],
        'total_chunks': 0,
        'errors': []
    }
    
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = Path(tmp_file.name)
            
            try:
                # Process the file
                result = document_processor.process_file(tmp_path)
                
                if result and result.get("status") == "success":
                    results['processed_files'].append({
                        "name": uploaded_file.name,
                        "chunks": result.get("chunks_created", 0),
                        "document_id": result.get("document_id")
                    })
                    results['total_chunks'] += result.get("chunks_created", 0)
                else:
                    results['errors'].append({
                        "name": uploaded_file.name,
                        "error": result.get("error", "Unknown error") if result else "Processing failed"
                    })
            
            finally:
                # Clean up temporary file
                if tmp_path.exists():
                    tmp_path.unlink()
            
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            results['errors'].append({
                "name": uploaded_file.name,
                "error": str(e)
            })
    
    status_text.text("File processing complete!")
    return results


def display_stats():
    """Display database statistics in sidebar."""
    try:
        stats = graph_db.get_graph_stats()
        
        st.sidebar.markdown("### 📊 Database Stats")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Documents", stats.get('documents', 0))
            st.metric("Relationships", stats.get('similarity_relations', 0))
        
        with col2:
            st.metric("Chunks", stats.get('chunks', 0))
            
    except Exception as e:
        st.sidebar.error(f"Could not fetch database stats: {e}")


def display_query_analysis(analysis: Dict[str, Any]):
    """Display query analysis results."""
    if not analysis:
        return
    
    with st.expander("🔍 Query Analysis", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Type:** " + analysis.get('query_type', 'Unknown'))
            st.write("**Complexity:** " + analysis.get('complexity', 'Unknown'))
        
        with col2:
            key_concepts = analysis.get('key_concepts', [])
            if key_concepts:
                st.write("**Key Concepts:** " + ", ".join(key_concepts))


def display_sources(sources: List[Dict[str, Any]]):
    """Display source chunks used in response."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} chunks used)", expanded=False):
        for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
            chunk_preview = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
            similarity = source.get('similarity', 0.0)
            
            st.markdown(f"""
            **Source {i}** (Similarity: {similarity:.3f})
            ```
            {chunk_preview}
            ```
            """)


def main():
    """Main Streamlit application."""
    # Title and description
    st.title("🚀 GraphRAG Pipeline")
    st.markdown("*Intelligent document assistant powered by LangGraph and Neo4j*")
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # RAG settings
    with st.sidebar.expander("🧠 RAG Settings", expanded=True):
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["simple", "graph_enhanced", "hybrid"],
            index=1
        )
        
        top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=5
        )
        
        temperature = st.slider(
            "Response creativity (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
    
    # Display database stats
    display_stats()
    
    # File upload section
    st.sidebar.markdown("### 📁 Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'txt', 'md'],
        accept_multiple_files=True,
        help="Upload documents to expand the knowledge base"
    )
    
    # Process uploaded files
    if uploaded_files and not st.session_state.processing_files:
        if st.sidebar.button("🚀 Process Files"):
            st.session_state.processing_files = True
            
            with st.sidebar.container():
                st.markdown("### � Processing Progress")
                progress_container = st.container()
                
                # Process files
                results = process_files_background(uploaded_files, progress_container)
                
                # Display results
                if results['processed_files']:
                    st.success(f"✅ Successfully processed {len(results['processed_files'])} files ({results['total_chunks']} chunks created)")
                    for file_info in results['processed_files']:
                        st.write(f"- 📄 {file_info['name']}: {file_info['chunks']} chunks")
                
                if results['errors']:
                    st.error(f"❌ Failed to process {len(results['errors'])} files")
                    for error_info in results['errors']:
                        st.write(f"- 📄 {error_info['name']}: {error_info['error']}")
                
                st.session_state.processing_files = False
    
    # Main chat interface
    st.markdown("### 💬 Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display additional data if available
            if "query_analysis" in message:
                display_query_analysis(message["query_analysis"])
            
            if "sources" in message:
                display_sources(message["sources"])
            
            if "graph_fig" in message:
                st.plotly_chart(message["graph_fig"], use_container_width=True)
    
    # Chat input
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            # Show processing indicator
            processing_placeholder = st.empty()
            processing_placeholder.markdown("🔍 Processing your query...")
            
            try:
                # Process query through RAG pipeline
                result = graph_rag.query(user_query)
                
                # Clear processing indicator
                processing_placeholder.empty()
                
                # Stream the response
                response_placeholder = st.empty()
                full_response = result['response']
                
                # Stream response word by word
                for partial_response in stream_response(full_response):
                    response_placeholder.markdown(partial_response)
                
                # Display query analysis
                if result.get('query_analysis'):
                    display_query_analysis(result['query_analysis'])
                
                # Display sources
                if result.get('sources'):
                    display_sources(result['sources'])
                
                # Add assistant message to session state
                message_data = {
                    "role": "assistant", 
                    "content": full_response,
                    "query_analysis": result.get('query_analysis'),
                    "sources": result.get('sources')
                }
                
                # Add graph visualization if requested
                if st.sidebar.checkbox("Show Query Graph", key="show_query_graph"):
                    try:
                        # Import here to avoid issues if dependencies aren't installed
                        from core.graph_viz import create_query_result_graph
                        
                        if result.get('sources'):
                            query_fig = create_query_result_graph(result['sources'])
                            st.plotly_chart(query_fig, use_container_width=True)
                            message_data["graph_fig"] = query_fig
                    except ImportError:
                        st.warning("Graph visualization dependencies not installed. Run: pip install plotly networkx")
                    except Exception as e:
                        st.warning(f"Could not create query graph: {e}")
                
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                processing_placeholder.empty()
                error_msg = f"❌ I encountered an error processing your request: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Knowledge Graph Visualization Section
    st.markdown("---")
    st.markdown("### 📊 Knowledge Graph Visualization")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("🔍 Show Full Knowledge Graph"):
            st.session_state.show_graph = True
    
    with col2:
        layout_algo = st.selectbox(
            "Layout Algorithm",
            ["spring", "circular", "kamada_kawai"],
            key="layout_select"
        )
    
    with col3:
        node_limit = st.number_input(
            "Max Nodes",
            min_value=10,
            max_value=500,
            value=100,
            key="node_limit"
        )
    
    if st.session_state.show_graph:
        try:
            # Import here to avoid issues if dependencies aren't installed
            from core.graph_viz import get_graph_data, create_plotly_graph
            
            with st.spinner("Loading knowledge graph..."):
                graph_data = get_graph_data(limit=node_limit)
                
                if graph_data['nodes']:
                    fig = create_plotly_graph(graph_data, layout_algorithm=layout_algo)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show graph statistics
                    st.info(f"📊 Graph contains {graph_data['total_nodes']} nodes and {graph_data['total_edges']} edges")
                else:
                    st.warning("No graph data available. Upload some documents first!")
                
        except ImportError:
            st.error("Graph visualization dependencies not installed. Run: pip install plotly networkx")
        except Exception as e:
            st.error(f"Error creating graph visualization: {e}")


if __name__ == "__main__":
    main()