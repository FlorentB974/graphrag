"""
Chainlit web interface for the GraphRAG pipeline.
"""
import asyncio
import logging
from pathlib import Path
from typing import List
import chainlit as cl
from chainlit.input_widget import Select, Slider
from rag.graph_rag import graph_rag
from ingestion.document_processor import document_processor
from core.graph_db import graph_db
from config.settings import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    await cl.Message(
        content="""# Welcome to GraphRAG Pipeline! 🚀

I'm your intelligent document assistant powered by LangGraph and Neo4j. Here's what I can do:

## 📚 **Document Management**
- Upload and process various file formats (PDF, DOCX, TXT, MD, etc.)
- Intelligent document chunking and embedding
- Graph-based relationship mapping

## 🧠 **Smart Retrieval**
- Vector similarity search
- Graph traversal for enhanced context
- Multi-step reasoning capabilities

## 📊 **Visualization**
- See the knowledge graph structure
- View relevant document chunks for each query
- Track reasoning paths

## 🎯 **Getting Started**
1. **Upload documents** using the file upload feature
2. **Ask questions** about your documents
3. **Explore relationships** through interactive queries

Try asking questions like:
- "What are the main concepts in the uploaded documents?"
- "How are these topics related?"
- "Can you compare different approaches mentioned?"

You can also upload documents directly in this chat to expand the knowledge base!
        """,
        author="GraphRAG Assistant"
    ).send()
    
    # Display current database stats
    try:
        stats = graph_db.get_graph_stats()
        await cl.Message(
            content=f"""📈 **Current Knowledge Base Stats:**
- 📄 Documents: {stats.get('documents', 0)}
- 🧩 Chunks: {stats.get('chunks', 0)}
- 🔗 Relationships: {stats.get('similarity_relations', 0)}
            """,
            author="System"
        ).send()
    except Exception as e:
        logger.warning(f"Could not fetch database stats: {e}")


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    try:
        # Check if message contains files
        if message.elements:
            await handle_file_uploads(message.elements)
            return
        
        # Process text query
        user_query = message.content
        
        # Show processing message
        processing_msg = cl.Message(content="🔍 Processing your query...", author="System")
        await processing_msg.send()
        
        # Process query through RAG pipeline
        result = await asyncio.to_thread(graph_rag.query, user_query)
        
        # Update processing message
        processing_msg.content = "✅ Query processed!"
        await processing_msg.update()
        
        # Send main response
        response_content = f"""## 💡 Answer

{result['response']}

---

## 📚 Sources ({len(result['sources'])} chunks used)
"""
        
        # Add sources information
        for i, source in enumerate(result['sources'][:5], 1):  # Limit to top 5 sources
            chunk_preview = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
            similarity = source.get('similarity', 0.0)
            
            response_content += f"""
**Source {i}** (Similarity: {similarity:.3f})
```
{chunk_preview}
```
"""
        
        await cl.Message(
            content=response_content,
            author="GraphRAG Assistant"
        ).send()
        
        # Send additional context if available
        if result.get('query_analysis'):
            analysis = result['query_analysis']
            context_content = f"""## 🔍 Query Analysis

- **Type**: {analysis.get('query_type', 'Unknown')}
- **Complexity**: {analysis.get('complexity', 'Unknown')}
- **Key Concepts**: {', '.join(analysis.get('key_concepts', []))}
- **Chunks Retrieved**: {len(result.get('retrieved_chunks', []))}
- **Graph Context Added**: {len(result.get('graph_context', [])) - len(result.get('retrieved_chunks', []))}
            """
            
            await cl.Message(
                content=context_content,
                author="System"
            ).send()
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(
            content=f"❌ I encountered an error processing your request: {str(e)}",
            author="System"
        ).send()


async def handle_file_uploads(elements: List[cl.CustomElement]):
    """Handle uploaded files."""
    try:
        processing_msg = cl.Message(content="📁 Processing uploaded files...", author="System")
        await processing_msg.send()
        
        processed_files = []
        total_chunks = 0
        
        for element in elements:
            if isinstance(element, cl.File):
                # Save uploaded file temporarily
                file_path = Path(f"temp_{element.name}")
                # Ensure content is bytes before writing (handle None and str)
                content = element.content
                if content is None:
                    content_bytes = b""
                elif isinstance(content, str):
                    content_bytes = content.encode("utf-8")
                else:
                    content_bytes = content
                with open(file_path, "wb") as f:
                    f.write(content_bytes)
                
                try:
                    # Process the file
                    result = await asyncio.to_thread(document_processor.process_file, file_path)
                    
                    if result and result.get("status") == "success":
                        processed_files.append({
                            "name": element.name,
                            "chunks": result.get("chunks_created", 0),
                            "document_id": result.get("document_id")
                        })
                        total_chunks += result.get("chunks_created", 0)
                    else:
                        processed_files.append({
                            "name": element.name,
                            "error": result.get("error", "Unknown error") if result else "Processing failed"
                        })
                
                finally:
                    # Clean up temporary file
                    if file_path.exists():
                        file_path.unlink()
        
        # Send results
        if processed_files:
            success_files = [f for f in processed_files if "error" not in f]
            error_files = [f for f in processed_files if "error" in f]
            
            result_content = f"✅ **File Processing Complete!**\n\n"
            
            if success_files:
                result_content += f"**Successfully processed {len(success_files)} files ({total_chunks} chunks created):**\n"
                for file_info in success_files:
                    result_content += f"- 📄 {file_info['name']}: {file_info['chunks']} chunks\n"
            
            if error_files:
                result_content += f"\n❌ **Failed to process {len(error_files)} files:**\n"
                for file_info in error_files:
                    result_content += f"- 📄 {file_info['name']}: {file_info['error']}\n"
            
            result_content += f"\n🎯 You can now ask questions about the uploaded documents!"
            
            await cl.Message(content=result_content, author="System").send()
        
    except Exception as e:
        logger.error(f"Error handling file uploads: {e}")
        await cl.Message(
            content=f"❌ Error processing files: {str(e)}",
            author="System"
        ).send()


@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates."""
    logger.info(f"Settings updated: {settings}")


# Settings configuration
settings_config = [
    Select(
        id="retrieval_mode",
        label="Retrieval Mode",
        values=["simple", "graph_enhanced", "hybrid"],
        initial_index=1,
    ),
    Slider(
        id="top_k",
        label="Number of chunks to retrieve",
        initial=5,
        min=1,
        max=10,
        step=1,
    ),
    Slider(
        id="temperature",
        label="Response creativity (temperature)",
        initial=0.7,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
]


if __name__ == "__main__":
    # Run the Chainlit app
    import chainlit.cli
    chainlit.cli.run_chainlit()