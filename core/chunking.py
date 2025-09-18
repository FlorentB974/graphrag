"""
Document chunking utilities for the RAG pipeline.
"""
import logging
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Handles document chunking with configurable parameters."""
    
    def __init__(self):
        """Initialize the document chunker."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_text(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks and return with metadata.
        
        Args:
            text: The text to chunk
            document_id: Identifier for the source document
            
        Returns:
            List of dictionaries containing chunk data
        """
        try:
            chunks = self.text_splitter.split_text(text)
            
            chunk_data = []
            for i, chunk in enumerate(chunks):
                chunk_info = {
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "content": chunk,
                    "chunk_index": i,
                    "document_id": document_id,
                    "metadata": {
                        "chunk_size": len(chunk),
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }
                chunk_data.append(chunk_info)
            
            logger.info(f"Successfully chunked document {document_id} into {len(chunks)} chunks")
            return chunk_data
            
        except Exception as e:
            logger.error(f"Failed to chunk document {document_id}: {e}")
            raise
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries with 'id' and 'content' keys
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            
            if not doc_id or not content:
                logger.warning(f"Skipping document with missing id or content: {doc}")
                continue
            
            chunks = self.chunk_text(content, doc_id)
            all_chunks.extend(chunks)
        
        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks")
        return all_chunks


# Global chunker instance
document_chunker = DocumentChunker()
