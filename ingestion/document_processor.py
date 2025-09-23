"""
Multi-format document processor for the RAG pipeline.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.chunking import document_chunker
from core.embeddings import embedding_manager
from core.graph_db import graph_db
from config.settings import settings
import concurrent.futures
from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.text_loader import TextLoader

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes documents of various formats and stores them in the graph database."""

    def __init__(self):
        """Initialize the document processor."""
        self.loaders = {
            ".pdf": PDFLoader(),
            ".docx": DOCXLoader(),
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".py": TextLoader(),
            ".js": TextLoader(),
            ".html": TextLoader(),
            ".css": TextLoader(),
        }

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and modification time."""
        content = f"{file_path}_{file_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()

    def _extract_metadata(
        self, file_path: Path, original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract metadata from file."""
        # Use original filename if provided, otherwise use file path name
        filename = original_filename if original_filename else file_path.name
        # Replace spaces with underscores for cleaner database storage
        if original_filename and " " in filename:
            filename = filename.replace(" ", "_")

        return {
            "filename": filename,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "file_extension": file_path.suffix,
            "created_at": file_path.stat().st_ctime,
            "modified_at": file_path.stat().st_mtime,
        }

    def process_file(
        self, file_path: Path, original_filename: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single file and store it in the graph database.

        Args:
            file_path: Path to the file to process
            original_filename: Optional original filename to preserve (useful for uploaded files)

        Returns:
            Processing result dictionary or None if failed
        """
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Get appropriate loader
            loader = self.loaders.get(file_path.suffix.lower())
            if not loader:
                logger.warning(f"No loader available for file type: {file_path.suffix}")
                return None

            # Generate document ID and extract metadata
            doc_id = self._generate_document_id(file_path)
            metadata = self._extract_metadata(file_path, original_filename)

            logger.info(f"Processing file: {file_path}")

            # Load document content
            content = loader.load(file_path)
            if not content:
                logger.warning(f"No content extracted from: {file_path}")
                return None

            # Create document node
            graph_db.create_document_node(doc_id, metadata)

            # Chunk the document
            chunks = document_chunker.chunk_text(content, doc_id)

            # Process each chunk in parallel to generate embeddings
            processed_chunks = []

            # Prepare inputs
            chunk_contents = [c["content"] for c in chunks]
            chunk_ids = [c["chunk_id"] for c in chunks]
            chunk_metas = [c.get("metadata", {}) for c in chunks]

            # Use ThreadPoolExecutor to perform blocking embedding calls concurrently.
            max_workers = getattr(settings, "embedding_concurrency")
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Map returns results in order of inputs
                    future_to_index = {executor.submit(embedding_manager.get_embedding, text): i for i, text in enumerate(chunk_contents)}
                    for future in concurrent.futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            embedding = future.result()
                        except Exception as e:
                            logger.error(f"Embedding failed for chunk {chunk_ids[idx]}: {e}")
                            embedding = []

                        # Store chunk in graph database
                        graph_db.create_chunk_node(
                            chunk_id=chunk_ids[idx],
                            doc_id=doc_id,
                            content=chunk_contents[idx],
                            embedding=embedding,
                            metadata=chunk_metas[idx],
                        )

                        processed_chunks.append(
                            {
                                "chunk_id": chunk_ids[idx],
                                "content": chunk_contents[idx],
                                "metadata": chunk_metas[idx],
                            }
                        )
            except Exception as e:
                logger.error(f"Failed to generate embeddings concurrently: {e}")
                # Fallback: process sequentially
                processed_chunks = []
                for chunk_data in chunks:
                    chunk_id = chunk_data["chunk_id"]
                    chunk_content = chunk_data["content"]
                    chunk_metadata = chunk_data.get("metadata", {})
                    try:
                        embedding = embedding_manager.get_embedding(chunk_content)
                    except Exception as e2:
                        logger.error(f"Embedding failed for chunk {chunk_id} in fallback: {e2}")
                        embedding = []

                    graph_db.create_chunk_node(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        content=chunk_content,
                        embedding=embedding,
                        metadata=chunk_metadata,
                    )

                    processed_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "content": chunk_content,
                            "metadata": chunk_metadata,
                        }
                    )

            # Create similarity relationships between chunks
            try:
                relationships_created = graph_db.create_chunk_similarities(doc_id)
                logger.info(
                    f"Created {relationships_created} similarity relationships for document {doc_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to create similarity relationships for document {doc_id}: {e}"
                )
                relationships_created = 0

            result = {
                "document_id": doc_id,
                "file_path": str(file_path),
                "chunks_created": len(processed_chunks),
                "similarity_relationships_created": relationships_created,
                "metadata": metadata,
                "status": "success",
            }

            logger.info(
                f"Successfully processed {file_path}: {len(processed_chunks)} chunks created"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return {"file_path": str(file_path), "status": "error", "error": str(e)}

    def process_directory(
        self, directory_path: Path, recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.

        Args:
            directory_path: Path to the directory to process
            recursive: Whether to process subdirectories

        Returns:
            List of processing results
        """
        results = []

        if not directory_path.exists() or not directory_path.is_dir():
            logger.error(f"Directory not found or not a directory: {directory_path}")
            return results

        # Find all supported files
        pattern = "**/*" if recursive else "*"
        for file_path in directory_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                result = self.process_file(file_path)
                if result:
                    results.append(result)

        logger.info(f"Processed directory {directory_path}: {len(results)} files")
        return results

    def process_multiple_files(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple files.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of processing results
        """
        results = []

        for file_path in file_paths:
            result = self.process_file(file_path)
            if result:
                results.append(result)

        logger.info(f"Processed {len(file_paths)} files: {len(results)} successful")
        return results

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.loaders.keys())


# Global document processor instance
document_processor = DocumentProcessor()
