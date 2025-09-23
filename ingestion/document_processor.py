"""
Multi-format document processor for the RAG pipeline.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.chunking import document_chunker
from core.embeddings import embedding_manager
from core.graph_db import graph_db
from config.settings import settings
import asyncio
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

    async def process_file_async(self, chunks: List[Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """Asynchronously embed chunks and store them in the graph DB.

        Args:
            chunks: list of chunk dicts
            doc_id: document id

        Returns:
            List of processed chunk summaries
        """
        processed_chunks: List[Dict[str, Any]] = []

        concurrency = getattr(settings, "embedding_concurrency")
        sem = asyncio.Semaphore(concurrency)

        async def _embed_and_store(chunk):
            content = chunk["content"]
            chunk_id = chunk["chunk_id"]
            metadata = chunk.get("metadata", {})

            async with sem:
                try:
                    embedding = await embedding_manager.aget_embedding(content)
                except Exception as e:
                    logger.error(f"Async embedding failed for {chunk_id}: {e}")
                    embedding = []

            # Persist to DB in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, graph_db.create_chunk_node, chunk_id, doc_id, content, embedding, metadata)

            return {"chunk_id": chunk_id, "content": content, "metadata": metadata}

        tasks = [asyncio.create_task(_embed_and_store(c)) for c in chunks]

        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                processed_chunks.append(res)
            except Exception as e:
                logger.error(f"Error in embedding task: {e}")

        return processed_chunks

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
            start_time = time.time()
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

            # Process chunks asynchronously with configurable concurrency
            try:
                # Use asyncio.run for a synchronous wrapper
                processed_chunks = asyncio.run(self.process_file_async(chunks, doc_id))
            except RuntimeError:
                # If an event loop is already running, get it and run until complete
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new task and wait for it
                    processed_chunks = loop.run_until_complete(self.process_file_async(chunks, doc_id))
                else:
                    processed_chunks = loop.run_until_complete(self.process_file_async(chunks, doc_id))

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
            # add processing duration
            duration = time.time() - start_time
            result["duration_seconds"] = duration

            # Print to stdout for quick feedback
            print(f"Processed {file_path} in {duration:.2f}s — {len(processed_chunks)} chunks")

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
