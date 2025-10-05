"""
Multi-format document processor for the RAG pipeline.
"""

import asyncio
import hashlib
import logging
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.chunking import document_chunker
from core.embeddings import embedding_manager
from core.entity_extraction import EntityExtractor
from core.graph_db import graph_db
from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.image_loader import ImageLoader
from ingestion.loaders.pptx_loader import PPTXLoader
from ingestion.loaders.text_loader import TextLoader
from ingestion.loaders.xlsx_loader import XLSXLoader

logger = logging.getLogger(__name__)


class EntityExtractionState(Enum):
    """States for entity extraction operations."""

    STARTING = "starting"
    LLM_EXTRACTION = "llm_extraction"
    EMBEDDING_GENERATION = "embedding_generation"
    DATABASE_OPERATIONS = "database_operations"
    VALIDATION = "validation"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class EntityExtractionStatus:
    """Status of an entity extraction operation."""

    operation_id: str
    document_id: str
    document_name: str
    state: EntityExtractionState
    started_at: float
    last_updated: float
    progress_info: Optional[str] = None
    error_message: Optional[str] = None


class DocumentProcessor:
    """Processes documents of various formats and stores them in the graph database."""

    def __init__(self):
        """Initialize the document processor."""
        # Initialize loaders with intelligent OCR support
        image_loader = ImageLoader()
        self.loaders = {
            ".pdf": PDFLoader(),  # PDF loader with intelligent OCR
            ".docx": DOCXLoader(),
            ".txt": TextLoader(),
            ".md": TextLoader(),
            ".py": TextLoader(),
            ".js": TextLoader(),
            ".html": TextLoader(),
            ".css": TextLoader(),
            ".csv": CSVLoader(),
            ".pptx": PPTXLoader(),
            ".xlsx": XLSXLoader(),
            ".xls": XLSXLoader(),  # Also support legacy Excel format
            # Add support for image files with intelligent OCR
            ".jpg": image_loader,
            ".jpeg": image_loader,
            ".png": image_loader,
            ".tiff": image_loader,
            ".bmp": image_loader,
        }
        
        # Smart OCR is always enabled and applied intelligently
        # No user configuration needed - OCR is applied only where necessary
        self.enable_quality_filtering = getattr(settings, "enable_quality_filtering", True)

        # Initialize entity extractor if enabled
        if settings.enable_entity_extraction:
            self.entity_extractor = EntityExtractor()
            logger.info("Entity extraction enabled")
        else:
            self.entity_extractor = None
            logger.info("Entity extraction disabled")

        # Track background entity extraction threads so the UI can detect ongoing work
        self._bg_entity_threads = []
        self._bg_lock = threading.Lock()

        # Track entity extraction operation states for more robust status checking
        self._entity_extraction_operations: Dict[str, EntityExtractionStatus] = {}
        self._operations_lock = threading.Lock()

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate a unique document ID based on file path and modification time."""
        content = f"{file_path}_{file_path.stat().st_mtime}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_entity_id(self, entity_name: str) -> str:
        """Generate a consistent entity ID from entity name."""
        return hashlib.md5(entity_name.lower().encode()).hexdigest()[:16]

    def _generate_operation_id(
        self, doc_id: str, operation_type: str = "entity_extraction"
    ) -> str:
        """Generate a unique operation ID."""
        timestamp = str(time.time())
        content = f"{doc_id}_{operation_type}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _start_entity_operation(self, doc_id: str, doc_name: str) -> str:
        """Start tracking an entity extraction operation."""
        operation_id = self._generate_operation_id(doc_id)
        with self._operations_lock:
            self._entity_extraction_operations[operation_id] = EntityExtractionStatus(
                operation_id=operation_id,
                document_id=doc_id,
                document_name=doc_name,
                state=EntityExtractionState.STARTING,
                started_at=time.time(),
                last_updated=time.time(),
            )
        return operation_id

    def _update_entity_operation(
        self,
        operation_id: str,
        state: EntityExtractionState,
        progress_info: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Update the state of an entity extraction operation."""
        with self._operations_lock:
            if operation_id in self._entity_extraction_operations:
                status = self._entity_extraction_operations[operation_id]
                status.state = state
                status.last_updated = time.time()
                if progress_info:
                    status.progress_info = progress_info
                if error_message:
                    status.error_message = error_message

    def _complete_entity_operation(self, operation_id: str):
        """Mark an entity extraction operation as completed and remove it from tracking."""
        with self._operations_lock:
            if operation_id in self._entity_extraction_operations:
                del self._entity_extraction_operations[operation_id]

    def _cleanup_stale_operations(self, max_age_seconds: float = 3600):
        """Remove operations that are too old (likely stale due to crashes)."""
        current_time = time.time()
        with self._operations_lock:
            stale_ops = [
                op_id
                for op_id, status in self._entity_extraction_operations.items()
                if current_time - status.started_at > max_age_seconds
            ]
            for op_id in stale_ops:
                logger.warning(f"Removing stale entity extraction operation: {op_id}")
                del self._entity_extraction_operations[op_id]

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

    def _derive_content_primary_type(self, file_extension: Optional[str]) -> str:
        """Derive a simple primary content type from a file extension.

        Returns a small set of canonical types (pdf, image, text, word, presentation,
        spreadsheet, unknown) used to populate the `content_primary_type` property
        on Document nodes.
        """
        ext = (file_extension or "").lower()
        mapping = {
            ".pdf": "pdf",
            ".docx": "word",
            ".doc": "word",
            ".txt": "text",
            ".md": "text",
            ".py": "text",
            ".js": "text",
            ".html": "text",
            ".css": "text",
            ".csv": "spreadsheet",
            ".xlsx": "spreadsheet",
            ".xls": "spreadsheet",
            ".pptx": "presentation",
            ".ppt": "presentation",
            ".jpg": "image",
            ".jpeg": "image",
            ".png": "image",
            ".tiff": "image",
            ".bmp": "image",
        }
        return mapping.get(ext, "unknown")

    async def process_file_async(
        self, chunks: List[Dict[str, Any]], doc_id: str, progress_callback=None
    ) -> List[Dict[str, Any]]:
        """Asynchronously embed chunks and store them in the graph DB.

        Args:
            chunks: list of chunk dicts
            doc_id: document id
            progress_callback: optional callback function to report progress (chunk_processed_count)

        Returns:
            List of processed chunk summaries
        """
        processed_chunks: List[Dict[str, Any]] = []
        processed_count = 0

        concurrency = getattr(settings, "embedding_concurrency")
        sem = asyncio.Semaphore(concurrency)

        async def _embed_and_store(chunk):
            nonlocal processed_count
            content = chunk["content"]
            chunk_id = chunk["chunk_id"]
            metadata = chunk.get("metadata", {})

            async with sem:
                try:
                    # Add small delay to prevent API flooding
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    embedding = await embedding_manager.aget_embedding(content)
                except Exception as e:
                    logger.error(f"Async embedding failed for {chunk_id}: {e}")
                    embedding = []

            # Persist to DB in a thread to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                graph_db.create_chunk_node,
                chunk_id,
                doc_id,
                content,
                embedding,
                metadata,
            )

            # Report progress
            processed_count += 1
            if progress_callback:
                progress_callback(processed_count)

            return {"chunk_id": chunk_id, "content": content, "metadata": metadata}

        tasks = [asyncio.create_task(_embed_and_store(c)) for c in chunks]

        for coro in asyncio.as_completed(tasks):
            try:
                res = await coro
                processed_chunks.append(res)
            except Exception as e:
                logger.error(f"Error in embedding task: {e}")

        return processed_chunks

    async def _create_entities_async(
        self, entity_dict, relationship_dict, doc_id_local: str
    ) -> tuple[int, int]:
        """Asynchronously create entities and relationships with parallel embedding generation."""
        created_entities = 0
        created_relationships = 0
        
        # Process entities in parallel with controlled concurrency
        concurrency = getattr(settings, "embedding_concurrency")
        sem = asyncio.Semaphore(concurrency)

        async def _create_single_entity(entity):
            nonlocal created_entities
            async with sem:
                try:
                    entity_id = self._generate_entity_id(entity.name)
                    # Add small delay to prevent API flooding
                    await asyncio.sleep(0.1)
                    # Use async entity creation
                    await graph_db.acreate_entity_node(
                        entity_id,
                        entity.name,
                        entity.type,
                        entity.description,
                        entity.importance_score,
                        entity.source_chunks or [],
                    )
                    # Link chunks to entity (run in executor to avoid blocking)
                    loop = asyncio.get_running_loop()
                    for chunk_id in entity.source_chunks or []:
                        try:
                            await loop.run_in_executor(
                                None,
                                graph_db.create_chunk_entity_relationship,
                                chunk_id,
                                entity_id,
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to create chunk-entity rel {chunk_id}->{entity_id}: {e}"
                            )
                    created_entities += 1
                    return entity_id
                except Exception as e:
                    logger.error(
                        f"Failed to persist entity {entity.name} for doc {doc_id_local}: {e}"
                    )
                    return None

        # Create all entities in parallel
        if entity_dict:
            tasks = [asyncio.create_task(_create_single_entity(entity)) for entity in entity_dict.values()]
            
            for coro in asyncio.as_completed(tasks):
                try:
                    await coro
                except Exception as e:
                    logger.error(f"Error in entity creation task: {e}")

        # Create relationships after all entities are created
        # Store entity relationships
        for relationships in relationship_dict.values():
            for relationship in relationships:
                try:
                    source_id = self._generate_entity_id(
                        relationship.source_entity
                    )
                    target_id = self._generate_entity_id(
                        relationship.target_entity
                    )
                    graph_db.create_entity_relationship(
                        entity_id1=source_id,
                        entity_id2=target_id,
                        relationship_type="RELATED_TO",
                        description=relationship.description,
                        strength=relationship.strength,
                        source_chunks=relationship.source_chunks or [],
                    )
                    created_relationships += 1
                except Exception as e:
                    logger.debug(
                        f"Failed to persist relationship for doc {doc_id_local}: {e}"
                    )

        return created_entities, created_relationships

    def process_file(
        self,
        file_path: Path,
        original_filename: Optional[str] = None,
        progress_callback=None,
        enable_quality_filtering: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single file and store it in the graph database.

        Args:
            file_path: Path to the file to process
            original_filename: Optional original filename to preserve (useful for uploaded files)
            progress_callback: Optional callback function to report chunk processing progress
            enable_quality_filtering: Optional override for quality filtering (if None, uses global setting)

        Returns:
            Processing result dictionary or None if failed
        """
        try:
            # Smart OCR is applied automatically by loaders
            use_quality_filtering = enable_quality_filtering if enable_quality_filtering is not None else self.enable_quality_filtering
            
            start_time = time.time()
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Get appropriate loader
            file_ext = file_path.suffix.lower()
            loader = self.loaders.get(file_ext)
            
            # Handle image files with intelligent OCR
            ocr_metadata = {}
            if isinstance(loader, ImageLoader):
                logger.info(f"Processing image file with intelligent OCR: {file_path}")
                result = loader.load_with_metadata(file_path)
                if not result or not result["content"]:
                    logger.info(f"No text content detected in image: {file_path}")
                    return None
                content = result["content"]
                ocr_metadata = result.get("metadata", {})
            elif isinstance(loader, PDFLoader):
                logger.info(f"Processing PDF file with intelligent OCR: {file_path}")
                result = loader.load_with_metadata(file_path)
                if not result or not result["content"]:
                    logger.warning(f"No content extracted from PDF: {file_path}")
                    return None
                content = result["content"]
                ocr_metadata = result.get("metadata", {})
            elif not loader:
                logger.warning(f"No loader available for file type: {file_ext}")
                return None
            else:
                # Load document content using appropriate loader
                logger.info(f"Processing file: {file_path}")
                content = loader.load(file_path)
                if not content:
                    logger.warning(f"No content extracted from: {file_path}")
                    return None

            # Generate document ID and extract metadata
            doc_id = self._generate_document_id(file_path)
            metadata = self._extract_metadata(file_path, original_filename)
            
            # Add OCR metadata if available
            if ocr_metadata:
                metadata.update(ocr_metadata)

            # Ensure content_primary_type is set (derive from file extension if missing)
            metadata["content_primary_type"] = metadata.get(
                "content_primary_type"
            ) or self._derive_content_primary_type(metadata.get("file_extension"))

            # Create document node
            graph_db.create_document_node(doc_id, metadata)

            # Chunk the document with enhanced processing
            if use_quality_filtering:
                chunks = document_chunker.chunk_text(
                    content,
                    doc_id,
                    enable_quality_filtering=use_quality_filtering,
                    enable_ocr_enhancement=False  # OCR already applied by smart loaders
                )
                logger.info(f"Used enhanced chunking (Quality filtering: {use_quality_filtering}) for {doc_id}")
            else:
                chunks = document_chunker.chunk_text(content, doc_id)
                logger.info(f"Used standard chunking for {doc_id}")

            # Process chunks asynchronously with configurable concurrency (embeddings + storing)
            try:
                # Use asyncio.run for a synchronous wrapper
                processed_chunks = asyncio.run(
                    self.process_file_async(chunks, doc_id, progress_callback)
                )
            except RuntimeError:
                # If an event loop is already running, get it and run until complete
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new task and wait for it
                    processed_chunks = loop.run_until_complete(
                        self.process_file_async(chunks, doc_id, progress_callback)
                    )
                else:
                    processed_chunks = loop.run_until_complete(
                        self.process_file_async(chunks, doc_id, progress_callback)
                    )

            # After chunk processing finishes, schedule entity extraction in background (if enabled)
            entity_count = 0
            relationship_count = 0

            should_extract_entities = settings.enable_entity_extraction
            if should_extract_entities and self.entity_extractor is None:
                # Initialize entity extractor if the global setting enabled it at startup
                logger.info(
                    "Initializing entity extractor for background processing..."
                )
                self.entity_extractor = EntityExtractor()

            if should_extract_entities and self.entity_extractor:
                extractor = self.entity_extractor
                # Prepare chunks for entity extraction
                chunks_for_extraction = [
                    {"chunk_id": chunk["chunk_id"], "content": chunk["content"]}
                    for chunk in chunks
                ]

                # Start background thread to perform entity extraction without blocking
                def _background_entity_worker(doc_id_local, chunks_local):
                    operation_id = None
                    try:
                        # Start tracking this operation
                        filename = (
                            original_filename if original_filename else file_path.name
                        )
                        operation_id = self._start_entity_operation(
                            doc_id_local, filename
                        )

                        logger.info(
                            f"Background entity extraction started for document {doc_id_local} (operation: {operation_id})"
                        )

                        # Phase 1: LLM extraction
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.LLM_EXTRACTION,
                            "Running LLM entity extraction",
                        )
                        try:
                            entity_dict, relationship_dict = asyncio.run(
                                extractor.extract_from_chunks(chunks_local)
                            )
                        except Exception as e:
                            logger.error(
                                f"Entity extractor failed in background for {doc_id_local}: {e}"
                            )
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.ERROR,
                                error_message=f"LLM extraction failed: {str(e)}",
                            )
                            return

                        # Phase 2: Embedding generation and database operations
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.EMBEDDING_GENERATION,
                            f"Generating embeddings for {len(entity_dict)} entities",
                        )

                        # Persist entities and relationships asynchronously
                        created_entities, created_relationships = asyncio.run(
                            self._create_entities_async(entity_dict, relationship_dict, doc_id_local)
                        )

                        # Phase 3: Database operations for relationships
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.DATABASE_OPERATIONS,
                            f"Creating {sum(len(rels) for rels in relationship_dict.values())} relationships",
                        )

                        # Store entity relationships
                        for relationships in relationship_dict.values():
                            for relationship in relationships:
                                try:
                                    source_id = self._generate_entity_id(
                                        relationship.source_entity
                                    )
                                    target_id = self._generate_entity_id(
                                        relationship.target_entity
                                    )
                                    graph_db.create_entity_relationship(
                                        entity_id1=source_id,
                                        entity_id2=target_id,
                                        relationship_type="RELATED_TO",
                                        description=relationship.description,
                                        strength=relationship.strength,
                                        source_chunks=relationship.source_chunks or [],
                                    )
                                    created_relationships += 1
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to persist relationship for doc {doc_id_local}: {e}"
                                    )

                        # Optionally create entity similarities for this document
                        try:
                            graph_db.create_entity_similarities(doc_id_local)
                        except Exception as e:
                            logger.debug(
                                f"Failed to create entity similarities for {doc_id_local}: {e}"
                            )

                        # Phase 4: Validation
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.VALIDATION,
                            "Validating entity embeddings",
                        )

                        # Validate entity embeddings after processing
                        try:
                            validation_results = graph_db.validate_entity_embeddings(
                                doc_id_local
                            )
                            if not validation_results["validation_passed"]:
                                logger.warning(
                                    f"Document {doc_id_local} has {validation_results['invalid_embeddings']} invalid entity embeddings"
                                )
                                # Optionally fix invalid embeddings
                                invalid_entity_ids = [
                                    entity["entity_id"]
                                    for entity in validation_results[
                                        "invalid_entity_details"
                                    ]
                                ]
                                if invalid_entity_ids:
                                    logger.info(
                                        f"Attempting to fix {len(invalid_entity_ids)} invalid entity embeddings..."
                                    )
                                    fix_results = graph_db.fix_invalid_embeddings(
                                        entity_ids=invalid_entity_ids
                                    )
                                    logger.info(
                                        f"Fixed {fix_results['entities_fixed']} entity embeddings"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"Failed to validate entity embeddings for document {doc_id_local}: {e}"
                            )

                        logger.info(
                            f"Background entity extraction finished for {doc_id_local}: {created_entities} entities, {created_relationships} relationships"
                        )
                    except Exception as e:
                        logger.error(
                            f"Unhandled error in background entity worker for {doc_id_local}: {e}"
                        )
                        if operation_id:
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.ERROR,
                                error_message=f"Unhandled error: {str(e)}",
                            )
                    finally:
                        # Complete the operation tracking
                        if operation_id:
                            self._complete_entity_operation(operation_id)

                        # Remove this thread from the tracking list when finished
                        try:
                            with self._bg_lock:
                                current = threading.current_thread()
                                if current in self._bg_entity_threads:
                                    self._bg_entity_threads.remove(current)
                        except Exception:
                            pass

                # Start thread and track it so the UI can detect background work
                t = threading.Thread(
                    target=_background_entity_worker,
                    args=(doc_id, chunks_for_extraction),
                    daemon=True,
                )
                with self._bg_lock:
                    self._bg_entity_threads.append(t)
                t.start()

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
                "entities_created": entity_count,
                "entity_relationships_created": relationship_count,
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
            print(
                f"Processed {file_path} in {duration:.2f}s — {len(processed_chunks)} chunks"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")
            return {"file_path": str(file_path), "status": "error", "error": str(e)}

    def is_entity_extraction_running(self) -> bool:
        """
        Return True if any background entity extraction operations are currently running.

        This function provides a comprehensive check that considers:
        1. Thread status (alive threads) - both new state-tracked and legacy threads
        2. Operation states (LLM extraction, embedding generation, database operations, validation)
        3. Thread names and activities to detect entity extraction work
        4. Cleanup of stale operations

        Returns:
            bool: True if any entity extraction is in progress, False otherwise
        """
        # First, clean up stale operations (older than 1 hour)
        self._cleanup_stale_operations()

        # Check thread status and operation states
        with self._bg_lock:
            # Clean up dead threads
            self._bg_entity_threads = [
                t for t in self._bg_entity_threads if t.is_alive()
            ]
            has_alive_threads = len(self._bg_entity_threads) > 0

            # Additional check: look for threads that might be doing entity work
            # but weren't tracked by the new system (legacy threads)
            import threading

            all_threads = threading.enumerate()
            entity_related_threads = []

            for thread in all_threads:
                # Check if thread is daemon and potentially doing entity extraction
                if thread.daemon and thread.is_alive():
                    try:
                        # Check thread name for entity-related work
                        thread_name = thread.name.lower()
                        if (
                            "entity" in thread_name
                            or "background" in thread_name
                            or "batch" in thread_name
                            or "global" in thread_name
                        ):
                            # Additional check: not the main thread or our current thread
                            if (
                                thread != threading.current_thread()
                                and thread != threading.main_thread()
                            ):
                                entity_related_threads.append(thread)
                    except (AttributeError, TypeError):
                        # Some threads might not have accessible name info
                        pass

            has_legacy_entity_threads = len(entity_related_threads) > 0

        with self._operations_lock:
            # Check if any operations are in non-completed states
            has_active_operations = any(
                status.state
                in [
                    EntityExtractionState.STARTING,
                    EntityExtractionState.LLM_EXTRACTION,
                    EntityExtractionState.EMBEDDING_GENERATION,
                    EntityExtractionState.DATABASE_OPERATIONS,
                    EntityExtractionState.VALIDATION,
                ]
                for status in self._entity_extraction_operations.values()
            )

        # Return True if any of these conditions are met
        return has_alive_threads or has_legacy_entity_threads or has_active_operations

    def get_entity_extraction_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about ongoing entity extraction operations.

        Returns:
            dict: Detailed status including active operations, their states, and progress
        """
        # Clean up stale operations first
        self._cleanup_stale_operations()

        with self._bg_lock:
            # Clean up dead threads
            self._bg_entity_threads = [
                t for t in self._bg_entity_threads if t.is_alive()
            ]
            thread_count = len(self._bg_entity_threads)

        with self._operations_lock:
            # Copy current operations to avoid locking issues
            operations_copy = dict(self._entity_extraction_operations)

        # Prepare detailed status information
        active_operations = []
        for op_id, status in operations_copy.items():
            runtime = time.time() - status.started_at
            active_operations.append(
                {
                    "operation_id": op_id,
                    "document_id": status.document_id,
                    "document_name": status.document_name,
                    "state": status.state.value,
                    "runtime_seconds": runtime,
                    "progress_info": status.progress_info,
                    "error_message": status.error_message,
                    "last_updated": status.last_updated,
                }
            )

        # Sort by start time (most recent first)
        active_operations.sort(key=lambda x: x["last_updated"], reverse=True)

        return {
            "is_running": len(active_operations) > 0 or thread_count > 0,
            "active_threads": thread_count,
            "active_operations": len(active_operations),
            "operations": active_operations,
            "states_summary": {
                state.value: sum(
                    1 for op in active_operations if op["state"] == state.value
                )
                for state in EntityExtractionState
            },
        }

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

    def process_file_chunks_only(
        self,
        file_path: Path,
        original_filename: Optional[str] = None,
        progress_callback=None,
        enable_quality_filtering: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a file for chunks only, without entity extraction.
        This is used for batch processing to avoid rate limiting.

        Args:
            file_path: Path to the file to process
            original_filename: Optional original filename to preserve
            progress_callback: Optional callback function to report chunk processing progress
            
            enable_quality_filtering: Optional override for quality filtering (if None, uses global setting)

        Returns:
            Processing result dictionary or None if failed
        """
        try:
            # Determine OCR settings (use provided params or fall back to global settings)
            # Smart OCR applied automatically
            use_quality_filtering = enable_quality_filtering if enable_quality_filtering is not None else self.enable_quality_filtering
            
            start_time = time.time()
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Get appropriate loader
            file_ext = file_path.suffix.lower()
            loader = self.loaders.get(file_ext)
            
            # Generate document ID and extract metadata (needed for all cases)
            doc_id = self._generate_document_id(file_path)
            metadata = self._extract_metadata(file_path, original_filename)
            
            logger.info(f"Processing file chunks only: {file_path}")
            
            # Handle image and PDF files with intelligent OCR
            ocr_metadata = {}
            if isinstance(loader, ImageLoader):
                logger.info(f"Processing image file with intelligent OCR: {file_path}")
                result = loader.load_with_metadata(file_path)
                if not result or not result["content"]:
                    logger.info(f"No text content detected in image: {file_path}")
                    return None
                content = result["content"]
                ocr_metadata = result.get("metadata", {})
            elif isinstance(loader, PDFLoader):
                logger.info(f"Processing PDF file with intelligent OCR: {file_path}")
                result = loader.load_with_metadata(file_path)
                if not result or not result["content"]:
                    logger.warning(f"No content extracted from PDF: {file_path}")
                    return None
                content = result["content"]
                ocr_metadata = result.get("metadata", {})
            elif not loader:
                logger.warning(f"No loader available for file type: {file_ext}")
                return None
            else:
                # Load document content using appropriate loader
                content = loader.load(file_path)
                if not content:
                    logger.warning(f"No content extracted from: {file_path}")
                    return None
            
            # Add OCR metadata if available
            if ocr_metadata:
                metadata.update(ocr_metadata)

            # Ensure content_primary_type is set (derive from file extension if missing)
            metadata["content_primary_type"] = metadata.get(
                "content_primary_type"
            ) or self._derive_content_primary_type(metadata.get("file_extension"))

            # Create document node
            graph_db.create_document_node(doc_id, metadata)

            # Chunk the document with enhanced processing
            if use_quality_filtering:
                chunks = document_chunker.chunk_text(
                    content,
                    doc_id,
                    enable_quality_filtering=use_quality_filtering,
                    enable_ocr_enhancement=False  # OCR already applied by smart loaders
                )
                logger.info(f"Used enhanced chunking (Quality filtering: {use_quality_filtering}) for {doc_id}")
            else:
                chunks = document_chunker.chunk_text(content, doc_id)
                logger.info(f"Used standard chunking for {doc_id}")

            # Process chunks asynchronously with configurable concurrency (embeddings + storing)
            try:
                # Use asyncio.run for a synchronous wrapper
                processed_chunks = asyncio.run(
                    self.process_file_async(chunks, doc_id, progress_callback)
                )
            except RuntimeError:
                # If an event loop is already running, get it and run until complete
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new task and wait for it
                    processed_chunks = loop.run_until_complete(
                        self.process_file_async(chunks, doc_id, progress_callback)
                    )
                else:
                    processed_chunks = loop.run_until_complete(
                        self.process_file_async(chunks, doc_id, progress_callback)
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

            # Validate chunk embeddings after processing
            try:
                validation_results = graph_db.validate_chunk_embeddings(doc_id)
                if not validation_results["validation_passed"]:
                    logger.warning(
                        f"Document {doc_id} has {validation_results['invalid_chunks']} invalid chunk embeddings"
                    )
                    # Optionally fix invalid embeddings
                    invalid_chunk_ids = [
                        chunk["chunk_id"]
                        for chunk in validation_results["invalid_chunk_details"]
                    ]
                    if invalid_chunk_ids:
                        logger.info(
                            f"Attempting to fix {len(invalid_chunk_ids)} invalid chunk embeddings..."
                        )
                        fix_results = graph_db.fix_invalid_embeddings(
                            chunk_ids=invalid_chunk_ids
                        )
                        logger.info(
                            f"Fixed {fix_results['chunks_fixed']} chunk embeddings"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to validate chunk embeddings for document {doc_id}: {e}"
                )

            result = {
                "document_id": doc_id,
                "file_path": str(file_path),
                "chunks_created": len(processed_chunks),
                "entities_created": 0,  # No entities in chunks-only mode
                "entity_relationships_created": 0,  # No entity relationships in chunks-only mode
                "similarity_relationships_created": relationships_created,
                "metadata": metadata,
                "status": "success",
            }

            logger.info(
                f"Successfully processed {file_path} (chunks only): {len(processed_chunks)} chunks created"
            )
            # add processing duration
            duration = time.time() - start_time
            result["duration_seconds"] = duration

            # Print to stdout for quick feedback
            print(
                f"Processed {file_path} in {duration:.2f}s — {len(processed_chunks)} chunks (chunks-only)"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to process file {file_path} (chunks only): {e}")
            return {"file_path": str(file_path), "status": "error", "error": str(e)}

    def process_batch_entities(
        self, processed_documents: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Process entity extraction for multiple documents in batch to avoid rate limiting.

        Args:
            processed_documents: List of document info dicts with document_id and file_name

        Returns:
            Dictionary with entity extraction statistics
        """
        if not settings.enable_entity_extraction:
            logger.info("Entity extraction disabled, skipping batch processing")
            return None

        # Initialize entity extractor if not already done
        if self.entity_extractor is None:
            logger.info("Initializing entity extractor for batch processing...")
            self.entity_extractor = EntityExtractor()

        logger.info(
            f"Starting batch entity extraction for {len(processed_documents)} documents"
        )

        def _batch_entity_worker(documents_list):
            """Background worker for batch entity extraction."""
            operation_ids = []
            try:
                # Get a reference to the entity extractor (should be initialized by now)
                extractor = self.entity_extractor
                if not extractor:
                    logger.error("Entity extractor not available in batch worker")
                    return None

                total_entities = 0
                total_relationships = 0
                by_document = {}

                for doc_info in documents_list:
                    doc_id = doc_info["document_id"]
                    file_name = doc_info["file_name"]
                    operation_id = None

                    try:
                        # Start tracking this operation
                        operation_id = self._start_entity_operation(doc_id, file_name)
                        operation_ids.append(operation_id)

                        logger.info(
                            f"Processing entities for document {doc_id} ({file_name}) - operation: {operation_id}"
                        )

                        # Get chunks for this document from the database
                        chunks_for_extraction = graph_db.get_document_chunks(doc_id)

                        if not chunks_for_extraction:
                            logger.warning(f"No chunks found for document {doc_id}")
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.ERROR,
                                error_message="No chunks found for document",
                            )
                            continue

                        # Phase 1: LLM extraction
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.LLM_EXTRACTION,
                            "Running LLM entity extraction",
                        )
                        entity_dict, relationship_dict = asyncio.run(
                            extractor.extract_from_chunks(chunks_for_extraction)
                        )

                        # Phase 2: Embedding generation and database operations
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.EMBEDDING_GENERATION,
                            f"Generating embeddings for {len(entity_dict)} entities",
                        )

                        # Persist entities and relationships asynchronously
                        created_entities, created_relationships = asyncio.run(
                            self._create_entities_async(entity_dict, relationship_dict, doc_id)
                        )

                        # Phase 3: Database operations for relationships
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.DATABASE_OPERATIONS,
                            f"Creating {sum(len(rels) for rels in relationship_dict.values())} relationships",
                        )

                        # Store entity relationships
                        for relationships in relationship_dict.values():
                            for relationship in relationships:
                                try:
                                    source_id = self._generate_entity_id(
                                        relationship.source_entity
                                    )
                                    target_id = self._generate_entity_id(
                                        relationship.target_entity
                                    )
                                    graph_db.create_entity_relationship(
                                        entity_id1=source_id,
                                        entity_id2=target_id,
                                        relationship_type="RELATED_TO",
                                        description=relationship.description,
                                        strength=relationship.strength,
                                        source_chunks=relationship.source_chunks or [],
                                    )
                                    created_relationships += 1
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to persist relationship for doc {doc_id}: {e}"
                                    )

                        # Optionally create entity similarities for this document
                        try:
                            graph_db.create_entity_similarities(doc_id)
                        except Exception as e:
                            logger.debug(
                                f"Failed to create entity similarities for {doc_id}: {e}"
                            )

                        # Phase 4: Validation
                        self._update_entity_operation(
                            operation_id,
                            EntityExtractionState.VALIDATION,
                            "Validating entity embeddings",
                        )

                        # Validate entity embeddings after processing
                        try:
                            validation_results = graph_db.validate_entity_embeddings(
                                doc_id
                            )
                            if not validation_results["validation_passed"]:
                                logger.warning(
                                    f"Document {doc_id} has {validation_results['invalid_embeddings']} invalid entity embeddings"
                                )
                                # Optionally fix invalid embeddings
                                invalid_entity_ids = [
                                    entity["entity_id"]
                                    for entity in validation_results[
                                        "invalid_entity_details"
                                    ]
                                ]
                                if invalid_entity_ids:
                                    logger.info(
                                        f"Attempting to fix {len(invalid_entity_ids)} invalid entity embeddings..."
                                    )
                                    fix_results = graph_db.fix_invalid_embeddings(
                                        entity_ids=invalid_entity_ids
                                    )
                                    logger.info(
                                        f"Fixed {fix_results['entities_fixed']} entity embeddings"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"Failed to validate entity embeddings for document {doc_id}: {e}"
                            )

                        logger.info(
                            f"Batch entity extraction finished for {doc_id}: {created_entities} entities, {created_relationships} relationships"
                        )

                        # Track stats
                        total_entities += created_entities
                        total_relationships += created_relationships
                        by_document[doc_id] = {
                            "entities": created_entities,
                            "relationships": created_relationships,
                        }

                    except Exception as e:
                        logger.error(
                            f"Failed to process entities for document {doc_id}: {e}"
                        )
                        if operation_id:
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.ERROR,
                                error_message=f"Failed to process document: {str(e)}",
                            )
                    finally:
                        # Complete the operation tracking for this document
                        if operation_id:
                            self._complete_entity_operation(operation_id)

                return {
                    "total_entities": total_entities,
                    "total_relationships": total_relationships,
                    "by_document": by_document,
                }

            except Exception as e:
                logger.error(f"Unhandled error in batch entity worker: {e}")
                return None
            finally:
                # Complete any remaining operations
                for op_id in operation_ids:
                    try:
                        self._complete_entity_operation(op_id)
                    except Exception:
                        pass

        # Start thread and track it so the UI can detect background work
        t = threading.Thread(
            target=_batch_entity_worker, args=(processed_documents,), daemon=True
        )
        with self._bg_lock:
            self._bg_entity_threads.append(t)
        t.start()

        # Return None for now since this is asynchronous - stats will be available later
        return None

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return list(self.loaders.keys())

    def estimate_chunks_from_files(self, uploaded_files: List[Any]) -> int:
        """
        Estimate total number of chunks that will be created from uploaded files.

        Args:
            uploaded_files: List of uploaded file objects with .name and .getvalue() methods

        Returns:
            Estimated total number of chunks
        """
        import tempfile

        total_chunks = 0

        for uploaded_file in uploaded_files:
            try:
                file_ext = Path(uploaded_file.name).suffix.lower()
                loader = self.loaders.get(file_ext)

                if not loader:
                    logger.warning(f"No loader available for file type: {file_ext}")
                    continue

                # Save file temporarily to load content
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=file_ext
                ) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)

                try:
                    # Load content and estimate chunks
                    content = loader.load(tmp_path)
                    if content:
                        # Use the same chunking logic to get accurate count
                        chunks = document_chunker.chunk_text(
                            content, f"temp_{uploaded_file.name}"
                        )
                        total_chunks += len(chunks)
                        logger.debug(
                            f"Estimated {len(chunks)} chunks for {uploaded_file.name}"
                        )
                finally:
                    # Clean up temporary file
                    if tmp_path.exists():
                        tmp_path.unlink()

            except Exception as e:
                logger.error(f"Error estimating chunks for {uploaded_file.name}: {e}")
                # Fallback estimation based on file size (rough estimate)
                try:
                    file_size = len(uploaded_file.getvalue())
                    estimated_chunks = max(
                        1, file_size // (settings.chunk_size * 2)
                    )  # Conservative estimate
                    total_chunks += estimated_chunks
                    logger.debug(
                        f"Fallback estimated {estimated_chunks} chunks for {uploaded_file.name}"
                    )
                except Exception as fallback_e:
                    logger.error(
                        f"Fallback estimation also failed for {uploaded_file.name}: {fallback_e}"
                    )
                    # Last resort: assume 1 chunk
                    total_chunks += 1

        logger.info(
            f"Estimated total of {total_chunks} chunks from {len(uploaded_files)} files"
        )
        return max(1, total_chunks)  # Ensure at least 1 chunk

    def extract_entities_for_all_documents(self) -> Optional[Dict[str, Any]]:
        """
        Extract entities for all documents that don't have entities yet.
        This is a global operation that processes all documents missing entity extraction.

        Returns:
            Dictionary with extraction statistics or None if extraction is disabled
        """
        if not settings.enable_entity_extraction:
            logger.info("Entity extraction disabled, skipping global extraction")
            return None

        # Initialize entity extractor if not already done
        if self.entity_extractor is None:
            logger.info("Initializing entity extractor for global extraction...")
            self.entity_extractor = EntityExtractor()

        # Get entity extraction status from database
        try:
            status = graph_db.get_entity_extraction_status()
            docs_without_entities = [
                doc
                for doc in status["documents"]
                if not doc["entities_extracted"] and doc["total_chunks"] > 0
            ]

            if not docs_without_entities:
                logger.info("All documents already have entities extracted")
                return {
                    "status": "no_action_needed",
                    "message": "All documents already have entities extracted",
                    "processed_documents": 0,
                }

            logger.info(
                f"Starting global entity extraction for {len(docs_without_entities)} documents"
            )

            # Prepare document list for batch processing
            documents_to_process = [
                {"document_id": doc["document_id"], "file_name": doc["filename"]}
                for doc in docs_without_entities
            ]

            # Start batch entity extraction in background
            def _global_entity_worker(documents_list):
                """Background worker for global entity extraction."""
                operation_ids = []
                try:
                    extractor = self.entity_extractor
                    if not extractor:
                        logger.error(
                            "Entity extractor not available in global extraction worker"
                        )
                        return None

                    total_entities = 0
                    total_relationships = 0
                    processed_docs = 0

                    for doc_info in documents_list:
                        doc_id = doc_info["document_id"]
                        file_name = doc_info["file_name"]
                        operation_id = None

                        try:
                            # Start tracking this operation
                            operation_id = self._start_entity_operation(
                                doc_id, file_name
                            )
                            operation_ids.append(operation_id)

                            logger.info(
                                f"Processing entities for document {doc_id} ({file_name}) - operation: {operation_id}"
                            )

                            # Get chunks for this document from the database
                            chunks_for_extraction = graph_db.get_document_chunks(doc_id)

                            if not chunks_for_extraction:
                                logger.warning(f"No chunks found for document {doc_id}")
                                self._update_entity_operation(
                                    operation_id,
                                    EntityExtractionState.ERROR,
                                    error_message="No chunks found for document",
                                )
                                continue

                            # Phase 1: LLM extraction
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.LLM_EXTRACTION,
                                "Running LLM entity extraction",
                            )
                            entity_dict, relationship_dict = asyncio.run(
                                extractor.extract_from_chunks(chunks_for_extraction)
                            )

                            # Phase 2: Embedding generation and database operations
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.EMBEDDING_GENERATION,
                                f"Generating embeddings for {len(entity_dict)} entities",
                            )

                            # Persist entities and relationships asynchronously
                            created_entities, created_relationships = asyncio.run(
                                self._create_entities_async(entity_dict, relationship_dict, doc_id)
                            )

                            # Phase 3: Database operations for relationships
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.DATABASE_OPERATIONS,
                                f"Creating {sum(len(rels) for rels in relationship_dict.values())} relationships",
                            )

                            # Store entity relationships
                            for relationships in relationship_dict.values():
                                for relationship in relationships:
                                    try:
                                        source_id = self._generate_entity_id(
                                            relationship.source_entity
                                        )
                                        target_id = self._generate_entity_id(
                                            relationship.target_entity
                                        )
                                        graph_db.create_entity_relationship(
                                            entity_id1=source_id,
                                            entity_id2=target_id,
                                            relationship_type="RELATED_TO",
                                            description=relationship.description,
                                            strength=relationship.strength,
                                            source_chunks=relationship.source_chunks
                                            or [],
                                        )
                                        created_relationships += 1
                                    except Exception as e:
                                        logger.debug(
                                            f"Failed to persist relationship for doc {doc_id}: {e}"
                                        )

                            # Optionally create entity similarities for this document
                            try:
                                graph_db.create_entity_similarities(doc_id)
                            except Exception as e:
                                logger.debug(
                                    f"Failed to create entity similarities for {doc_id}: {e}"
                                )

                            # Phase 4: Validation
                            self._update_entity_operation(
                                operation_id,
                                EntityExtractionState.VALIDATION,
                                "Validating entity embeddings",
                            )

                            # Validate entity embeddings after processing
                            try:
                                validation_results = (
                                    graph_db.validate_entity_embeddings(doc_id)
                                )
                                if not validation_results["validation_passed"]:
                                    logger.warning(
                                        f"Document {doc_id} has {validation_results['invalid_embeddings']} invalid entity embeddings"
                                    )
                                    # Optionally fix invalid embeddings
                                    invalid_entity_ids = [
                                        entity["entity_id"]
                                        for entity in validation_results[
                                            "invalid_entity_details"
                                        ]
                                    ]
                                    if invalid_entity_ids:
                                        logger.info(
                                            f"Attempting to fix {len(invalid_entity_ids)} invalid entity embeddings..."
                                        )
                                        fix_results = graph_db.fix_invalid_embeddings(
                                            entity_ids=invalid_entity_ids
                                        )
                                        logger.info(
                                            f"Fixed {fix_results['entities_fixed']} entity embeddings"
                                        )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to validate entity embeddings for document {doc_id}: {e}"
                                )

                            logger.info(
                                f"Global entity extraction finished for {doc_id}: {created_entities} entities, {created_relationships} relationships"
                            )

                            # Track stats
                            total_entities += created_entities
                            total_relationships += created_relationships
                            processed_docs += 1

                        except Exception as e:
                            logger.error(
                                f"Failed to process entities for document {doc_id}: {e}"
                            )
                            if operation_id:
                                self._update_entity_operation(
                                    operation_id,
                                    EntityExtractionState.ERROR,
                                    error_message=f"Failed to process document: {str(e)}",
                                )
                        finally:
                            # Complete the operation tracking for this document
                            if operation_id:
                                self._complete_entity_operation(operation_id)

                    logger.info(
                        f"Global entity extraction completed: {processed_docs} documents, {total_entities} entities, {total_relationships} relationships"
                    )

                except Exception as e:
                    logger.error(
                        f"Unhandled error in global entity extraction worker: {e}"
                    )
                finally:
                    # Complete any remaining operations
                    for op_id in operation_ids:
                        try:
                            self._complete_entity_operation(op_id)
                        except Exception:
                            pass

                    # Remove this thread from the tracking list when finished
                    try:
                        with self._bg_lock:
                            current = threading.current_thread()
                            if current in self._bg_entity_threads:
                                self._bg_entity_threads.remove(current)
                    except Exception:
                        pass

            # Start thread and track it so the UI can detect background work
            t = threading.Thread(
                target=_global_entity_worker, args=(documents_to_process,), daemon=True
            )
            with self._bg_lock:
                self._bg_entity_threads.append(t)
            t.start()

            return {
                "status": "started",
                "message": f"Started entity extraction for {len(docs_without_entities)} documents",
                "processed_documents": len(docs_without_entities),
            }

        except Exception as e:
            logger.error(f"Failed to start global entity extraction: {e}")
            return {
                "status": "error",
                "message": f"Failed to start entity extraction: {str(e)}",
                "processed_documents": 0,
            }


# Global document processor instance
document_processor = DocumentProcessor()
