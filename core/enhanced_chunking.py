"""
Enhanced document chunking utilities with OCR and quality assessment.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings
from core.ocr import ocr_processor

logger = logging.getLogger(__name__)


class EnhancedDocumentChunker:
    """Handles document chunking with OCR support and quality assessment."""

    def __init__(self):
        """Initialize the enhanced document chunker."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        
        # OCR and quality settings
        self.enable_quality_filtering = True
        self.enable_ocr_enhancement = True

    def chunk_text(self, text: str, document_id: str, enable_quality_filtering: Optional[bool] = None, enable_ocr_enhancement: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with quality assessment and OCR enhancement.

        Args:
            text: The text to chunk
            document_id: Identifier for the source document
            enable_quality_filtering: Override for quality filtering (if None, uses instance setting)
            enable_ocr_enhancement: Override for OCR enhancement (if None, uses instance setting)

        Returns:
            List of dictionaries containing chunk data with quality metrics
        """
        try:
            # Determine settings to use (provided parameters or instance defaults)
            use_quality_filtering = enable_quality_filtering if enable_quality_filtering is not None else self.enable_quality_filtering
            use_ocr_enhancement = enable_ocr_enhancement if enable_ocr_enhancement is not None else self.enable_ocr_enhancement
            
            chunks = self.text_splitter.split_text(text)
            chunk_data = []
            processed_chunks = 0
            filtered_chunks = 0

            for i, chunk in enumerate(chunks):
                # Assess chunk quality (only if quality filtering is enabled)
                if use_quality_filtering:
                    quality_assessment = ocr_processor.assess_chunk_quality(chunk)
                else:
                    # Default quality assessment when filtering is disabled
                    quality_assessment = {
                        "quality_score": 1.0,
                        "reason": "Quality filtering disabled",
                        "needs_ocr": False,
                        "metrics": {
                            "total_chars": len(chunk),
                            "text_ratio": 1.0,
                            "whitespace_ratio": 0.0,
                            "fragmentation_ratio": 0.0,
                            "has_artifacts": False,
                        }
                    }
                
                # Create base chunk info
                chunk_info = {
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "content": chunk,
                    "chunk_index": i,
                    "document_id": document_id,
                    "metadata": {
                        "chunk_size": len(chunk),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "quality_score": quality_assessment["quality_score"],
                        "quality_reason": quality_assessment["reason"],
                        # Flatten quality metrics for Neo4j compatibility
                        "total_chars": quality_assessment["metrics"]["total_chars"],
                        "text_ratio": quality_assessment["metrics"]["text_ratio"],
                        "whitespace_ratio": quality_assessment["metrics"]["whitespace_ratio"],
                        "fragmentation_ratio": quality_assessment["metrics"]["fragmentation_ratio"],
                        "has_artifacts": quality_assessment["metrics"]["has_artifacts"],
                        "processing_method": "standard",
                    },
                }

                # Apply quality filtering if enabled
                if use_quality_filtering and quality_assessment["needs_ocr"]:
                    # For now, we'll keep the chunk but mark it for potential removal after entity extraction
                    chunk_info["metadata"]["needs_review"] = True
                    chunk_info["metadata"]["quality_warning"] = quality_assessment["reason"]
                    logger.debug(f"Chunk {i} flagged for review: {quality_assessment['reason']}")

                # Add OCR enhancement metadata if applicable (only if OCR enhancement is enabled)
                if use_ocr_enhancement and ("OCR" in chunk or "Images/Diagrams" in chunk):
                    chunk_info["metadata"]["processing_method"] = "ocr_enhanced"
                    chunk_info["metadata"]["contains_ocr"] = True
                elif not use_ocr_enhancement:
                    chunk_info["metadata"]["processing_method"] = "standard"
                    chunk_info["metadata"]["contains_ocr"] = False
                    
                chunk_data.append(chunk_info)
                processed_chunks += 1

            logger.info(
                f"Successfully chunked document {document_id} into {processed_chunks} chunks "
                f"({filtered_chunks} filtered for quality)"
            )
            
            return chunk_data

        except Exception as e:
            logger.error(f"Failed to chunk document {document_id}: {e}")
            raise

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents with enhanced processing.

        Args:
            documents: List of document dictionaries with 'id' and 'content' keys

        Returns:
            List of all chunks from all documents with quality metrics
        """
        all_chunks = []
        total_quality_issues = 0

        for doc in documents:
            doc_id = doc.get("id")
            content = doc.get("content", "")

            if not doc_id or not content:
                logger.warning(f"Skipping document with missing id or content: {doc}")
                continue

            chunks = self.chunk_text(content, doc_id)
            
            # Track quality issues
            quality_issues = sum(
                1 for chunk in chunks
                if chunk["metadata"].get("needs_review", False)
            )
            total_quality_issues += quality_issues
            
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} total chunks "
            f"({total_quality_issues} with quality issues)"
        )
        
        return all_chunks

    def post_entity_quality_filter(
        self,
        chunks: List[Dict[str, Any]],
        entity_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks based on quality assessment and entity extraction results.
        
        This method is called after entity extraction to remove chunks that:
        1. Have poor quality scores
        2. Produced no entities or relationships
        3. Are likely noise from scanned documents
        
        Args:
            chunks: List of chunk dictionaries
            entity_results: Results from entity extraction with entity counts per chunk
            
        Returns:
            Filtered list of chunks
        """
        if not self.enable_quality_filtering:
            return chunks
            
        filtered_chunks = []
        removed_count = 0
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["content"]
            
            # Get entity extraction results for this chunk
            entity_count = entity_results.get(chunk_id, {}).get("entity_count", 0)
            relationship_count = entity_results.get(chunk_id, {}).get("relationship_count", 0)
            
            # Determine if chunk should be removed
            should_remove = ocr_processor.should_remove_chunk(
                chunk_text, entity_count, relationship_count
            )
            
            if should_remove:
                logger.info(f"Removing low-quality chunk: {chunk_id}")
                removed_count += 1
                # Add removal metadata for tracking
                chunk["metadata"]["removed"] = True
                chunk["metadata"]["removal_reason"] = "poor_quality_no_entities"
            else:
                filtered_chunks.append(chunk)
        
        logger.info(f"Quality filtering removed {removed_count} chunks, kept {len(filtered_chunks)} chunks")
        return filtered_chunks

    def get_quality_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a quality summary for a set of chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Dictionary with quality statistics
        """
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            return {"total_chunks": 0, "quality_stats": {}}
        
        # Calculate quality statistics
        quality_scores = [chunk["metadata"].get("quality_score", 0.0) for chunk in chunks]
        ocr_chunks = sum(1 for chunk in chunks if chunk["metadata"].get("contains_ocr", False))
        review_chunks = sum(1 for chunk in chunks if chunk["metadata"].get("needs_review", False))
        removed_chunks = sum(1 for chunk in chunks if chunk["metadata"].get("removed", False))
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)
        
        return {
            "total_chunks": total_chunks,
            "quality_stats": {
                "average_quality_score": avg_quality,
                "min_quality_score": min_quality,
                "max_quality_score": max_quality,
                "ocr_enhanced_chunks": ocr_chunks,
                "chunks_needing_review": review_chunks,
                "removed_chunks": removed_chunks,
                "quality_distribution": {
                    "high_quality": sum(1 for score in quality_scores if score >= 0.8),
                    "medium_quality": sum(1 for score in quality_scores if 0.5 <= score < 0.8),
                    "low_quality": sum(1 for score in quality_scores if score < 0.5),
                }
            }
        }


# Global enhanced chunker instance
enhanced_document_chunker = EnhancedDocumentChunker()
