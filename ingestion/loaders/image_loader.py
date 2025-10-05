"""
Image loader that processes standalone images with intelligent OCR.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.ocr import ocr_processor

logger = logging.getLogger(__name__)


class ImageLoader:
    """Loads content from image files with intelligent OCR detection."""

    def __init__(self):
        """Initialize the image loader."""
        self.processor = ocr_processor

    def load(self, file_path: Path) -> Optional[str]:
        """
        Load text content from an image file using intelligent OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text content or None if no text found
        """
        result = self.load_with_metadata(file_path)
        return result["content"] if result else None

    def load_with_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load image content with intelligent OCR and detailed metadata.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with content and OCR metadata
        """
        try:
            logger.info(f"Processing image with intelligent OCR: {file_path}")
            
            # Use OCR processor to analyze and extract from image
            result = self.processor.process_standalone_image(file_path)
            
            if not result["content"]:
                logger.info(f"No text content found in image: {file_path}")
                return None

            # Create metadata - flatten for Neo4j compatibility
            ocr_metadata = result["ocr_metadata"]
            content_analysis = ocr_metadata.get("content_analysis", {})
            
            metadata = {
                "processing_method": "image_ocr",
                "file_type": "standalone_image",
                "ocr_applied": ocr_metadata.get("ocr_applied", 0),
                "ocr_items_count": len(ocr_metadata.get("ocr_items", [])),
                # Flatten content analysis fields
                "content_primary_type": content_analysis.get("primary_type", "unknown"),
                "content_needs_ocr": content_analysis.get("needs_ocr", False),
            }

            # Log processing result
            if ocr_metadata.get("ocr_applied", 0) > 0:
                content_type = metadata["content_primary_type"]
                logger.info(f"OCR extracted text from {content_type} image: {file_path}")
            
            return {
                "content": result["content"],
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Failed to load image with OCR {file_path}: {e}")
            return None
