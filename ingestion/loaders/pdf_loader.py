"""
Enhanced PDF loader with intelligent OCR integration.
Only applies OCR to images, diagrams, and scanned content, preserving readable text.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pypdf import PdfReader

from core.ocr import ocr_processor

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads content from PDF files with intelligent OCR application."""

    def __init__(self):
        """Initialize the PDF loader."""
        self.processor = ocr_processor

    def load(self, file_path: Path) -> Optional[str]:
        """
        Load text content from a PDF file with intelligent OCR.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content or None if failed
        """
        result = self.load_with_metadata(file_path)
        return result["content"] if result else None

    def load_with_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load PDF content with intelligent OCR and detailed metadata.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with content and OCR metadata
        """
        try:
            logger.info(f"Processing PDF with intelligent OCR: {file_path}")
            
            # Use OCR processor to intelligently handle the PDF
            result = self.processor.process_pdf_intelligently(file_path)
            
            if not result["content"]:
                logger.warning(f"No content extracted from PDF: {file_path}")
                return None

            # Create comprehensive metadata - flatten for Neo4j compatibility
            ocr_metadata = result["ocr_metadata"]
            processing_summary = ocr_metadata.get("processing_summary", {})
            
            metadata = {
                "processing_method": "ocr",
                "total_pages": ocr_metadata.get("total_pages", 0),
                "pages_processed": ocr_metadata.get("pages_processed", 0),
                "ocr_applied_pages": ocr_metadata.get("ocr_applied", 0),
                "readable_text_pages": ocr_metadata.get("readable_text_pages", 0),
                # Flatten processing summary fields
                "summary_total_pages": processing_summary.get("total_pages", 0),
                "summary_readable_pages": processing_summary.get("readable_pages", 0),
                "summary_ocr_pages": processing_summary.get("ocr_pages", 0),
                "summary_image_pages": processing_summary.get("image_pages", 0),
                "summary_mixed_pages": processing_summary.get("mixed_pages", 0),
                # Convert OCR items list to a simple count for now
                "ocr_items_count": len(ocr_metadata.get("ocr_items", [])),
            }

            # Log processing summary
            logger.info(
                f"PDF processing completed for {file_path}: "
                f"Total: {metadata['summary_total_pages']} pages, "
                f"Readable: {metadata['summary_readable_pages']}, "
                f"OCR: {metadata['summary_ocr_pages']}, "
                f"Images: {metadata['summary_image_pages']}"
            )

            return {
                "content": result["content"],
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Failed to load PDF with OCR {file_path}: {e}")
            return None

    def load_with_ocr(self, file_path: Path, enable_ocr: bool = True) -> Optional[str]:
        """
        Legacy method for compatibility.
        Now uses intelligent OCR regardless of enable_ocr parameter.

        Args:
            file_path: Path to the PDF file
            enable_ocr: Ignored (intelligent OCR always applies intelligently)

        Returns:
            Extracted text content or None if failed
        """
        if not enable_ocr:
            # If OCR is explicitly disabled, fall back to basic text extraction
            try:
                reader = PdfReader(str(file_path))
                text_content = []
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                if not text_content:
                    return None
                    
                return "\n\n".join(text_content)
                
            except Exception as e:
                logger.error(f"Basic PDF text extraction failed for {file_path}: {e}")
                return None
        else:
            # Use intelligent OCR (default behavior)
            return self.load(file_path)
