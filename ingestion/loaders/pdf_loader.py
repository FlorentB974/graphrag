"""
Enhanced PDF document loader with OCR support.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader

from core.ocr import ocr_processor

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads content from PDF files with OCR support for scanned documents."""

    def __init__(self):
        """Initialize the enhanced PDF loader."""
        self.use_ocr_fallback = True  # Whether to use OCR for scanned pages
        self.ocr_threshold = 0.6  # Quality threshold below which OCR is applied

    def load(self, file_path: Path) -> Optional[str]:
        """
        Load text content from a PDF file with OCR support.

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content or None if failed
        """
        # Use global OCR setting by default
        from config.settings import settings
        return self.load_with_ocr(file_path, enable_ocr=settings.enable_ocr)

    def load_with_ocr(self, file_path: Path, enable_ocr: bool = True) -> Optional[str]:
        """
        Load text content from a PDF file with configurable OCR support.

        Args:
            file_path: Path to the PDF file
            enable_ocr: Whether to use OCR for scanned pages and images

        Returns:
            Extracted text content or None if failed
        """
        try:
            reader = PdfReader(str(file_path))
            text_content = []
            scanned_pages = []

            if enable_ocr:
                # OCR enabled: identify scanned pages for OCR processing
                logger.debug(f"OCR enabled: analyzing pages for quality in {file_path}")
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        
                        # Assess if this page is scanned
                        if page_text.strip() and not ocr_processor._is_scanned_pdf_page(page_text):
                            # Good quality text extraction
                            text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                            logger.debug(f"Page {page_num + 1}: Using regular text extraction")
                        else:
                            # Poor quality or no text - mark for OCR
                            scanned_pages.append(page_num)
                            logger.debug(f"Page {page_num + 1}: Marked for OCR processing")
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1} in {file_path}: {e}")
                        scanned_pages.append(page_num)  # Mark for OCR as fallback
                        continue
            else:
                # OCR disabled: extract all text normally without quality assessment
                logger.debug(f"OCR disabled: extracting all text normally from {file_path}")
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Any text is good enough when OCR is disabled
                            text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                            logger.debug(f"Page {page_num + 1}: Using regular text extraction (OCR disabled)")
                        else:
                            logger.debug(f"Page {page_num + 1}: No text found (OCR disabled, skipping)")
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1} in {file_path}: {e}")
                        continue

            # Second pass: OCR processing for scanned pages (only if OCR is enabled)
            if enable_ocr and scanned_pages and self.use_ocr_fallback:
                logger.info(f"Processing {len(scanned_pages)} scanned pages with OCR")
                
                try:
                    # Process the entire PDF with OCR to get scanned pages
                    ocr_text = ocr_processor.process_scanned_pdf(file_path)
                    
                    if ocr_text:
                        # Parse OCR text to extract individual pages
                        ocr_pages = self._parse_ocr_pages(ocr_text)
                        
                        # Add OCR text for scanned pages
                        for page_num in scanned_pages:
                            if page_num < len(ocr_pages):
                                ocr_page_text = ocr_pages[page_num]
                                if ocr_page_text.strip():
                                    text_content.append(f"--- Page {page_num + 1} (OCR) ---\n{ocr_page_text}")
                                    logger.debug(f"Page {page_num + 1}: Added OCR text")
                                else:
                                    logger.warning(f"Page {page_num + 1}: No OCR text extracted")
                            else:
                                logger.warning(f"Page {page_num + 1}: OCR page index out of range")
                                
                except Exception as e:
                    logger.error(f"OCR processing failed for {file_path}: {e}")
                    # Continue without OCR text
            elif enable_ocr and scanned_pages:
                logger.info(f"OCR fallback disabled, skipping {len(scanned_pages)} scanned pages in {file_path}")

            # Third pass: Extract and OCR images/diagrams (only if OCR is enabled)
            if enable_ocr:
                try:
                    image_text = ocr_processor.process_images_in_pdf(file_path)
                    if image_text and image_text.strip():
                        text_content.append(f"--- Extracted from Images/Diagrams ---\n{image_text}")
                        logger.info(f"Added text extracted from images in {file_path}")
                except Exception as e:
                    logger.warning(f"Image extraction failed for {file_path}: {e}")

            if not text_content:
                logger.warning(f"No text content extracted from PDF: {file_path}")
                return None

            # Combine all text content
            full_text = "\n\n".join(text_content)
            
            # Log summary
            total_pages = len(reader.pages)
            ocr_pages = len(scanned_pages)
            logger.info(
                f"Successfully loaded PDF: {file_path} "
                f"({total_pages} pages, {ocr_pages} with OCR, "
                f"{len(text_content)} text sections)"
            )
            
            return full_text

        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            return None

    def _parse_ocr_pages(self, ocr_text: str) -> List[str]:
        """
        Parse OCR text output to extract individual pages.
        
        Args:
            ocr_text: Full OCR text output
            
        Returns:
            List of text content for each page
        """
        try:
            # Split by page markers
            import re
            pages = re.split(r'--- Page \d+ ---\n?', ocr_text)
            
            # Remove empty first element (before first page marker)
            if pages and not pages[0].strip():
                pages = pages[1:]
                
            return pages
            
        except Exception as e:
            logger.error(f"Failed to parse OCR pages: {e}")
            return []

    def load_with_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load PDF content with processing metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with content and processing metadata
        """
        try:
            content = self.load(file_path)
            if not content:
                return None
                
            # Count OCR sections
            ocr_sections = content.count("(OCR)")
            image_sections = 1 if "Extracted from Images/Diagrams" in content else 0
            
            return {
                "content": content,
                "metadata": {
                    "ocr_pages_processed": ocr_sections,
                    "image_sections_extracted": image_sections,
                    "processing_method": "enhanced_pdf_loader"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to load PDF with metadata {file_path}: {e}")
            return None
