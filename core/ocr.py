"""
OCR processing for document ingestion pipeline.
Handles text extraction from images, scanned documents, and diagrams.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing for extracting text from images and scanned documents."""

    def __init__(self):
        """Initialize the OCR processor."""
        # Configure tesseract if needed
        self.tesseract_config = "--oem 3 --psm 6"  # Default config for mixed text
        
        # Quality thresholds
        self.MIN_TEXT_RATIO = 0.1  # Minimum ratio of alphanumeric chars to total chars
        self.MIN_CHUNK_LENGTH = 50  # Minimum meaningful chunk length
        self.MAX_WHITESPACE_RATIO = 0.7  # Maximum ratio of whitespace characters
        
        # Image preprocessing settings
        self.IMAGE_DPI = 300  # DPI for PDF to image conversion
        self.ENHANCE_IMAGES = True  # Whether to apply image enhancement
        
        logger.info("OCR processor initialized with tesseract")

    def _is_scanned_pdf_page(self, page_text: str) -> bool:
        """
        Detect if a PDF page is likely scanned based on text quality.
        
        Args:
            page_text: Extracted text from PDF page
            
        Returns:
            True if page appears to be scanned, False otherwise
        """
        if not page_text or len(page_text.strip()) < 10:
            return True
            
        # Calculate text quality metrics
        total_chars = len(page_text)
        alpha_chars = sum(1 for c in page_text if c.isalnum())
        whitespace_chars = sum(1 for c in page_text if c.isspace())
        
        # Calculate ratios
        text_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0
        
        # Check for common scanned document artifacts
        has_ocr_artifacts = bool(re.search(r'[^\x00-\x7F]+', page_text))  # Non-ASCII chars
        has_fragmented_words = len(re.findall(r'\b\w{1,2}\b', page_text)) > total_chars * 0.1
        
        # Determine if page is scanned
        is_scanned = (
            text_ratio < self.MIN_TEXT_RATIO
            or whitespace_ratio > self.MAX_WHITESPACE_RATIO
            or has_ocr_artifacts
            or has_fragmented_words
        )
        
        logger.debug(
            f"Page scan detection - text_ratio: {text_ratio:.3f}, "
            f"whitespace_ratio: {whitespace_ratio:.3f}, "
            f"artifacts: {has_ocr_artifacts}, fragmented: {has_fragmented_words}, "
            f"is_scanned: {is_scanned}"
        )
        
        return is_scanned

    def _enhance_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image enhancement techniques to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply adaptive thresholding to handle varying lighting
            threshold = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return image

    def _extract_text_from_image(self, image: Image.Image, enhance: bool = True) -> str:
        """
        Extract text from a PIL Image using OCR.
        
        Args:
            image: PIL Image object
            enhance: Whether to apply image enhancement
            
        Returns:
            Extracted text
        """
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Apply enhancement if requested
            if enhance and self.ENHANCE_IMAGES:
                img_array = self._enhance_image_for_ocr(img_array)
                # Convert back to PIL Image
                enhanced_image = Image.fromarray(img_array)
            else:
                enhanced_image = image
            
            # Perform OCR
            text = pytesseract.image_to_string(enhanced_image, config=self.tesseract_config)
            
            # Clean up the extracted text
            text = text.strip()
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Clean up excessive newlines
            text = re.sub(r' +', ' ', text)  # Clean up excessive spaces
            
            return text
            
        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return ""

    def process_scanned_pdf(self, file_path: Path) -> Optional[str]:
        """
        Process a scanned PDF using OCR.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            logger.info(f"Processing scanned PDF with OCR: {file_path}")
            
            # Convert PDF pages to images
            images = convert_from_path(
                str(file_path),
                dpi=self.IMAGE_DPI,
                fmt='RGB'
            )
            
            extracted_pages = []
            for page_num, image in enumerate(images):
                try:
                    logger.debug(f"Processing page {page_num + 1}/{len(images)}")
                    
                    # Extract text from the page image
                    page_text = self._extract_text_from_image(image, enhance=True)
                    
                    if page_text.strip():
                        extracted_pages.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    else:
                        logger.warning(f"No text extracted from page {page_num + 1}")
                        
                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    continue
            
            if not extracted_pages:
                logger.warning(f"No text content extracted from scanned PDF: {file_path}")
                return None
                
            full_text = "\n\n".join(extracted_pages)
            logger.info(f"Successfully processed scanned PDF: {file_path} ({len(images)} pages)")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to process scanned PDF {file_path}: {e}")
            return None

    def extract_images_from_pdf(self, file_path: Path) -> List[Tuple[int, Image.Image]]:
        """
        Extract images/diagrams from PDF pages by converting pages to images.
        This is a simpler approach that works more reliably across different PDF types.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of tuples (page_number, image)
        """
        try:
            # Convert PDF pages to images for image analysis
            # This approach is more reliable than trying to extract embedded images
            images = convert_from_path(
                str(file_path),
                dpi=150,  # Lower DPI for image detection
                fmt='RGB'
            )
            
            extracted_images = []
            for page_num, image in enumerate(images):
                # For now, we treat each page as a potential image
                # In the future, we could add image detection algorithms here
                extracted_images.append((page_num + 1, image))
            
            logger.info(f"Converted {len(extracted_images)} PDF pages to images: {file_path}")
            return extracted_images
            
        except Exception as e:
            logger.error(f"Failed to extract images from PDF {file_path}: {e}")
            return []

    def process_images_in_pdf(self, file_path: Path) -> Optional[str]:
        """
        Extract and OCR all images/diagrams found in a PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text from all images or None if no images found
        """
        try:
            extracted_images = self.extract_images_from_pdf(file_path)
            
            if not extracted_images:
                return None
                
            image_texts = []
            for page_num, image in extracted_images:
                try:
                    # Extract text from the image
                    image_text = self._extract_text_from_image(image, enhance=True)
                    
                    if image_text.strip():
                        image_texts.append(f"--- Image from Page {page_num} ---\n{image_text}")
                        
                except Exception as e:
                    logger.error(f"Failed to OCR image from page {page_num}: {e}")
                    continue
            
            if not image_texts:
                return None
                
            combined_text = "\n\n".join(image_texts)
            logger.info(f"Extracted text from {len(image_texts)} images in PDF: {file_path}")
            return combined_text
            
        except Exception as e:
            logger.error(f"Failed to process images in PDF {file_path}: {e}")
            return None

    def process_standalone_image(self, file_path: Path) -> Optional[str]:
        """
        Process a standalone image file with OCR.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Extracted text or None if failed
        """
        try:
            logger.info(f"Processing standalone image with OCR: {file_path}")
            
            # Open and process the image
            with Image.open(file_path) as image:
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Extract text
                text = self._extract_text_from_image(image, enhance=True)
                
                if text.strip():
                    logger.info(f"Successfully extracted text from image: {file_path}")
                    return text
                else:
                    logger.warning(f"No text found in image: {file_path}")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return None

    def assess_chunk_quality(self, chunk_text: str) -> Dict[str, Any]:
        """
        Assess the quality of a text chunk to determine if OCR might help.
        
        Args:
            chunk_text: Text content of the chunk
            
        Returns:
            Dictionary with quality assessment results
        """
        if not chunk_text:
            return {
                "quality_score": 0.0,
                "needs_ocr": True,
                "reason": "Empty chunk",
                "metrics": {}
            }
        
        # Calculate quality metrics
        total_chars = len(chunk_text)
        alpha_chars = sum(1 for c in chunk_text if c.isalnum())
        whitespace_chars = sum(1 for c in chunk_text if c.isspace())
        
        text_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0
        
        # Check for OCR artifacts
        has_artifacts = bool(re.search(r'[^\x00-\x7F]+', chunk_text))
        fragmented_words = len(re.findall(r'\b\w{1,2}\b', chunk_text))
        fragmentation_ratio = fragmented_words / len(chunk_text.split()) if chunk_text.split() else 0
        
        # Calculate overall quality score (0-1)
        quality_score = text_ratio * 0.4 + (1 - whitespace_ratio) * 0.3 + (1 - fragmentation_ratio) * 0.3
        
        # Penalties for artifacts and short length
        if has_artifacts:
            quality_score *= 0.8
        if total_chars < self.MIN_CHUNK_LENGTH:
            quality_score *= 0.7
            
        # Determine if OCR is needed
        needs_ocr = (
            quality_score < 0.6
            or text_ratio < self.MIN_TEXT_RATIO
            or whitespace_ratio > self.MAX_WHITESPACE_RATIO
            or fragmentation_ratio > 0.3
        )
        
        reason = []
        if text_ratio < self.MIN_TEXT_RATIO:
            reason.append(f"Low text ratio ({text_ratio:.2f})")
        if whitespace_ratio > self.MAX_WHITESPACE_RATIO:
            reason.append(f"High whitespace ratio ({whitespace_ratio:.2f})")
        if fragmentation_ratio > 0.3:
            reason.append(f"High fragmentation ({fragmentation_ratio:.2f})")
        if has_artifacts:
            reason.append("OCR artifacts detected")
        if total_chars < self.MIN_CHUNK_LENGTH:
            reason.append("Too short")
            
        return {
            "quality_score": quality_score,
            "needs_ocr": needs_ocr,
            "reason": "; ".join(reason) if reason else "Good quality",
            "metrics": {
                "total_chars": total_chars,
                "text_ratio": text_ratio,
                "whitespace_ratio": whitespace_ratio,
                "fragmentation_ratio": fragmentation_ratio,
                "has_artifacts": has_artifacts
            }
        }

    def should_remove_chunk(self, chunk_text: str, entity_count: int = 0, relationship_count: int = 0) -> bool:
        """
        Determine if a chunk should be removed due to poor quality and lack of entities.
        
        Args:
            chunk_text: Text content of the chunk
            entity_count: Number of entities extracted from the chunk
            relationship_count: Number of relationships extracted from the chunk
            
        Returns:
            True if chunk should be removed, False otherwise
        """
        quality_assessment = self.assess_chunk_quality(chunk_text)
        
        # Remove chunk if it's very low quality AND has no entities/relationships
        should_remove = (
            quality_assessment["quality_score"] < 0.3
            and entity_count == 0
            and relationship_count == 0
            and len(chunk_text.strip()) < self.MIN_CHUNK_LENGTH
        )
        
        if should_remove:
            logger.info(
                f"Marking chunk for removal - Quality: {quality_assessment['quality_score']:.2f}, "
                f"Entities: {entity_count}, Relationships: {relationship_count}"
            )
        
        return should_remove


# Global OCR processor instance
ocr_processor = OCRProcessor()
