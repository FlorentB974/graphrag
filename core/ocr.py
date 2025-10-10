"""
Smart OCR processor that automatically detects when OCR is needed.
Handles images, diagrams, and scanned content while preserving readable text.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class OCRProcessor:
    """Intelligently determines when and what to OCR in documents."""

    def __init__(self):
        """Initialize the smart OCR processor."""
        # Ensure poppler is available in PATH
        self._setup_poppler_path()

        # Content detection thresholds
        self.MIN_TEXT_RATIO = 0.15  # Minimum ratio of alphanumeric chars to total chars
        self.MAX_WHITESPACE_RATIO = 0.65  # Maximum ratio of whitespace characters
        self.MIN_CHUNK_LENGTH = 30  # Minimum meaningful chunk length
        self.MIN_WORDS_PER_LINE = 2  # Minimum words per line to consider readable text

        # Image analysis settings
        self.IMAGE_DPI = 300  # DPI for PDF to image conversion
        self.ANALYSIS_DPI = 150  # Lower DPI for content analysis (faster)

        # OCR configuration
        self.tesseract_config = "--oem 3 --psm 6"

        logger.info("Smart OCR processor initialized")

    def _setup_poppler_path(self):
        """Ensure poppler utilities are available in PATH."""
        try:
            # Common poppler installation paths
            homebrew_bin = "/opt/homebrew/bin"
            macports_bin = "/opt/local/bin"

            current_path = os.environ.get("PATH", "")

            # Check if poppler is already accessible
            import subprocess

            try:
                subprocess.run(["pdftoppm", "-v"], capture_output=True, check=True)
                logger.info("Poppler already available in PATH")
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                pass

            # Try to add homebrew bin to PATH if it exists and contains pdftoppm
            if os.path.exists(os.path.join(homebrew_bin, "pdftoppm")):
                if homebrew_bin not in current_path:
                    os.environ["PATH"] = f"{homebrew_bin}:{current_path}"
                    logger.info(f"Added {homebrew_bin} to PATH for poppler access")
                return

            # Try macportsrrr
            if os.path.exists(os.path.join(macports_bin, "pdftoppm")):
                if macports_bin not in current_path:
                    os.environ["PATH"] = f"{macports_bin}:{current_path}"
                    logger.info(f"Added {macports_bin} to PATH for poppler access")
                return

            logger.warning(
                "Poppler not found in common installation paths. OCR may fail for PDFs requiring image conversion."
            )

        except Exception as e:
            logger.warning(
                f"Failed to setup poppler PATH: {e}. OCR may fail for some PDFs."
            )

    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Analyze text quality to determine if it's readable or needs OCR.

        Args:
            text: Text content to analyze

        Returns:
            Dictionary with quality metrics and assessment
        """
        if not text or len(text.strip()) < 5:
            return {
                "is_readable": False,
                "quality_score": 0.0,
                "reason": "Empty or too short",
                "metrics": {"total_chars": len(text) if text else 0},
            }

        # Calculate quality metrics
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalnum())
        whitespace_chars = sum(1 for c in text if c.isspace())
        lines = text.split("\n")

        # Text composition metrics
        text_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        whitespace_ratio = whitespace_chars / total_chars if total_chars > 0 else 0

        # Line structure analysis
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        avg_words_per_line = 0
        if non_empty_lines:
            total_words = sum(len(line.split()) for line in non_empty_lines)
            avg_words_per_line = total_words / len(non_empty_lines)

        # Pattern detection
        has_ocr_artifacts = bool(re.search(r"[^\x00-\x7F]+", text))  # Non-ASCII chars
        has_fragmented_words = len(re.findall(r"\b\w{1,2}\b", text)) > total_chars * 0.1
        has_excessive_spaces = "   " in text  # Multiple consecutive spaces

        # Calculate overall quality score (0-1)
        quality_score = (
            text_ratio * 0.4
            + (1 - whitespace_ratio) * 0.3
            + min(avg_words_per_line / 5, 1) * 0.3
        )

        # Apply penalties
        if has_ocr_artifacts:
            quality_score *= 0.8
        if has_fragmented_words:
            quality_score *= 0.7
        if has_excessive_spaces:
            quality_score *= 0.9
        if total_chars < self.MIN_CHUNK_LENGTH:
            quality_score *= 0.6

        # Determine if text is readable
        is_readable = (
            quality_score >= 0.5
            and text_ratio >= self.MIN_TEXT_RATIO
            and whitespace_ratio <= self.MAX_WHITESPACE_RATIO
            and avg_words_per_line >= self.MIN_WORDS_PER_LINE
            and not (has_fragmented_words and has_ocr_artifacts)
        )

        # Generate reason
        reasons = []
        if text_ratio < self.MIN_TEXT_RATIO:
            reasons.append(f"Low text ratio ({text_ratio:.2f})")
        if whitespace_ratio > self.MAX_WHITESPACE_RATIO:
            reasons.append(f"High whitespace ratio ({whitespace_ratio:.2f})")
        if avg_words_per_line < self.MIN_WORDS_PER_LINE:
            reasons.append(f"Few words per line ({avg_words_per_line:.1f})")
        if has_fragmented_words:
            reasons.append("Fragmented words detected")
        if has_ocr_artifacts:
            reasons.append("OCR artifacts detected")
        if total_chars < self.MIN_CHUNK_LENGTH:
            reasons.append("Content too short")

        reason = "; ".join(reasons) if reasons else "Good quality text"

        return {
            "is_readable": is_readable,
            "quality_score": quality_score,
            "reason": reason,
            "metrics": {
                "total_chars": total_chars,
                "text_ratio": text_ratio,
                "whitespace_ratio": whitespace_ratio,
                "avg_words_per_line": avg_words_per_line,
                "lines": len(lines),
                "has_ocr_artifacts": has_ocr_artifacts,
                "has_fragmented_words": has_fragmented_words,
            },
        }

    def assess_chunk_quality(self, chunk: str) -> Dict[str, Any]:
        """
        Assess the quality of a text chunk to determine if it needs OCR or should be filtered.

        This is a public method used by the chunking module to evaluate chunk quality.

        Args:
            chunk: Text chunk to assess

        Returns:
            Dictionary with quality assessment including:
            - quality_score: float between 0-1
            - reason: string describing quality issues
            - needs_ocr: boolean indicating if OCR might improve quality
            - metrics: detailed quality metrics
        """
        # Use the internal text quality analysis
        analysis = self._analyze_text_quality(chunk)

        # Determine if OCR might help (for chunks with poor quality)
        needs_ocr = not analysis["is_readable"] and analysis["quality_score"] < 0.3

        # Map metrics to match expected format in chunking module
        metrics = {
            "total_chars": analysis["metrics"]["total_chars"],
            "text_ratio": analysis["metrics"]["text_ratio"],
            "whitespace_ratio": analysis["metrics"]["whitespace_ratio"],
            "fragmentation_ratio": (
                1.0 if analysis["metrics"]["has_fragmented_words"] else 0.0
            ),
            "has_artifacts": analysis["metrics"]["has_ocr_artifacts"],
        }

        return {
            "quality_score": analysis["quality_score"],
            "reason": analysis["reason"],
            "needs_ocr": needs_ocr,
            "metrics": metrics,
        }

    def should_remove_chunk(
        self, chunk_text: str, entity_count: int = 0, relationship_count: int = 0
    ) -> bool:
        """
        Determine if a chunk should be removed based on quality and entity extraction results.

        Args:
            chunk_text: The text content of the chunk
            entity_count: Number of entities extracted from this chunk
            relationship_count: Number of relationships extracted from this chunk

        Returns:
            True if the chunk should be removed, False otherwise
        """
        # Assess chunk quality
        quality_assessment = self.assess_chunk_quality(chunk_text)

        # Remove if quality is very poor AND no meaningful entities were extracted
        if (
            quality_assessment["quality_score"] < 0.2
            and entity_count == 0
            and relationship_count == 0
        ):
            return True

        # Remove if the chunk is too short and has no entities
        if len(chunk_text.strip()) < self.MIN_CHUNK_LENGTH and entity_count == 0:
            return True

        # Keep the chunk otherwise
        return False

    def _detect_image_content(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze an image to detect if it contains text, diagrams, or other content.

        Args:
            image: Image as numpy array

        Returns:
            Dictionary with content type analysis
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()

            # Basic image statistics
            height, width = gray.shape
            total_pixels = height * width

            # Edge detection to find structural content
            edges = cv2.Canny(gray, 50, 150)
            edge_pixel_ratio = np.sum(edges > 0) / total_pixels

            # Brightness analysis
            mean_brightness = gray.mean()
            brightness_std = gray.std()

            # Text detection using connected components
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find connected components (potential text regions)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            # Analyze component characteristics
            text_like_components = 0
            for i in range(1, num_labels):  # Skip background (label 0)
                x, y, w, h, area = stats[i]
                aspect_ratio = w / h if h > 0 else 0

                # Text-like components are usually rectangular with certain aspect ratios
                if 0.1 <= aspect_ratio <= 10 and 50 <= area <= total_pixels * 0.1:
                    text_like_components += 1

            text_component_ratio = text_like_components / max(num_labels - 1, 1)

            # Determine content type
            content_types = []
            confidence_scores = {}

            # Text content detection (improved sensitivity)
            if text_component_ratio > 0.05 or edge_pixel_ratio < 0.05:
                content_types.append("text")
                confidence_scores["text"] = min(text_component_ratio * 3, 1.0)

            # Diagram/structural content detection
            if edge_pixel_ratio > 0.1 and brightness_std > 30:
                content_types.append("diagram")
                confidence_scores["diagram"] = min(edge_pixel_ratio * 2, 1.0)

            # Scanned page detection (improved sensitivity for low-quality scans)
            if (
                brightness_std > 20 and text_component_ratio > 0.001
            ):  # Much more sensitive
                content_types.append("scanned_page")
                confidence_scores["scanned_page"] = min(
                    (brightness_std / 80) * (text_component_ratio * 50), 1.0
                )

            # Image/photo detection (low text components, varied brightness)
            if text_component_ratio < 0.02 and brightness_std > 20:
                content_types.append("image")
                confidence_scores["image"] = 1.0 - text_component_ratio

            # Default to mixed if unclear
            if not content_types:
                content_types.append("mixed")
                confidence_scores["mixed"] = 0.5

            # Select primary content type
            primary_type = max(content_types, key=lambda t: confidence_scores.get(t, 0))

            # Determine if OCR is needed with improved logic
            needs_ocr = (
                primary_type in ["text", "diagram", "scanned_page"]
                # Also apply OCR if image has potential text characteristics
                or (
                    primary_type == "image"
                    and (
                        brightness_std > 25  # Good contrast suggesting text
                        or text_component_ratio > 0.001  # Some structured content
                        or edge_pixel_ratio > 0.03
                    )
                )  # Some structured edges
            )

            return {
                "primary_type": primary_type,
                "content_types": content_types,
                "confidence_scores": confidence_scores,
                "needs_ocr": needs_ocr,
                "metrics": {
                    "edge_pixel_ratio": edge_pixel_ratio,
                    "text_component_ratio": text_component_ratio,
                    "mean_brightness": mean_brightness,
                    "brightness_std": brightness_std,
                    "total_components": num_labels - 1,
                },
            }

        except Exception as e:
            logger.error(f"Image content detection failed: {e}")
            return {
                "primary_type": "unknown",
                "content_types": ["unknown"],
                "confidence_scores": {"unknown": 0.5},
                "needs_ocr": True,  # Default to OCR for safety
                "metrics": {},
            }

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
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            return cleaned

        except Exception as e:
            logger.warning(f"Image enhancement failed, using original: {e}")
            return image

    def _extract_text_from_image(
        self, image: Image.Image, content_type: str = "mixed"
    ) -> Optional[str]:
        """
        Extract text from a PIL Image using OCR with content-aware configuration.

        Args:
            image: PIL Image object
            content_type: Type of content detected in image

        Returns:
            Extracted text or None if failed
        """
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)

            # Apply enhancement based on content type
            if content_type in ["diagram", "scanned_page"]:
                img_array = self._enhance_image_for_ocr(img_array)
                enhanced_image = Image.fromarray(img_array)
            else:
                enhanced_image = image

            # Adjust OCR config based on content type
            if content_type == "diagram":
                config = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?()[]{}/-+=*&%$#@"
            elif content_type == "scanned_page":
                config = "--oem 3 --psm 4"  # Assume single column of text
            else:
                config = self.tesseract_config  # Default config

            # Perform OCR
            text = pytesseract.image_to_string(enhanced_image, config=config)

            # Clean up the extracted text
            text = text.strip()
            text = re.sub(r"\n\s*\n", "\n\n", text)  # Clean up excessive newlines
            text = re.sub(r" +", " ", text)  # Clean up excessive spaces

            return text if text else None

        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")
            return None

    def analyze_pdf_content(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze PDF content to determine which parts need OCR.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with analysis results for each page
        """
        try:
            reader = PdfReader(str(file_path))
            analysis_results = {
                "pages": [],
                "summary": {
                    "total_pages": len(reader.pages),
                    "readable_pages": 0,
                    "ocr_pages": 0,
                    "image_pages": 0,
                    "mixed_pages": 0,
                },
            }

            # Analyze each page
            for page_num, page in enumerate(reader.pages):
                try:
                    # Extract existing text
                    page_text = page.extract_text()

                    # Analyze text quality
                    text_analysis = self._analyze_text_quality(page_text)

                    page_result = {
                        "page_number": page_num + 1,
                        "has_text": bool(page_text.strip()),
                        "text_analysis": text_analysis,
                        "needs_ocr": not text_analysis["is_readable"],
                        "content_types": [],
                        "ocr_items": [],
                    }

                    # If text is not readable, analyze the page as an image
                    if not text_analysis["is_readable"]:
                        try:
                            # Convert page to image for analysis
                            images = convert_from_path(
                                str(file_path),
                                first_page=page_num + 1,
                                last_page=page_num + 1,
                                dpi=self.ANALYSIS_DPI,
                                fmt="RGB",
                            )

                            if images:
                                img_array = np.array(images[0])
                                image_analysis = self._detect_image_content(img_array)

                                page_result["image_analysis"] = image_analysis
                                page_result["content_types"] = image_analysis[
                                    "content_types"
                                ]

                                # Determine OCR strategy
                                if image_analysis["needs_ocr"]:
                                    ocr_item = {
                                        "type": image_analysis["primary_type"],
                                        "source": "full_page",
                                        "confidence": image_analysis[
                                            "confidence_scores"
                                        ].get(image_analysis["primary_type"], 0.5),
                                    }
                                    page_result["ocr_items"].append(ocr_item)

                        except Exception as e:
                            logger.warning(
                                f"Failed to analyze page {page_num + 1} as image: {e}"
                            )
                            # Default to OCR for safety
                            page_result["ocr_items"].append(
                                {
                                    "type": "scanned_page",
                                    "source": "full_page",
                                    "confidence": 0.5,
                                }
                            )

                    # Update summary
                    if text_analysis["is_readable"]:
                        analysis_results["summary"]["readable_pages"] += 1
                    elif any(
                        item["type"] in ["text", "scanned_page"]
                        for item in page_result["ocr_items"]
                    ):
                        analysis_results["summary"]["ocr_pages"] += 1
                    elif any(
                        item["type"] == "image" for item in page_result["ocr_items"]
                    ):
                        analysis_results["summary"]["image_pages"] += 1
                    else:
                        analysis_results["summary"]["mixed_pages"] += 1

                    analysis_results["pages"].append(page_result)

                except Exception as e:
                    logger.error(f"Failed to analyze page {page_num + 1}: {e}")
                    # Add error page with default OCR
                    error_page = {
                        "page_number": page_num + 1,
                        "has_text": False,
                        "text_analysis": {
                            "is_readable": False,
                            "quality_score": 0.0,
                            "reason": "Analysis failed",
                        },
                        "needs_ocr": True,
                        "content_types": ["unknown"],
                        "ocr_items": [
                            {
                                "type": "unknown",
                                "source": "full_page",
                                "confidence": 0.5,
                            }
                        ],
                        "error": str(e),
                    }
                    analysis_results["pages"].append(error_page)
                    analysis_results["summary"]["mixed_pages"] += 1

            logger.info(f"Analyzed PDF {file_path}: {analysis_results['summary']}")
            return analysis_results

        except Exception as e:
            logger.error(f"Failed to analyze PDF content: {e}")
            return {
                "pages": [],
                "summary": {
                    "total_pages": 0,
                    "readable_pages": 0,
                    "ocr_pages": 0,
                    "image_pages": 0,
                    "mixed_pages": 0,
                },
                "error": str(e),
            }

    def process_pdf_intelligently(self, file_path: Path) -> Dict[str, Any]:
        """
        Process PDF with intelligent OCR application.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with extracted content and processing metadata
        """
        try:
            # First, analyze the PDF to determine OCR strategy
            analysis = self.analyze_pdf_content(file_path)

            if not analysis["pages"]:
                logger.warning(f"No analyzable pages in PDF: {file_path}")
                return {
                    "content": None,
                    "ocr_metadata": {"error": "No analyzable pages"},
                }

            reader = PdfReader(str(file_path))
            text_content = []
            ocr_metadata = {
                "total_pages": len(reader.pages),
                "pages_processed": 0,
                "ocr_applied": 0,
                "ocr_items": [],
                "readable_text_pages": 0,
                "processing_summary": analysis["summary"],
            }

            # Process each page based on analysis
            for page_analysis in analysis["pages"]:
                page_num = page_analysis["page_number"] - 1  # Convert to 0-based index

                try:
                    if page_analysis["text_analysis"]["is_readable"]:
                        # Use existing readable text
                        page_text = reader.pages[page_num].extract_text()
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                        ocr_metadata["readable_text_pages"] += 1
                        logger.debug(f"Page {page_num + 1}: Using readable text")

                    elif page_analysis["ocr_items"]:
                        # Apply OCR to specific items
                        for ocr_item in page_analysis["ocr_items"]:
                            try:
                                # Convert page to image for OCR
                                images = convert_from_path(
                                    str(file_path),
                                    first_page=page_num + 1,
                                    last_page=page_num + 1,
                                    dpi=self.IMAGE_DPI,
                                    fmt="RGB",
                                )

                                if images:
                                    # Extract text using OCR
                                    ocr_text = self._extract_text_from_image(
                                        images[0], ocr_item["type"]
                                    )

                                    if ocr_text and ocr_text.strip():
                                        ocr_label = f"Page {page_num + 1} ({ocr_item['type'].title()})"
                                        text_content.append(
                                            f"--- {ocr_label} ---\n{ocr_text}"
                                        )

                                        # Track OCR usage
                                        ocr_metadata["ocr_items"].append(
                                            {
                                                "page": page_num + 1,
                                                "type": ocr_item["type"],
                                                "source": ocr_item["source"],
                                                "confidence": ocr_item["confidence"],
                                                "text_length": len(ocr_text),
                                            }
                                        )
                                        ocr_metadata["ocr_applied"] += 1

                                        logger.debug(
                                            f"Page {page_num + 1}: OCR applied for {ocr_item['type']}"
                                        )
                                    else:
                                        logger.warning(
                                            f"Page {page_num + 1}: OCR produced no text for {ocr_item['type']}"
                                        )

                            except Exception as e:
                                logger.error(
                                    f"OCR failed for page {page_num + 1}, item {ocr_item['type']}: {e}"
                                )
                                continue
                    else:
                        logger.debug(
                            f"Page {page_num + 1}: Skipped (no readable text or OCR items)"
                        )

                    ocr_metadata["pages_processed"] += 1

                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")
                    continue

            # Combine all content
            if not text_content:
                logger.warning(f"No content extracted from PDF: {file_path}")
                return {"content": None, "ocr_metadata": ocr_metadata}

            full_content = "\n\n".join(text_content)

            # Log processing summary
            logger.info(
                f"Smart OCR processing completed for {file_path}: "
                f"{ocr_metadata['pages_processed']} pages processed, "
                f"{ocr_metadata['ocr_applied']} pages with OCR, "
                f"{ocr_metadata['readable_text_pages']} pages with readable text"
            )

            return {"content": full_content, "ocr_metadata": ocr_metadata}

        except Exception as e:
            logger.error(f"Smart OCR processing failed for {file_path}: {e}")
            return {
                "content": None,
                "ocr_metadata": {
                    "error": str(e),
                    "total_pages": 0,
                    "pages_processed": 0,
                    "ocr_applied": 0,
                },
            }

    def process_standalone_image(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a standalone image file with smart OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            logger.info(f"Processing standalone image with smart OCR: {file_path}")

            # Open and analyze the image
            with Image.open(file_path) as image:
                # Convert to RGB if necessary
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Analyze content type
                img_array = np.array(image)
                content_analysis = self._detect_image_content(img_array)

                ocr_metadata = {
                    "file_type": "standalone_image",
                    "content_analysis": content_analysis,
                    "ocr_applied": 0,
                    "ocr_items": [],
                }

                # Apply OCR if needed
                if content_analysis["needs_ocr"]:
                    ocr_text = self._extract_text_from_image(
                        image, content_analysis["primary_type"]
                    )

                    if ocr_text and ocr_text.strip():
                        ocr_metadata["ocr_applied"] = 1
                        ocr_metadata["ocr_items"].append(
                            {
                                "type": content_analysis["primary_type"],
                                "source": "full_image",
                                "confidence": content_analysis["confidence_scores"].get(
                                    content_analysis["primary_type"], 0.5
                                ),
                                "text_length": len(ocr_text),
                            }
                        )

                        logger.info(
                            f"Successfully extracted text from image: {file_path}"
                        )
                        return {"content": ocr_text, "ocr_metadata": ocr_metadata}
                    else:
                        logger.warning(f"OCR produced no text for image: {file_path}")
                        return {"content": None, "ocr_metadata": ocr_metadata}
                else:
                    logger.info(f"Image does not contain OCR-able content: {file_path}")
                    return {"content": None, "ocr_metadata": ocr_metadata}

        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return {
                "content": None,
                "ocr_metadata": {
                    "error": str(e),
                    "file_type": "standalone_image",
                    "ocr_applied": 0,
                },
            }


# Global OCR processor instance
ocr_processor = OCRProcessor()
