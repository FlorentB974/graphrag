"""
PDF document loader.
"""
import logging
from pathlib import Path
from typing import Optional
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFLoader:
    """Loads content from PDF files."""
    
    def load(self, file_path: Path) -> Optional[str]:
        """
        Load text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            reader = PdfReader(str(file_path))
            
            text_content = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} in {file_path}: {e}")
                    continue
            
            if not text_content:
                logger.warning(f"No text content extracted from PDF: {file_path}")
                return None
            
            full_text = "\n\n".join(text_content)
            logger.info(f"Successfully loaded PDF: {file_path} ({len(reader.pages)} pages)")
            return full_text
            
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            return None