"""
Text document loader.
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TextLoader:
    """Loads content from plain text files."""
    
    def load(self, file_path: Path) -> Optional[str]:
        """
        Load text content from a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Text content or None if failed
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                        
                        if not content.strip():
                            logger.warning(f"Empty file: {file_path}")
                            return None
                        
                        logger.info(f"Successfully loaded text file: {file_path} (encoding: {encoding})")
                        return content
                        
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"Could not decode file with any supported encoding: {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            return None