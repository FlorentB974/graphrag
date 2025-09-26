"""
Document loaders module initialization.
"""

from .csv_loader import CSVLoader
from .docx_loader import DOCXLoader
from .pdf_loader import PDFLoader
from .pptx_loader import PPTXLoader
from .text_loader import TextLoader
from .xlsx_loader import XLSXLoader

__all__ = [
    "CSVLoader",
    "DOCXLoader",
    "PDFLoader",
    "PPTXLoader",
    "TextLoader",
    "XLSXLoader",
]
