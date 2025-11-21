"""
Document loaders module initialization.
"""

from .csv_loader import CSVLoader
from .docx_loader import DOCXLoader
from .marker_pdf_loader import MarkerPdfLoader
from .pptx_loader import PPTXLoader
from .text_loader import TextLoader
from .xlsx_loader import XLSXLoader

__all__ = [
    "CSVLoader",
    "DOCXLoader",
    "MarkerPdfLoader",
    "PPTXLoader",
    "TextLoader",
    "XLSXLoader",
]
