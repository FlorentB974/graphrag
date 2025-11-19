"""Marker-based PDF loader."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

logger = logging.getLogger(__name__)


class MarkerPdfLoader:
    """Loads PDF files using the Marker PDF-to-Markdown converter."""

    def __init__(
        self,
        *,
        torch_device: Optional[str] = None,
        converter: Optional[PdfConverter] = None,
    ) -> None:
        self.device = torch_device or os.environ.get("TORCH_DEVICE")
        self._converter: Optional[PdfConverter] = converter

    def _create_converter(self) -> PdfConverter:
        """Create a Marker PDF converter with the configured device."""
        try:
            artifact_dict = create_model_dict(device=self.device)
            return PdfConverter(artifact_dict=artifact_dict)
        except Exception as exc:  # pragma: no cover - import-time failures
            logger.error("Failed to initialize Marker PDF converter: %s", exc)
            raise

    def load(self, file_path: Path) -> Optional[str]:
        """Load PDF content as markdown text."""
        result = self.load_with_metadata(file_path)
        return result["content"] if result else None

    def load_with_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load PDF content with metadata about the parsing process."""
        try:
            converter = self._get_converter()
            rendered = converter(str(file_path))
            text, _, images = text_from_rendered(rendered)
            text = (text or "").strip()
            if not text:
                logger.warning("Marker did not return any text for PDF: %s", file_path)
                return None

            images_count = len(images or [])
            metadata: Dict[str, Any] = {
                "source": str(file_path),
                "parser": "marker",
                "processing_method": "marker_pdf",  # aligns with previous naming style
                "images_count": images_count,
            }
            if self.device:
                metadata["torch_device"] = self.device

            return {"content": text, "metadata": metadata}
        except Exception as exc:
            logger.error("Failed to convert PDF %s using Marker: %s", file_path, exc)
            return None

    def _get_converter(self) -> PdfConverter:
        """Lazily initialize the Marker converter to avoid heavy startup costs."""
        if self._converter is None:
            self._converter = self._create_converter()
        return self._converter
