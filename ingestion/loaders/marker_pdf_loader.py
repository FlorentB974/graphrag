"""Marker-based PDF loader."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

try:  # pragma: no cover - optional dependency for type checking
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
except ImportError:  # pragma: no cover - allow tests without Marker installed
    ConfigParser = None  # type: ignore[assignment]
    PdfConverter = Any  # type: ignore[assignment]

    def create_model_dict(*_: Any, **__: Any) -> Any:
        raise ImportError("marker-pdf is required to convert PDF files")

    def text_from_rendered(*_: Any, **__: Any) -> Any:
        raise ImportError("marker-pdf is required to convert PDF files")

logger = logging.getLogger(__name__)


class MarkerPdfLoader:
    """Loads PDF files using the Marker PDF-to-Markdown converter."""

    def __init__(
        self,
        *,
        torch_device: Optional[str] = None,
        converter: Optional[PdfConverter] = None,
        llm_provider: Literal["auto", "gemini", "openai"] | None = "auto",
        use_llm: bool = True,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self.device = torch_device or os.environ.get("TORCH_DEVICE")
        self._converter: Optional[PdfConverter] = converter
        self.use_llm = use_llm
        self.llm_provider, self.llm_api_key = self._select_llm_provider(
            llm_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
        )
        self.llm_configured = bool(self.llm_provider and self.llm_api_key)

    def _create_converter(self) -> PdfConverter:
        """Create a Marker PDF converter with the configured device."""
        if ConfigParser is None:
            raise ImportError("marker-pdf is required to convert PDF files")
        try:
            config: Dict[str, Any] = {
                "output_format": "markdown",
                "use_llm": self.use_llm,
                "disable_image_extraction": True,
            }

            if self.use_llm and self.llm_provider == "gemini" and self.llm_api_key:
                config["gemini_api_key"] = self.llm_api_key
            elif self.use_llm and self.llm_provider == "openai" and self.llm_api_key:
                config["openai_api_key"] = self.llm_api_key
                config["llm_service"] = "marker.services.openai.OpenAIService"

            config_parser = ConfigParser(config)
            config_dict = config_parser.generate_config_dict()
            artifact_dict = create_model_dict(device=self.device)

            return PdfConverter(
                config=config_dict,
                artifact_dict=artifact_dict,
                processor_list=config_parser.get_processors(),
                renderer=config_parser.get_renderer(),
                llm_service=config_parser.get_llm_service(),
            )
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
                "use_llm": self.use_llm,
                "llm_provider": self.llm_provider,
                "llm_configured": self.llm_configured,
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

    def _select_llm_provider(
        self,
        llm_provider: Literal["auto", "gemini", "openai"] | None,
        *,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Determine which LLM provider to use based on env vars and config.

        Preference order when llm_provider="auto": Gemini (GEMINI_API_KEY or
        GOOGLE_API_KEY) is chosen before OpenAI (OPENAI_API_KEY). If neither key
        is found we fall back to non-LLM mode.
        """

        if not self.use_llm:
            return None, None

        def _get_gemini_key() -> Optional[str]:
            return gemini_api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )

        def _get_openai_key() -> Optional[str]:
            return openai_api_key or os.environ.get("OPENAI_API_KEY")

        if llm_provider is None:
            return None, None

        if llm_provider == "gemini":
            key = _get_gemini_key()
            return ("gemini", key) if key else (None, None)

        if llm_provider == "openai":
            key = _get_openai_key()
            return ("openai", key) if key else (None, None)

        # Automatic selection prefers Gemini keys over OpenAI keys when present.
        gemini_key = _get_gemini_key()
        if gemini_key:
            return "gemini", gemini_key

        openai_key = _get_openai_key()
        if openai_key:
            return "openai", openai_key

        return None, None
