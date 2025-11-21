"""Tests for the Marker-based PDF loader."""

from pathlib import Path

import pytest

from ingestion.loaders.marker_pdf_loader import MarkerPdfLoader


class DummyConverter:
    """Simple stub that mimics the Marker converter interface."""

    def __init__(self) -> None:
        self.called_with: Path | None = None

    def __call__(self, file_path: str):  # pragma: no cover - trivial
        self.called_with = Path(file_path)
        return {"rendered": True}


def test_marker_pdf_loader_returns_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_converter = DummyConverter()
    sample_pdf_path = tmp_path / "sample.pdf"
    sample_pdf_path.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    def fake_text_from_rendered(rendered):  # pragma: no cover - trivial helper
        assert rendered == {"rendered": True}
        return ("# Sample\n\nContent", None, ["image-1"])

    monkeypatch.setattr(
        "ingestion.loaders.marker_pdf_loader.text_from_rendered",
        fake_text_from_rendered,
    )

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    loader = MarkerPdfLoader(converter=dummy_converter)
    result = loader.load_with_metadata(sample_pdf_path)

    assert result is not None
    assert result["content"].startswith("# Sample")
    metadata = result["metadata"]
    assert metadata["parser"] == "marker"
    assert metadata["source"].endswith("sample.pdf")
    assert metadata["images_count"] == 1
    assert metadata["use_llm"] is True
    assert metadata["llm_provider"] == "openai"
    assert metadata["llm_configured"] is True
    # Ensure fallback load() helper returns the same text
    assert loader.load(sample_pdf_path) == result["content"]


def test_marker_pdf_loader_prefers_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gem-123")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    loader = MarkerPdfLoader(converter=DummyConverter())

    assert loader.llm_provider == "gemini"
    assert loader.llm_api_key == "gem-123"
    assert loader.llm_configured is True


def test_marker_pdf_loader_disables_llm_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    loader = MarkerPdfLoader(converter=DummyConverter())

    assert loader.llm_provider is None
    assert loader.llm_api_key is None
    assert loader.llm_configured is False
