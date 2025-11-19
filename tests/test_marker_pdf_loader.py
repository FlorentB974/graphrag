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

    loader = MarkerPdfLoader(converter=dummy_converter)
    result = loader.load_with_metadata(sample_pdf_path)

    assert result is not None
    assert result["content"].startswith("# Sample")
    metadata = result["metadata"]
    assert metadata["parser"] == "marker"
    assert metadata["source"].endswith("sample.pdf")
    assert metadata["images_count"] == 1
    # Ensure fallback load() helper returns the same text
    assert loader.load(sample_pdf_path) == result["content"]
