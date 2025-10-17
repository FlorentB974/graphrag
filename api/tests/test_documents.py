"""Tests for document metadata and preview endpoints."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.graph_db import graph_db


@pytest.fixture()
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


def test_get_document_metadata_success(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "id": "doc-123",
        "title": "Sample Document",
        "file_name": "sample.pdf",
        "mime_type": "application/pdf",
        "preview_url": None,
        "uploaded_at": "2025-10-11T12:00:00+00:00",
        "uploader": {"id": "user-1", "name": "Ada"},
        "chunks": [
            {"id": "chunk-1", "text": "Hello", "index": 0, "offset": 0, "score": None},
            {"id": "chunk-2", "text": "World", "index": 1, "offset": 60, "score": 0.9},
        ],
        "entities": [
            {"type": "PERSON", "text": "Ada", "count": 1, "positions": [0]},
        ],
        "quality_scores": {"total": 0.95},
        "related_documents": [{"id": "doc-456", "title": "Other"}],
        "metadata": {"file_extension": ".pdf"},
    }

    def fake_get_document_details(document_id: str):
        assert document_id == "doc-123"
        return payload

    monkeypatch.setattr(graph_db, "get_document_details", fake_get_document_details)

    response = client.get("/api/documents/doc-123")
    assert response.status_code == 200
    assert response.json()["id"] == payload["id"]
    assert response.json()["chunks"][0]["text"] == "Hello"


def test_get_document_metadata_not_found(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_document_details(document_id: str):
        raise ValueError("Document not found")

    monkeypatch.setattr(graph_db, "get_document_details", fake_get_document_details)

    response = client.get("/api/documents/missing")
    assert response.status_code == 404


def test_get_document_preview_stream(tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello preview", encoding="utf-8")

    def fake_get_document_file_info(document_id: str):
        return {
            "file_name": "sample.txt",
            "file_path": str(file_path),
            "mime_type": "text/plain",
            "preview_url": None,
        }

    monkeypatch.setattr(graph_db, "get_document_file_info", fake_get_document_file_info)

    response = client.get("/api/documents/doc-123/preview")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == "Hello preview"


def test_get_document_preview_redirect(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    preview_url = "https://example.com/doc.pdf"

    def fake_get_document_file_info(document_id: str):
        return {
            "file_name": "doc.pdf",
            "file_path": None,
            "mime_type": "application/pdf",
            "preview_url": preview_url,
        }

    monkeypatch.setattr(graph_db, "get_document_file_info", fake_get_document_file_info)

    response = client.get("/api/documents/doc-123/preview", follow_redirects=False)
    assert response.status_code in (302, 303, 307)
    assert response.headers["location"] == preview_url
