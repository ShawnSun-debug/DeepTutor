"""Tests for the batch document import endpoint (/api/imports/documents).

The single seam under test is the HTTP endpoint: a TestClient over a bare
FastAPI with the real routers mounted at /api and Co-Writer storage pointed at
a temporary directory. Nothing here reaches into the store to prove a result;
every assertion is either the JSON the endpoint returns or a follow-up request
through the same seam.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import deeptutor.services.config as _dt_config

# The Co-Writer router loads main.yaml at import time; a worktree has no
# data/user/settings, so stub the loader before importing it.
_dt_config.load_config_with_main = lambda *_a, **_k: {
    "paths": {},
    "logging": {},
    "system": {"language": "en"},
}

from deeptutor.api.routers import co_writer as co_writer_router
from deeptutor.api.routers import imports as imports_router
from deeptutor.co_writer.storage import CoWriterStorage


class _Paths:
    """The three Co-Writer paths the storage layer asks for."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def get_co_writer_docs_dir(self) -> Path:
        return self.root / "documents"

    def get_co_writer_doc_root(self, doc_id: str) -> Path:
        return self.get_co_writer_docs_dir() / f"doc_{doc_id}"

    def get_co_writer_doc_manifest(self, doc_id: str) -> Path:
        return self.get_co_writer_doc_root(doc_id) / "manifest.json"


@pytest.fixture
def storage(tmp_path: Path) -> CoWriterStorage:
    return CoWriterStorage(_Paths(tmp_path))


@pytest.fixture
def client(storage: CoWriterStorage, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(imports_router, "get_co_writer_storage", lambda: storage)
    monkeypatch.setattr(co_writer_router, "get_co_writer_storage", lambda: storage)
    app = FastAPI()
    # Production mounts this router at /api/imports (see web/lib/imports-api.ts);
    # /api alone would collide with the Co-Writer documents routes.
    app.include_router(imports_router.router, prefix="/api/imports")
    app.include_router(co_writer_router.router, prefix="/api")
    return TestClient(app)


def _markdown(name: str, body: bytes):
    return ("files", (name, body, "text/markdown"))


def _paragraph(text: str = "Hello") -> dict:
    return {
        "id": "a",
        "type": "paragraph",
        "props": {},
        "content": [{"type": "text", "text": text, "styles": {}}],
        "children": [],
    }


def test_one_markdown_file_becomes_an_editable_page(client: TestClient) -> None:
    """A .md file lands as one ordinary page, not a read-only blob."""
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("Lecture Notes.md", b"# Fourier\n\nTransform basics.\n")],
    )
    assert response.status_code == 200, response.text

    payload = response.json()
    assert len(payload["files"]) == 1
    result = payload["files"][0]
    assert result["filename"] == "Lecture Notes.md"
    assert result["destination"] == "authored_page"
    assert result["outcome"] == "imported"
    assert result["reason"] is None

    doc_id = result["document_id"]
    page = client.get(f"/api/documents/{doc_id}")
    assert page.status_code == 200
    document = page.json()
    assert document["title"] == "Lecture Notes"
    assert document["content"] == "# Fourier\n\nTransform basics.\n"
    # Markdown page with no block tree: the browser promotes it on first open,
    # exactly like a DOCX import.
    assert document["blocks"] is None

    # Editable through the same editor path an authored page uses.
    edited = client.put(
        f"/api/documents/{doc_id}",
        json={"blocks": [_paragraph()], "expected_revision": document["revision"]},
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["content"] == "Hello"


def test_line_endings_are_normalised_to_lf(client: TestClient) -> None:
    """Windows and legacy-Mac endings must arrive exactly as typed in the editor."""
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("crlf.md", b"# A\r\n\r\nB\rC\r\n")],
    )
    assert response.status_code == 200
    document = client.get(f"/api/documents/{response.json()['files'][0]['document_id']}").json()
    assert document["content"] == "# A\n\nB\nC\n"


def test_title_falls_back_to_content_when_the_name_is_unusable(client: TestClient) -> None:
    """A punctuation-only stem is not a usable title, so the content names the page."""
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("---.md", b"# Real Heading\n\nBody text.\n")],
    )
    assert response.status_code == 200
    document = client.get(f"/api/documents/{response.json()['files'][0]['document_id']}").json()
    assert document["title"] == "Real Heading"


def test_title_is_trimmed_and_length_capped(client: TestClient) -> None:
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("  " + "a" * 200 + "  .md", b"# body\n")],
    )
    assert response.status_code == 200
    document = client.get(f"/api/documents/{response.json()['files'][0]['document_id']}").json()
    assert document["title"] == "a" * 120


def test_markdown_extension_is_case_insensitive(client: TestClient) -> None:
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("Upper.MD", b"# body\n"), _markdown("long.markdown", b"# body\n")],
    )
    assert response.status_code == 200
    assert [item["outcome"] for item in response.json()["files"]] == ["imported", "imported"]


def test_gbk_text_is_decoded(client: TestClient) -> None:
    """Older Chinese notes are GBK, not UTF-8; both must arrive readable."""
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("gbk.md", "# 笔记".encode("gbk"))],
    )
    assert response.status_code == 200
    document = client.get(f"/api/documents/{response.json()['files'][0]['document_id']}").json()
    assert document["content"] == "# 笔记"


def test_one_bad_file_never_fails_the_batch(client: TestClient) -> None:
    """Unsupported, empty, blank and oversized-everything files fail individually."""
    response = client.post(
        "/api/imports/documents",
        files=[
            _markdown("good.md", b"# Good\n\nFine.\n"),
            _markdown("notes.txt", b"plain text"),
            _markdown("empty.md", b""),
            _markdown("blank.md", b"\n\n   \n"),
            _markdown("toolong.md", b"x" * 600_001),
            _markdown("second.md", b"# Second\n"),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    outcomes = {item["filename"]: item for item in payload["files"]}
    # Order is the order the user picked the files in.
    assert [item["filename"] for item in payload["files"]] == list(outcomes)
    assert outcomes["good.md"]["outcome"] == "imported"
    assert outcomes["second.md"]["outcome"] == "imported"
    assert outcomes["notes.txt"]["outcome"] == "skipped"
    assert outcomes["notes.txt"]["reason"] == "unsupported extension (supported: .md, .markdown)"
    assert outcomes["empty.md"]["outcome"] == "skipped"
    assert outcomes["empty.md"]["reason"] == "empty"
    assert outcomes["blank.md"]["outcome"] == "skipped"
    assert outcomes["blank.md"]["reason"] == "empty"
    assert outcomes["toolong.md"]["outcome"] == "failed"
    assert "600,000 character limit" in outcomes["toolong.md"]["reason"]
    assert payload["imported"] == 2
    assert payload["skipped"] == 3
    assert payload["failed"] == 1
    assert payload["destination"] == "authored_page"
    for item in payload["files"]:
        assert item["destination"] == "authored_page"
    assert outcomes["notes.txt"]["document_id"] is None


def test_file_over_the_upload_ceiling_fails_with_the_limit_stated(client: TestClient) -> None:
    """The 20 MB ceiling is enforced on real bytes, not the declared size."""
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("big.md", b"x" * (20 * 1024 * 1024 + 1)), _markdown("ok.md", b"# ok\n")],
    )
    assert response.status_code == 200
    big, ok = response.json()["files"]
    assert big["outcome"] == "failed"
    assert "20 MB limit" in big["reason"]
    assert ok["outcome"] == "imported"


def test_malformed_requests_are_client_errors(client: TestClient) -> None:
    """Only a request with nothing to report per file is an HTTP error."""
    assert client.post("/api/imports/documents").status_code == 400
    unknown = client.post(
        "/api/imports/documents",
        data={"destination": "knowledge_base"},
        files=[_markdown("a.md", b"# a\n")],
    )
    assert unknown.status_code == 400
    assert "authored_page" in unknown.json()["detail"]


def test_parent_page_is_passed_through(client: TestClient) -> None:
    """Import can target an existing page tree."""
    parent = client.post("/api/documents", json={"title": "Parent", "blocks": []}).json()
    response = client.post(
        "/api/imports/documents",
        data={"parent_id": parent["id"]},
        files=[_markdown("Child.md", b"# child\n")],
    )
    assert response.status_code == 200
    child = client.get(f"/api/documents/{response.json()['files'][0]['document_id']}").json()
    assert child["parent_id"] == parent["id"]


def test_a_batch_where_every_file_fails_is_still_a_200(client: TestClient) -> None:
    """A bad batch is not a bad request: there is a per-file outcome for each."""
    response = client.post(
        "/api/imports/documents",
        files=[_markdown("a.txt", b"x"), _markdown("b.txt", b"y")],
    )
    assert response.status_code == 200
    assert response.json()["imported"] == 0
    assert response.json()["skipped"] == 2


def test_unknown_parent_page_fails_each_file_and_creates_nothing(client: TestClient) -> None:
    """The parent is per request, so a missing one fails the batch per file."""
    response = client.post(
        "/api/imports/documents",
        data={"parent_id": "deadbeef1234"},
        files=[_markdown("orphan.md", b"# orphan\n"), _markdown("also.md", b"# also\n")],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["failed"] == 2
    assert payload["imported"] == 0
    for item in payload["files"]:
        assert item["outcome"] == "failed"
        assert item["document_id"] is None
        assert "Parent page does not exist" in item["reason"]
    # Nothing was created, so nothing can be opened.
    assert client.get("/api/documents").json()["documents"] == []
