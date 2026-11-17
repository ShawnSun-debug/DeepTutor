"""HTML import into an authored page (POST /api/imports/documents).

The single seam under test is the HTTP endpoint, exactly as in
``test_imports_documents.py``: a TestClient over a bare FastAPI with the real
routers mounted at /api and Co-Writer storage pointed at a temporary
directory. Every assertion reads the page back through the same seam, so the
converter's internals are free to change.

HTML is the one format that needs real conversion: without it an .html file
would land as a page full of raw tags.
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
    monkeypatch.setattr(imports_router, "_document_store", lambda: storage)
    monkeypatch.setattr(co_writer_router, "get_co_writer_storage", lambda: storage)
    app = FastAPI()
    app.include_router(imports_router.router, prefix="/api/imports")
    app.include_router(co_writer_router.router, prefix="/api")
    return TestClient(app)


def _html(name: str, body: bytes):
    return ("files", (name, body, "text/html"))


def _import_one(client: TestClient, name: str, body: bytes) -> dict:
    """Import a single file and return its per-file result entry."""
    response = client.post("/api/imports/documents", files=[_html(name, body)])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["files"]) == 1
    return payload["files"][0]


def _page(client: TestClient, result: dict) -> dict:
    """Read the imported page back through the same seam."""
    assert result["document_id"] is not None
    page = client.get(f"/api/documents/{result['document_id']}")
    assert page.status_code == 200, page.text
    return page.json()


def test_headings_and_paragraphs_become_markdown(client: TestClient) -> None:
    """The simplest real export: a title, a subtitle, one paragraph."""
    result = _import_one(
        client,
        "Fourier.html",
        b"<html><body><h1>Fourier</h1><h2>Basics</h2>"
        b"<p>Transform &amp; inverse transform.</p></body></html>",
    )
    assert result["outcome"] == "imported"
    assert result["reason"] is None

    document = _page(client, result)
    assert document["content"] == "# Fourier\n\n## Basics\n\nTransform & inverse transform."
    assert document["blocks"] is None
    assert document["title"] == "Fourier"
    # No source tag survives the conversion.
    assert "<" not in document["content"]


def test_inline_markup_lists_quotes_and_code_survive(client: TestClient) -> None:
    """Every construct the spec names, in one page, converted exactly."""
    result = _import_one(
        client,
        "Lecture.html",
        (
            "<html><body>"
            "<h1>Lecture</h1>"
            "<p>Use <strong>bold</strong>, <em>italic</em> and "
            "<code>inline()</code>.</p>"
            '<p>See <a href="https://example.com/x">the notes</a>.</p>'
            "<h2>Steps</h2>"
            "<ol><li>First</li><li>Second</li></ol>"
            "<ul><li>Alpha</li><li>Beta</li></ul>"
            "<blockquote><p>Quoted wisdom.</p></blockquote>"
            "<pre><code>def f():\n    return 1\n</code></pre>"
            "</body></html>"
        ).encode(),
    )
    assert result["outcome"] == "imported", result

    document = _page(client, result)
    expected = (
        "# Lecture\n\n"
        "Use **bold**, *italic* and `inline()`.\n\n"
        "See [the notes](https://example.com/x).\n\n"
        "## Steps\n\n"
        "1. First\n"
        "2. Second\n\n"
        "- Alpha\n"
        "- Beta\n\n"
        "> Quoted wisdom.\n\n"
        "```\n"
        "def f():\n"
        "    return 1\n"
        "```"
    )
    assert document["content"] == expected
    assert "<" not in document["content"]


def test_tables_become_readable_rows_rather_than_vanishing(client: TestClient) -> None:
    """A table may degrade, but its cells must still be readable."""
    result = _import_one(
        client,
        "Glossary.html",
        (
            "<table>"
            "<tr><th>Term</th><th>Meaning</th></tr>"
            "<tr><td>Fourier</td><td>Frequency</td></tr>"
            "<tr><td>Legacy</td></tr>"
            "</table>"
        ).encode(),
    )
    assert result["outcome"] == "imported", result
    assert _page(client, result)["content"] == (
        "| Term | Meaning |\n| --- | --- |\n| Fourier | Frequency |\n| Legacy |  |"
    )


def test_unknown_elements_keep_their_text_and_scripts_do_not_leak(
    client: TestClient,
) -> None:
    """Unrecognised containers contribute their text; code and CSS never do."""
    result = _import_one(
        client,
        "styled.html",
        (
            "<html><head><title>Ignored</title>"
            "<style>body { color: red }</style></head>"
            '<body><div class="wrap"><span style="color:red">Red</span> '
            "<my-widget>Widget</my-widget></div>"
            "<script>var x = 1;</script>"
            "<p>Real text.</p></body></html>"
        ).encode(),
    )
    assert result["outcome"] == "imported", result
    assert _page(client, result)["content"] == "Red Widget\n\nReal text."


def test_malformed_markup_degrades_instead_of_failing(client: TestClient) -> None:
    """Unclosed everything still imports; only the formatting suffers."""
    result = _import_one(
        client,
        "broken.html",
        ("<html><body><h1>Broken<p>Still <b>here<ul><li>one<li>two<pre>unclosed code").encode(),
    )
    assert result["outcome"] == "imported", result
    assert result["reason"] is None

    content = _page(client, result)["content"]
    assert content.startswith("# Broken")
    for fragment in ("Still here", "one", "two", "unclosed code"):
        assert fragment in content, content
    assert "<" not in content


def test_html_without_readable_text_is_reported_as_empty(client: TestClient) -> None:
    """A page that converts to nothing must not land as a blank page."""
    response = client.post(
        "/api/imports/documents",
        files=[
            _html(
                "style-only.html",
                b"<html><head><style>body { color: red }</style></head>"
                b"<body><!-- nothing here --></body></html>",
            ),
            _html(
                "title-only.html",
                b"<html><head><title>Just a title</title></head><body></body></html>",
            ),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["imported"] == 0
    assert payload["skipped"] == 2
    for item in payload["files"]:
        assert item["outcome"] == "skipped"
        assert item["reason"] == "empty"
        assert item["document_id"] is None
    # Nothing was created, so nothing can be opened.
    assert client.get("/api/documents").json()["documents"] == []


def test_htm_extension_case_and_gbk_decoding(client: TestClient) -> None:
    """An older Chinese export is GBK, and .htm/.HTML are the same format."""
    response = client.post(
        "/api/imports/documents",
        files=[
            _html("笔记.htm", "<h1>笔记</h1><p>中文内容</p>".encode("gbk")),
            _html("Upper.HTML", b"<p>upper</p>"),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert [item["outcome"] for item in payload["files"]] == ["imported", "imported"]

    notes = client.get(f"/api/documents/{payload['files'][0]['document_id']}").json()
    assert notes["content"] == "# 笔记\n\n中文内容"
    assert notes["title"] == "笔记"
