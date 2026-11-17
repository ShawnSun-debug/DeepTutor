"""Tests for importing files into collections as records.

The single seam is the HTTP contract: a TestClient over a bare FastAPI with the
real imports and notebook routers mounted, and the notebook manager pointed at a
temporary directory. Every assertion is either the JSON the import endpoint
returns or a follow-up request through the collection surface itself — the record
has to be listed, openable and exportable exactly like a saved favorite, so the
tests prove that through /api/notebooks rather than by reading the JSON file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import deeptutor.services.config as _dt_config

# The Co-Writer module loads main.yaml at import time; a worktree has no
# data/user/settings, so stub the loader before importing the routers.
_dt_config.load_config_with_main = lambda *_a, **_k: {
    "paths": {},
    "logging": {},
    "system": {"language": "en"},
}

import deeptutor.agents.notebook as notebook_agents
from deeptutor.api.routers import imports as imports_router
from deeptutor.api.routers import notebook as notebook_router
from deeptutor.services.notebook.service import NotebookManager

COLLECTION = "collection"


class _SummarizerMustNotRun:
    """Stands in for the summarisation agent so any use of it is a failure.

    A collection record with an empty summary is what triggers the agent, so an
    import that let a record land without one would try to build this class and
    raise. Counting the constructions as well means the promise is checked even
    if a future caller swallows the exception.
    """

    calls = 0

    def __init__(self, *_args, **_kwargs) -> None:
        type(self).calls += 1
        raise AssertionError("import must not construct the summarisation agent")


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NotebookManager:
    instance = NotebookManager(base_dir=str(tmp_path / "notebook"))
    monkeypatch.setattr(imports_router, "notebook_manager", instance)
    monkeypatch.setattr(notebook_router, "notebook_manager", instance)
    return instance


@pytest.fixture
def client(manager: NotebookManager) -> TestClient:
    app = FastAPI()
    # Production mounts imports at /api/imports (see web/lib/imports-api.ts) and
    # the collections surface at /api/notebooks.
    app.include_router(imports_router.router, prefix="/api/imports")
    app.include_router(notebook_router.router, prefix="/api")
    return TestClient(app)


@pytest.fixture
def collection(manager: NotebookManager) -> str:
    return manager.create_notebook(name="Physics")["id"]


def _file(name: str, body: bytes):
    return ("files", (name, body, "text/markdown"))


def _import(client: TestClient, files: list, **form):
    return client.post(
        "/api/imports/documents",
        data={"destination": COLLECTION, **form},
        files=files,
    )


def test_one_file_becomes_one_record(client: TestClient, collection: str) -> None:
    """A file lands as one ordinary record: listed, openable, exportable."""
    response = _import(
        client,
        [_file("Fourier Notes.md", b"# Fourier Transform\n\nIt decomposes signals.\n")],
        collection_ids=collection,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["destination"] == COLLECTION
    assert payload["imported"] == 1

    result = payload["files"][0]
    assert result["filename"] == "Fourier Notes.md"
    assert result["outcome"] == "imported"
    assert result["reason"] is None
    assert result["collection_ids"] == [collection]
    assert result["record_id"]
    # Authored pages are the other destination; this result is not one.
    assert result["document_id"] is None

    # Listed like any other collection.
    listing = client.get("/api/notebooks").json()["notebooks"]
    assert [item["record_count"] for item in listing] == [1]

    # Openable: the record carries the file's title and body.
    detail = client.get(f"/api/notebooks/{collection}").json()
    record = detail["records"][0]
    assert record["id"] == result["record_id"]
    assert record["title"] == "Fourier Notes"
    assert record["output"] == "# Fourier Transform\n\nIt decomposes signals.\n"
    assert record["summary"] == "Fourier Transform"

    # Exportable: the collection's Markdown export includes the imported body.
    export = client.get(f"/api/notebooks/{collection}/export")
    assert export.status_code == 200
    assert "## Fourier Notes" in export.text
    assert "It decomposes signals." in export.text


def test_record_type_is_the_import_value(client: TestClient, collection: str) -> None:
    """The vocabulary gains an import value instead of borrowing an unrelated one."""
    _import(client, [_file("a.md", b"# A\n")], collection_ids=collection)
    record = client.get(f"/api/notebooks/{collection}").json()["records"][0]
    assert record["type"] == "import"

    stats = client.get("/api/notebooks/statistics").json()
    assert stats["records_by_type"]["import"] == 1
    assert sum(stats["records_by_type"].values()) == stats["total_records"] == 1


def test_a_large_drop_makes_no_summarisation_call(
    client: TestClient, collection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promise that matters: fifty files cost zero model calls.

    Both references to the agent class are replaced with something that raises,
    so a summary generated by a model would fail the import instead of quietly
    costing money.
    """
    _SummarizerMustNotRun.calls = 0
    monkeypatch.setattr(notebook_agents, "NotebookSummarizeAgent", _SummarizerMustNotRun)
    monkeypatch.setattr(notebook_router, "NotebookSummarizeAgent", _SummarizerMustNotRun)

    files = [
        _file(f"note-{index}.md", f"# Note {index}\n\nBody {index}.\n".encode())
        for index in range(50)
    ]
    response = _import(client, files, collection_ids=collection)
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["imported"] == 50
    assert payload["failed"] == 0
    assert _SummarizerMustNotRun.calls == 0
    # Every record still carries a locally derived summary, which is what keeps
    # the list readable without the agent.
    records = client.get(f"/api/notebooks/{collection}").json()["records"]
    assert [record["summary"] for record in records] == [f"Note {index}" for index in range(50)]


def test_summary_falls_back_to_the_first_non_empty_line(
    client: TestClient, collection: str
) -> None:
    """A file with no heading still gets a summary from its opening text."""
    response = _import(
        client,
        [_file("plain.txt", b"\n\n   Green tea, two minutes.\n\nMore detail.\n")],
        collection_ids=collection,
    )
    assert response.status_code == 200, response.text
    record = client.get(f"/api/notebooks/{collection}").json()["records"][0]
    assert record["title"] == "plain"
    assert record["summary"] == "Green tea, two minutes."


def test_summary_is_truncated(client: TestClient, collection: str) -> None:
    """A long opening line is cut, so the collection list stays scannable."""
    _import(client, [_file("long.md", ("a" * 300).encode())], collection_ids=collection)
    record = client.get(f"/api/notebooks/{collection}").json()["records"][0]
    assert record["summary"] == "a" * 240 + "\u2026"


def test_one_request_can_target_several_collections(
    client: TestClient, manager: NotebookManager
) -> None:
    """One request may target several collections, and each file lands once per target."""
    first = manager.create_notebook(name="One")["id"]
    second = manager.create_notebook(name="Two")["id"]

    response = _import(
        client,
        [_file("shared.md", b"# Shared\n\nSame body.\n")],
        collection_ids=[first, second],
    )
    assert response.status_code == 200, response.text
    result = response.json()["files"][0]
    assert result["outcome"] == "imported"
    assert result["collection_ids"] == [first, second]

    for notebook_id in (first, second):
        records = client.get(f"/api/notebooks/{notebook_id}").json()["records"]
        assert [record["id"] for record in records] == [result["record_id"]]


def test_unknown_collection_fails_the_file_and_creates_nothing(client: TestClient) -> None:
    """The add-record funnel skips unknown notebooks silently; the import must not."""
    response = _import(client, [_file("lost.md", b"# Lost\n")], collection_ids=["deadbeef"])
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["imported"] == 0
    assert payload["failed"] == 1
    item = payload["files"][0]
    assert item["outcome"] == "failed"
    assert "deadbeef" in item["reason"]
    assert item["record_id"] is None
    assert item["collection_ids"] is None
    # Nothing was created, so nothing can be listed.
    assert client.get("/api/notebooks").json()["notebooks"] == []


def test_a_partial_target_is_reported_rather_than_hidden(
    client: TestClient, manager: NotebookManager
) -> None:
    """A record that reached one of two targets says where it did not go."""
    good = manager.create_notebook(name="Good")["id"]
    response = _import(
        client,
        [_file("partial.md", b"# Partial\n")],
        collection_ids=[good, "no-such-collection"],
    )
    item = response.json()["files"][0]
    assert item["outcome"] == "imported"
    assert item["collection_ids"] == [good]
    assert "no-such-collection" in item["reason"]
    assert len(client.get(f"/api/notebooks/{good}").json()["records"]) == 1


def test_a_request_without_a_target_fails_each_file(client: TestClient) -> None:
    """No target is a per-file failure, the same rule a missing parent page follows."""
    response = _import(client, [_file("a.md", b"# a\n"), _file("b.md", b"# b\n")])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["failed"] == 2
    for item in payload["files"]:
        assert item["outcome"] == "failed"
        assert item["reason"] == "No target collection was given."


def test_each_file_becomes_its_own_record(client: TestClient, collection: str) -> None:
    response = _import(
        client,
        [_file("one.md", b"# One\n"), _file("two.md", b"# Two\n")],
        collection_ids=collection,
    )
    payload = response.json()
    assert payload["imported"] == 2
    record_ids = [item["record_id"] for item in payload["files"]]
    assert len(set(record_ids)) == 2
    records = client.get(f"/api/notebooks/{collection}").json()["records"]
    assert [record["title"] for record in records] == ["one", "two"]


def test_unsupported_empty_and_oversized_files_keep_their_own_outcome(
    client: TestClient, collection: str
) -> None:
    """One bad file never fails the batch, and never becomes a blank record."""
    response = _import(
        client,
        [
            _file("good.md", b"# Good\n\nFine.\n"),
            _file("scan.pdf", b"%PDF-1.4"),
            _file("empty.md", b""),
            _file("blank.md", b"\n\n   \n"),
            _file("toolong.md", b"x" * 600_001),
            _file("second.txt", b"# Second\n"),
        ],
        collection_ids=collection,
    )
    assert response.status_code == 200, response.text
    outcomes = {item["filename"]: item for item in response.json()["files"]}

    assert outcomes["good.md"]["outcome"] == "imported"
    assert outcomes["second.txt"]["outcome"] == "imported"
    assert outcomes["scan.pdf"]["outcome"] == "skipped"
    assert outcomes["scan.pdf"]["reason"] == (
        "unsupported extension (supported: .md, .markdown, .txt)"
    )
    assert outcomes["empty.md"]["outcome"] == "skipped"
    assert outcomes["empty.md"]["reason"] == "empty"
    assert outcomes["blank.md"]["outcome"] == "skipped"
    assert outcomes["toolong.md"]["outcome"] == "failed"
    assert "600,000 character limit" in outcomes["toolong.md"]["reason"]

    payload = response.json()
    assert (payload["imported"], payload["skipped"], payload["failed"]) == (2, 3, 1)
    # Only the two usable files became records.
    assert len(client.get(f"/api/notebooks/{collection}").json()["records"]) == 2
