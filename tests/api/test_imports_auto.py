"""Tests for the automatic destination (/api/imports/documents, destination=auto).

The one seam is the HTTP endpoint, exactly as for the other import tests: a
TestClient over a bare FastAPI with the real routers mounted, Co-Writer storage
and the notebook/knowledge managers pointed at temporary directories, and the
indexing task replaced by a recorder. Nothing here reads a store directly to
prove a result unless the claim *is* about the store: every routing decision is
asserted on the JSON the endpoint returns, and every landing is then confirmed
through the surface that owns the store.

What these tests pin down:

1. A mixed drop is routed per file by type, and a request can reach all three
   stores at once.
2. The response carries a grouped view whose totals add up to the flat list.
3. An explicit destination overrides the automatic choice for every file.
4. A type nothing reasonable accepts fails per file, names what *is* accepted,
   and never guesses a store; a knowledge-base format with no knowledge base
   named fails the same way instead of picking one.
5. Duplicate content inside one request is skipped predictably and says which
   copy was kept; nothing is deduplicated across requests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import deeptutor.services.config as _dt_config

# The Co-Writer and knowledge routers load main.yaml at import time; a worktree
# has no data/user/settings, so stub the loader before importing them.
_dt_config.load_config_with_main = lambda *_a, **_k: {
    "paths": {},
    "logging": {},
    "system": {"language": "en"},
}

# Imported after the stub above, on purpose.
from deeptutor.api.routers import co_writer as co_writer_router  # noqa: E402
from deeptutor.api.routers import imports as imports_router  # noqa: E402
from deeptutor.api.routers import knowledge as knowledge_router  # noqa: E402
from deeptutor.api.routers import notebook as notebook_router  # noqa: E402
from deeptutor.co_writer.storage import CoWriterStorage  # noqa: E402
from deeptutor.knowledge.manager import KnowledgeBaseManager  # noqa: E402
from deeptutor.services.notebook.service import NotebookManager  # noqa: E402

AUTO = "auto"
KB_NAME = "kb"


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


class _TaskRecorder:
    """Stands in for the indexing task; records what was queued instead."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return None


@pytest.fixture(autouse=True)
def _disable_pocketbase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deeptutor.services.pocketbase_client.is_pocketbase_enabled", lambda: False)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CoWriterStorage:
    instance = CoWriterStorage(_Paths(tmp_path / "co-writer"))
    monkeypatch.setattr(imports_router, "_document_store", lambda: instance)
    monkeypatch.setattr(co_writer_router, "get_co_writer_storage", lambda: instance)
    return instance


@pytest.fixture
def notebooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NotebookManager:
    instance = NotebookManager(base_dir=str(tmp_path / "notebook"))
    monkeypatch.setattr(imports_router, "_collection_manager", lambda: instance)
    monkeypatch.setattr(notebook_router, "notebook_manager", instance)
    return instance


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KnowledgeBaseManager:
    instance = KnowledgeBaseManager(base_dir=str(tmp_path / "kbs"))
    monkeypatch.setattr(knowledge_router, "get_kb_manager", lambda: instance)
    return instance


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _TaskRecorder:
    instance = _TaskRecorder()
    monkeypatch.setattr(knowledge_router, "run_upload_processing_task", instance)
    return instance


@pytest.fixture
def client(storage: CoWriterStorage, manager: KnowledgeBaseManager) -> TestClient:
    app = FastAPI()
    # Production mounts imports at /api/imports (see web/lib/notes-api.ts) and
    # the two stores it can reach at /api and /api/notebooks.
    app.include_router(imports_router.router, prefix="/api/imports")
    app.include_router(co_writer_router.router, prefix="/api")
    app.include_router(knowledge_router.router, prefix="/api")
    app.include_router(notebook_router.router, prefix="/api")
    return TestClient(app)


def seed_kb(
    manager: KnowledgeBaseManager,
    name: str = KB_NAME,
    *,
    provider: str = "llamaindex",
    extra: dict | None = None,
) -> Path:
    """Create a ready knowledge base the manager and the route both accept."""
    kb_dir = manager.base_dir / name
    (kb_dir / "raw").mkdir(parents=True, exist_ok=True)
    entry = {"path": name, "rag_provider": provider, "status": "ready", **(extra or {})}
    manager.config.setdefault("knowledge_bases", {})[name] = entry
    manager._save_config()
    (kb_dir / "metadata.json").write_text(
        json.dumps({"rag_provider": provider, "needs_reindex": False}), encoding="utf-8"
    )
    return kb_dir


def _file(name: str, body: bytes):
    return ("files", (name, body, "application/octet-stream"))


def _auto(client: TestClient, files: list, **form):
    return client.post(
        "/api/imports/documents",
        data={"destination": AUTO, **form},
        files=files,
    )


def _by_name(payload: dict) -> dict[str, dict]:
    return {item["filename"]: item for item in payload["files"]}


def _raw(manager: KnowledgeBaseManager, name: str = KB_NAME) -> Path:
    return manager.base_dir / name / "raw"


def _page(client: TestClient, result: dict) -> dict:
    """Open the page a notes result reports, through the Co-Writer surface."""
    response = client.get(f"/api/documents/{result['document_id']}")
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 1. A mixed drop is routed per file by type
# ---------------------------------------------------------------------------


def test_a_mixed_drop_is_routed_by_type_and_reaches_all_three_stores(
    client: TestClient,
    manager: KnowledgeBaseManager,
    notebooks: NotebookManager,
    recorder: _TaskRecorder,
) -> None:
    """Markdown, HTML and a PDF land in three different stores in one request."""
    seed_kb(manager)
    collection = notebooks.create_notebook(name="Physics")["id"]

    response = _auto(
        client,
        [
            _file("notes.md", b"# Fourier\n\nIt decomposes signals.\n"),
            _file("page.html", b"<h1>Converted</h1><p>Body text.</p>"),
            _file("paper.pdf", b"%PDF-1.4 first"),
        ],
        collection_ids=collection,
        target=KB_NAME,
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    # The request-level destination is still what was asked for; each file
    # reports where it actually went.
    assert payload["destination"] == AUTO
    assert payload["imported"] == 3
    assert payload["skipped"] == 0
    assert payload["failed"] == 0
    assert [item["filename"] for item in payload["files"]] == [
        "notes.md",
        "page.html",
        "paper.pdf",
    ]
    assert not any(item["destination"] == AUTO for item in payload["files"])

    landed = _by_name(payload)
    # Markdown with a collection named is an explicit collection choice.
    assert landed["notes.md"]["destination"] == "collection"
    assert landed["notes.md"]["collection_ids"] == [collection]
    assert landed["notes.md"]["outcome"] == "imported"
    # HTML needs the authored-page converter, so notes is the only real home.
    assert landed["page.html"]["destination"] == "authored_page"
    assert landed["page.html"]["outcome"] == "imported"
    # Neither note nor record holds a PDF; the knowledge base does.
    assert landed["paper.pdf"]["destination"] == "knowledge_base"
    assert landed["paper.pdf"]["outcome"] == "imported"

    # Every landing is confirmed against the store that owns it.
    records = client.get(f"/api/notebooks/{collection}").json()["records"]
    assert [record["title"] for record in records] == ["notes"]
    assert records[0]["output"] == "# Fourier\n\nIt decomposes signals.\n"

    page = _page(client, landed["page.html"])
    assert page["title"] == "page"
    assert "<h1>" not in page["content"]
    assert "Converted" in page["content"]

    assert (_raw(manager) / "paper.pdf").read_bytes() == b"%PDF-1.4 first"
    assert len(recorder.calls) == 1
    assert [Path(p).name for p in recorder.calls[0]["uploaded_file_paths"]] == ["paper.pdf"]

    # And the store with nothing routed to it is absent, not an empty group.
    assert set(payload["groups"]) == {"authored_page", "collection", "knowledge_base"}


def test_markdown_and_text_without_a_collection_land_in_notes(
    client: TestClient, notebooks: NotebookManager
) -> None:
    """With no collection named, the automatic choice is notes; nothing invents one."""
    response = _auto(
        client,
        [_file("a.md", b"# A\n"), _file("b.txt", b"plain body\n")],
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["imported"] == 2
    assert {item["destination"] for item in payload["files"]} == {"authored_page"}
    landed = _by_name(payload)
    assert _page(client, landed["a.md"])["content"] == "# A\n"
    assert _page(client, landed["b.txt"])["content"] == "plain body\n"
    # The collection store was never touched: no collection exists afterwards.
    assert client.get("/api/notebooks").json()["notebooks"] == []


def test_html_goes_to_notes_even_when_collections_are_named(
    client: TestClient, notebooks: NotebookManager
) -> None:
    """A collection record holds text; raw tags would be worse than a page."""
    collection = notebooks.create_notebook(name="Physics")["id"]

    response = _auto(
        client,
        [_file("doc.html", b"<h1>Kept as a page</h1>")],
        collection_ids=collection,
    )
    assert response.status_code == 200, response.text
    result = response.json()["files"][0]
    assert result["destination"] == "authored_page"
    assert result["outcome"] == "imported"
    assert client.get(f"/api/notebooks/{collection}").json()["records"] == []


def test_the_knowledge_base_batch_stays_one_job_in_auto_mode(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """Routing does not split the knowledge base's one-job-per-batch contract."""
    seed_kb(manager)

    response = _auto(
        client,
        [_file("one.pdf", b"%PDF-1.4 one"), _file("two.pdf", b"%PDF-1.4 two")],
        target=KB_NAME,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["imported"] == 2
    assert len(recorder.calls) == 1
    assert payload["task_id"] == recorder.calls[0]["task_id"]
    assert payload["groups"]["knowledge_base"]["task_id"] == payload["task_id"]


# ---------------------------------------------------------------------------
# 2. The grouped view
# ---------------------------------------------------------------------------


def test_the_grouped_view_totals_every_destination(
    client: TestClient,
    manager: KnowledgeBaseManager,
    notebooks: NotebookManager,
    recorder: _TaskRecorder,
) -> None:
    """Groups aggregate files and outcomes, and add up to the flat list."""
    seed_kb(manager)
    collection = notebooks.create_notebook(name="Physics")["id"]

    response = _auto(
        client,
        [
            _file("ok.md", b"# ok\n"),
            _file("a.html", b"<h1>A</h1>"),
            _file("b.pdf", b"%PDF-1.4 b"),
            _file("payload.exe", b"MZ"),
            _file("empty.md", b""),
        ],
        collection_ids=collection,
        target=KB_NAME,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    groups = payload["groups"]

    # A file no store can take belongs to no group, but it is still counted.
    assert set(groups) == {"authored_page", "collection", "knowledge_base"}

    # a.html landed in notes; ok.md and the empty Markdown went to the collection
    # because collections were named, and only one of them could be read.
    notes = groups["authored_page"]
    assert (notes["files"], notes["imported"], notes["skipped"], notes["failed"]) == (1, 1, 0, 0)
    collections = groups["collection"]
    assert (collections["files"], collections["imported"], collections["skipped"]) == (2, 1, 1)
    kb = groups["knowledge_base"]
    assert (kb["files"], kb["imported"], kb["failed"]) == (1, 1, 0)

    # The groups are a partition of the routed files, not a second opinion.
    assert sum(group["files"] for group in groups.values()) == 4
    # The only ungrouped file is the one that reached no store, so it is the only
    # outcome the groups do not carry.
    assert (
        sum(group["imported"] for group in groups.values()),
        sum(group["skipped"] for group in groups.values()),
    ) == (payload["imported"], payload["skipped"])
    assert sum(group["failed"] for group in groups.values()) == payload["failed"] - 1
    assert sum(group["files"] for group in groups.values()) + 1 == len(payload["files"])
    for group in groups.values():
        assert group["imported"] + group["skipped"] + group["failed"] == group["files"]


def test_a_single_destination_still_reports_one_group(client: TestClient) -> None:
    """The grouped field is always present, so a client reads one shape."""
    payload = client.post(
        "/api/imports/documents",
        files=[_file("a.md", b"# a\n")],
    ).json()
    assert payload["destination"] == "authored_page"
    assert set(payload["groups"]) == {"authored_page"}
    assert payload["groups"]["authored_page"] == {
        "destination": "authored_page",
        "files": 1,
        "imported": 1,
        "skipped": 0,
        "failed": 0,
        "task_id": None,
    }


# ---------------------------------------------------------------------------
# 3. An explicit destination overrides the automatic choice
# ---------------------------------------------------------------------------


def test_an_explicit_destination_overrides_auto_for_every_file(
    client: TestClient, manager: KnowledgeBaseManager, notebooks: NotebookManager
) -> None:
    """One named destination means every file obeys that destination's rules."""
    seed_kb(manager)
    collection = notebooks.create_notebook(name="Physics")["id"]
    files = [_file("a.md", b"# a\n"), _file("b.html", b"<h1>b</h1>"), _file("c.pdf", b"%PDF-1.4")]

    notes = client.post(
        "/api/imports/documents",
        data={"destination": "authored_page"},
        files=files,
    )
    assert notes.status_code == 200, notes.text
    notes_payload = notes.json()
    assert set(notes_payload["groups"]) == {"authored_page"}
    assert {item["destination"] for item in notes_payload["files"]} == {"authored_page"}
    # The destination's own format rule decides, not auto's routing.
    assert _by_name(notes_payload)["c.pdf"]["outcome"] == "skipped"

    kb = client.post(
        "/api/imports/documents",
        data={"destination": "knowledge_base", "target": KB_NAME},
        files=[_file("a.md", b"# a\n"), _file("c.pdf", b"%PDF-1.4")],
    )
    assert kb.status_code == 200, kb.text
    kb_payload = kb.json()
    assert set(kb_payload["groups"]) == {"knowledge_base"}
    assert kb_payload["imported"] == 2
    assert collection not in kb_payload["groups"]

    records = client.post(
        "/api/imports/documents",
        data={"destination": "collection", "collection_ids": collection},
        files=[_file("d.md", b"# d\n")],
    )
    assert records.status_code == 200, records.text
    assert set(records.json()["groups"]) == {"collection"}


# ---------------------------------------------------------------------------
# 4. Nothing is guessed: unroutable types and unnamed knowledge bases fail
# ---------------------------------------------------------------------------


def test_a_type_no_store_accepts_fails_and_lists_the_supported_formats(
    client: TestClient, manager: KnowledgeBaseManager
) -> None:
    """A binary is not guessed into a store; the reason names what would work."""
    response = _auto(client, [_file("payload.exe", b"MZ")])
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["imported"] == 0
    assert payload["failed"] == 1
    result = payload["files"][0]
    assert result["outcome"] == "failed"
    assert result["destination"] == ""
    assert result["reason"].startswith("unsupported extension (supported: ")
    # The list is every format auto can route, so the user knows what to try.
    for extension in (".md", ".txt", ".html", ".pdf"):
        assert extension in result["reason"]
    # Nothing landed anywhere and no group was invented for it.
    assert payload["groups"] == {}
    assert not (_raw(manager) / "payload.exe").exists()


def test_a_knowledge_base_format_without_a_target_fails_instead_of_guessing(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """A PDF has a reasonable home, but auto will not pick a knowledge base."""
    seed_kb(manager)

    response = _auto(client, [_file("paper.pdf", b"%PDF-1.4"), _file("ok.md", b"# ok\n")])
    assert response.status_code == 200, response.text
    payload = response.json()

    landed = _by_name(payload)
    assert landed["ok.md"]["outcome"] == "imported"
    assert landed["ok.md"]["destination"] == "authored_page"
    assert landed["paper.pdf"]["outcome"] == "failed"
    assert landed["paper.pdf"]["destination"] == "knowledge_base"
    assert "knowledge base" in landed["paper.pdf"]["reason"].lower()
    assert "target" in landed["paper.pdf"]["reason"]
    # The reason is the missing choice, not a format complaint.
    assert "unsupported extension" not in landed["paper.pdf"]["reason"]

    # Nothing was written and nothing was queued.
    assert recorder.calls == []
    assert payload["task_id"] is None
    assert not (_raw(manager) / "paper.pdf").exists()


def test_an_unknown_knowledge_base_named_in_auto_mode_is_a_request_error(
    client: TestClient, manager: KnowledgeBaseManager
) -> None:
    """A name that resolves to no knowledge base is a wrong request, not a bad file."""
    seed_kb(manager)

    response = _auto(client, [_file("paper.pdf", b"%PDF-1.4")], target="missing-kb")
    assert response.status_code == 404
    assert "missing-kb" in response.json()["detail"]


def test_an_unknown_collection_in_auto_mode_fails_only_that_file(
    client: TestClient,
) -> None:
    """A collection is per file, so a bad id does not sink the rest of the drop."""
    response = _auto(
        client,
        [_file("a.md", b"# a\n"), _file("b.html", b"<h1>b</h1>")],
        collection_ids=["deadbeef"],
    )
    assert response.status_code == 200, response.text
    landed = _by_name(response.json())
    assert landed["a.md"]["outcome"] == "failed"
    assert "deadbeef" in landed["a.md"]["reason"]
    assert landed["b.html"]["outcome"] == "imported"


# ---------------------------------------------------------------------------
# 5. Duplicates inside one request
# ---------------------------------------------------------------------------


def test_a_second_copy_of_the_same_content_is_skipped_and_names_the_first(
    client: TestClient, storage: CoWriterStorage
) -> None:
    """Two identical drops in one request import once, and the result says which."""
    response = _auto(
        client,
        [
            _file("first.md", b"# Same content\n"),
            _file("second.md", b"# Same content\n"),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    first, second = payload["files"]
    assert first["outcome"] == "imported"
    assert first["filename"] == "first.md"
    assert second["outcome"] == "skipped"
    assert second["reason"] == (
        'duplicate of "first.md": only the first copy of identical content in this '
        "request was kept."
    )
    assert payload["imported"] == 1
    assert payload["skipped"] == 1
    # Only one page exists; the duplicate was never written.
    assert len(client.get("/api/documents").json()["documents"]) == 1
    # The duplicate is counted in the group it would have gone to.
    assert payload["groups"]["authored_page"] == {
        "destination": "authored_page",
        "files": 2,
        "imported": 1,
        "skipped": 1,
        "failed": 0,
        "task_id": None,
    }


def test_duplicate_detection_uses_content_not_the_filename(
    client: TestClient,
) -> None:
    """The same bytes under two names is one import; different bytes are two."""
    response = _auto(
        client,
        [
            _file("a.md", b"# body\n"),
            _file("a-copy.txt", b"# body\n"),
            _file("b.md", b"# other body\n"),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    outcomes = [item["outcome"] for item in payload["files"]]
    assert outcomes == ["imported", "skipped", "imported"]
    assert 'duplicate of "a.md"' in payload["files"][1]["reason"]


def test_two_empty_files_are_not_treated_as_duplicates(
    client: TestClient,
) -> None:
    """An empty file is empty, which is a better reason than "duplicate"."""
    response = _auto(client, [_file("a.md", b""), _file("b.md", b"")])
    assert response.status_code == 200, response.text
    payload = response.json()
    for item in payload["files"]:
        assert item["outcome"] == "skipped"
        assert item["reason"] == "empty"


def test_duplicates_are_not_deduplicated_across_requests(client: TestClient) -> None:
    """Deduplication is scoped to one request, so re-importing still works."""
    first = _auto(client, [_file("a.md", b"# same\n")]).json()
    second = _auto(client, [_file("a.md", b"# same\n")]).json()
    assert first["imported"] == 1
    assert second["imported"] == 1
    assert len(client.get("/api/documents").json()["documents"]) == 2


def test_a_duplicate_pointing_at_a_failed_first_copy_still_says_which_was_kept(
    client: TestClient,
) -> None:
    """The rule is positional, so it holds even when the first copy did not land."""
    response = _auto(
        client,
        [
            _file("too-big.md", b"x" * 600_001),
            _file("also-too-big.md", b"x" * 600_001),
        ],
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    first, second = payload["files"]
    assert first["outcome"] == "failed"
    assert "600,000 character limit" in first["reason"]
    assert second["outcome"] == "skipped"
    assert 'duplicate of "too-big.md"' in second["reason"]
