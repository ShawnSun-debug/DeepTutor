"""Tests for `deeptutor import` — the command-line face of the import spine.

The command must not reimplement importing. It posts multipart form data at
`POST /api/imports/documents` in-process, against the real
`deeptutor.api.routers.imports` router, so these tests stand up the same
seam `tests/api/test_imports_auto.py` uses — Co-Writer storage, the notebook
manager and the knowledge-base manager pointed at `tmp_path`, and the
indexing task replaced by a recorder — and assert on what the command prints,
the exit code it returns, and that each file really landed in the store its
destination owns. A landing through a store patched on the router module is the
proof that the router's own code ran.

What these tests pin down:

1. Each of the three destinations imports through one command.
2. `auto` routes each file by type, exactly as the endpoint's table does.
3. Every file's outcome and reason is printed, grouped by destination.
4. The exit code reflects whether every single file imported.
5. `--help` names the supported formats and the destinations.
6. `--json` is the endpoint's response, parseable as-is.
7. A wrong invocation is a usage error (exit 2) with a readable message.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from typer.testing import CliRunner

from deeptutor.api.routers import imports as imports_router
from deeptutor.co_writer.storage import CoWriterStorage
from deeptutor.knowledge.manager import KnowledgeBaseManager
import deeptutor.services.config as _dt_config
from deeptutor.services.notebook.service import NotebookManager
from deeptutor_cli.main import app

runner = CliRunner()

KB_NAME = "kb"


@pytest.fixture(autouse=True)
def _stub_main_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The knowledge router loads main.yaml at import time; a worktree has none.

    The CLI imports that router lazily on its first knowledge-base import, so the
    stub has to be in place for the whole test rather than only at collection
    time (see the note in tests/api/test_imports_auto.py).
    """
    monkeypatch.setattr(
        _dt_config,
        "load_config_with_main",
        lambda *_a, **_k: {"paths": {}, "logging": {}, "system": {"language": "en"}},
    )


@pytest.fixture(autouse=True)
def _disable_pocketbase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deeptutor.services.pocketbase_client.is_pocketbase_enabled", lambda: False)


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


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CoWriterStorage:
    instance = CoWriterStorage(_Paths(tmp_path / "co-writer"))
    monkeypatch.setattr(imports_router, "_document_store", lambda: instance)
    return instance


@pytest.fixture
def notebooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> NotebookManager:
    instance = NotebookManager(base_dir=str(tmp_path / "notebook"))
    monkeypatch.setattr(imports_router, "_collection_manager", lambda: instance)
    return instance


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> KnowledgeBaseManager:
    from deeptutor.api.routers import knowledge as knowledge_router

    instance = KnowledgeBaseManager(base_dir=str(tmp_path / "kbs"))
    monkeypatch.setattr(knowledge_router, "get_kb_manager", lambda: instance)
    return instance


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _TaskRecorder:
    from deeptutor.api.routers import knowledge as knowledge_router

    instance = _TaskRecorder()
    monkeypatch.setattr(knowledge_router, "run_upload_processing_task", instance)
    return instance


@pytest.fixture(autouse=True)
def _isolated_stores(
    _stub_main_config: None,
    storage: CoWriterStorage,
    notebooks: NotebookManager,
    manager: KnowledgeBaseManager,
    recorder: _TaskRecorder,
) -> None:
    """Point all three stores at tmp_path for every test in this file.

    Autouse because the endpoint resolves its stores through helpers on the
    router module: a test that only reads the command's output would otherwise
    import into the developer's real notes store. The API tests get this for free
    by always going through their client fixture.
    """


def seed_kb(manager: KnowledgeBaseManager, name: str = KB_NAME) -> Path:
    """Create a ready knowledge base the manager and the route both accept."""
    kb_dir = manager.base_dir / name
    (kb_dir / "raw").mkdir(parents=True, exist_ok=True)
    entry = {"path": name, "rag_provider": "llamaindex", "status": "ready"}
    manager.config.setdefault("knowledge_bases", {})[name] = entry
    manager._save_config()
    (kb_dir / "metadata.json").write_text(
        json.dumps({"rag_provider": "llamaindex", "needs_reindex": False}), encoding="utf-8"
    )
    return kb_dir


def write(directory: Path, name: str, body: bytes | str) -> Path:
    path = directory / name
    if isinstance(body, str):
        path.write_text(body, encoding="utf-8")
    else:
        path.write_bytes(body)
    return path


def import_files(*args: str):
    return runner.invoke(app, ["import", *args])


# ---------------------------------------------------------------------------
# 1. Each destination imports through the one command
# ---------------------------------------------------------------------------


def test_to_notes_imports_a_markdown_file_as_a_page(
    storage: CoWriterStorage, tmp_path: Path
) -> None:
    source = write(tmp_path, "lesson.md", "# Fourier\n\nIt decomposes signals.\n")

    result = import_files(str(source), "--to", "notes")

    assert result.exit_code == 0, result.output
    assert "imported lesson.md" in result.output
    # The landing is the proof the real router ran: the store is patched on the
    # router module, which only its authored-page importer calls.
    documents = storage.list_documents()
    assert [document.title for document in documents] == ["lesson"]
    assert storage.load_document(documents[0].id).content == "# Fourier\n\nIt decomposes signals.\n"


def test_to_collection_imports_a_file_as_a_record(
    notebooks: NotebookManager, tmp_path: Path
) -> None:
    collection = notebooks.create_notebook(name="Physics")["id"]
    source = write(tmp_path, "notes.md", "# Fourier\n\nIt decomposes signals.\n")

    result = import_files(str(source), "--to", "collection", "--collection", collection)

    assert result.exit_code == 0, result.output
    assert "imported notes.md" in result.output
    records = notebooks.get_records(collection)
    assert [record["title"] for record in records] == ["notes"]
    assert records[0]["output"] == "# Fourier\n\nIt decomposes signals.\n"


def test_to_knowledge_base_imports_a_file_and_queues_one_job(
    manager: KnowledgeBaseManager, recorder: _TaskRecorder, tmp_path: Path
) -> None:
    kb_dir = seed_kb(manager)
    source = write(tmp_path, "paper.pdf", b"%PDF-1.4 first")

    result = import_files(str(source), "--to", "knowledge-base", "--kb", KB_NAME)

    assert result.exit_code == 0, result.output
    assert "imported paper.pdf" in result.output
    assert (kb_dir / "raw" / "paper.pdf").read_bytes() == b"%PDF-1.4 first"
    assert len(recorder.calls) == 1
    assert "Indexing task:" in result.output


# ---------------------------------------------------------------------------
# 2. auto routes per file, and the per-file report says where each one went
# ---------------------------------------------------------------------------


def test_auto_routes_each_file_by_type_into_all_three_stores(
    storage: CoWriterStorage,
    notebooks: NotebookManager,
    manager: KnowledgeBaseManager,
    recorder: _TaskRecorder,
    tmp_path: Path,
) -> None:
    kb_dir = seed_kb(manager)
    collection = notebooks.create_notebook(name="Physics")["id"]
    markdown = write(tmp_path, "notes.md", "# Fourier\n")
    html = write(tmp_path, "page.html", "<h1>Converted</h1><p>Body text.</p>")
    pdf = write(tmp_path, "paper.pdf", b"%PDF-1.4 first")

    result = import_files(
        str(markdown),
        str(html),
        str(pdf),
        "--to",
        "auto",
        "--collection",
        collection,
        "--kb",
        KB_NAME,
    )

    assert result.exit_code == 0, result.output
    # Markdown with a collection named is an explicit collection choice; HTML has
    # only one readable home; a PDF needs the knowledge base.
    assert "collection" in result.output
    assert "authored_page" in result.output
    assert "knowledge_base" in result.output
    assert [record["title"] for record in notebooks.get_records(collection)] == ["notes"]
    assert [document.title for document in storage.list_documents()] == ["page"]
    assert (kb_dir / "raw" / "paper.pdf").read_bytes() == b"%PDF-1.4 first"


def test_the_per_file_line_carries_the_endpoints_reason(
    storage: CoWriterStorage, tmp_path: Path
) -> None:
    ok = write(tmp_path, "ok.md", "# ok\n")
    binary = write(tmp_path, "payload.exe", b"MZ")

    result = import_files(str(ok), str(binary), "--to", "auto")

    # A binary is not guessed into a store: it fails, with the endpoint's own
    # reason, and the exit code says the batch was not a full success.
    assert result.exit_code == 1, result.output
    assert "imported ok.md" in result.output
    assert "failed payload.exe — unsupported extension (supported: " in result.output
    assert ".md" in result.output


def test_a_skipped_file_also_makes_the_command_fail(tmp_path: Path) -> None:
    good = write(tmp_path, "good.md", "# good\n")
    empty = write(tmp_path, "empty.md", "")

    result = import_files(str(good), str(empty), "--to", "notes")

    assert result.exit_code == 1, result.output
    assert "skipped empty.md — empty" in result.output
    assert "1 imported, 1 skipped, 0 failed" in result.output


def test_exit_code_is_zero_only_when_every_file_imported(tmp_path: Path) -> None:
    first = write(tmp_path, "a.md", "# a\n")
    second = write(tmp_path, "b.txt", "plain body\n")

    result = import_files(str(first), str(second), "--to", "notes")

    assert result.exit_code == 0, result.output
    assert "2 imported, 0 skipped, 0 failed" in result.output


# ---------------------------------------------------------------------------
# 3. --help is the discovery surface
# ---------------------------------------------------------------------------


def test_help_lists_the_formats_the_destinations_and_the_endpoint() -> None:
    result = import_files("--help")

    assert result.exit_code == 0, result.output
    # Destinations.
    for destination in ("notes", "collection", "knowledge-base", "auto"):
        assert destination in result.output
    # Formats: notes and collections take text; a knowledge base takes the
    # engine's own indexable formats.
    for extension in (".md", ".markdown", ".mdown", ".txt", ".html", ".htm", ".pdf", ".docx"):
        assert extension in result.output
    # And the command says which endpoint it goes through, so "one import spine"
    # is visible from the command line.
    assert "/api/imports/documents" in result.output


# ---------------------------------------------------------------------------
# 4. --json is the endpoint response, verbatim
# ---------------------------------------------------------------------------


def test_json_output_is_the_endpoint_response(storage: CoWriterStorage, tmp_path: Path) -> None:
    ok = write(tmp_path, "ok.md", "# ok\n")
    binary = write(tmp_path, "payload.exe", b"MZ")

    result = import_files(str(ok), str(binary), "--to", "auto", "--json")

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    # The same shape DocumentImportResponse defines, untouched.
    assert payload["destination"] == "auto"
    assert payload["imported"] == 1
    assert payload["failed"] == 1
    assert [item["filename"] for item in payload["files"]] == ["ok.md", "payload.exe"]
    assert payload["files"][0]["outcome"] == "imported"
    assert payload["files"][1]["outcome"] == "failed"
    assert payload["files"][1]["reason"].startswith("unsupported extension (supported: ")
    assert set(payload["groups"]) == {"authored_page"}
    assert payload["groups"]["authored_page"]["imported"] == 1


# ---------------------------------------------------------------------------
# 5. A wrong invocation is a usage error with a readable message
# ---------------------------------------------------------------------------


def test_a_missing_path_is_a_usage_error(tmp_path: Path) -> None:
    result = import_files(str(tmp_path / "nope.md"), "--to", "notes")

    assert result.exit_code == 2, result.output
    assert "nope.md" in result.output


def test_a_directory_is_refused_instead_of_being_walked(tmp_path: Path) -> None:
    folder = tmp_path / "drop"
    folder.mkdir()

    result = import_files(str(folder), "--to", "notes")

    assert result.exit_code == 2, result.output
    assert "not a file" in result.output


def test_collection_destination_without_a_collection_is_a_usage_error(tmp_path: Path) -> None:
    source = write(tmp_path, "a.md", "# a\n")

    result = import_files(str(source), "--to", "collection")

    assert result.exit_code == 2, result.output
    assert "--collection" in result.output


def test_knowledge_base_destination_without_a_knowledge_base_is_a_usage_error(
    tmp_path: Path,
) -> None:
    source = write(tmp_path, "a.pdf", b"%PDF-1.4")

    result = import_files(str(source), "--to", "knowledge-base")

    assert result.exit_code == 2, result.output
    assert "--kb" in result.output


def test_notes_destination_refuses_a_collection(tmp_path: Path) -> None:
    source = write(tmp_path, "a.md", "# a\n")

    result = import_files(str(source), "--to", "notes", "--collection", "abc")

    assert result.exit_code == 2, result.output
    assert "--collection" in result.output


def test_an_unknown_destination_is_a_usage_error(tmp_path: Path) -> None:
    source = write(tmp_path, "a.md", "# a\n")

    result = import_files(str(source), "--to", "notebooks")

    assert result.exit_code == 2, result.output
    assert "notes" in result.output and "knowledge-base" in result.output


def test_an_unknown_knowledge_base_is_reported_as_a_bad_request(
    manager: KnowledgeBaseManager, tmp_path: Path
) -> None:
    seed_kb(manager)
    source = write(tmp_path, "paper.pdf", b"%PDF-1.4")

    result = import_files(str(source), "--to", "knowledge-base", "--kb", "missing-kb")

    assert result.exit_code == 2, result.output
    assert "missing-kb" in result.output
