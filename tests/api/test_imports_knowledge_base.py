"""Tests for importing files into a knowledge base (/api/imports/documents).

The one seam is the import HTTP endpoint. The knowledge manager runs on a
temporary base dir and the indexing task is replaced by a recorder, so every
assertion is either the JSON the endpoint returns, the bytes on disk under
``<kb>/raw/``, or the arguments handed to the pipeline that does the real
parsing and embedding.

What this does NOT prove: no index is built and no embedding server is
contacted, so "the file is retrievable afterwards" rests on the claim that the
same task, arguments and staging directory are used as a native upload (see
``test_import_is_indistinguishable_from_a_native_upload``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

import deeptutor.services.config as _dt_config

# The knowledge router loads main.yaml at import time; a worktree (and CI) has
# no data/user/settings, so stub the loader before importing anything.
_dt_config.load_config_with_main = lambda *_a, **_k: {
    "paths": {},
    "logging": {},
    "system": {"language": "en"},
}

# Imported after the stub above, on purpose.
from deeptutor.api.routers import imports as imports_router  # noqa: E402
from deeptutor.api.routers import knowledge as knowledge_router  # noqa: E402
from deeptutor.knowledge.manager import KnowledgeBaseManager  # noqa: E402

KB_NAME = "kb"


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
def manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> KnowledgeBaseManager:
    """A real manager whose base dir is a throwaway directory."""
    manager = KnowledgeBaseManager(base_dir=str(tmp_path / "kbs"))
    monkeypatch.setattr(knowledge_router, "get_kb_manager", lambda: manager)
    return manager


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _TaskRecorder:
    recorder = _TaskRecorder()
    monkeypatch.setattr(knowledge_router, "run_upload_processing_task", recorder)
    return recorder


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(imports_router.router, prefix="/api/imports")
    app.include_router(knowledge_router.router, prefix="/api")
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
    entry = {
        "path": name,
        "rag_provider": provider,
        "status": "ready",
        **(extra or {}),
    }
    manager.config.setdefault("knowledge_bases", {})[name] = entry
    manager._save_config()
    (kb_dir / "metadata.json").write_text(
        json.dumps({"rag_provider": provider, "needs_reindex": False}), encoding="utf-8"
    )
    return kb_dir


def _file(name: str, body: bytes):
    return ("files", (name, body, "application/octet-stream"))


def _import(client: TestClient, files, target: str = KB_NAME):
    return client.post(
        "/api/imports/documents",
        data={"destination": "knowledge_base", "target": target},
        files=files,
    )


def _raw(manager: KnowledgeBaseManager, name: str = KB_NAME) -> Path:
    return manager.base_dir / name / "raw"


def _outcomes(payload: dict) -> dict[str, dict]:
    return {item["filename"]: item for item in payload["files"]}


# ---------------------------------------------------------------------------
# 1. One file, one landing place, one job
# ---------------------------------------------------------------------------


def test_a_markdown_file_lands_in_the_knowledge_base_raw_directory(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """The bytes land where an upload lands, and one indexing task is queued."""
    seed_kb(manager)

    response = _import(client, [_file("Lecture One.md", b"# Fourier\n")])

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["destination"] == "knowledge_base"
    assert payload["imported"] == 1
    assert payload["failed"] == 0
    result = payload["files"][0]
    assert result["filename"] == "Lecture One.md"
    assert result["outcome"] == "imported"
    assert result["reason"] is None
    assert result["stored_paths"] == ["Lecture One.md"]

    stored = _raw(manager) / "Lecture One.md"
    assert stored.read_bytes() == b"# Fourier\n"

    # Exactly one indexing task, covering the accepted file.
    assert len(recorder.calls) == 1
    queued = recorder.calls[0]
    assert queued["kb_name"] == KB_NAME
    assert queued["base_dir"] == str(manager.base_dir)
    assert queued["rag_provider"] == "llamaindex"
    assert queued["uploaded_file_paths"] == [str(stored)]
    # Not a folder sync: those are the only callers that pass a folder id.
    assert queued.get("folder_id") is None
    assert payload["task_id"] == queued["task_id"]


def test_the_batch_is_one_indexing_job_not_one_per_file(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """Fifty files must not become fifty indexing jobs against one index."""
    seed_kb(manager)

    response = _import(
        client,
        [_file(f"note-{index}.md", b"# body\n") for index in range(5)],
    )

    assert response.status_code == 200
    assert response.json()["imported"] == 5
    assert len(recorder.calls) == 1
    assert len(recorder.calls[0]["uploaded_file_paths"]) == 5


# ---------------------------------------------------------------------------
# 2. A rejected file carries the knowledge base's own reason; the batch goes on
# ---------------------------------------------------------------------------


def test_a_file_the_engine_rejects_fails_alone_with_the_engine_reason(
    client: TestClient,
    manager: KnowledgeBaseManager,
    recorder: _TaskRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PageIndex accepts PDFs and office documents; one bad format is one result."""
    seed_kb(manager, provider="pageindex")
    monkeypatch.setattr(
        "deeptutor.services.rag.pipelines.pageindex.config.is_pageindex_configured",
        lambda: True,
    )

    response = _import(
        client,
        [
            _file("notes.md", b"# ok\n"),
            _file("page.html", b"<h1>not for pageindex</h1>"),
            _file("second.md", b"# two\n"),
        ],
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    files = _outcomes(payload)
    assert payload["imported"] == 2
    assert payload["failed"] == 1

    rejected = files["page.html"]
    assert rejected["outcome"] == "failed"
    # The engine's own sentence, naming the file it refused.
    assert "PageIndex knowledge bases accept:" in rejected["reason"]
    assert "page.html" in rejected["reason"]
    assert rejected["stored_paths"] is None

    # The batch continued after the rejected file: the one after it still landed.
    assert (_raw(manager) / "notes.md").exists()
    assert (_raw(manager) / "second.md").exists()
    assert not (_raw(manager) / "page.html").exists()

    # And only the accepted files went to the pipeline.
    assert len(recorder.calls) == 1
    assert sorted(Path(p).name for p in recorder.calls[0]["uploaded_file_paths"]) == [
        "notes.md",
        "second.md",
    ]


def test_an_unsupported_file_type_fails_alone(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    seed_kb(manager)

    response = _import(
        client,
        [_file("kept.md", b"# kept\n"), _file("payload.exe", b"MZ"), _file("also.md", b"# also\n")],
    )

    assert response.status_code == 200
    payload = response.json()
    files = _outcomes(payload)
    assert files["payload.exe"]["outcome"] == "failed"
    assert "unsupported file type" in files["payload.exe"]["reason"].lower()
    assert payload["imported"] == 2


def test_a_batch_where_every_file_is_rejected_queues_no_job(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """Nothing accepted means no indexing job and no status flip."""
    seed_kb(manager)

    response = _import(client, [_file("a.exe", b"MZ"), _file("b.exe", b"MZ")])

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported"] == 0
    assert payload["failed"] == 2
    assert payload["task_id"] is None
    assert recorder.calls == []
    entry = json.loads((manager.base_dir / "kb_config.json").read_text(encoding="utf-8"))[
        "knowledge_bases"
    ][KB_NAME]
    assert entry["status"] == "ready"


def test_two_files_cannot_share_one_raw_path(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """The second copy of a name is refused; the first is kept."""
    seed_kb(manager)

    response = _import(client, [_file("notes.md", b"first"), _file("notes.md", b"second")])

    assert response.status_code == 200
    first, second = response.json()["files"]
    assert first["outcome"] == "imported"
    assert second["outcome"] == "failed"
    assert "Duplicate filename" in second["reason"]
    assert (_raw(manager) / "notes.md").read_bytes() == b"first"
    assert len(recorder.calls) == 1
    assert len(recorder.calls[0]["uploaded_file_paths"]) == 1


# ---------------------------------------------------------------------------
# 3. Batch-level refusals are still reported per file, in the KB's own words
# ---------------------------------------------------------------------------


def test_an_unusable_embedding_binding_is_reported_per_file_in_the_kbs_words(
    client: TestClient,
    manager: KnowledgeBaseManager,
    recorder: _TaskRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken binding must not be paraphrased into a generic import error."""
    seed_kb(manager, extra={"embedding_selection": {"profile_id": "gone", "model_id": "gone"}})
    monkeypatch.setattr(
        "deeptutor.services.rag.embedding_binding.load_catalog",
        lambda: {"services": {"embedding": {"profiles": []}}},
    )

    # What the knowledge-base surface says for the same knowledge base.
    native = client.post(f"/api/knowledge-bases/{KB_NAME}/upload", files=[_file("a.md", b"# a\n")])
    assert native.status_code == 409, native.text
    reason = native.json()["detail"]

    response = _import(client, [_file("a.md", b"# a\n"), _file("b.md", b"# b\n")])

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["imported"] == 0
    assert payload["failed"] == 2
    assert payload["task_id"] is None
    for item in payload["files"]:
        assert item["outcome"] == "failed"
        assert item["reason"] == reason
        assert item["stored_paths"] is None
    assert recorder.calls == []
    assert not (_raw(manager) / "a.md").exists()


def test_a_read_only_knowledge_base_refuses_every_file_with_its_explanation(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """A vault pointer has no raw/ to receive files, so the batch is refused."""
    name = "vault"
    manager.config.setdefault("knowledge_bases", {})[name] = {
        "path": name,
        "type": "obsidian",
        "vault_path": str(manager.base_dir / "_vault"),
        "rag_provider": "llamaindex",
        "status": "ready",
    }
    manager._save_config()

    native = client.post(f"/api/knowledge-bases/{name}/upload", files=[_file("a.md", b"# a\n")])
    assert native.status_code == 409, native.text
    reason = native.json()["detail"]

    response = _import(client, [_file("a.md", b"# a\n"), _file("b.md", b"# b\n")], target=name)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["failed"] == 2
    for item in payload["files"]:
        assert item["outcome"] == "failed"
        assert item["reason"] == reason
        assert "read-only" in item["reason"]
    assert recorder.calls == []
    # A read-only pointer never grows a raw/ directory.
    assert not (manager.base_dir / name / "raw").exists()


# ---------------------------------------------------------------------------
# 4. The target is user data: verbatim, and only outer whitespace is trimmed
# ---------------------------------------------------------------------------


def test_the_knowledge_base_name_is_matched_verbatim(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """Names are case-sensitive, so the name cannot ride inside the destination."""
    seed_kb(manager, name="Study Notes")

    # The destination enum is case-insensitive; the name is not touched at all.
    upper = client.post(
        "/api/imports/documents",
        data={"destination": "Knowledge_Base", "target": "Study Notes"},
        files=[_file("a.md", b"# a\n")],
    )
    assert upper.status_code == 200, upper.text
    assert upper.json()["imported"] == 1

    # Outer whitespace is a paste artifact, not part of the name.
    padded = _import(client, [_file("b.md", b"# b\n")], target="  Study Notes  ")
    assert padded.status_code == 200
    assert padded.json()["imported"] == 1

    # A different case is a different name: no silent fallback, no guess.
    wrong = _import(client, [_file("c.md", b"# c\n")], target="study notes")
    assert wrong.status_code == 404
    assert "study notes" in wrong.json()["detail"]


def test_missing_target_is_a_request_error(
    client: TestClient, manager: KnowledgeBaseManager
) -> None:
    """A destination that names no knowledge base has nothing to report per file."""
    seed_kb(manager)

    response = client.post(
        "/api/imports/documents",
        data={"destination": "knowledge_base"},
        files=[_file("a.md", b"# a\n")],
    )

    assert response.status_code == 400
    assert "target" in response.json()["detail"]


def test_an_unknown_knowledge_base_is_a_404_naming_it(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    seed_kb(manager)

    response = _import(client, [_file("a.md", b"# a\n")], target="missing-kb")

    assert response.status_code == 404
    assert "missing-kb" in response.json()["detail"]
    assert recorder.calls == []


# ---------------------------------------------------------------------------
# 5. Indistinguishable from an upload, and bounded by the KB's limits
# ---------------------------------------------------------------------------


def test_import_is_indistinguishable_from_a_native_upload(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """Same staging directory, same pipeline arguments, same task namespace."""
    seed_kb(manager)

    native = client.post(
        f"/api/knowledge-bases/{KB_NAME}/upload", files=[_file("native.md", b"# same bytes\n")]
    )
    assert native.status_code == 200, native.text
    imported = _import(client, [_file("via-import.md", b"# same bytes\n")])
    assert imported.status_code == 200, imported.text

    assert len(recorder.calls) == 2
    native_call, import_call = recorder.calls

    # The two calls ask the pipeline for exactly the same kind of work.
    assert set(native_call) == set(import_call)
    for key in (
        "kb_name",
        "base_dir",
        "rag_provider",
        "owner",
        "storage_workspace_id",
        "accepted_indexing_snapshot",
        "folder_root",
    ):
        assert native_call.get(key) == import_call.get(key), key

    # Both stage into <kb>/raw and both run the same task type.
    native_path = Path(native_call["uploaded_file_paths"][0])
    import_path = Path(import_call["uploaded_file_paths"][0])
    assert native_path.parent == import_path.parent == _raw(manager)
    assert (_raw(manager) / "native.md").read_bytes() == (
        _raw(manager) / "via-import.md"
    ).read_bytes()

    native_type, _, _ = native_call["task_id"].partition("_")
    import_type, _, _ = import_call["task_id"].partition("_")
    assert native_type == import_type == "kb"


def test_the_knowledge_bases_own_size_limit_applies_not_the_import_ceiling(
    client: TestClient, manager: KnowledgeBaseManager, recorder: _TaskRecorder
) -> None:
    """A knowledge base takes documents the 20 MB page-import ceiling refuses."""
    seed_kb(manager)
    big = b"x" * (20 * 1024 * 1024 + 1)

    response = _import(client, [_file("big.pdf", big)])

    assert response.status_code == 200, response.text
    result = response.json()["files"][0]
    assert result["outcome"] == "imported", result["reason"]
    assert (_raw(manager) / "big.pdf").stat().st_size == len(big)
