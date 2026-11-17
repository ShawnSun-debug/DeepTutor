from __future__ import annotations

import asyncio
from datetime import datetime
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - optional dependency in lightweight envs
    FastAPI = None
    TestClient = None

pytestmark = pytest.mark.skipif(
    FastAPI is None or TestClient is None, reason="fastapi not installed"
)

if FastAPI is not None and TestClient is not None:
    import importlib

    knowledge_router_module = importlib.import_module("deeptutor.api.routers.knowledge")
    router = knowledge_router_module.router
else:  # pragma: no cover - optional dependency in lightweight envs
    knowledge_router_module = None
    router = None


class _FolderManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.config = {
            "knowledge_bases": {
                "kb": {
                    "path": "kb",
                    "rag_provider": "llamaindex",
                    "needs_reindex": False,
                    "status": "ready",
                }
            }
        }
        self.config_file = base_dir / "kb_config.json"
        self.folder = {
            "id": "folder-1",
            "path": str(base_dir / "notes"),
            "added_at": "2026-09-09T10:00:00",
            "file_count": 2,
            "last_sync": None,
        }
        self.update_calls: list[tuple[str, str, list[str]]] = []
        self.mtime_calls: list[dict[str, str] | None] = []
        self.detect_result = {
            "new_files": [],
            "modified_files": [],
            "new_count": 0,
            "modified_count": 0,
        }

    def _load_config(self) -> dict:
        return self.config

    def list_knowledge_bases(self) -> list[str]:
        return list(self.config["knowledge_bases"])

    def get_default(self) -> str:
        return "kb"

    def get_linked_folders(self, _kb_name: str) -> list[dict]:
        return [self.folder]

    def link_folder(self, _kb_name: str, _folder_path: str) -> dict:
        return self.folder

    def unlink_folder(self, _kb_name: str, _folder_id: str) -> bool:
        return True

    def detect_folder_changes(self, _kb_name: str, _folder_id: str) -> dict:
        return self.detect_result

    def update_folder_sync_state(
        self,
        kb_name: str,
        folder_id: str,
        files: list[str],
        source_mtimes: dict[str, str] | None = None,
    ) -> None:
        self.update_calls.append((kb_name, folder_id, files))
        self.mtime_calls.append(source_mtimes)

    def update_kb_status(self, name: str, status: str, progress: dict | None = None) -> None:
        self.config["knowledge_bases"][name]["status"] = status
        self.config["knowledge_bases"][name]["progress"] = progress or {}


def _build_app() -> FastAPI:
    if FastAPI is None or router is None:  # pragma: no cover - guarded by pytestmark
        raise RuntimeError("fastapi is not installed")
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


def _patch_manager(monkeypatch: pytest.MonkeyPatch, manager: _FolderManager) -> None:
    monkeypatch.setattr(knowledge_router_module, "get_kb_manager", lambda: manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "get_current_user",
        lambda: None,
    )
    monkeypatch.setattr(
        knowledge_router_module,
        "resolve_kb",
        lambda _name: SimpleNamespace(name="kb", base_dir=manager.base_dir),
    )
    monkeypatch.setattr(
        knowledge_router_module,
        "manager_for_resource",
        lambda _resource: manager,
    )


def test_linked_folder_routes_return_explicit_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _FolderManager(tmp_path)
    _patch_manager(monkeypatch, manager)

    with TestClient(_build_app()) as client:
        linked = client.post(
            "/api/knowledge-bases/kb/link-folder",
            json={"folder_path": str(tmp_path / "notes")},
        )
        listed = client.get("/api/knowledge-bases/kb/linked-folders")

    assert linked.status_code == 200
    assert linked.json()["last_sync"] is None
    assert listed.status_code == 200
    assert listed.json()[0]["last_sync"] is None


def test_noop_sync_records_success_and_returns_sync_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _FolderManager(tmp_path)
    _patch_manager(monkeypatch, manager)

    with TestClient(_build_app()) as client:
        response = client.post("/api/knowledge-bases/kb/sync-folder/folder-1")

    assert response.status_code == 200
    assert response.json() == {
        "message": "No new or modified files to sync",
        "folder_path": str(tmp_path / "notes"),
        "files": [],
        "new_files": 0,
        "modified_files": 0,
        "file_count": 0,
        "task_id": None,
        "removed": [],
        "removed_files": 0,
    }
    assert manager.update_calls == [("kb", "folder-1", [])]


def test_queued_sync_returns_counts_and_task_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _FolderManager(tmp_path)
    manager.detect_result = {
        "new_files": [str(tmp_path / "notes" / "new.md")],
        "modified_files": [str(tmp_path / "notes" / "changed.md")],
        "new_count": 1,
        "modified_count": 1,
    }
    _patch_manager(monkeypatch, manager)
    monkeypatch.setattr(
        knowledge_router_module,
        "_build_unique_task_id",
        lambda *_args: "sync-task",
    )

    async def _noop_task(*_args, **_kwargs):
        return None

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _noop_task)

    with TestClient(_build_app()) as client:
        response = client.post("/api/knowledge-bases/kb/sync-folder/folder-1")

    assert response.status_code == 200
    assert response.json()["new_files"] == 1
    assert response.json()["modified_files"] == 1
    assert response.json()["file_count"] == 2
    assert response.json()["files"] == [
        str(tmp_path / "notes" / "new.md"),
        str(tmp_path / "notes" / "changed.md"),
    ]
    assert response.json()["task_id"] == "sync-task"


def test_link_folder_rejects_connected_kb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manager = _FolderManager(tmp_path)
    manager.config["knowledge_bases"]["kb"]["type"] = "linked"
    _patch_manager(monkeypatch, manager)

    with TestClient(_build_app()) as client:
        response = client.post(
            "/api/knowledge-bases/kb/link-folder",
            json={"folder_path": str(tmp_path / "notes")},
        )

    assert response.status_code == 409


def test_unlink_folder_rejects_connected_kb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _FolderManager(tmp_path)
    manager.config["knowledge_bases"]["kb"]["type"] = "linked"
    _patch_manager(monkeypatch, manager)

    with TestClient(_build_app()) as client:
        response = client.delete("/api/knowledge-bases/kb/linked-folders/folder-1")

    assert response.status_code == 409


def test_completed_folder_task_records_source_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _FolderManager(tmp_path)
    _patch_manager(monkeypatch, manager)
    source_root = tmp_path / "notes"
    source_path = source_root / "nested" / "note.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("note", encoding="utf-8")
    staged_path = tmp_path / "kb" / "raw" / "nested" / "note.md"

    class _SuccessfulAdder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def add_documents(self, _paths, **_kwargs):
            return [staged_path]

        async def process_new_documents(self, _staged):
            return SimpleNamespace(
                processed_files=[staged_path],
                processed_count=1,
                has_failures=False,
            )

        def update_metadata(self, _count) -> None:
            pass

    monkeypatch.setattr(knowledge_router_module, "DocumentAdder", _SuccessfulAdder)

    asyncio.run(
        knowledge_router_module.run_upload_processing_task(
            kb_name="kb",
            base_dir=str(tmp_path),
            uploaded_file_paths=[str(source_path)],
            task_id="folder-success-test",
            rag_provider="llamaindex",
            folder_id="folder-1",
            folder_root=str(source_root),
        )
    )

    assert manager.update_calls == [("kb", "folder-1", [str(source_path)])]
    assert manager.mtime_calls == [
        {str(source_path): datetime.fromtimestamp(source_path.stat().st_mtime).isoformat()}
    ]


def test_empty_completed_folder_task_advances_sync_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _FolderManager(tmp_path)
    _patch_manager(monkeypatch, manager)
    source_path = tmp_path / "notes" / "note.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("note", encoding="utf-8")

    class _EmptyAdder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def add_documents(self, _paths, **_kwargs):
            return []

        def update_metadata(self, _count) -> None:
            raise AssertionError("empty tasks do not need metadata updates")

    monkeypatch.setattr(knowledge_router_module, "DocumentAdder", _EmptyAdder)

    asyncio.run(
        knowledge_router_module.run_upload_processing_task(
            kb_name="kb",
            base_dir=str(tmp_path),
            uploaded_file_paths=[str(source_path)],
            task_id="folder-empty-test",
            rag_provider="llamaindex",
            folder_id="folder-1",
            folder_root=str(tmp_path / "notes"),
        )
    )

    assert manager.update_calls == [("kb", "folder-1", [str(source_path)])]
    assert manager.mtime_calls == [
        {str(source_path): datetime.fromtimestamp(source_path.stat().st_mtime).isoformat()}
    ]


def test_failed_folder_task_does_not_advance_sync_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = _FolderManager(tmp_path)
    manager.folder["last_sync"] = "previous-success"
    _patch_manager(monkeypatch, manager)
    staged_path = tmp_path / "kb" / "raw" / "note.md"

    class _FailingAdder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def add_documents(self, _paths, **_kwargs):
            return [staged_path]

        async def process_new_documents(self, _staged):
            return SimpleNamespace(
                processed_files=[],
                processed_count=0,
                has_failures=True,
                failed_count=1,
                failure_summary=lambda: "note.md: indexing failed",
                failures=[SimpleNamespace(file_path=staged_path, error="indexing failed")],
            )

        def update_metadata(self, _count) -> None:
            raise AssertionError("failed tasks must not update metadata")

    monkeypatch.setattr(knowledge_router_module, "DocumentAdder", _FailingAdder)

    asyncio.run(
        knowledge_router_module.run_upload_processing_task(
            kb_name="kb",
            base_dir=str(tmp_path),
            uploaded_file_paths=[str(tmp_path / "notes" / "note.md")],
            task_id="folder-failure-test",
            rag_provider="llamaindex",
            folder_id="folder-1",
            folder_root=str(tmp_path / "notes"),
        )
    )

    assert manager.update_calls == []
    assert manager.folder["last_sync"] == "previous-success"


# ---------------------------------------------------------------------------
# Folder integrity, removals and the one-run-per-source guard.
#
# These drive the real KnowledgeBaseManager on tmp_path so the change detection
# deciding "new / modified / unchanged / removed" is the production one. Only
# the indexing task is replaced by a recorder: parsing, embedding and indexing
# have their own tests and are not what this route owns.
# ---------------------------------------------------------------------------


class _LinkedFolderKB:
    """A real KB manager with one linked folder and a recorder for dispatch."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from deeptutor.knowledge.manager import KnowledgeBaseManager

        base = tmp_path / "kbs"
        base.mkdir()
        self.manager = KnowledgeBaseManager(base_dir=str(base))
        (base / "kb").mkdir()
        self.manager.register_knowledge_base("kb")
        self.kb_dir = base / "kb"

        self.source = tmp_path / "notes"
        self.source.mkdir()
        self.keep = self.source / "keep.md"
        self.keep.write_text("keep", encoding="utf-8")
        self.drop = self.source / "drop.md"
        self.drop.write_text("drop", encoding="utf-8")

        self.folder = self.manager.link_folder("kb", str(self.source))
        self.dispatched: list[dict] = []

        monkeypatch.setattr(knowledge_router_module, "get_kb_manager", lambda: self.manager)
        monkeypatch.setattr(knowledge_router_module, "get_current_user", lambda: None)
        monkeypatch.setattr(
            knowledge_router_module, "run_upload_processing_task", self._record_dispatch
        )

    async def _record_dispatch(self, **kwargs) -> None:
        self.dispatched.append(kwargs)

    def sync(self, client: TestClient):
        return client.post(f"/api/knowledge-bases/kb/sync-folder/{self.folder['id']}")

    def record_synced(self, *paths: Path) -> None:
        """Apply what a completed indexing run records, so the next scan sees them."""
        self.manager.update_folder_sync_state("kb", self.folder["id"], [str(p) for p in paths])

    def stored_folder(self) -> dict:
        return self.manager.get_linked_folders("kb")[0]


def _advance_mtime(path: Path, seconds: float) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + seconds))


def paths_of(dispatch: dict) -> list[str]:
    return [str(path) for path in dispatch["uploaded_file_paths"]]


@pytest.fixture()
def linked_kb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _LinkedFolderKB:
    return _LinkedFolderKB(tmp_path, monkeypatch)


def test_sync_dispatches_new_and_modified_files_only(linked_kb: _LinkedFolderKB) -> None:
    with TestClient(_build_app()) as client:
        first = linked_kb.sync(client)
        assert first.status_code == 200
        assert sorted(paths_of(linked_kb.dispatched[0])) == sorted(
            [str(linked_kb.keep), str(linked_kb.drop)]
        )
        assert linked_kb.dispatched[0]["folder_root"] == str(linked_kb.source)

        # The completed run records both files; one of them then changes on disk.
        linked_kb.record_synced(linked_kb.keep, linked_kb.drop)
        _advance_mtime(linked_kb.drop, 60)

        second = linked_kb.sync(client)

    body = second.json()
    assert second.status_code == 200
    assert body["new_files"] == 0
    assert body["modified_files"] == 1
    assert body["files"] == [str(linked_kb.drop)]
    # The unchanged file is not re-indexed.
    assert paths_of(linked_kb.dispatched[1]) == [str(linked_kb.drop)]
    assert linked_kb.dispatched[1]["folder_root"] == str(linked_kb.source)


def test_sync_reports_removed_files_and_keeps_their_imported_copies(
    linked_kb: _LinkedFolderKB,
) -> None:
    imported = linked_kb.kb_dir / "raw" / "drop.md"
    imported.parent.mkdir(parents=True)
    imported.write_text("imported", encoding="utf-8")
    linked_kb.record_synced(linked_kb.keep, linked_kb.drop)
    linked_kb.drop.unlink()

    with TestClient(_build_app()) as client:
        response = linked_kb.sync(client)

    body = response.json()
    assert response.status_code == 200
    assert body["removed_files"] == 1
    assert body["removed"] == [str(linked_kb.drop)]
    # The knowledge base's existing deletion semantics stay explicit: a sync
    # never deletes an imported document behind the user's back.
    assert imported.read_text(encoding="utf-8") == "imported"
    assert "no longer in the folder" in body["message"]
    assert "knowledge base" in body["message"]
    assert linked_kb.dispatched == []
    assert linked_kb.stored_folder()["last_sync"] is not None


def test_sync_refuses_a_linked_folder_that_is_gone(linked_kb: _LinkedFolderKB) -> None:
    linked_kb.record_synced(linked_kb.keep, linked_kb.drop)
    synced_at = linked_kb.stored_folder()["last_sync"]
    shutil.rmtree(linked_kb.source)

    with TestClient(_build_app()) as client:
        response = linked_kb.sync(client)

    assert response.status_code == 409
    assert str(linked_kb.source) in response.json()["detail"]
    # A failed scan must not look like a successful sync.
    assert linked_kb.stored_folder()["last_sync"] == synced_at
    assert linked_kb.dispatched == []


def test_sync_refuses_a_linked_folder_that_lost_every_file(
    linked_kb: _LinkedFolderKB,
) -> None:
    linked_kb.record_synced(linked_kb.keep, linked_kb.drop)
    synced_at = linked_kb.stored_folder()["last_sync"]
    linked_kb.keep.unlink()
    linked_kb.drop.unlink()

    with TestClient(_build_app()) as client:
        response = linked_kb.sync(client)

    assert response.status_code == 409
    assert str(linked_kb.source) in response.json()["detail"]
    assert linked_kb.stored_folder()["last_sync"] == synced_at
    assert linked_kb.dispatched == []


def test_sync_of_a_new_folder_without_supported_files_is_a_successful_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    linked_kb = _LinkedFolderKB(tmp_path, monkeypatch)
    linked_kb.keep.unlink()
    linked_kb.drop.unlink()

    with TestClient(_build_app()) as client:
        response = linked_kb.sync(client)

    assert response.status_code == 200
    assert response.json()["file_count"] == 0
    assert response.json()["task_id"] is None
    assert linked_kb.dispatched == []


def test_a_second_sync_while_the_first_is_running_dispatches_nothing(
    linked_kb: _LinkedFolderKB, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    started = threading.Event()
    dispatched: list[dict] = []

    async def _blocking(**kwargs) -> None:
        dispatched.append(kwargs)
        started.set()
        await asyncio.to_thread(gate.wait, 30)

    monkeypatch.setattr(knowledge_router_module, "run_upload_processing_task", _blocking)

    first: dict = {}
    with TestClient(_build_app()) as client:

        def _run_first() -> None:
            first["response"] = linked_kb.sync(client)

        worker = threading.Thread(target=_run_first, daemon=True)
        worker.start()
        try:
            assert started.wait(30), "the first sync never reached its indexing task"
            second = linked_kb.sync(client)
        finally:
            gate.set()
            worker.join(60)

    assert first["response"].status_code == 200
    assert second.status_code == 409
    assert "already" in second.json()["detail"].lower()
    assert len(dispatched) == 1


def test_a_failure_before_the_task_is_queued_gives_the_lock_back(
    linked_kb: _LinkedFolderKB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source must not stay "already syncing" because bookkeeping failed.

    The lock is handed to the background task, so its only release used to sit
    in a `try` that started *after* the task id, the progress entry and the
    queueing call: a failure in any of them left the OS byte lock held, and
    every later sync of that source answered 409 until the process restarted.

    The failed request's traceback is retained on purpose below. It keeps the
    route's frame - and with it the lock handle nothing released - alive, which
    is exactly what an error reporter does in production; without that
    reference the frame is collected and the OS closes the handle for us.
    """
    from fastapi import BackgroundTasks, HTTPException

    real_build = knowledge_router_module._build_unique_task_id

    def _build(*_args, **_kwargs):
        raise RuntimeError("could not build the task id")

    monkeypatch.setattr(knowledge_router_module, "_build_unique_task_id", _build)

    async def _attempt() -> None:
        await knowledge_router_module.sync_folder("kb", linked_kb.folder["id"], BackgroundTasks())

    with pytest.raises(HTTPException) as failed:
        asyncio.run(_attempt())

    monkeypatch.setattr(knowledge_router_module, "_build_unique_task_id", real_build)

    # `failed` keeps the exception and its context, and with them the route's
    # frame and the lock handle in it - so a lock nobody released is still held
    # right here, exactly as it would be behind an error reporter.
    with TestClient(_build_app()) as client:
        retried = linked_kb.sync(client)

    assert failed.value.status_code == 500
    assert "could not build the task id" in str(failed.value.detail)
    # Nothing was queued, so the source is not locked: the retry starts.
    assert retried.status_code == 200, retried.text
    assert retried.json()["task_id"] is not None
    # The failed attempt dispatched nothing; the retry dispatched the batch once.
    assert len(linked_kb.dispatched) == 1


def test_a_failed_prune_is_logged_with_the_folder_name_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning names the folder; a stray `f` used to render as `f'<id>'`."""
    manager = _FolderManager(tmp_path)

    def _broken(*_args, **_kwargs):
        raise RuntimeError("sync state is unreadable")

    manager.prune_folder_sync_state = _broken
    with caplog.at_level("WARNING", logger=knowledge_router_module.__name__):
        knowledge_router_module._record_reported_removals(
            manager, "kb", "folder-1", [str(tmp_path / "gone.md")]
        )

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "from folder 'folder-1' of KB 'kb'" in message
    assert "f'folder-1'" not in message
    assert "sync state is unreadable" in message


def test_a_reported_removal_is_not_reported_again(linked_kb: _LinkedFolderKB) -> None:
    """A removal is reported once, then forgotten instead of nagging forever.

    With a startup sync this is what keeps a launch from re-announcing a
    deletion the user already saw. Forgetting the tracked path is not a
    delete: the imported copy stays in the knowledge base.
    """
    imported = linked_kb.kb_dir / "raw" / "drop.md"
    imported.parent.mkdir(parents=True)
    imported.write_text("imported", encoding="utf-8")
    linked_kb.record_synced(linked_kb.keep, linked_kb.drop)
    linked_kb.drop.unlink()

    with TestClient(_build_app()) as client:
        first = linked_kb.sync(client)
        second = linked_kb.sync(client)

    assert first.status_code == 200
    assert first.json()["removed"] == [str(linked_kb.drop)]
    assert second.status_code == 200
    assert second.json()["removed"] == []
    assert second.json()["removed_files"] == 0
    assert "no longer in the folder" not in second.json()["message"]
    # Pruning the tracked set must never touch what was imported.
    assert imported.read_text(encoding="utf-8") == "imported"
    stored = linked_kb.stored_folder()["synced_files"]
    assert str(linked_kb.keep) in stored
    assert str(linked_kb.drop) not in stored
