"""Ticket #7 — an Obsidian vault can become a retrievable knowledge base.

The seam is the HTTP route ``POST /api/knowledge-bases/connect-obsidian``:
everything asserted here is observable from the response plus the state the
knowledge manager leaves behind in a ``tmp_path`` store. Real embedding and
FAISS work needs a localhost provider, so the first sync is asserted as
*queued* work (the folder scan handed to the ingestion task), never executed.
"""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from deeptutor.api.routers import knowledge as router
from deeptutor.knowledge.kb_types import OBSIDIAN_KB_TYPE, supports_rag_retrieval
from deeptutor.knowledge.manager import KnowledgeBaseManager
from tests.services.workspace.test_data_scope import account as account  # noqa: F401

CONNECT = "/api/knowledge-bases/connect-obsidian"


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """Point the route at a throwaway knowledge store."""
    store = tmp_path / "knowledge_bases"
    kb_manager = KnowledgeBaseManager(base_dir=str(store))
    monkeypatch.setattr(router, "kb_manager", kb_manager)
    return kb_manager


@pytest.fixture
def queued(monkeypatch):
    """Capture the sync job the route dispatches instead of running it."""
    calls: list[dict] = []

    async def record(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(router, "run_upload_processing_task", record)
    return calls


@pytest.fixture
def vault(tmp_path):
    """A two-folder vault whose subfolders share a note file name."""
    root = tmp_path / "vault"
    (root / "daily").mkdir(parents=True)
    (root / "projects").mkdir()
    (root / "daily" / "note.md").write_text("# daily note\nentry", encoding="utf-8")
    (root / "projects" / "note.md").write_text("# project note\nentry", encoding="utf-8")
    (root / "index.md").write_text("# index\nhome", encoding="utf-8")
    return root


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router.router, prefix="/api")
    return TestClient(app)


def _persisted_entry(manager: KnowledgeBaseManager, kb_name: str) -> dict:
    path = manager.base_dir / "kb_config.json"
    return json.loads(path.read_text(encoding="utf-8"))["knowledge_bases"][kb_name]


def test_indexed_vault_creates_a_retrievable_kb_with_a_queued_first_sync(
    account, manager, queued, vault
):
    with _client() as client:
        response = client.post(
            CONNECT,
            json={"name": "notes", "vault_path": str(vault), "mode": "indexed"},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "indexed"

    kb_name = body["name"]
    entry = _persisted_entry(manager, kb_name)
    assert entry.get("type") != OBSIDIAN_KB_TYPE
    assert supports_rag_retrieval(entry) is True
    assert entry["rag_provider"] == "llamaindex"

    folders = manager.get_linked_folders(kb_name)
    assert [folder["path"] for folder in folders] == [str(vault)]

    assert len(queued) == 1
    job = queued[0]
    assert job["kb_name"] == kb_name
    assert job["rag_provider"] == "llamaindex"
    assert job["folder_root"] == str(vault)
    assert job["folder_id"] == folders[0]["id"]
    assert body["task_id"] == job["task_id"]
    # Same file name in two subfolders survives as two distinct sources.
    assert sorted(job["uploaded_file_paths"]) == sorted(
        [
            str(vault / "daily" / "note.md"),
            str(vault / "index.md"),
            str(vault / "projects" / "note.md"),
        ]
    )


def test_live_mode_is_unchanged_and_still_unretrievable(account, manager, queued, vault):
    with _client() as client:
        response = client.post(CONNECT, json={"name": "notes", "vault_path": str(vault)})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "live"
    assert body["name"] == "notes"
    assert body["vault_path"] == str(vault)

    entry = _persisted_entry(manager, "notes")
    assert entry["type"] == OBSIDIAN_KB_TYPE
    assert supports_rag_retrieval(entry) is False
    assert manager.get_linked_folders("notes") == []
    assert queued == []


def test_both_modes_coexist_for_one_vault(account, manager, queued, vault):
    with _client() as client:
        live = client.post(CONNECT, json={"name": "notes", "vault_path": str(vault)})
        indexed = client.post(
            CONNECT,
            json={"name": "notes", "vault_path": str(vault), "mode": "indexed"},
        )

    assert live.status_code == 200 and indexed.status_code == 200
    assert live.json()["name"] != indexed.json()["name"]
    assert set(manager.list_knowledge_bases()) == {live.json()["name"], indexed.json()["name"]}
    assert _persisted_entry(manager, live.json()["name"])["vault_path"] == str(vault)
    assert manager.get_linked_folders(live.json()["name"]) == []
    assert [f["path"] for f in manager.get_linked_folders(indexed.json()["name"])] == [str(vault)]
    assert len(queued) == 1


def test_reconnecting_an_indexed_vault_reuses_it(account, manager, queued, vault):
    payload = {"name": "notes", "vault_path": str(vault), "mode": "indexed"}
    with _client() as client:
        first = client.post(CONNECT, json=payload)
        again = client.post(CONNECT, json=payload)
        renamed = client.post(
            CONNECT, json={"vault_path": str(vault), "mode": "indexed", "name": "other"}
        )

    assert first.status_code == 200, first.text
    kb_name = first.json()["name"]
    assert first.json()["already_connected"] is False
    assert first.json()["documents_queued"] == 3

    for repeat in (again, renamed):
        assert repeat.status_code == 200, repeat.text
        assert repeat.json()["already_connected"] is True
        assert repeat.json()["name"] == kb_name
        assert repeat.json()["task_id"] is None
        assert repeat.json()["documents_queued"] == 0

    # One knowledge base, one folder source, one sync job — no duplicates.
    assert manager.list_knowledge_bases() == [kb_name]
    assert len(manager.get_linked_folders(kb_name)) == 1
    assert len(queued) == 1


@pytest.mark.parametrize("kind", ["missing", "not-a-directory"])
def test_indexed_mode_rejects_a_bad_vault_path_before_registering(
    account, manager, queued, vault, tmp_path, kind
):
    bad_path = tmp_path / ("absent-vault" if kind == "missing" else "vault-file")
    if kind == "not-a-directory":
        bad_path.write_text("# not a vault", encoding="utf-8")

    with _client() as client:
        response = client.post(
            CONNECT, json={"name": "notes", "vault_path": str(bad_path), "mode": "indexed"}
        )

    assert response.status_code == 400, response.text
    assert manager.list_knowledge_bases() == []
    assert queued == []


def test_unknown_mode_is_rejected(account, manager, queued, vault):
    with _client() as client:
        response = client.post(
            CONNECT, json={"name": "notes", "vault_path": str(vault), "mode": "mirror"}
        )

    assert response.status_code == 400, response.text
    assert manager.list_knowledge_bases() == []
    assert queued == []
