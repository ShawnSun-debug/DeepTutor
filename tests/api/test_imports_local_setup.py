"""What a knowledge base says when it cannot embed, and how to run it locally.

The one seam is HTTP: the import endpoint, the upload endpoint and knowledge-base
creation, mounted on a bare FastAPI app with the manager on a temporary base
directory. Every assertion is either the JSON a user would see or the bytes on
disk.

What this does NOT prove: that any embedding server is running, and that the
frontend renders these sentences (no UI test here).
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
from deeptutor.services.config.model_catalog import ModelCatalogService  # noqa: E402

LOCAL_ENDPOINT = "http://localhost:11434/api/embed"
PUBLIC_ENDPOINT = "https://api.openai.com/v1/embeddings"
MODEL = "nomic-embed-text"
KB_NAME = "kb"

LOCAL_SERVERS = ("Ollama", "vLLM", "LM Studio", "Lemonade")


class _TaskRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return None


class _CatalogService(ModelCatalogService):
    """Stands in for the Settings model catalog, keeping its read logic."""

    def __init__(self, catalog: dict) -> None:
        self._catalog = json.loads(json.dumps(catalog))

    def load(self) -> dict:
        return json.loads(json.dumps(self._catalog))


def _catalog(*, base_url: str = "", api_key: str = "") -> dict:
    profile = {
        "id": "profile-embedding",
        "name": "Embedding",
        "binding": "openai",
        "base_url": base_url,
        "api_key": api_key,
        "models": [{"id": "model-embedding", "name": MODEL, "model": MODEL}],
    }
    return {
        "version": 1,
        "services": {
            "llm": {"active_profile_id": None, "active_model_id": None, "profiles": []},
            "embedding": {
                "active_profile_id": "profile-embedding",
                "active_model_id": "model-embedding",
                "profiles": [profile],
            },
            "search": {"active_profile_id": None, "profiles": []},
        },
    }


EMPTY_CATALOG = {"version": 1, "services": {"embedding": {"profiles": []}}}


@pytest.fixture(autouse=True)
def _disable_pocketbase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deeptutor.services.pocketbase_client.is_pocketbase_enabled", lambda: False)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> KnowledgeBaseManager:
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


def _use_catalog(monkeypatch: pytest.MonkeyPatch, catalog: dict) -> None:
    """Every reader of Settings: the binding gate and the runtime resolver."""
    service = _CatalogService(catalog)
    monkeypatch.setattr(
        "deeptutor.services.rag.embedding_binding.load_catalog", lambda: service.load()
    )
    monkeypatch.setattr(
        "deeptutor.services.config.provider_runtime.get_model_catalog_service", lambda: service
    )
    monkeypatch.setattr("deeptutor.services.config.get_model_catalog_service", lambda: service)


def seed_kb(manager: KnowledgeBaseManager, *, extra: dict | None = None) -> Path:
    kb_dir = manager.base_dir / KB_NAME
    (kb_dir / "raw").mkdir(parents=True, exist_ok=True)
    entry = {"path": KB_NAME, "rag_provider": "llamaindex", "status": "ready", **(extra or {})}
    manager.config.setdefault("knowledge_bases", {})[KB_NAME] = entry
    manager._save_config()
    (kb_dir / "metadata.json").write_text(
        json.dumps({"rag_provider": "llamaindex", "needs_reindex": False}), encoding="utf-8"
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


# ---------------------------------------------------------------------------
# 1. A blocked batch names the local way out, once per file
# ---------------------------------------------------------------------------


def test_a_blocked_batch_tells_every_file_how_to_run_fully_local(
    client: TestClient,
    manager: KnowledgeBaseManager,
    recorder: _TaskRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_kb(manager, extra={"embedding_selection": {"profile_id": "gone", "model_id": "gone"}})
    monkeypatch.setattr(
        "deeptutor.services.rag.embedding_binding.load_catalog",
        lambda: {"services": {"embedding": {"profiles": []}}},
    )

    native = client.post(f"/api/knowledge-bases/{KB_NAME}/upload", files=[_file("a.md", b"# a\n")])
    assert native.status_code == 409, native.text
    reason = native.json()["detail"]
    for server in LOCAL_SERVERS:
        assert server in reason, reason
    assert "no API key" in reason
    assert "localhost" in reason.lower()
    assert "network access" in reason.lower()

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
    assert not (manager.base_dir / KB_NAME / "raw" / "a.md").exists()


# ---------------------------------------------------------------------------
# 2. Creating a knowledge base that cannot embed yet says what to do
# ---------------------------------------------------------------------------


def test_creating_a_knowledge_base_without_an_embedding_provider_explains_the_local_way(
    client: TestClient,
    manager: KnowledgeBaseManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_catalog(monkeypatch, EMPTY_CATALOG)

    response = client.post(
        "/api/knowledge-bases", data={"name": "fresh", "rag_provider": "llamaindex"}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    setup = body["embedding_setup"]
    assert setup["configured"] is False
    assert setup["requires_network"] is None
    for server in LOCAL_SERVERS:
        assert server in body["message"], body["message"]
    assert "no API key" in body["message"]
    assert "localhost" in body["message"].lower()
    assert body["name"] == "fresh"


# ---------------------------------------------------------------------------
# 3. The verdict is about the resolved endpoint, not a generic failure
# ---------------------------------------------------------------------------


def test_a_remote_embedding_model_without_a_key_is_flagged_as_needing_network(
    client: TestClient,
    manager: KnowledgeBaseManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_catalog(monkeypatch, _catalog(base_url=PUBLIC_ENDPOINT, api_key=""))

    response = client.post(
        "/api/knowledge-bases",
        data={
            "name": "remote",
            "rag_provider": "llamaindex",
            "embedding_model": json.dumps(
                {"profile_id": "profile-embedding", "model_id": "model-embedding"}
            ),
        },
    )

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "API key" in detail
    assert "needs network access" in detail
    assert PUBLIC_ENDPOINT in detail
    for server in LOCAL_SERVERS:
        assert server in detail, detail


# ---------------------------------------------------------------------------
# 4. A keyless localhost model is accepted, and the import lands and queues
# ---------------------------------------------------------------------------


def test_a_local_keyless_model_is_accepted_and_its_import_is_queued(
    client: TestClient,
    manager: KnowledgeBaseManager,
    recorder: _TaskRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_catalog(monkeypatch, _catalog(base_url=LOCAL_ENDPOINT, api_key=""))

    created = client.post(
        "/api/knowledge-bases",
        data={
            "name": "local-kb",
            "rag_provider": "llamaindex",
            "embedding_model": json.dumps(
                {"profile_id": "profile-embedding", "model_id": "model-embedding"}
            ),
        },
    )

    assert created.status_code == 200, created.text
    setup = created.json()["embedding_setup"]
    assert setup["configured"] is True
    assert setup["is_local"] is True
    assert setup["requires_network"] is False
    assert setup["needs_api_key"] is False
    entry = json.loads((manager.base_dir / "kb_config.json").read_text(encoding="utf-8"))[
        "knowledge_bases"
    ]["local-kb"]
    assert entry["embedding_selection"] == {
        "profile_id": "profile-embedding",
        "model_id": "model-embedding",
    }

    imported = _import(client, [_file("notes.md", b"# local\n")], target="local-kb")

    assert imported.status_code == 200, imported.text
    payload = imported.json()
    assert payload["imported"] == 1, payload
    assert payload["task_id"] is not None
    assert len(recorder.calls) == 1
    assert (manager.base_dir / "local-kb" / "raw" / "notes.md").read_bytes() == b"# local\n"
