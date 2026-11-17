"""One automatic folder sync per linked source after application start (#9).

The pass must reuse the manual sync route (detect -> dispatch -> mark queued),
never sit on the startup path, never let a failure escape into the startup
sequence, and record a failure where other indexing failures are recorded,
attributed to its source.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
from types import SimpleNamespace

import pytest

try:
    import fastapi  # noqa: F401
except Exception:  # pragma: no cover - optional dependency in lightweight envs
    fastapi = None

from deeptutor.knowledge.manager import KnowledgeBaseManager

pytestmark = pytest.mark.skipif(fastapi is None, reason="fastapi not installed")

STARTUP_MODULE = "deeptutor.api.startup_folder_sync"
ROUTER_MODULE = "deeptutor.api.routers.knowledge"


def _import(name: str):
    return importlib.import_module(name)


class _StartupWorld:
    """A real KB manager with two linked folder sources and a dispatch recorder."""

    def __init__(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch
        self.base = tmp_path / "kbs"
        self.base.mkdir()
        self.manager = KnowledgeBaseManager(base_dir=str(self.base))
        (self.base / "kb").mkdir()
        self.manager.register_knowledge_base("kb")
        self.kb_dir = self.base / "kb"

        self.source_a = tmp_path / "notes-a"
        self.source_a.mkdir()
        self.a1 = self.source_a / "a1.md"
        self.a1.write_text("a1", encoding="utf-8")
        self.source_b = tmp_path / "notes-b"
        self.source_b.mkdir()
        self.b1 = self.source_b / "b1.md"
        self.b1.write_text("b1", encoding="utf-8")
        self.folder_a = self.manager.link_folder("kb", str(self.source_a))
        self.folder_b = self.manager.link_folder("kb", str(self.source_b))

        self.dispatched: list[dict] = []
        self.router = _import(ROUTER_MODULE)
        self.startup = _import(STARTUP_MODULE)
        monkeypatch.setattr(self.router, "get_kb_manager", lambda: self.manager)
        monkeypatch.setattr(self.router, "get_current_user", lambda: None)
        monkeypatch.setattr(self.router, "run_upload_processing_task", self._record)
        monkeypatch.setattr(
            self.router, "_build_unique_task_id", lambda *_args, **_kwargs: "startup-sync"
        )
        monkeypatch.setattr(self.startup, "_sync", self.startup.StartupFolderSync())
        self.use_settings()

    async def _record(self, **kwargs) -> None:
        self.dispatched.append(kwargs)

    def use_settings(self, **system) -> None:
        """Point the pass at one system-settings payload."""
        payload = dict(system)
        self.monkeypatch.setattr(
            self.startup,
            "get_runtime_settings_service",
            lambda: SimpleNamespace(load_system=lambda **_kwargs: dict(payload)),
        )

    def progress(self) -> dict:
        return json.loads((self.kb_dir / ".progress.json").read_text(encoding="utf-8"))

    def kb_config(self) -> dict:
        return json.loads((self.base / "kb_config.json").read_text(encoding="utf-8"))

    def folder_ids(self) -> set[str]:
        return {str(item["folder_id"]) for item in self.dispatched}


@pytest.fixture()
def world(tmp_path, monkeypatch: pytest.MonkeyPatch) -> _StartupWorld:
    return _StartupWorld(tmp_path, monkeypatch)


def _paths(dispatch: dict) -> list[str]:
    return [str(path) for path in dispatch["uploaded_file_paths"]]


def test_one_sync_is_scheduled_per_linked_source(world: _StartupWorld) -> None:
    """Each linked source is dispatched once, through the manual sync route."""
    outcomes = asyncio.run(world.startup.run_startup_folder_sync(manager=world.manager))

    assert [item.status for item in outcomes] == ["synced", "synced"]
    assert world.folder_ids() == {world.folder_a["id"], world.folder_b["id"]}
    assert sorted(_paths(dispatch) for dispatch in world.dispatched) == [
        [str(world.a1)],
        [str(world.b1)],
    ]
    assert {dispatch["folder_root"] for dispatch in world.dispatched} == {
        str(world.source_a),
        str(world.source_b),
    }
    # Queued through the route, so the KB carries the same live status a
    # manual sync shows between acceptance and the indexing run.
    queued = world.kb_config()["knowledge_bases"]["kb"]
    assert queued["status"] == "processing"
    assert "Syncing" in queued["progress"]["message"]


def test_nothing_is_synced_when_the_setting_is_off(world: _StartupWorld) -> None:
    world.use_settings(startup_folder_sync_enabled=False)

    outcomes = asyncio.run(world.startup.run_startup_folder_sync(manager=world.manager))

    assert outcomes == []
    assert world.dispatched == []


def test_a_missing_setting_keeps_the_startup_sync_enabled(world: _StartupWorld) -> None:
    """Default enabled: an install whose system.json predates the key still syncs."""
    world.use_settings()

    outcomes = asyncio.run(world.startup.run_startup_folder_sync(manager=world.manager))

    assert len(outcomes) == 2
    assert world.folder_ids() == {world.folder_a["id"], world.folder_b["id"]}


def test_two_passes_in_one_start_sync_each_source_once(world: _StartupWorld) -> None:
    """A restart that re-enters the pass must not stack a second sync per source."""

    async def _twice() -> list:
        first = await world.startup.run_startup_folder_sync(manager=world.manager)
        second = await world.startup.run_startup_folder_sync(manager=world.manager)
        return [first, second]

    first, second = asyncio.run(_twice())

    assert len(first) == 2
    assert second == []
    assert len(world.dispatched) == 2


def test_scheduling_returns_before_the_pass_runs(world: _StartupWorld) -> None:
    """Startup must not wait: the scheduler hands back a pending task."""
    finished = asyncio.Event()
    gate = asyncio.Event()

    async def _slow() -> list:
        await gate.wait()
        finished.set()
        return []

    world.monkeypatch.setattr(world.startup._sync, "run", _slow)

    async def _scenario() -> None:
        task = world.startup.schedule_startup_folder_sync()
        assert task is not None
        assert not finished.is_set(), "scheduling waited for the sync"
        assert not task.done()
        gate.set()
        await task
        # One pass per process: a second call never starts another.
        assert world.startup.schedule_startup_folder_sync() is None

    asyncio.run(_scenario())
    assert finished.is_set()


def test_disabled_setting_schedules_nothing(world: _StartupWorld) -> None:
    world.use_settings(startup_folder_sync_enabled=False)

    async def _scenario():
        return world.startup.schedule_startup_folder_sync()

    assert asyncio.run(_scenario()) is None
    assert world.dispatched == []


def test_a_raising_sync_is_recorded_against_its_source_and_never_escapes(
    world: _StartupWorld,
) -> None:
    """A sync that raises leaves the start successful and names its source."""

    async def _explode(**kwargs) -> None:
        if kwargs["folder_id"] == world.folder_a["id"]:
            raise RuntimeError("index exploded")
        world.dispatched.append(kwargs)

    world.monkeypatch.setattr(world.router, "run_upload_processing_task", _explode)

    outcomes = asyncio.run(world.startup.run_startup_folder_sync(manager=world.manager))

    assert [item.status for item in outcomes] == ["failed", "synced"]
    assert world.folder_a["path"] in outcomes[0].detail
    assert "index exploded" in outcomes[0].detail
    # The failure is visible where indexing failures live, attributed to the
    # source, and the pass still synced the healthy source.
    progress = world.progress()
    assert progress["stage"] == "error"
    assert str(world.source_a) in progress["error"]
    assert world.kb_config()["knowledge_bases"]["kb"]["status"] == "error"
    assert world.folder_ids() == {world.folder_b["id"]}
    # A crashed run must not leave the source locked for the next start.
    from deeptutor.knowledge.folder_sync_lock import FolderSyncLock

    lock = FolderSyncLock.acquire(world.kb_dir, world.folder_a["id"])
    assert lock is not None
    lock.release()


def test_a_linked_folder_that_is_gone_is_recorded_not_raised(world: _StartupWorld) -> None:
    """The route refuses a missing folder with 409; startup swallows it."""
    shutil.rmtree(world.source_a)

    outcomes = asyncio.run(world.startup.run_startup_folder_sync(manager=world.manager))

    assert [item.status for item in outcomes] == ["failed", "synced"]
    assert str(world.source_a) in outcomes[0].detail
    assert world.folder_ids() == {world.folder_b["id"]}
    assert str(world.source_a) in world.progress()["error"]
