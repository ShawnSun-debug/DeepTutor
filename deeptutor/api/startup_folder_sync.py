"""One best-effort folder sync per linked source after application start (#9).

A linked folder changes when the user is not looking at DeepTutor, so a manual
click is the wrong trigger for the most common case: a restart. This module
schedules exactly one pass per process, off the startup path, and reuses the
manual sync route so detect -> dispatch -> "queued" stays one implementation.

Why a fire-and-forget task instead of an awaited startup step: the pass parses,
embeds and indexes for as long as the folder is large, and startup must finish
without it. The task reference is kept only so the loop does not garbage-
collect it; nothing on the startup path ever awaits it, and every failure is
caught inside the task, so a broken folder cannot fail application start.

One pending sync per source per start: a source is claimed once per process
(the second pass in the same process finds nothing left to claim), and the
per-source lock the manual route already takes refuses a run that a previous
start is still holding. Two starts therefore cost at most one pass of work.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
import logging
from pathlib import Path

from fastapi import BackgroundTasks, HTTPException

from deeptutor.api.routers import knowledge as knowledge_routes
from deeptutor.knowledge.progress_tracker import ProgressStage, ProgressTracker
from deeptutor.services.config.runtime_settings import get_runtime_settings_service

logger = logging.getLogger(__name__)

SETTING_KEY = "startup_folder_sync_enabled"

# Keeps scheduled passes alive; a task nobody references can be collected mid-run.
_TASKS: set[asyncio.Task] = set()


@dataclass(frozen=True)
class StartupSyncOutcome:
    """What one source's startup sync did."""

    kb_name: str
    folder_id: str
    folder_path: str
    # "synced" queued an indexing run, "unchanged" had nothing to index,
    # "failed" was refused or raised and was recorded against the source.
    status: str
    files: tuple[str, ...] = field(default_factory=tuple)
    removed: tuple[str, ...] = field(default_factory=tuple)
    detail: str = ""


def startup_sync_enabled() -> bool:
    """Read the runtime switch; a payload without the key keeps it enabled."""
    return bool(get_runtime_settings_service().load_system().get(SETTING_KEY, True))


class StartupFolderSync:
    """Schedules and runs the single startup pass of this process."""

    def __init__(self) -> None:
        self._scheduled = False
        self._claimed: set[tuple[str, str]] = set()

    def schedule(self) -> asyncio.Task | None:
        """Queue the pass and return immediately; never raises into startup."""
        if self._scheduled:
            return None
        self._scheduled = True
        if not self._allowed():
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.info("No running event loop; skipping the startup folder sync")
            return None
        task = loop.create_task(self.run())
        _TASKS.add(task)
        task.add_done_callback(_TASKS.discard)
        return task

    async def run(self, *, manager=None) -> list[StartupSyncOutcome]:
        """Sync every linked source once and record what failed. Never raises."""
        # Re-read the switch here as well: driving the pass directly must honour
        # it too, so the disabled case is provable through this seam.
        if not self._allowed():
            return []
        try:
            manager = manager if manager is not None else knowledge_routes.get_kb_manager()
            targets = self._targets(manager)
        except Exception:
            logger.warning("Could not enumerate linked folders for the startup sync", exc_info=True)
            return []

        outcomes: list[StartupSyncOutcome] = []
        for kb_name, folder in targets:
            if not self._claim(kb_name, folder):
                continue
            outcomes.append(await self._sync_source(manager, kb_name, folder))
        if outcomes:
            logger.info(
                "Startup folder sync finished: %s source(s), %s failed",
                len(outcomes),
                sum(outcome.status == "failed" for outcome in outcomes),
            )
        self._record_last_failures(manager, outcomes)
        return outcomes

    @staticmethod
    def _targets(manager) -> list[tuple[str, dict]]:
        """Every (kb, linked folder) pair, read without letting one KB stop the pass."""
        targets: list[tuple[str, dict]] = []
        for kb_name in list(manager.list_knowledge_bases()):
            try:
                folders = manager.get_linked_folders(kb_name) or []
            except Exception:
                logger.warning(
                    "Skipping the linked folders of KB '%s' in the startup sync",
                    kb_name,
                    exc_info=True,
                )
                continue
            targets.extend((kb_name, folder) for folder in folders if folder.get("id"))
        return targets

    @staticmethod
    def _allowed() -> bool:
        """Read the runtime switch; an unreadable setting never starts work."""
        try:
            if startup_sync_enabled():
                return True
        except Exception:
            logger.warning("Could not read the startup folder sync setting", exc_info=True)
            return False
        logger.info("Startup folder sync is disabled")
        return False

    def _claim(self, kb_name: str, folder: dict) -> bool:
        key = (kb_name, str(folder.get("id") or ""))
        if key in self._claimed:
            return False
        self._claimed.add(key)
        return True

    async def _sync_source(self, manager, kb_name: str, folder: dict) -> StartupSyncOutcome:
        """Run the manual sync route in-process and drain what it queued.

        The route owns detection, the writable/connected guards, the per-source
        lock and the "queued" status write; driving it instead of copying it is
        what keeps a startup sync and a user-triggered one the same operation.
        It hands its indexing work to FastAPI background tasks, which normal
        requests drain after the response; here the pass drains them itself so
        the sources are synced one after another (a KB takes one indexing lock
        at a time).
        """
        folder_id = str(folder.get("id") or "")
        folder_path = str(folder.get("path") or "")
        background = BackgroundTasks()
        try:
            response = await knowledge_routes.sync_folder(kb_name, folder_id, background)
        except HTTPException as exc:
            # Missing or emptied folder (409), lock held (409), gone KB (404):
            # the route refuses the sync, and startup records instead of raising.
            return self._failed(manager, kb_name, folder, str(exc.detail))
        except Exception as exc:
            return self._failed(manager, kb_name, folder, str(exc) or exc.__class__.__name__)

        if not background.tasks:
            return StartupSyncOutcome(
                kb_name=kb_name,
                folder_id=folder_id,
                folder_path=folder_path,
                status="unchanged",
                removed=tuple(response.removed),
            )

        try:
            await background()
        except Exception as exc:
            # run_upload_processing_task records its own indexing failures;
            # reaching here means the run itself crashed.
            return self._failed(manager, kb_name, folder, str(exc) or exc.__class__.__name__)

        return StartupSyncOutcome(
            kb_name=kb_name,
            folder_id=folder_id,
            folder_path=folder_path,
            status="synced",
            files=tuple(response.files),
            removed=tuple(response.removed),
        )

    def _failed(self, manager, kb_name: str, folder: dict, reason: str) -> StartupSyncOutcome:
        folder_path = str(folder.get("path") or "")
        detail = f"{folder_path}: {reason}"
        logger.warning("Startup folder sync of '%s' failed: %s", folder_path, reason)
        self._record(
            manager,
            kb_name,
            message=f"Startup sync of '{folder_path}' failed: {reason}",
            error=detail,
            file_name=folder_path,
        )
        return StartupSyncOutcome(
            kb_name=kb_name,
            folder_id=str(folder.get("id") or ""),
            folder_path=folder_path,
            status="failed",
            detail=detail,
        )

    def _record_last_failures(self, manager, outcomes: list[StartupSyncOutcome]) -> None:
        """Leave each affected KB's indexing surface showing its failure.

        The progress slot is per knowledge base, so a later healthy source would
        overwrite an earlier failure's record. Re-stating the failures at the
        end keeps the surface truthful without inventing a second one.
        """
        failed: dict[str, list[StartupSyncOutcome]] = {}
        for outcome in outcomes:
            if outcome.status == "failed":
                failed.setdefault(outcome.kb_name, []).append(outcome)
        for kb_name, items in failed.items():
            paths = ", ".join(item.folder_path for item in items)
            self._record(
                manager,
                kb_name,
                message=f"Startup sync failed for {len(items)} linked source(s): {paths}",
                error=" | ".join(item.detail for item in items),
            )

    @staticmethod
    def _record(manager, kb_name: str, *, message: str, error: str, file_name: str = "") -> None:
        """Record through the same progress tracker other indexing failures use."""
        try:
            tracker = ProgressTracker(kb_name, Path(manager.base_dir))
            tracker.update(ProgressStage.ERROR, message=message, file_name=file_name, error=error)
        except Exception:
            logger.warning(
                "Could not record a startup sync failure for KB '%s'", kb_name, exc_info=True
            )


_sync = StartupFolderSync()


def schedule_startup_folder_sync() -> asyncio.Task | None:
    """Queue the startup pass; returns None when disabled or already queued."""
    return _sync.schedule()


async def run_startup_folder_sync(*, manager=None) -> list[StartupSyncOutcome]:
    """Run the pass to completion (the test seam; production schedules it)."""
    return await _sync.run(manager=manager)


async def cancel_startup_folder_sync() -> None:
    """Stop a pass still running at shutdown so it cannot outlive the app."""
    pending = [task for task in _TASKS if not task.done()]
    for task in pending:
        task.cancel()
    for task in pending:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
