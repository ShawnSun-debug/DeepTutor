"""One nonblocking sync per linked folder source, shared across worker processes.

The backend can serve from several uvicorn workers (backend_workers), so an
in-process set of "currently syncing" keys would still let two workers index the
same folder at once. This follows the pattern of
deeptutor/services/rag/pipelines/lightrag/write_lock.py: a nonblocking
byte-range lock on a per-source lock file. The second caller is refused instead
of queued, and the OS drops the lock when the holder exits, so a crashed sync
never leaves a source permanently "already syncing".
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import IO

LOCK_DIR_NAME = ".folder-sync-locks"


def _lock_path(kb_dir: Path, folder_id: str) -> Path:
    key = hashlib.sha256(f"{kb_dir.resolve()}:{folder_id}".encode("utf-8")).hexdigest()
    return kb_dir.parent / LOCK_DIR_NAME / f"{key}.lock"


class FolderSyncLock:
    """A held sync lock for one linked folder; release() is idempotent."""

    def __init__(self, handle: IO[bytes]) -> None:
        self._handle: IO[bytes] | None = handle

    @classmethod
    def acquire(cls, kb_dir: str | Path, folder_id: str) -> "FolderSyncLock | None":
        """Take the source's lock without waiting; None when a run holds it."""
        lock_path = _lock_path(Path(kb_dir), folder_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600), "r+b")
        try:
            _lock(handle)
        except OSError:
            # Locked by another run, in this process or in another one.
            handle.close()
            return None
        return cls(handle)

    def release(self) -> None:
        """Give the lock back; safe to call twice, and from a different frame."""
        handle, self._handle = self._handle, None
        if handle is None or handle.closed:
            return
        try:
            _unlock(handle)
        finally:
            handle.close()


def _lock(handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: IO[bytes]) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
