"""
Capped upload reader
====================

One streaming reader for every import route that accepts a file upload.

Uploads are read in bounded chunks so an oversized body is rejected as soon as
the ceiling is crossed instead of being buffered whole first. The reader raises
format-neutral exceptions rather than HTTP errors, because a batch import route
reports a bad file per file (see deeptutor/api/routers/imports.py) while a
single-file route maps the same failure onto a 400.
"""

from __future__ import annotations

from fastapi import UploadFile

# Generous ceilings — they exist to stop runaway payloads (OOM / surprise
# LLM bills), not to constrain normal documents.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024


class UploadError(ValueError):
    """Base class for a rejected upload.

    Messages are user-facing: single-file routes surface str(exc) as the HTTP
    detail, batch routes surface it as the per-file reason.
    """


class UploadTooLarge(UploadError):
    def __init__(self, filename: str, limit_bytes: int = MAX_UPLOAD_BYTES) -> None:
        self.filename = filename
        self.limit_bytes = limit_bytes
        super().__init__(f"{filename} exceeds the {limit_bytes // (1024 * 1024)} MB limit.")


class UploadEmpty(UploadError):
    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(f"{filename} is empty.")


async def read_upload_bytes(
    file: UploadFile,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    chunk_bytes: int = UPLOAD_CHUNK_BYTES,
) -> bytes:
    """Read the upload into memory, raising once it passes max_bytes.

    The declared UploadFile.size is not trusted: it is client-reported and
    absent on chunked uploads, so the ceiling is enforced on the bytes actually
    read.
    """
    filename = (file.filename or "").strip()
    chunks: list[bytes] = []
    written = 0
    while chunk := await file.read(chunk_bytes):
        written += len(chunk)
        if written > max_bytes:
            raise UploadTooLarge(filename, max_bytes)
        chunks.append(chunk)
    if written == 0:
        raise UploadEmpty(filename)
    return b"".join(chunks)
