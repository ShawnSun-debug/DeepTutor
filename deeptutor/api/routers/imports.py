"""
Bring external material into the user's learning space.

Two families live here, both of them batch-oriented and both reporting what
happened per item rather than failing the whole request:

* ``POST /chat-history`` imports conversations from ChatGPT exports and
  external coding CLIs (Claude Code, Codex) as re-openable sessions.
* ``POST /documents`` imports files into a destination store. It is the
  shared import spine: one per-file result contract, one destination field,
  and one handler per destination. Today the only destination is an
  authored page; later destinations are additive entries in ``_IMPORTERS``.

Reading the user's local ``~/.claude`` / ``~/.codex`` or ChatGPT export happens
in the browser — those files live on the user's machine, not the server. The
browser normalizes each conversation to the small JSON shape below and POSTs
it here; this router only validates and persists. Imported sessions
share the session tables with native chats (so the chat loop can re-open and
continue them) but carry an ``imported_`` id prefix that keeps them in their
own Space category. Re-importing the same folder is idempotent (dedup by id).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
import logging
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, field_validator

from deeptutor.api.utils.upload_reader import (
    UploadEmpty,
    UploadTooLarge,
    read_upload_bytes,
)
from deeptutor.co_writer.storage import get_co_writer_storage
from deeptutor.services.session import (
    get_sqlite_session_store,
    make_imported_session_id,
)
from deeptutor.utils.file_types import FileTypeRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# Browser adapters only emit these; reject anything else so a malformed payload
# can never seed an unsupported provider category.
_ALLOWED_SOURCES = {"chatgpt", "claude_code", "codex"}

# Defensive ceilings — a single import request should never grow unbounded.
_MAX_SESSIONS_PER_REQUEST = 1000
_MAX_MESSAGES_PER_SESSION = 10000


class ImportedMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = ""
    created_at: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImportedSession(BaseModel):
    external_id: str = Field(..., min_length=1, max_length=256)
    title: str = ""
    source_cwd: str = ""
    created_at: float
    updated_at: float
    messages: list[ImportedMessage] = Field(default_factory=list)


class ChatHistoryImportRequest(BaseModel):
    source: str = Field(..., min_length=1)
    # All sessions in one request belong to one agent (a named, scoped slice of
    # a source folder). Empty when the client hasn't adopted the agent model yet
    # — the backend stays backwards-compatible by simply omitting attribution.
    agent_id: str = Field(default="", max_length=256)
    agent_name: str = Field(default="", max_length=256)
    sessions: list[ImportedSession] = Field(default_factory=list)

    @field_validator("source")
    @classmethod
    def _normalize_source(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized not in _ALLOWED_SOURCES:
            raise ValueError(f"Unsupported import source: {value!r}")
        return normalized


@router.post("/chat-history")
async def import_chat_history(payload: ChatHistoryImportRequest) -> dict[str, Any]:
    if not payload.sessions:
        raise HTTPException(status_code=400, detail="No sessions to import")
    if len(payload.sessions) > _MAX_SESSIONS_PER_REQUEST:
        raise HTTPException(
            status_code=413,
            detail=f"Too many sessions in one request (max {_MAX_SESSIONS_PER_REQUEST})",
        )

    store = get_sqlite_session_store()
    imported = 0
    skipped = 0
    results: list[dict[str, Any]] = []

    for incoming in payload.sessions:
        # Drop content-less rows (e.g. tool-only turns the adapter could not
        # reduce to text) so the transcript stays a clean human conversation.
        messages = [m for m in incoming.messages if (m.content or "").strip()]
        if not messages:
            skipped += 1
            results.append(
                {"external_id": incoming.external_id, "imported": False, "reason": "empty"}
            )
            continue
        if len(messages) > _MAX_MESSAGES_PER_SESSION:
            messages = messages[:_MAX_MESSAGES_PER_SESSION]

        session_id = make_imported_session_id(payload.source, incoming.external_id)
        import_meta: dict[str, Any] = {
            "source": payload.source,
            "source_cwd": incoming.source_cwd,
            "external_id": incoming.external_id,
        }
        if payload.agent_id:
            import_meta["agent_id"] = payload.agent_id
        if payload.agent_name:
            import_meta["agent_name"] = payload.agent_name
        preferences = {"import": import_meta}
        try:
            result = await store.import_session(
                session_id,
                incoming.title,
                incoming.created_at,
                incoming.updated_at,
                preferences,
                [m.model_dump() for m in messages],
            )
        except Exception:
            logger.exception("failed to import session %s", incoming.external_id)
            skipped += 1
            results.append(
                {"external_id": incoming.external_id, "imported": False, "reason": "error"}
            )
            continue

        if result.get("imported"):
            imported += 1
        else:
            skipped += 1
        results.append({"external_id": incoming.external_id, **result})

    return {"imported": imported, "skipped": skipped, "sessions": results}


@router.get("/chat-history")
async def list_imported_chat_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    store = get_sqlite_session_store()
    sessions = await store.list_imported_sessions(limit=limit, offset=offset)
    return {"sessions": sessions}


# ─────────────────────────────────────────────────────────────────────────────
# Document import — the shared import spine
# ─────────────────────────────────────────────────────────────────────────────

# A destination is a store, not a format; the format comes from the file
# extension. Later destinations are additive entries here, not new shapes.
DESTINATION_AUTHORED_PAGE = "authored_page"

# One request is the batch a user picked in a file dialog, so the ceiling only
# exists to keep a malformed client from pinning a worker.
_MAX_IMPORT_FILES = 200

# The authored page destination speaks Markdown only. Plain text and HTML
# belong to their own destinations and are deliberately absent here.
_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})

# Matches the Co-Writer edit ceiling: a longer page could be imported but
# never edited afterwards, so it is refused rather than silently truncated.
_MAX_IMPORTED_CHARS = 600_000
_MAX_TITLE_CHARS = 120

_UNSUPPORTED_EXTENSION_REASON = "unsupported extension (supported: .md, .markdown)"


class ImportFileResult(BaseModel):
    """What happened to one file. Every destination reports through this."""

    filename: str
    destination: str
    outcome: Literal["imported", "skipped", "failed"]
    reason: str | None = None
    document_id: str | None = None


class DocumentImportResponse(BaseModel):
    """One outcome per file, plus the batch totals a UI shows."""

    destination: str
    imported: int
    skipped: int
    failed: int
    files: list[ImportFileResult]


DocumentImporter = Callable[..., Awaitable[ImportFileResult]]


def _authored_page_title(filename: str) -> str | None:
    """Title from the file stem, or None to let storage derive one.

    An empty or punctuation-only stem (".md", "...") is not a usable name, so
    the store falls back to the first line of the content instead of creating
    a page called ".md".
    """
    stem = Path(filename).stem.strip()
    if not stem or not any(character.isalnum() for character in stem):
        return None
    return stem[:_MAX_TITLE_CHARS]


def _normalise_newlines(text: str) -> str:
    """Windows and legacy-Mac line endings become LF.

    FileTypeRouter.decode_bytes decodes but does not normalise, so imported
    text would otherwise differ from text typed in the editor.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


async def _import_authored_page(
    file: UploadFile, *, destination: str, parent_id: str | None
) -> ImportFileResult:
    """Import one file as a Markdown page with no block tree.

    No server-side Markdown-to-blocks conversion happens here: DOCX import
    also writes Markdown only, and the browser promotes the page to blocks the
    first time it is opened.
    """
    filename = (file.filename or "").strip()
    if not filename:
        return ImportFileResult(
            filename="",
            destination=destination,
            outcome="failed",
            reason="The upload has no filename.",
        )
    if Path(filename).suffix.lower() not in _MARKDOWN_EXTENSIONS:
        return ImportFileResult(
            filename=filename,
            destination=destination,
            outcome="skipped",
            reason=_UNSUPPORTED_EXTENSION_REASON,
        )
    try:
        data = await read_upload_bytes(file)
    except UploadTooLarge as exc:
        return ImportFileResult(
            filename=filename, destination=destination, outcome="failed", reason=str(exc)
        )
    except UploadEmpty:
        return ImportFileResult(
            filename=filename, destination=destination, outcome="skipped", reason="empty"
        )

    content = _normalise_newlines(FileTypeRouter.decode_bytes(data))
    if not content.strip():
        # Whitespace-only is empty for a reader, so it would land as a blank page.
        return ImportFileResult(
            filename=filename, destination=destination, outcome="skipped", reason="empty"
        )
    if len(content) > _MAX_IMPORTED_CHARS:
        return ImportFileResult(
            filename=filename,
            destination=destination,
            outcome="failed",
            reason=f"{filename} exceeds the {_MAX_IMPORTED_CHARS:,} character limit.",
        )

    try:
        document = get_co_writer_storage().create_document(
            title=_authored_page_title(filename), content=content, parent_id=parent_id
        )
    except ValueError as exc:
        return ImportFileResult(
            filename=filename, destination=destination, outcome="failed", reason=str(exc)
        )
    except Exception:
        logger.exception("failed to import %s as an authored page", filename)
        return ImportFileResult(
            filename=filename,
            destination=destination,
            outcome="failed",
            reason="The page could not be created.",
        )
    return ImportFileResult(
        filename=filename,
        destination=destination,
        outcome="imported",
        document_id=document.id,
    )


_IMPORTERS: dict[str, DocumentImporter] = {
    DESTINATION_AUTHORED_PAGE: _import_authored_page,
}


@router.post("/documents", response_model=DocumentImportResponse)
async def import_documents(
    files: list[UploadFile] | None = File(default=None),
    destination: str = Form(default=DESTINATION_AUTHORED_PAGE),
    parent_id: str | None = Form(default=None),
) -> DocumentImportResponse:
    """Import a batch of files and report one outcome per file.

    A bad file never fails the batch: it gets its own skipped or failed entry
    and the rest still import. Only a malformed request — no files, too many
    files, an unknown destination — is an HTTP error, because there is nothing
    per-file to report for it.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files to import")
    if len(files) > _MAX_IMPORT_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request (max {_MAX_IMPORT_FILES})",
        )
    normalized = (destination or "").strip().lower()
    importer = _IMPORTERS.get(normalized)
    if importer is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown import destination {destination!r}; "
                f"supported: {', '.join(sorted(_IMPORTERS))}"
            ),
        )

    results = [
        await importer(upload, destination=normalized, parent_id=parent_id) for upload in files
    ]
    counts = {"imported": 0, "skipped": 0, "failed": 0}
    for result in results:
        counts[result.outcome] += 1
    return DocumentImportResponse(destination=normalized, files=results, **counts)
