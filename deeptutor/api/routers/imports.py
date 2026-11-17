"""
Bring external material into the user's learning space.

Two families live here, both of them batch-oriented and both reporting what
happened per item rather than failing the whole request:

* ``POST /chat-history`` imports conversations from ChatGPT exports and
  external coding CLIs (Claude Code, Codex) as re-openable sessions.
* ``POST /documents`` imports files into a destination store. It is the
  shared import spine: one per-file result contract, one destination field,
  and one handler per destination. Two destinations exist today — an authored
  page and a collection record — and later ones are additive entries in
  ``_IMPORTERS``.

Reading the user's local ``~/.claude`` / ``~/.codex`` or ChatGPT export happens
in the browser — those files live on the user's machine, not the server. The
browser normalizes each conversation to the small JSON shape below and POSTs
it here; this router only validates and persists. Imported sessions
share the session tables with native chats (so the chat loop can re-open and
continue them) but carry an ``imported_`` id prefix that keeps them in their
own Space category. Re-importing the same folder is idempotent (dedup by id).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, field_validator

from deeptutor.api.utils.upload_reader import (
    UploadEmpty,
    UploadTooLarge,
    read_upload_bytes,
)
from deeptutor.co_writer.html_converter import HtmlConversionError, html_to_markdown
from deeptutor.co_writer.storage import get_co_writer_storage
from deeptutor.services.notebook import notebook_manager
from deeptutor.services.notebook.service import RecordType
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
#
# ``destination`` is a closed, case-insensitive enum, so normalising it is safe.
# Anything that identifies user data — a knowledge-base name, a collection id —
# is case-sensitive and must travel in its own verbatim field instead of being
# smuggled through this one.
DESTINATION_AUTHORED_PAGE = "authored_page"
DESTINATION_COLLECTION = "collection"

# One request is the batch a user picked in a file dialog, so the ceiling only
# exists to keep a malformed client from pinning a worker.
_MAX_IMPORT_FILES = 200

# The authored page destination speaks Markdown, plain text and HTML. Markdown
# and plain text are stored as they are; HTML is converted to Markdown first.
#
# Ordered tuples rather than sets: this same list is rendered verbatim in the
# "unsupported extension" reason, so there is exactly one source of truth for
# what is accepted and no way for the message to drift behind it.
_MARKDOWN_EXTENSIONS = (".md", ".markdown", ".mdown")
_PLAIN_TEXT_EXTENSIONS = (".txt",)
_HTML_EXTENSIONS = (".html", ".htm")
_AUTHORED_PAGE_EXTENSIONS = _MARKDOWN_EXTENSIONS + _PLAIN_TEXT_EXTENSIONS + _HTML_EXTENSIONS

# A collection record stores text, so the Markdown and plain-text families
# belong here. HTML is absent on purpose: it needs the authored-page converter
# to become readable Markdown, and a record holding raw tags would be worse than
# a refusal. Built from the same tuples the authored page uses, so the accepted
# list and the reason rendered below cannot drift apart.
_COLLECTION_EXTENSIONS = _MARKDOWN_EXTENSIONS + _PLAIN_TEXT_EXTENSIONS

# A record summary is one scannable line in a list, not a document.
_MAX_RECORD_SUMMARY_CHARS = 240

# Matches the Co-Writer edit ceiling: a longer page could be imported but
# never edited afterwards, so it is refused rather than silently truncated.
_MAX_IMPORTED_CHARS = 600_000
_MAX_TITLE_CHARS = 120

_UNSUPPORTED_EXTENSION_REASON = (
    f"unsupported extension (supported: {', '.join(_AUTHORED_PAGE_EXTENSIONS)})"
)
_COLLECTION_UNSUPPORTED_EXTENSION_REASON = (
    f"unsupported extension (supported: {', '.join(_COLLECTION_EXTENSIONS)})"
)


class ImportFileResult(BaseModel):
    """What happened to one file. Every destination reports through this."""

    filename: str
    destination: str
    outcome: Literal["imported", "skipped", "failed"]
    reason: str | None = None
    # Authored page only: the page the file became.
    document_id: str | None = None
    # Collection only: the record that was created. Every collection in one
    # batch shares this id, matching how the collection manager stamps records.
    record_id: str | None = None
    # Collection only: where the record actually landed. This is the manager's
    # own "added to" answer, so a collection that did not receive the record is
    # visible here instead of being assumed.
    collection_ids: list[str] | None = None


class DocumentImportResponse(BaseModel):
    """One outcome per file, plus the batch totals a UI shows."""

    destination: str
    imported: int
    skipped: int
    failed: int
    files: list[ImportFileResult]


@dataclass(frozen=True)
class ImportTarget:
    """Where the batch is going, resolved once for every file in it.

    ``destination`` names the store; the remaining fields are that store's own
    identifiers. Keeping them in one object means a new destination adds a field
    here instead of widening every handler's signature.
    """

    destination: str
    parent_id: str | None = None
    # Collection ids are user identifiers and may be case-sensitive: they are
    # carried verbatim here, never through the lowercased destination enum.
    collection_ids: tuple[str, ...] = ()


DocumentImporter = Callable[[UploadFile, ImportTarget], Awaitable[ImportFileResult]]


def _title_from_filename(filename: str) -> str | None:
    """Title from the file stem, or None when the stem is not a usable name.

    An empty or punctuation-only stem (".md", "...") is not a usable name: an
    authored page lets storage derive the title from its first line, and a
    collection record derives one from its content.
    """
    stem = Path(filename).stem.strip()
    if not stem or not any(character.isalnum() for character in stem):
        return None
    return stem[:_MAX_TITLE_CHARS]


def _strip_leading_bom(text: str) -> str:
    """Drop one leading byte-order mark.

    FileTypeRouter.decode_bytes tries plain utf-8 before utf-8-sig, so a UTF-8
    BOM decodes cleanly and stays glued to the text as an invisible first
    character. Only a *leading* mark is a BOM; a U+FEFF anywhere else is
    content and is left alone.
    """
    return text[1:] if text.startswith("\ufeff") else text


def _normalise_newlines(text: str) -> str:
    """Windows and legacy-Mac line endings become LF.

    FileTypeRouter.decode_bytes decodes but does not normalise, so imported
    text would otherwise differ from text typed in the editor.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


async def _import_authored_page(file: UploadFile, target: ImportTarget) -> ImportFileResult:
    """Import one file as a page with no block tree.

    No server-side Markdown-to-blocks conversion happens here: DOCX import
    also writes Markdown only, and the browser promotes the page to blocks the
    first time it is opened. Plain text is stored verbatim — the page holds
    exactly the file's characters, so no escaping or wrapper markup is added.
    """
    filename = (file.filename or "").strip()
    if not filename:
        return ImportFileResult(
            filename="",
            destination=target.destination,
            outcome="failed",
            reason="The upload has no filename.",
        )
    suffix = Path(filename).suffix.lower()
    if suffix not in _AUTHORED_PAGE_EXTENSIONS:
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="skipped",
            reason=_UNSUPPORTED_EXTENSION_REASON,
        )
    try:
        data = await read_upload_bytes(file)
    except UploadTooLarge as exc:
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="failed", reason=str(exc)
        )
    except UploadEmpty:
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="skipped", reason="empty"
        )

    if suffix in _HTML_EXTENSIONS:
        # The parser is CPU-bound on a large page, so keep it off the loop.
        try:
            content = await asyncio.to_thread(html_to_markdown, data, filename)
        except HtmlConversionError as exc:
            return ImportFileResult(
                filename=filename,
                destination=target.destination,
                outcome="failed",
                reason=str(exc),
            )
    else:
        content = FileTypeRouter.decode_bytes(data)
    content = _normalise_newlines(_strip_leading_bom(content))
    if not content.strip():
        # Whitespace-only is empty for a reader, so it would land as a blank page.
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="skipped", reason="empty"
        )
    if len(content) > _MAX_IMPORTED_CHARS:
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="failed",
            reason=f"{filename} exceeds the {_MAX_IMPORTED_CHARS:,} character limit.",
        )

    try:
        document = get_co_writer_storage().create_document(
            title=_title_from_filename(filename),
            content=content,
            parent_id=target.parent_id,
        )
    except ValueError as exc:
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="failed", reason=str(exc)
        )
    except Exception:
        logger.exception("failed to import %s as an authored page", filename)
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="failed",
            reason="The page could not be created.",
        )
    return ImportFileResult(
        filename=filename,
        destination=target.destination,
        outcome="imported",
        document_id=document.id,
    )


_HEADING = re.compile(r"#{1,6}\s+")


def _local_summary(content: str, fallback: str) -> str:
    """Derive a one-line summary without asking a model.

    A record's summary is what keeps a collection list scannable, and an empty
    one is also the exact value that makes the collection route call the
    summarisation agent. Importing a drop of files must cost nothing, so the
    summary is taken from the content itself instead.

    The first heading wins, and otherwise the first non-empty line does.
    """
    summary = ""
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING.match(stripped):
            heading = stripped.lstrip("#").strip()
            if heading:
                summary = heading
                break
            continue
        summary = stripped
        break
    if not summary:
        summary = fallback
    if len(summary) > _MAX_RECORD_SUMMARY_CHARS:
        summary = summary[:_MAX_RECORD_SUMMARY_CHARS].rstrip() + "\u2026"
    return summary


async def _import_collection(file: UploadFile, target: ImportTarget) -> ImportFileResult:
    """Import one file as one record in every targeted collection.

    The record goes through the collection manager's own add-record funnel, so
    it is afterwards indistinguishable from a saved favorite. That funnel
    silently skips a collection it cannot find, which is why the manager's
    answer is checked: an import that landed nowhere is a failure with a reason,
    never an empty success.
    """
    filename = (file.filename or "").strip()
    if not filename:
        return ImportFileResult(
            filename="",
            destination=target.destination,
            outcome="failed",
            reason="The upload has no filename.",
        )
    if Path(filename).suffix.lower() not in _COLLECTION_EXTENSIONS:
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="skipped",
            reason=_COLLECTION_UNSUPPORTED_EXTENSION_REASON,
        )
    if not target.collection_ids:
        # Mirrors the authored page's missing-parent rule: the target belongs to
        # the whole batch, so a missing one fails each file rather than the
        # request, and nothing is ever reported as imported.
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="failed",
            reason="No target collection was given.",
        )
    try:
        data = await read_upload_bytes(file)
    except UploadTooLarge as exc:
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="failed", reason=str(exc)
        )
    except UploadEmpty:
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="skipped", reason="empty"
        )

    content = _normalise_newlines(FileTypeRouter.decode_bytes(data))
    if not content.strip():
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="skipped", reason="empty"
        )
    if len(content) > _MAX_IMPORTED_CHARS:
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="failed",
            reason=f"{filename} exceeds the {_MAX_IMPORTED_CHARS:,} character limit.",
        )

    title = _title_from_filename(filename) or _local_summary(content, filename)
    try:
        added = notebook_manager.add_record(
            notebook_ids=list(target.collection_ids),
            record_type=RecordType.IMPORT,
            title=title,
            user_query="",
            output=content,
            summary=_local_summary(content, title),
            metadata={"source_filename": filename},
        )
    except ValueError as exc:
        return ImportFileResult(
            filename=filename, destination=target.destination, outcome="failed", reason=str(exc)
        )
    except Exception:
        logger.exception("failed to import %s as a collection record", filename)
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="failed",
            reason="The record could not be saved.",
        )

    landed = [str(notebook_id) for notebook_id in added.get("added_to_notebooks") or []]
    if not landed:
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="failed",
            reason=(
                f"No such collection: {', '.join(target.collection_ids)}. The record was not added."
            ),
        )
    missing = [notebook_id for notebook_id in target.collection_ids if notebook_id not in landed]
    return ImportFileResult(
        filename=filename,
        destination=target.destination,
        outcome="imported",
        # A partial landing is stated rather than passed off as complete: the
        # outcome names where the record went and what it never reached.
        reason=f"Not added to: {', '.join(missing)} (no such collection)." if missing else None,
        record_id=str(added["record"]["id"]),
        collection_ids=landed,
    )


_IMPORTERS: dict[str, DocumentImporter] = {
    DESTINATION_AUTHORED_PAGE: _import_authored_page,
    DESTINATION_COLLECTION: _import_collection,
}


def _unique_collection_ids(raw: list[str] | None) -> tuple[str, ...]:
    """Collection ids stay verbatim: trimmed, deduplicated, never case-folded.

    They identify user data, so they must not travel through the lowercased
    destination enum. Duplicates are dropped because adding one record twice to
    the same collection would make the batch ambiguous to read back.
    """
    unique: dict[str, None] = {}
    for value in raw or []:
        candidate = (value or "").strip()
        if candidate:
            unique.setdefault(candidate, None)
    return tuple(unique)


@router.post("/documents", response_model=DocumentImportResponse)
async def import_documents(
    files: list[UploadFile] | None = File(default=None),
    destination: str = Form(default=DESTINATION_AUTHORED_PAGE),
    parent_id: str | None = Form(default=None),
    # A repeatable multipart field: one request may target several
    # collections, and FastAPI accepts a plain list for repeated form keys but
    # not the `list[str] | None` union, which it reads as a JSON body.
    collection_ids: list[str] = Form(default=[]),
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

    target = ImportTarget(
        destination=normalized,
        parent_id=parent_id,
        collection_ids=_unique_collection_ids(collection_ids),
    )
    results = [await importer(upload, target) for upload in files]
    counts = {"imported": 0, "skipped": 0, "failed": 0}
    for result in results:
        counts[result.outcome] += 1
    return DocumentImportResponse(destination=normalized, files=results, **counts)
