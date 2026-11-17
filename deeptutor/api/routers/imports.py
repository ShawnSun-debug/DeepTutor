"""
Bring external material into the user's learning space.

Two families live here, both of them batch-oriented and both reporting what
happened per item rather than failing the whole request:

* ``POST /chat-history`` imports conversations from ChatGPT exports and
  external coding CLIs (Claude Code, Codex) as re-openable sessions.
* ``POST /documents`` imports files into a destination store. It is the
  shared import spine: one per-file result contract, one destination field,
  and one handler per destination. An authored page is the default
  destination; a knowledge base hands its files straight to the existing
  upload and indexing pipeline. Later destinations are additive entries in
  ``_IMPORTERS`` / ``_BATCH_IMPORTERS``.

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
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field, field_validator

from deeptutor.api.utils.upload_reader import (
    UploadEmpty,
    UploadTooLarge,
    read_upload_bytes,
)
from deeptutor.services.session import (
    get_sqlite_session_store,
    make_imported_session_id,
)
from deeptutor.utils.file_types import FileTypeRouter
from deeptutor.utils.html_markdown import HtmlConversionError, html_to_markdown

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
DESTINATION_KNOWLEDGE_BASE = "knowledge_base"

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

# Matches the Co-Writer edit ceiling: a longer page could be imported but
# never edited afterwards, so it is refused rather than silently truncated.
_MAX_IMPORTED_CHARS = 600_000
_MAX_TITLE_CHARS = 120

_UNSUPPORTED_EXTENSION_REASON = (
    f"unsupported extension (supported: {', '.join(_AUTHORED_PAGE_EXTENSIONS)})"
)


class ImportFileResult(BaseModel):
    """What happened to one file. Every destination reports through this."""

    filename: str
    destination: str
    outcome: Literal["imported", "skipped", "failed"]
    reason: str | None = None
    # Authored pages: the id of the page that was created.
    document_id: str | None = None
    # Destinations with a layout of their own (a knowledge base's ``raw/``)
    # report where the file landed. One upload can become several files,
    # because a `.zip` is expanded into its members, so this is a list.
    stored_paths: list[str] | None = None


class DocumentImportResponse(BaseModel):
    """One outcome per file, plus the batch totals a UI shows."""

    destination: str
    imported: int
    skipped: int
    failed: int
    files: list[ImportFileResult]
    # The background job a destination queued for this batch, when it queues
    # one; a UI can follow its progress instead of inventing its own spinner.
    task_id: str | None = None


DocumentImporter = Callable[..., Awaitable[ImportFileResult]]
DocumentBatchImporter = Callable[..., Awaitable["_BatchImportOutcome"]]


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


def _document_store() -> Any:
    """Return the authored-page store, resolved on demand.

    The store belongs to the co-writer feature while this router belongs to the server
    feature, so importing it at module scope would make the whole API server depend on a
    feature that may not be installed (see tests/modules/test_isolated_features.py).
    Resolving through this one helper keeps a single seam for tests to patch.
    """

    from deeptutor.co_writer.storage import get_co_writer_storage

    return get_co_writer_storage()


async def _import_authored_page(
    file: UploadFile, *, destination: str, parent_id: str | None
) -> ImportFileResult:
    """Import one Markdown or plain-text file as a page with no block tree.

    No server-side Markdown-to-blocks conversion happens here: DOCX import
    also writes Markdown only, and the browser promotes the page to blocks the
    first time it is opened. Plain text is stored verbatim — the page holds
    exactly the file's characters, so no escaping or wrapper markup is added.
    """
    filename = (file.filename or "").strip()
    if not filename:
        return ImportFileResult(
            filename="",
            destination=destination,
            outcome="failed",
            reason="The upload has no filename.",
        )
    suffix = Path(filename).suffix.lower()
    if suffix not in _AUTHORED_PAGE_EXTENSIONS:
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

    if suffix in _HTML_EXTENSIONS:
        # The parser is CPU-bound on a large page, so keep it off the loop.
        try:
            content = await asyncio.to_thread(html_to_markdown, data, filename)
        except HtmlConversionError as exc:
            return ImportFileResult(
                filename=filename,
                destination=destination,
                outcome="failed",
                reason=str(exc),
            )
    else:
        content = FileTypeRouter.decode_bytes(data)
    content = _normalise_newlines(_strip_leading_bom(content))
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
        store = _document_store()
    except ImportError:
        return ImportFileResult(
            filename=filename,
            destination=destination,
            outcome="failed",
            reason="The notes feature is not available in this installation.",
        )

    try:
        document = store.create_document(
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


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge-base destination
# ─────────────────────────────────────────────────────────────────────────────

# The upload route's own task type, so an imported file turns up in exactly the
# progress surfaces an uploaded one does.
_KB_IMPORT_TASK_TYPE = "kb_upload"

# Verbatim from ``upload_files``: a paraphrase would tell the user something
# the knowledge base never said. The equivalence test locks this by comparing it
# with the upload route's own 409 for the same knowledge base.
_EMBEDDING_BINDING_UNAVAILABLE = (
    "The bound embedding model is unavailable or changed. Check the model settings or "
    "select a model to re-index this knowledge base."
)


@dataclass
class _BatchImportOutcome:
    """A batch-shaped destination's answer: per-file results and its one job."""

    results: list[ImportFileResult]
    task_id: str | None = None


def _kb_result(upload: UploadFile, outcome: Literal["failed"], reason: str) -> ImportFileResult:
    return ImportFileResult(
        filename=(upload.filename or "").strip(),
        destination=DESTINATION_KNOWLEDGE_BASE,
        outcome=outcome,
        reason=reason,
    )


async def _import_knowledge_base(
    files: list[UploadFile],
    *,
    target: str | None,
    background_tasks: BackgroundTasks,
) -> _BatchImportOutcome:
    """Submit a batch of files to an existing knowledge base.

    The knowledge base owns ingestion: this function resolves the target, runs
    the same batch-level gates ``upload_files`` runs, stages the accepted files
    with the same save helper, and queues the same processing task. Nothing
    here parses, chunks or embeds anything, and nothing is stored anywhere
    except the knowledge base's own ``raw/`` directory, so an imported file is
    indistinguishable afterwards from an uploaded one.

    Batch-level rejections (a read-only pointer, an unusable embedding binding,
    an engine that is not ready) are reported once per file carrying the
    knowledge base's own reason: the batch cannot succeed, but there is still a
    per-file answer to give and nothing is written or queued.

    The import ceiling does not apply to a knowledge base: its own engine's
    format and size rules are the ones that decide, exactly as for an upload.
    """
    # A knowledge-base name is user data and case-sensitive, while
    # ``destination`` is a lowercased enum — so the name arrives in ``target`
    # and is used verbatim. Only outer whitespace, which is a paste artifact and
    # never part of a name, is trimmed.
    kb_name = (target or "").strip()
    if not kb_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "The knowledge_base destination requires 'target': the name of the "
                "knowledge base to import into."
            ),
        )

    # Resolve the pipeline's helpers at call time instead of copying them: a fix
    # in the upload route must reach this one, and importing the knowledge
    # router eagerly would drag the whole RAG stack into this router's import.
    from deeptutor.api.routers import knowledge as kb_routes
    from deeptutor.services.workspace.knowledge import workspace_id_for_kb_base_dir
    from deeptutor.services.workspace.models import WorkspaceError

    try:
        manager, kb_name, kb_base_dir = kb_routes._writable_kb(kb_name)
    except WorkspaceError as exc:
        # A workspace problem is not a missing knowledge base; keep the reason.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError:
        raise HTTPException(
            status_code=404, detail=f"Knowledge base '{kb_name}' not found"
        ) from None

    # A name that resolves to no knowledge base is a request to the wrong place,
    # not one file's problem: there is no knowledge base whose reason it is.
    kb_entry = kb_routes._load_kb_entry_or_404(manager, kb_name)

    try:
        kb_routes._assert_kb_writable_or_409(kb_name, kb_entry)
        kb_provider = kb_routes._validate_registered_provider(
            kb_entry.get("rag_provider") or kb_routes.DEFAULT_PROVIDER
        )

        from contextlib import nullcontext

        from deeptutor.services.embedding.config import embedding_config_scope
        from deeptutor.services.rag.embedding_binding import binding_status

        binding_state, bound_config = binding_status(kb_entry)
        if binding_state in {"missing", "changed", "unconfigured"}:
            raise HTTPException(status_code=409, detail=_EMBEDDING_BINDING_UNAVAILABLE)
        with embedding_config_scope(bound_config) if bound_config else nullcontext():
            kb_routes._assert_provider_ready(kb_provider)
            accepted_snapshot = kb_routes._freeze_append_indexing(
                kb_name, kb_base_dir, kb_provider, None
            )
    except HTTPException as exc:
        # The knowledge base refuses this batch for its own reason. Every file
        # carries that reason, and nothing is written or queued.
        reason = str(exc.detail)
        return _BatchImportOutcome(results=[_kb_result(f, "failed", reason) for f in files])

    raw_dir = manager.get_knowledge_base_path(kb_name) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted_extensions = FileTypeRouter.accepted_upload_extensions()
    # `.zip` is a container: its members are validated against the set when it
    # is expanded, and the archive itself is never indexed.
    upload_extensions = accepted_extensions | {".zip"} if accepted_extensions else set()

    results: list[ImportFileResult] = []
    accepted_paths: list[str] = []
    taken: dict[str, UploadFile] = {}

    for upload in files:
        filename = (upload.filename or "").strip()
        try:
            # The knowledge base's engine rule, asked about this file alone, so
            # one rejected format does not fail the rest of the batch.
            kb_routes._enforce_provider_formats(kb_provider, [upload])
            validated = kb_routes._validate_upload_batch(
                [upload], allowed_extensions=upload_extensions
            )
            first = taken.get(validated[0]["path"])
            if first is not None:
                # Let the upload surface word its own duplicate report rather
                # than paraphrasing it: two files cannot share one name in raw/.
                kb_routes._validate_upload_batch(
                    [first, upload], allowed_extensions=upload_extensions
                )
        except HTTPException as exc:
            results.append(_kb_result(upload, "failed", str(exc.detail)))
            continue

        try:
            stored, stored_paths = await kb_routes._save_uploaded_files_off_loop(
                [upload], raw_dir, allowed_extensions=upload_extensions
            )
        except HTTPException as exc:
            results.append(_kb_result(upload, "failed", str(exc.detail)))
            continue

        taken[validated[0]["path"]] = upload
        accepted_paths.extend(stored_paths)
        results.append(
            ImportFileResult(
                filename=filename,
                destination=DESTINATION_KNOWLEDGE_BASE,
                outcome="imported",
                stored_paths=stored,
            )
        )

    task_id: str | None = None
    if accepted_paths:
        # One job for the accepted set, exactly as an upload of the same files
        # would queue: indexing a knowledge base is a batch operation, not a
        # per-file one.
        task_id = kb_routes._build_unique_task_id(_KB_IMPORT_TASK_TYPE, kb_name)
        kb_routes.get_task_stream_manager().ensure_task(task_id)
        kb_routes._mark_kb_queued_for_processing(
            manager,
            kb_name,
            task_id,
            f"Processing {len(accepted_paths)} imported file(s)...",
        )
        background_tasks.add_task(
            kb_routes.run_upload_processing_task,
            accepted_indexing_snapshot=accepted_snapshot,
            kb_name=kb_name,
            base_dir=str(kb_base_dir),
            uploaded_file_paths=accepted_paths,
            task_id=task_id,
            rag_provider=kb_provider,
            owner=kb_routes.get_current_user(),
            storage_workspace_id=workspace_id_for_kb_base_dir(kb_base_dir),
        )
    return _BatchImportOutcome(results=results, task_id=task_id)


# Most destinations answer per file; a knowledge base answers per batch because
# one indexing job covers every file accepted in the request.
_BATCH_IMPORTERS: dict[str, DocumentBatchImporter] = {
    DESTINATION_KNOWLEDGE_BASE: _import_knowledge_base,
}


@router.post("/documents", response_model=DocumentImportResponse)
async def import_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] | None = File(default=None),
    destination: str = Form(default=DESTINATION_AUTHORED_PAGE),
    parent_id: str | None = Form(default=None),
    target: str | None = Form(default=None),
) -> DocumentImportResponse:
    """Import a batch of files and report one outcome per file.

    A bad file never fails the batch: it gets its own skipped or failed entry
    and the rest still import. Only a malformed request — no files, too many
    files, an unknown destination, a destination missing the thing it names —
    is an HTTP error, because there is nothing per-file to report for it.

    ``target`` names the instance inside the destination: the knowledge base
    to import into. It is used verbatim, never case-folded, because a
    knowledge-base name is user data (see the note above ``_IMPORTERS``).
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files to import")
    if len(files) > _MAX_IMPORT_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request (max {_MAX_IMPORT_FILES})",
        )
    normalized = (destination or "").strip().lower()
    batch_importer = _BATCH_IMPORTERS.get(normalized)
    importer = _IMPORTERS.get(normalized)
    if batch_importer is None and importer is None:
        supported = ", ".join(sorted({*_IMPORTERS, *_BATCH_IMPORTERS}))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown import destination {destination!r}; supported: {supported}",
        )

    task_id: str | None = None
    if batch_importer is not None:
        outcome = await batch_importer(files, target=target, background_tasks=background_tasks)
        results, task_id = outcome.results, outcome.task_id
    else:
        results = [
            await importer(upload, destination=normalized, parent_id=parent_id)  # type: ignore[misc]
            for upload in files
        ]
    counts = {"imported": 0, "skipped": 0, "failed": 0}
    for result in results:
        counts[result.outcome] += 1
    return DocumentImportResponse(destination=normalized, files=results, task_id=task_id, **counts)
