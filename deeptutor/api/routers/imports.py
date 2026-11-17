"""
Bring external material into the user's learning space.

Two families live here, both of them batch-oriented and both reporting what
happened per item rather than failing the whole request:

* ``POST /chat-history`` imports conversations from ChatGPT exports and
  external coding CLIs (Claude Code, Codex) as re-openable sessions.
* ``POST /documents`` imports files into a destination store. It is the
  shared import spine: one per-file result contract, one destination field,
  and one handler per destination. Three destinations exist today — an authored
  page, a collection record, and a knowledge base — and later ones are additive
  entries in ``_IMPORTERS`` (one file at a time) or ``_BATCH_IMPORTERS`` (a
  destination that answers the whole batch at once and queues one job for it).
  A fourth value, ``auto``, is not a store: it picks one of those three per file
  and then dispatches each store once (see ``_import_automatically``).

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
import hashlib
import logging
from pathlib import Path
import re
from typing import Any, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
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
DESTINATION_COLLECTION = "collection"
DESTINATION_KNOWLEDGE_BASE = "knowledge_base"
# Not a store: a request-scoped instruction to pick one of the three above per
# file. It never appears in a per-file result, which always reports the store the
# file actually went to.
DESTINATION_AUTO = "auto"

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
    # Destinations with a layout of their own (a knowledge base's ``raw/``)
    # report where the file landed. One upload can become several files,
    # because a `.zip` is expanded into its members, so this is a list.
    stored_paths: list[str] | None = None


class ImportDestinationGroup(BaseModel):
    """One destination's share of a batch, so a UI can render it as a section."""

    destination: str
    files: int
    imported: int
    skipped: int
    failed: int
    # The background job this destination queued for the batch, when it queues
    # one. Only a batch-shaped destination has one; the knowledge base owns it.
    task_id: str | None = None


class DocumentImportResponse(BaseModel):
    """One outcome per file, plus the batch totals a UI shows."""

    # What the request asked for, normalised — so ``auto`` stays visible here
    # rather than being replaced by one of the stores it chose from. Where each
    # file actually went is on its own result and in ``groups``.
    destination: str
    imported: int
    skipped: int
    failed: int
    files: list[ImportFileResult]
    # The same outcomes rolled up per store, keyed by the destination the file
    # landed in and ordered as the destinations first appear in ``files``.
    # Always present, even for a single explicit destination, so a client reads
    # one shape. A file no destination accepted has no group: it went nowhere.
    groups: dict[str, ImportDestinationGroup] = Field(default_factory=dict)
    # The background job a destination queued for this batch, when it queues
    # one; a UI can follow its progress instead of inventing its own spinner.
    task_id: str | None = None


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
# A batch-shaped destination answers the whole request at once and may queue one job
# for it, so its signature is not the per-file one.
DocumentBatchImporter = Callable[..., Awaitable["_BatchImportOutcome"]]


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


def _document_store() -> Any:
    """Return the authored-page store, resolved on demand.

    The store belongs to the co-writer feature while this router belongs to the server
    feature, so importing it at module scope would make the whole API server depend on a
    feature that may not be installed (see tests/modules/test_isolated_features.py).
    Resolving through one helper keeps a single seam for tests to patch.
    """

    from deeptutor.co_writer.storage import get_co_writer_storage

    return get_co_writer_storage()


def _collection_manager() -> Any:
    """Return the collection manager, resolved on demand.

    Same reason as `_document_store`: the collections live in the notebook feature.
    """

    from deeptutor.services.notebook import notebook_manager

    return notebook_manager


def _record_type_import() -> Any:
    """Return the record type that marks an imported file, resolved on demand.

    Same reason as `_document_store`: the vocabulary lives in the notebook feature.
    """

    from deeptutor.services.notebook.service import RecordType

    return RecordType.IMPORT


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
        store = _document_store()
    except ImportError:
        return ImportFileResult(
            filename=filename,
            destination=target.destination,
            outcome="failed",
            reason="The notes feature is not available in this installation.",
        )

    try:
        document = store.create_document(
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

    # Same order as the authored page: drop the decoded BOM first, or a record's
    # body and its list summary would both start with an invisible character.
    content = _normalise_newlines(_strip_leading_bom(FileTypeRouter.decode_bytes(data)))
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
        added = _collection_manager().add_record(
            notebook_ids=list(target.collection_ids),
            record_type=_record_type_import(),
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
    "The bound embedding model is unavailable or changed: the model was deleted, its settings "
    "changed, or it no longer resolves. Restore it in Settings > Embedding models, or select a "
    "model to re-index this knowledge base. A localhost embedding server (Ollama, vLLM, "
    "LM Studio or Lemonade) satisfies a fully local deployment and needs no API key; a model on "
    "a remote API needs network access and its API key."
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


@dataclass(frozen=True)
class _ResolvedKnowledgeBase:
    """The knowledge base a batch is going into, resolved once for the batch."""

    manager: Any
    name: str
    base_dir: Path
    entry: dict[str, Any]


def _resolve_knowledge_base(kb_name: str) -> _ResolvedKnowledgeBase:
    """Resolve the named knowledge base, or raise why it cannot be used.

    Split out from `_import_knowledge_base` because a name that does not resolve
    has to be known *before* anything is written: `auto` routes notes and
    collections first, and a 404 raised after those were written told the user
    the whole request had failed while their pages already existed, so the
    corrected retry created them twice.

    The `HTTPException` carries the reason the request itself would give (404
    for a name that resolves to nothing, 409 for a workspace or access
    problem), so a per-file failure can repeat it verbatim.
    """
    # Resolve the pipeline's helpers at call time instead of copying them: a fix
    # in the upload route must reach this one, and importing the knowledge
    # router eagerly would drag the whole RAG stack into this router's import.
    from deeptutor.api.routers import knowledge as kb_routes
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
    return _ResolvedKnowledgeBase(
        manager=manager, name=kb_name, base_dir=kb_base_dir, entry=kb_entry
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

    resolved = _resolve_knowledge_base(kb_name)
    manager, kb_name, kb_base_dir = resolved.manager, resolved.name, resolved.base_dir
    kb_entry = resolved.entry

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


# -----------------------------------------------------------------------------
# Automatic destination - one store per file, chosen by type
# -----------------------------------------------------------------------------

# The routing table. It is a table, not a heuristic: for one request, the same
# file type always lands in the same store, and the rows below are the whole of
# it.
#
#   extension family                            needs             destination
#   ------------------------------------------  ----------------  --------------
#   .html .htm                                  -                 authored_page
#   .md .markdown .mdown .txt                   no collection     authored_page
#   .md .markdown .mdown .txt                   collections       collection
#   anything the knowledge-base engine accepts  a knowledge base  knowledge_base
#     (pdf, docx, pptx, xlsx, epub, images, and  named
#     every text/code format that is not a
#     note or record format)
#   everything else                             -                 none: failed,
#                                                                 listing what is
#                                                                 accepted
#
# Why the table reads this way:
#
# * HTML has exactly one readable home. The authored page is the only
#   destination with an HTML-to-Markdown converter; a collection record would
#   hold raw tags, and a knowledge base would index markup nobody proof-read.
# * Markdown and plain text are the ambiguous pair, so they take the cheapest
#   home: notes, which needs no context at all - no parent, no collection, no
#   knowledge base. Naming collections is the explicit way to send them
#   elsewhere, and an explicit collection outranks the type rule exactly as a
#   named destination outranks every automatic decision.
# * Everything else can only be indexed, so it goes to the knowledge base - but
#   only when one is named. Auto never picks a knowledge base or a collection on
#   the user's behalf: a file with no nameable home fails and says what it needs.
# * A container (.zip) is deliberately absent: its members decide their own
#   destination, which cannot be seen before the archive is expanded, so it has
#   to name a destination explicitly.
# * A file whose extension is not in the table is never "close enough" to one
#   that is: it fails with the accepted list instead of being guessed into a
#   store that would not read it back correctly.

# The reason a knowledge-base format gives when the request named no knowledge
# base. It is a missing choice, not a format complaint, and it names the field
# that carries the choice.
_AUTO_NEEDS_KNOWLEDGE_BASE_REASON = (
    "This format can only be imported into a knowledge base. Name the knowledge "
    "base to import it into ('target'), or choose another destination for the drop."
)

# Files are hashed in one chunk per read: large enough that a document of any
# normal size is a couple of reads, small enough that a second full copy of it
# never sits in memory.
_DIGEST_CHUNK_BYTES = 1024 * 1024


def _knowledge_base_extensions() -> frozenset[str]:
    """Every extension the knowledge-base engine accepts, resolved on demand.

    The engine is the authority on its own formats, so this asks it rather than
    copying a list that would drift. It is resolved per request, not at import
    time, because building the parse service is optional-dependency work the API
    server must not do while it is being imported. An installation where the
    service cannot be built simply has no knowledge-base formats to route to.
    """
    try:
        accepted = FileTypeRouter.accepted_upload_extensions()
    except Exception:  # noqa: BLE001 - an unbuildable engine means "no formats"
        logger.warning("could not resolve knowledge-base formats; auto cannot route them")
        return frozenset()
    return frozenset(extension.lower() for extension in accepted)


def _auto_supported_extensions(kb_extensions: frozenset[str]) -> tuple[str, ...]:
    """The extensions auto can route, in the order the reason renders them.

    Notes and records come first because they are the reader-visible stores; the
    engine's own formats follow, sorted so the message is stable between runs.
    """
    ordered: dict[str, None] = {}
    for extension in (*_AUTHORED_PAGE_EXTENSIONS, *_COLLECTION_EXTENSIONS):
        ordered.setdefault(extension, None)
    for extension in sorted(kb_extensions):
        ordered.setdefault(extension, None)
    return tuple(ordered)


def _auto_unsupported_reason(kb_extensions: frozenset[str]) -> str:
    """The refusal for a type no store accepts, listing what would be accepted."""
    supported = ", ".join(_auto_supported_extensions(kb_extensions))
    return f"unsupported extension (supported: {supported})"


def _auto_destination(
    filename: str,
    *,
    collection_ids: tuple[str, ...],
    kb_extensions: frozenset[str],
) -> str | None:
    """The store auto picks for one filename, or None when no store accepts it.

    The table this implements is documented above _knowledge_base_extensions;
    this is its only implementation, and every row is covered by
    tests/api/test_imports_auto.py.
    """
    suffix = Path(filename).suffix.lower()
    if not suffix:
        return None
    if suffix in _HTML_EXTENSIONS:
        return DESTINATION_AUTHORED_PAGE
    if suffix in _COLLECTION_EXTENSIONS:
        return DESTINATION_COLLECTION if collection_ids else DESTINATION_AUTHORED_PAGE
    if suffix in kb_extensions:
        return DESTINATION_KNOWLEDGE_BASE
    return None


async def _content_digest(upload: UploadFile) -> str:
    """The upload's SHA-256, with the stream rewound for whoever reads it next.

    Empty content returns an empty string rather than the digest of nothing, so
    an empty file keeps the importer's own "empty" reason instead of being called
    a duplicate of every other empty file in the drop.
    """
    digest = hashlib.sha256()
    read = 0
    await upload.seek(0)
    while chunk := await upload.read(_DIGEST_CHUNK_BYTES):
        read += len(chunk)
        digest.update(chunk)
    await upload.seek(0)
    return digest.hexdigest() if read else ""


async def _import_automatically(
    files: list[UploadFile],
    *,
    parent_id: str | None,
    collection_ids: tuple[str, ...],
    target: str | None,
    background_tasks: BackgroundTasks,
) -> tuple[list[ImportFileResult], str | None]:
    """Route every file by type, then dispatch each store once for the batch.

    The result list is in the request's file order, because that is the order the
    user picked the files in; the dispatching is grouped instead, so the
    knowledge base still receives one batch and queues one indexing job for it.

    Explicit context outranks the type table: a named destination never reaches
    here at all, and named collections move Markdown and plain text to the
    collection (see the table above). Nothing is guessed - a type no store
    accepts, and a knowledge-base format with no knowledge base named, are
    per-file failures that say what is missing.

    Duplicate content inside one request is skipped: the first copy is the one
    that is imported, and every later copy names it. Nothing is remembered
    between requests, so re-importing the same file still works.

    The knowledge-base target is resolved before the first store is dispatched,
    so a name that resolves to nothing fails only the files planned into it and
    the rest of the batch still lands. Resolving it last is what produced a 404
    for a request that had already written pages: the user retried the corrected
    name and imported the same Markdown twice.
    """
    kb_extensions = _knowledge_base_extensions()
    kb_name = (target or "").strip()

    results: list[ImportFileResult | None] = [None] * len(files)
    planned: list[str | None] = [None] * len(files)

    for index, upload in enumerate(files):
        filename = (upload.filename or "").strip()
        if not filename:
            results[index] = ImportFileResult(
                filename="",
                destination="",
                outcome="failed",
                reason="The upload has no filename.",
            )
            continue
        destination = _auto_destination(
            filename, collection_ids=collection_ids, kb_extensions=kb_extensions
        )
        if destination is None:
            results[index] = ImportFileResult(
                filename=filename,
                destination="",
                outcome="failed",
                reason=_auto_unsupported_reason(kb_extensions),
            )
            continue
        planned[index] = destination

    # Duplicates are resolved in request order, before anything is written, so no
    # store ever sees a second copy of the same bytes.
    seen: dict[str, str] = {}
    for index, upload in enumerate(files):
        destination = planned[index]
        if destination is None:
            continue
        digest = await _content_digest(upload)
        if not digest:
            continue
        kept = seen.get(digest)
        if kept is None:
            seen[digest] = (upload.filename or "").strip()
            continue
        results[index] = ImportFileResult(
            filename=(upload.filename or "").strip(),
            destination=destination,
            outcome="skipped",
            reason=(
                f'duplicate of "{kept}": only the first copy of identical content '
                "in this request was kept."
            ),
        )
        planned[index] = None

    # The knowledge base is the one destination whose name can be wrong while
    # every other file in the batch is fine, and it used to be resolved last:
    # notes and collections were written first, then a name that resolved to
    # nothing raised a 404 that escaped this function, so the user was told the
    # whole request had failed while their pages existed - and the corrected
    # retry created them again.
    #
    # Resolving it here, before any store is dispatched, makes a wrong name cost
    # nothing: the files planned into it fail one by one with the knowledge
    # base's own reason and the rest of the batch proceeds, so nothing is ever
    # written that the response does not report.
    kb_indexes = [
        index
        for index, planned_destination in enumerate(planned)
        if planned_destination == DESTINATION_KNOWLEDGE_BASE
    ]
    kb_rejection: str | None = None
    if kb_indexes:
        if not kb_name:
            kb_rejection = _AUTO_NEEDS_KNOWLEDGE_BASE_REASON
        else:
            try:
                _resolve_knowledge_base(kb_name)
            except HTTPException as exc:
                # The knowledge base refuses the whole batch for its own
                # reason, and there is still a per-file answer to give.
                kb_rejection = str(exc.detail)
    if kb_rejection is not None:
        for index in kb_indexes:
            results[index] = ImportFileResult(
                filename=(files[index].filename or "").strip(),
                # A named knowledge base that does not resolve refused the file,
                # so the result names the destination that refused it. A file
                # with no knowledge base named went nowhere and claims no
                # destination - which is what keeps it out of `groups`.
                destination=DESTINATION_KNOWLEDGE_BASE if kb_name else "",
                outcome="failed",
                reason=kb_rejection,
            )
            planned[index] = None

    # Per-file destinations: one handler call per file, each reporting its own
    # landing. Iterating the registry keeps a future destination routable here as
    # soon as the table names it and the handler exists.
    for destination, importer in _IMPORTERS.items():
        indexes = [
            index
            for index, planned_destination in enumerate(planned)
            if planned_destination == destination
        ]
        if not indexes:
            continue
        import_target = ImportTarget(
            destination=destination,
            parent_id=parent_id,
            collection_ids=collection_ids,
        )
        for index in indexes:
            results[index] = await importer(files[index], import_target)

    task_id: str | None = None
    if kb_indexes and kb_rejection is None:
        # One call, one batch, one indexing job - the same contract an explicit
        # knowledge-base import has. It resolves the target again, so its own
        # failure is caught here rather than escaping: between the check above
        # and this call the knowledge base can be deleted, made read-only, or
        # have its workspace moved, and none of those is a request-level error
        # once the rest of the batch has an answer to report.
        try:
            outcome = await _import_knowledge_base(
                [files[index] for index in kb_indexes],
                target=target,
                background_tasks=background_tasks,
            )
        except HTTPException as exc:
            for index in kb_indexes:
                results[index] = _kb_result(files[index], "failed", str(exc.detail))
        else:
            task_id = outcome.task_id
            for index, result in zip(kb_indexes, outcome.results, strict=True):
                results[index] = result

    missing = [index for index, result in enumerate(results) if result is None]
    if missing:  # pragma: no cover - defensive: every planned index is dispatched
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: no import result for file(s) {missing}",
        )
    return [result for result in results if result is not None], task_id


def _group_results(
    results: list[ImportFileResult],
    *,
    task_id: str | None,
    task_destination: str | None,
) -> dict[str, ImportDestinationGroup]:
    """Roll per-file outcomes up per destination, in the order they first appear."""
    groups: dict[str, ImportDestinationGroup] = {}
    for result in results:
        if not result.destination:
            # No store accepted this file, so there is no destination to group it
            # under. It is still counted in the flat totals and reported per file.
            continue
        group = groups.get(result.destination)
        if group is None:
            group = ImportDestinationGroup(
                destination=result.destination,
                files=0,
                imported=0,
                skipped=0,
                failed=0,
            )
            groups[result.destination] = group
        group.files += 1
        if result.outcome == "imported":
            group.imported += 1
        elif result.outcome == "skipped":
            group.skipped += 1
        else:
            group.failed += 1
    if task_id is not None and task_destination in groups:
        # The job belongs to the destination that queued it, so the group carries
        # it rather than the response growing a second, ambiguous field.
        groups[task_destination].task_id = task_id
    return groups


@router.post("/documents", response_model=DocumentImportResponse)
async def import_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] | None = File(default=None),
    destination: str = Form(default=DESTINATION_AUTHORED_PAGE),
    parent_id: str | None = Form(default=None),
    # A repeatable multipart field: one request may target several
    # collections, and FastAPI accepts a plain list for repeated form keys but
    # not the `list[str] | None` union, which it reads as a JSON body.
    collection_ids: list[str] = Form(default=[]),
    target: str | None = Form(default=None),
) -> DocumentImportResponse:
    """Import a batch of files and report one outcome per file.

    A bad file never fails the batch: it gets its own skipped or failed entry
    and the rest still import. Only a malformed request — no files, too many
    files, an unknown destination, a destination missing the thing it names —
    is an HTTP error, because there is nothing per-file to report for it.

    `target` names the instance inside the destination: the knowledge base to
    import into. It is used verbatim, never case-folded, because a
    knowledge-base name is user data (see the note above `_IMPORTERS`).

    `destination="auto"` routes each file by type instead of sending the whole
    batch to one store: HTML and (without named collections) Markdown and plain
    text become notes, Markdown and plain text with named collections become
    records, and everything the knowledge-base engine accepts goes to the named
    knowledge base. The routing table and its reasoning live above
    `_knowledge_base_extensions`. Every file still reports the store it
    actually landed in, and `groups` rolls those results up per destination.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files to import")
    if len(files) > _MAX_IMPORT_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files in one request (max {_MAX_IMPORT_FILES})",
        )
    normalized = (destination or "").strip().lower()
    # "auto" is not a store, so it has no handler of its own: it picks a store per
    # file and then dispatches each one through the two registries below.
    automatic = normalized == DESTINATION_AUTO
    # Two registries, because a destination either answers per file or answers
    # the batch as a whole (a knowledge base queues one job for all of it).
    batch_importer = _BATCH_IMPORTERS.get(normalized)
    importer = _IMPORTERS.get(normalized)
    if not automatic and batch_importer is None and importer is None:
        supported = ", ".join(sorted({*_IMPORTERS, *_BATCH_IMPORTERS, DESTINATION_AUTO}))
        raise HTTPException(
            status_code=400,
            detail=f"Unknown import destination {destination!r}; supported: {supported}",
        )

    targets = _unique_collection_ids(collection_ids)
    results: list[ImportFileResult]
    task_id: str | None = None
    # Which destination the batch's one job belongs to, so the grouped view can
    # carry it. Only a batch-shaped destination queues a job.
    task_destination: str | None = None
    if automatic:
        results, task_id = await _import_automatically(
            files,
            parent_id=parent_id,
            collection_ids=targets,
            target=target,
            background_tasks=background_tasks,
        )
        task_destination = DESTINATION_KNOWLEDGE_BASE
    elif batch_importer is not None:
        outcome = await batch_importer(files, target=target, background_tasks=background_tasks)
        results, task_id = outcome.results, outcome.task_id
        task_destination = normalized
    elif importer is not None:
        import_target = ImportTarget(
            destination=normalized,
            parent_id=parent_id,
            collection_ids=targets,
        )
        results = [await importer(upload, import_target) for upload in files]
    else:  # pragma: no cover - unreachable, the guard above rejects every other name
        raise HTTPException(status_code=400, detail=f"Unknown import destination {destination!r}")

    counts = {"imported": 0, "skipped": 0, "failed": 0}
    for result in results:
        counts[result.outcome] += 1
    return DocumentImportResponse(
        destination=normalized,
        files=results,
        task_id=task_id,
        groups=_group_results(results, task_id=task_id, task_destination=task_destination),
        **counts,
    )
