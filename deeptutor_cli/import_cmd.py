"""`deeptutor import` - the command-line face of the import spine.

The command deliberately owns no importing logic. It builds the real
`deeptutor.api.routers.imports` router into a bare ASGI application and posts
to `POST /api/imports/documents` in-process, so the format rules, the per-file
result contract and the automatic routing are the endpoint's, not a copy of them.
That is the whole design: one import spine, two front doors.

Two consequences are worth keeping in mind:

* Nothing here may import `deeptutor.api` at module scope. The server module is
  optional and `deeptutor_cli` has to stay importable without it, so the router is
  imported inside `_post`, after `require_module("server")` has said whether it exists.
* The destination names the CLI exposes are aliases, never a second enum: every
  value in `_DESTINATIONS` is one the endpoint already understands, and nothing
  else is accepted.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import logging
from pathlib import Path

import typer

from .common import console

# CLI spelling -> the endpoint's own destination value (see DESTINATION_* in
# deeptutor/api/routers/imports.py). An alias table, not a second enum.
_DESTINATIONS = {
    "notes": "authored_page",
    "collection": "collection",
    "knowledge-base": "knowledge_base",
    "auto": "auto",
}

# The one endpoint this command uses. It is named in the help text as well, so
# "the CLI and the web app import the same way" is visible from the terminal.
_IMPORTS_PREFIX = "/api/imports"
_ENDPOINT = f"{_IMPORTS_PREFIX}/documents"

_SERVER_MODULE_HINT = (
    "The import command needs the server module, because it goes through the same "
    "HTTP endpoint the web app uses. Run: deeptutor module install server"
)


class _EndpointUnavailable(RuntimeError):
    """The import endpoint cannot be built in this installation."""


def register(app: typer.Typer) -> None:
    @app.command("import")
    def import_files(
        files: list[Path] = typer.Argument(..., help="Files to import."),
        to: str = typer.Option(
            "auto",
            "--to",
            help=(
                "Destination: notes | collection | knowledge-base | auto. "
                "notes accepts .md .markdown .mdown .txt .html .htm; "
                "a collection accepts .md .markdown .mdown .txt; "
                "a knowledge base accepts what its engine indexes "
                "(.pdf .docx .pptx .xlsx .epub, images, and text/code formats). "
                "auto picks one of the three per file, by type."
            ),
        ),
        collection: list[str] = typer.Option(
            [],
            "--collection",
            help=(
                "Collection id to import into; repeat for several. Required with --to collection."
            ),
        ),
        kb: str | None = typer.Option(
            None, "--kb", help="Knowledge base name. Required with --to knowledge-base."
        ),
        parent: str | None = typer.Option(
            None, "--parent", help="Parent page id for notes (--to notes or auto only)."
        ),
        as_json: bool = typer.Option(
            False,
            "--json",
            help="Print the endpoint's response as JSON instead of the per-file report.",
        ),
    ) -> None:
        """Import files into your notes, a collection, or a knowledge base.

        Every file reports its own outcome and reason, and the exit code is 0 only
        when every file imported. The command posts to /api/imports/documents
        in-process - the same endpoint the web app uses - so the format rules, the
        per-file contract and the automatic routing are the endpoint's own.
        """
        destination, collection_ids, target, parent_id = _resolve(to, collection, kb, parent)
        sources = _existing_files(files)

        try:
            status, payload = asyncio.run(
                _post(
                    sources,
                    destination=destination,
                    collection_ids=collection_ids,
                    target=target,
                    parent_id=parent_id,
                )
            )
        except _EndpointUnavailable as exc:
            _error(str(exc))
            raise typer.Exit(code=1) from exc

        if status >= 400:
            # A whole-request rejection: the endpoint had nothing per file to
            # report, so the reason is the request's own detail.
            _error(_detail(payload, status))
            raise typer.Exit(code=2 if status < 500 else 1)

        if as_json:
            # The endpoint's response, verbatim and machine readable.
            typer.echo(json.dumps(payload, indent=2))
        else:
            _report(payload)
        raise typer.Exit(code=_exit_code(payload))


@contextmanager
def _quiet_httpx():
    """Keep httpx's per-request INFO line out of a one-shot command's output.

    httpx logs every request at INFO. For a command whose whole output is
    the per-file report, that line is noise on stderr -- and it is the one
    thing a user piping the report does not want. The level is restored
    afterwards so calling this from a library does not change global state.
    """

    logger = logging.getLogger("httpx")
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(previous)


async def _post(
    sources: list[Path],
    *,
    destination: str,
    collection_ids: list[str],
    target: str | None,
    parent_id: str | None,
) -> tuple[int, dict]:
    """POST the drop at the real import router, in this process.

    The application is a bare FastAPI with only the imports router mounted: the
    full `deeptutor.api.main` app would drag every router (and their settings
    files) into a command that needs exactly one of them. The router object is the
    real one, mounted at the same prefix the server uses, so this is the endpoint.
    """
    try:
        from deeptutor.modules.catalog import ModuleUnavailable, require_module

        require_module("server")
        from fastapi import FastAPI
        import httpx

        from deeptutor.api.routers import imports as imports_router
    except (ImportError, ModuleUnavailable) as exc:
        raise _EndpointUnavailable(_SERVER_MODULE_HINT) from exc

    application = FastAPI()
    application.include_router(imports_router.router, prefix=_IMPORTS_PREFIX)

    # httpx expands a list value into one repeated form field, which is exactly
    # how the endpoint reads a repeatable collection_ids.
    form: dict[str, object] = {"destination": destination}
    if parent_id:
        form["parent_id"] = parent_id
    if target:
        form["target"] = target
    if collection_ids:
        form["collection_ids"] = list(collection_ids)

    uploads = [
        ("files", (source.name, source.read_bytes(), "application/octet-stream"))
        for source in sources
    ]

    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://deeptutor.local") as client:
        with _quiet_httpx():
            response = await client.post(_ENDPOINT, data=form, files=uploads)

    try:
        payload = response.json()
    except ValueError:
        payload = {"detail": response.text}
    return response.status_code, payload


def _resolve(
    to: str,
    collection: list[str],
    kb: str | None,
    parent: str | None,
) -> tuple[str, list[str], str | None, str | None]:
    """Map the CLI's names onto the endpoint's form fields, or refuse the mix.

    The destinations are mutually exclusive because each store's own identifiers
    only mean something to that store: a collection id sent to a knowledge-base
    import would be silently ignored, and a silent no-op is worse than a refusal.
    `auto` is the one destination that takes all three as context, because it is
    the one that chooses per file.
    """
    key = (to or "").strip().lower()
    if key not in _DESTINATIONS:
        raise typer.BadParameter(
            f"unknown destination {to!r}; choose one of: notes, collection, knowledge-base, auto",
            param_hint="--to",
        )
    collections = [value for value in collection if value.strip()]
    target = (kb or "").strip() or None
    parent_id = (parent or "").strip() or None

    if key == "collection":
        if not collections:
            raise typer.BadParameter(
                "--to collection needs at least one --collection <id>.", param_hint="--collection"
            )
        if target:
            raise typer.BadParameter(
                "--kb is for --to knowledge-base or auto; a collection holds records.",
                param_hint="--kb",
            )
    elif key == "knowledge-base":
        if not target:
            raise typer.BadParameter("--to knowledge-base needs --kb <name>.", param_hint="--kb")
        if collections:
            raise typer.BadParameter(
                "--collection is for --to collection or auto; a knowledge base is not a "
                "collection.",
                param_hint="--collection",
            )
    else:  # notes or auto
        if key == "notes" and collections:
            raise typer.BadParameter(
                "--collection is for --to collection or auto; --to notes would ignore it.",
                param_hint="--collection",
            )
        if key == "notes" and target:
            raise typer.BadParameter(
                "--kb is for --to knowledge-base or auto; --to notes would ignore it.",
                param_hint="--kb",
            )

    if parent_id and key in {"collection", "knowledge-base"}:
        raise typer.BadParameter(
            "--parent only applies to notes (--to notes or auto).", param_hint="--parent"
        )
    return _DESTINATIONS[key], collections, target, parent_id


def _existing_files(files: list[Path]) -> list[Path]:
    """The files to send, or a readable refusal.

    A directory is refused rather than walked: the endpoint's unit is one file
    with one outcome, and quietly expanding a folder would import a tree the user
    never listed.
    """
    for path in files:
        if not path.exists():
            raise typer.BadParameter(f"{path} does not exist.", param_hint="FILES")
        if not path.is_file():
            raise typer.BadParameter(
                f"{path} is not a file; pass the files themselves.", param_hint="FILES"
            )
    return list(files)


def _exit_code(payload: dict) -> int:
    """0 only when every file imported; 1 when any was skipped or failed."""
    results = payload.get("files") or []
    return 0 if results and all(item.get("outcome") == "imported" for item in results) else 1


def _line(item: dict) -> str:
    """One file's report line: outcome, filename, and the reason when there is one."""
    line = f"{item.get('outcome', '?')} {item.get('filename', '')}".rstrip()
    reason = item.get("reason")
    return f"{line} — {reason}" if reason else line


def _report(payload: dict) -> None:
    """Print every file's outcome, grouped by the store it landed in."""
    results = payload.get("files") or []
    by_destination: dict[str, list[dict]] = {}
    unrouted: list[dict] = []
    for item in results:
        destination = item.get("destination") or ""
        if destination:
            by_destination.setdefault(destination, []).append(item)
        else:
            # No store accepted the file, so the endpoint gave it no group.
            unrouted.append(item)

    for destination, group in (payload.get("groups") or {}).items():
        _print(
            f"{destination}: {group['imported']} imported, "
            f"{group['skipped']} skipped, {group['failed']} failed",
            style="bold",
        )
        for item in by_destination.get(destination, []):
            _print(f"  {_line(item)}")
    if unrouted:
        _print(f"unrouted: {len(unrouted)} failed", style="bold")
        for item in unrouted:
            _print(f"  {_line(item)}")

    _print(
        f"Total: {payload['imported']} imported, {payload['skipped']} skipped, "
        f"{payload['failed']} failed",
        style="bold",
    )
    task_id = payload.get("task_id")
    if task_id:
        _print(f"Indexing task: {task_id}")


def _detail(payload: dict, status: int) -> str:
    """The endpoint's own rejection reason, or an honest fallback."""
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, str) and detail.strip():
        return detail
    if detail is not None:
        return json.dumps(detail)
    return f"the import endpoint returned HTTP {status}."


def _print(text: str, *, style: str | None = None) -> None:
    # A filename or a reason may contain square brackets or markup characters, so
    # nothing printed here is parsed as rich markup, and long reasons stay on one
    # line instead of being wrapped into an unreadable block.
    console.print(text, style=style, markup=False, highlight=False, soft_wrap=True)


def _error(message: str) -> None:
    console.print(
        f"Error: {message}", style="bold red", markup=False, highlight=False, soft_wrap=True
    )


__all__ = ["register"]
