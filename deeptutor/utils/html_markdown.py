"""HTML -> Markdown conversion shared by the authored-page importer.

A saved web page or an HTML export must arrive as readable Markdown, not as a
page full of raw tags, and importing one must not pull a new runtime dependency
into the process — so this is built on the standard library's `html.parser`.

This lives in `deeptutor/utils` on purpose: code that no module manifest claims is owned
by the core module, whereas everything under `deeptutor/co_writer` belongs to the
co-writer feature and may not be imported by the API server at module scope. See
tests/modules/test_isolated_features.py.

The shape mirrors `docx_converter`: one bytes-in / Markdown-out function and one
dedicated exception, so a conversion fault becomes a per-file failure in the
import batch instead of a 500 for the whole request.

The converter is deliberately forgiving. `HTMLParser` already tolerates unclosed
markup, and every construct here degrades to plainer Markdown rather than
raising: malformed input costs formatting, never the import. Content is never
dropped along with an element — an element this module does not understand
contributes its text.
"""

from __future__ import annotations

from html.parser import HTMLParser
import re

from deeptutor.utils.file_types import FileTypeRouter


class HtmlConversionError(Exception):
    """A file could not be read as HTML at all (not: badly formatted HTML)."""


# Characters that would otherwise be read as Markdown syntax. Applied to text
# nodes only, never to markers this module emits itself.
_ESCAPE_RE = re.compile(r"([\\`*_\[\]~|])")

# Runs of horizontal whitespace collapse to one space, the way a browser
# renders them. Newlines survive: they come from <br> and from <pre>.
_HSPACE_RE = re.compile(r"[^\S\n]+")

# Elements that end the current inline run and start a new block.
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "body",
        "caption",
        "dd",
        "div",
        "dl",
        "dt",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "header",
        "hr",
        "html",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)

_HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}

# Inline elements that wrap their content rather than start a block. Anything
# else that is not block-level is simply transparent: its text flows through.
_FRAME_TAGS = frozenset({"a", "b", "code", "del", "em", "i", "s", "strike", "strong"})
_EMPHASIS = {
    "b": "**",
    "strong": "**",
    "i": "*",
    "em": "*",
    "s": "~~",
    "strike": "~~",
    "del": "~~",
}

# Never content: a browser does not show these either. Everything else that
# is not understood is transparent, so its children still reach the page.
_SKIP_TAGS = frozenset({"head", "noscript", "script", "style", "template", "title"})

# Which destinations may become a link. This mirrors
# SAFE_MARKDOWN_PROTOCOL_REGEX in web/lib/markdown-display.ts, which is what the
# rendered note applies to every href: the converter is the first of the two
# layers, so a stored page never carries a scheme the renderer would strip.
# `attachment` is a real scheme here (generated-file cards), and a relative
# destination or an #anchor has no scheme at all, so both are always kept.
_SAFE_LINK_SCHEMES = frozenset({"http", "https", "irc", "ircs", "mailto", "xmpp", "attachment"})

# A scheme as a URL parser reads it: the run of scheme characters before the
# first colon. Anchored, so a colon after a path/query/fragment separator leaves
# the destination relative ("./notes/x.md", "?q=a:b", "#a:b").
_SCHEME_RE = re.compile(r"^([a-z][a-z0-9+.-]*):", re.IGNORECASE)

# Characters a browser strips from a URL before reading its scheme, so a scheme
# hidden behind one of them is the same scheme and must be judged as it.
_URL_NOISE_RE = re.compile(r"[\x00-\x20\x7f]+")

# The one scheme-bearing destination an image may carry, mirroring
# SAFE_RASTER_DATA_IMAGE_REGEX in web/lib/markdown-display.ts. A saved page's
# inline screenshot is passive raster data and worth keeping; `data:text/html`
# is markup smuggled through a URL and `data:image/svg+xml` can carry script.
_SAFE_IMAGE_DATA_RE = re.compile(
    r"^data:image/(?:png|jpe?g|gif|webp|bmp|tiff?|avif);base64,[a-z0-9+/=\s]+$",
    re.IGNORECASE,
)

_INDENT = "  "


def html_to_markdown(data: bytes, filename: str = "page.html") -> str:
    """Convert an HTML document to Markdown.

    Decoding goes through `FileTypeRouter.decode_bytes`, the same path the rest of
    the importer uses, so a GBK page decodes the way a GBK Markdown file does.
    """
    if not data:
        raise HtmlConversionError(f"{filename} is empty.")
    text = FileTypeRouter.decode_bytes(data)
    try:
        parser = _HtmlToMarkdown()
        parser.feed(text)
        parser.close()
        return parser.markdown()
    except HtmlConversionError:
        raise
    except Exception as exc:  # a parser fault is a file fault, never a 500
        raise HtmlConversionError(f"{filename} could not be read as HTML.") from exc


class _HtmlToMarkdown(HTMLParser):
    """Streaming HTML -> Markdown renderer.

    Inline content accumulates in buffers; a wrapping element owns a buffer of
    its own so an empty or unclosed tag cannot lose the text inside it.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # (list-serial or None, rendered block) — a shared serial means two
        # blocks are consecutive items of one list and stay tight.
        self._blocks: list[tuple[int | None, str]] = []
        self._inline: list[str] = []
        self._frames: list[dict[str, object]] = []
        self._raw_depth = 0
        self._heading = 0
        self._pending_marker: str | None = None
        self._list_stack: list[dict[str, object]] = []
        self._list_serial = 0
        self._quote_depth = 0
        self._in_pre = False
        self._pre_parts: list[str] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._in_cell = False
        self._cell_parts: list[str] = []
        self._skipping: str | None = None

    # -- output ---------------------------------------------------------
    def markdown(self) -> str:
        if self._in_pre:
            # An unclosed <pre> would otherwise swallow everything after it.
            self._end_pre()
        if self._table is not None:
            # Same for a table: `</table>` is the only thing that used to emit
            # one, so a table that was never closed (or a file truncated inside
            # it) dropped every cell it had collected while the file still
            # reported `imported`. The rows are the only copy of that content.
            self._end_table()
        self._flush()
        parts: list[str] = []
        previous: int | None = None
        for item_id, block in self._blocks:
            if not block:
                continue
            if parts:
                tight = item_id is not None and item_id == previous
                parts.append("\n" if tight else "\n\n")
            parts.append(block)
            previous = item_id
        return "".join(parts).strip()

    # -- token handlers -------------------------------------------------
    def handle_data(self, data: str) -> None:
        if self._skipping:
            return
        if self._in_pre:
            self._pre_parts.append(data)
        elif self._raw_depth:
            # Inside <code>: escaping would be rendered literally.
            self._inline.append(data)
        else:
            self._inline.append(_escape_md(data))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skipping:
            # An unclosed <head> must not swallow the body.
            if tag == "body":
                self._skipping = None
            return
        if self._in_pre:
            return
        if tag in _SKIP_TAGS:
            self._skipping = tag
            return
        if tag in _HEADING_TAGS:
            self._flush()
            self._heading = _HEADING_TAGS[tag]
        elif tag == "br":
            self._inline.append("\n")
        elif tag == "img":
            self._append_image(attrs)
        elif tag in ("ul", "ol"):
            self._flush()
            if not self._list_stack:
                self._list_serial += 1
            self._list_stack.append({"ordered": tag == "ol", "count": 0})
        elif tag == "li":
            self._flush()
            self._begin_item()
        elif tag == "blockquote":
            self._flush()
            self._quote_depth += 1
        elif tag == "pre":
            self._flush()
            self._in_pre = True
            self._pre_parts = []
        elif tag == "table":
            if self._table is not None:
                # Tables do not nest. Emitting the open one keeps the rows it
                # already collected instead of losing them to the reset below.
                self._end_table()
            self._flush()
            self._table = []
            self._row = None
        elif tag == "tr" and self._table is not None:
            self._flush()
            self._row = []
        elif tag in ("td", "th") and self._table is not None:
            self._flush()
            self._in_cell = True
            self._cell_parts = []
        elif tag in _FRAME_TAGS:
            self._open_frame(tag, attrs)
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self._skipping:
            if tag == self._skipping or (self._skipping == "head" and tag == "body"):
                self._skipping = None
            return
        if self._in_pre:
            if tag == "pre":
                self._end_pre()
            return
        if tag in _HEADING_TAGS:
            level = self._heading or _HEADING_TAGS[tag]
            self._heading = 0
            self._flush(prefix="#" * level + " ")
        elif tag in ("ul", "ol"):
            self._flush()
            if self._list_stack:
                self._list_stack.pop()
        elif tag == "li":
            self._flush()
        elif tag == "blockquote":
            self._flush()
            self._quote_depth = max(0, self._quote_depth - 1)
        elif tag in ("td", "th") and self._in_cell:
            self._end_cell()
        elif tag == "tr" and self._table is not None:
            self._end_row()
        elif tag == "table" and self._table is not None:
            self._end_table()
        elif tag in _FRAME_TAGS:
            self._close_frame(tag)
        elif tag in _BLOCK_TAGS:
            self._flush()

    # -- inline buffers -------------------------------------------------
    def _open_frame(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = _attr(attrs, "href") if tag == "a" else ""
        self._frames.append({"tag": tag, "saved": self._inline, "href": href})
        self._inline = []
        if tag == "code":
            self._raw_depth += 1

    def _close_frame(self, tag: str) -> None:
        # A stray close tag is ignored; the text it would have wrapped stays.
        if not self._frames or self._frames[-1]["tag"] != tag:
            return
        frame = self._frames.pop()
        text = "".join(self._inline)
        self._inline = frame["saved"]  # type: ignore[assignment]
        if tag == "code":
            self._raw_depth = max(0, self._raw_depth - 1)
        self._inline.append(_wrap(frame, text))

    def _drain_frames(self) -> None:
        """Flatten unclosed wrappers at a block boundary without losing text."""
        while self._frames:
            frame = self._frames.pop()
            text = "".join(self._inline)
            self._inline = frame["saved"]  # type: ignore[assignment]
            self._inline.append(text)
            if frame["tag"] == "code":
                self._raw_depth = max(0, self._raw_depth - 1)

    # -- lists, quotes and code blocks -----------------------------------
    def _begin_item(self) -> None:
        if not self._list_stack:
            return
        frame = self._list_stack[-1]
        depth = len(self._list_stack) - 1
        if frame["ordered"]:
            frame["count"] = int(frame["count"]) + 1  # type: ignore[call-overload]
            marker = f"{frame['count']}. "
        else:
            marker = "- "
        self._pending_marker = _INDENT * depth + marker

    def _end_pre(self) -> None:
        self._in_pre = False
        text = "".join(self._pre_parts).replace("\r\n", "\n").replace("\r", "\n")
        self._pre_parts = []
        body = text.strip("\n").rstrip()
        if not body:
            return
        # A bullet whose item is nothing but a code block loses the bullet;
        # keeping it would leak the marker onto an unrelated later block.
        self._pending_marker = None
        self._blocks.append((None, "```\n" + body + "\n```"))

    def _append_image(self, attrs: list[tuple[str, str | None]]) -> None:
        source = _attr(attrs, "src")
        alt = _attr(attrs, "alt")
        if source:
            target = _link_target(source, image=True)
            if target:
                self._inline.append(f"![{_escape_md(alt)}]({target})")
                return
        if alt:
            # A refused source loses the picture, never its caption.
            self._inline.append(_escape_md(alt))

    # -- tables -----------------------------------------------------------
    def _end_cell(self) -> None:
        self._flush()
        self._in_cell = False
        # A newline or a stray pipe inside a cell would break the row.
        cell = " ".join(" ".join(self._cell_parts).split())
        self._cell_parts = []
        if self._row is not None:
            self._row.append(cell)
        elif cell:
            # Cells outside a table are malformed input, not content to drop.
            self._blocks.append((None, cell))

    def _end_row(self) -> None:
        if self._in_cell:
            self._end_cell()
        self._flush()
        if self._table is not None and self._row and any(cell for cell in self._row):
            self._table.append(self._row)
        self._row = None

    def _end_table(self) -> None:
        if self._in_cell:
            self._end_cell()
        if self._row is not None:
            self._end_row()
        rows = [row for row in (self._table or []) if any(cell for cell in row)]
        self._table = None
        if not rows:
            return
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        lines = [
            "| " + " | ".join(padded[0]) + " |",
            "| " + " | ".join("---" for _ in padded[0]) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
        self._blocks.append((None, "\n".join(lines)))

    # -- block assembly --------------------------------------------------
    def _flush(self, prefix: str = "") -> None:
        self._drain_frames()
        text = _normalise("".join(self._inline))
        self._inline = []
        if not text:
            # The marker stays pending for a block nested inside the item.
            return
        if self._in_cell:
            self._cell_parts.append(text)
            return
        if not prefix and self._heading:
            # A heading whose close tag never arrives still reads as a heading;
            # the alternative is silently demoting it to a paragraph.
            prefix = "#" * self._heading + " "
            self._heading = 0
        marker = self._pending_marker
        self._pending_marker = None
        lines = text.split("\n")
        first = next((index for index, line in enumerate(lines) if line), None)
        item_id = None
        if first is not None:
            lines[first] = (marker or "") + prefix + lines[first]
            if marker is not None:
                item_id = self._list_serial
        if self._quote_depth:
            lines = [_quote(line, self._quote_depth) for line in lines]
        self._blocks.append((item_id, "\n".join(lines)))


def _wrap(frame: dict[str, object], text: str) -> str:
    if not text.strip():
        return text
    tag = str(frame["tag"])
    if tag == "a":
        target = _link_target(str(frame.get("href") or ""))
        if not target:
            return text
        return _surround(text, "[", f"]({target})")
    if tag == "code":
        # A backtick inside the span would end it early: keep it as plain text.
        return text if "`" in text else _surround(text, "`", "`")
    marker = _EMPHASIS[tag]
    return _surround(text, marker, marker)


def _surround(text: str, opening: str, closing: str) -> str:
    """Wrap the non-blank core so surrounding spaces stay outside the markers."""
    core = text.strip()
    if not core:
        return text
    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    return f"{lead}{opening}{core}{closing}{trail}"


def _quote(line: str, depth: int) -> str:
    if line:
        return ("> " * depth + line).rstrip()
    return "> " * (depth - 1) + ">"


def _normalise(text: str) -> str:
    """Collapse whitespace the way a browser would render it."""
    text = text.replace("\xa0", " ")
    lines = [_HSPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _escape_md(text: str) -> str:
    return _ESCAPE_RE.sub(r"\\\1", text)


def _attr(attrs: list[tuple[str, str | None]], name: str) -> str:
    for key, value in attrs:
        if key == name:
            return (value or "").strip()
    return ""


def _link_target(href: str, *, image: bool = False) -> str:
    """The destination to write into a link, or "" when it must not be one.

    A destination the web renderer would refuse - `javascript:`, `vbscript:`,
    `data:text/html` and every other scheme outside `_SAFE_LINK_SCHEMES` -
    yields no link at all. The text the element wrapped is content and is kept
    by the caller, which is this module's rule for anything it cannot
    represent: a refused destination costs the link, never the words.
    """
    href = (href or "").strip()
    if not href or not _destination_is_allowed(href, image=image):
        return ""
    # A destination with spaces or angle brackets must be wrapped or it ends
    # the Markdown link early.
    if any(character in href for character in " ()<>"):
        return f"<{href}>"
    return href


def _destination_is_allowed(href: str, *, image: bool) -> bool:
    """Whether a destination is scheme-free (relative) or carries a safe scheme."""
    if image and _SAFE_IMAGE_DATA_RE.match(href):
        return True
    match = _SCHEME_RE.match(_URL_NOISE_RE.sub("", href))
    return match is None or match.group(1).lower() in _SAFE_LINK_SCHEMES


__all__ = ["HtmlConversionError", "html_to_markdown"]
