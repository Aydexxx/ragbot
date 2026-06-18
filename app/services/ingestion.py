"""Document ingestion: text extraction and chunking.

Pure text processing — no network calls. Text is split at paragraph/sentence
boundaries where possible and packed into ``chunk_size`` windows with
``overlap`` between consecutive windows, never cutting a word in half.
"""

from __future__ import annotations

import bisect
import io
import re
from typing import NamedTuple

from pypdf import PdfReader

from app.models.chunk import Chunk

_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}
_PDF_EXTENSIONS = {".pdf"}

#: File extensions :func:`extract` can handle. Exposed for callers (e.g. the
#: health endpoint) that need to advertise this without duplicating the list.
SUPPORTED_EXTENSIONS: list[str] = sorted(_PDF_EXTENSIONS | _TEXT_EXTENSIONS)

#: Separator joined between pages of a PDF. Page start offsets are computed
#: against text assembled with exactly this separator.
_PAGE_SEPARATOR = "\n\n"


class Extraction(NamedTuple):
    """Extracted document text plus, for paginated formats, page boundaries.

    ``page_starts`` holds the character offset at which each page begins in
    ``text`` (so page *i*, 0-based, spans ``page_starts[i]`` up to
    ``page_starts[i + 1]``). It is ``None`` for formats without pages, where
    character offsets are the only locator available.
    """

    text: str
    page_starts: list[int] | None

# Paragraph break (blank line) or sentence end (punctuation + whitespace,
# followed by what looks like the start of a new sentence). This is a
# heuristic, not a full NLP sentence tokenizer.
_BOUNDARY_RE = re.compile(r"\n\s*\n+" r"|(?<=[.!?])\s+(?=[A-Z0-9\"'(])")

_WORD_RE = re.compile(r"\S+")


class UnsupportedFileTypeError(ValueError):
    """Raised when extract_text is given a file type it cannot handle."""


def extract(file_bytes: bytes, filename: str) -> Extraction:
    """Extract text and page boundaries from raw bytes, keyed off the extension.

    Args:
        file_bytes: Raw file content.
        filename: Original filename; only its extension is used.

    Returns:
        An :class:`Extraction` with the full text and, for PDFs, the start
        offset of each page (``None`` for unpaginated TXT/MD).

    Raises:
        UnsupportedFileTypeError: If the extension isn't PDF/TXT/MD.
    """
    suffix = _suffix(filename)

    if suffix in _PDF_EXTENSIONS:
        return _extract_pdf(file_bytes)
    if suffix in _TEXT_EXTENSIONS:
        return Extraction(
            text=file_bytes.decode("utf-8", errors="replace"), page_starts=None
        )

    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix or filename}'. Supported: {SUPPORTED_EXTENSIONS}"
    )


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text only; see :func:`extract` for text plus page offsets."""
    return extract(file_bytes, filename).text


def assign_pages(chunks: list[Chunk], page_starts: list[int] | None) -> list[Chunk]:
    """Set each chunk's ``page`` from its start offset, in place.

    A chunk is attributed to the page it *starts* on (a chunk straddling a page
    break still belongs to one page). When ``page_starts`` is ``None`` the
    format has no pages and every ``page`` is left as ``None``.

    Returns the same list for convenient chaining.
    """
    if page_starts is None:
        return chunks
    for chunk in chunks:
        chunk.page = _page_for_offset(page_starts, chunk.char_start)
    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split text into overlapping chunks.

    Splits at paragraph/sentence boundaries where possible, then greedily
    packs those pieces into windows of at most ``chunk_size`` characters.
    Consecutive windows share roughly ``overlap`` characters of trailing
    context. A single sentence/paragraph longer than ``chunk_size`` falls
    back to a word-boundary split so output never cuts a word in half.

    Args:
        text: The text to chunk.
        chunk_size: Maximum length, in characters, of each chunk.
        overlap: Approximate character overlap between consecutive chunks.

    Returns:
        Chunks in document order, with sequential ``index`` and char offsets
        into ``text``. Empty for empty/whitespace-only input.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and less than chunk_size")

    if not text.strip():
        return []

    spans: list[tuple[int, int]] = []
    for start, end in _split_atoms(text):
        if end - start <= chunk_size:
            spans.append((start, end))
        else:
            spans.extend(_split_words(text, start, end, chunk_size))

    chunks: list[Chunk] = []
    window: list[tuple[int, int]] = []
    window_len = 0
    i = 0
    while i < len(spans):
        start, end = spans[i]
        length = end - start
        added_len = length if not window else length + 1  # +1 for join space
        if window and window_len + added_len > chunk_size:
            chunks.append(_finalize(text, window, len(chunks)))
            window, window_len = _carry_overlap(window, overlap)
            continue
        window.append((start, end))
        window_len += added_len
        i += 1

    if window:
        chunks.append(_finalize(text, window, len(chunks)))

    return chunks


def _suffix(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _extract_pdf(file_bytes: bytes) -> Extraction:
    """Extract PDF text page by page, recording where each page starts.

    Pages are joined with ``_PAGE_SEPARATOR``; ``page_starts[i]`` is the offset
    of page *i* in the joined text, so a chunk's start offset maps back to its
    source page.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    page_texts = [page.extract_text() or "" for page in reader.pages]

    page_starts: list[int] = []
    pos = 0
    for page_text in page_texts:
        page_starts.append(pos)
        pos += len(page_text) + len(_PAGE_SEPARATOR)

    return Extraction(
        text=_PAGE_SEPARATOR.join(page_texts), page_starts=page_starts
    )


def _page_for_offset(page_starts: list[int], offset: int) -> int:
    """Return the 1-based page number containing ``offset``.

    ``page_starts`` is ascending, so the page is the last start that is not
    past ``offset``. Offsets before the first page (shouldn't happen) clamp to
    page 1.
    """
    idx = bisect.bisect_right(page_starts, offset) - 1
    return max(idx, 0) + 1


def _split_atoms(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of paragraph/sentence atoms in ``text``.

    Each span is trimmed to exclude leading/trailing whitespace, so atoms
    never start or end mid-word.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for match in _BOUNDARY_RE.finditer(text):
        if match.start() > pos:
            spans.append(_trim_span(text, pos, match.start()))
        pos = match.end()
    if pos < len(text):
        spans.append(_trim_span(text, pos, len(text)))
    return [span for span in spans if span[0] < span[1]]


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _split_words(
    text: str, start: int, end: int, chunk_size: int
) -> list[tuple[int, int]]:
    """Split an oversized atom into chunk_size-ish windows on word boundaries.

    A single word longer than ``chunk_size`` is kept intact and may exceed
    ``chunk_size`` — never cutting mid-word takes priority over the size cap.
    """
    word_spans = [
        (start + m.start(), start + m.end())
        for m in _WORD_RE.finditer(text[start:end])
    ]

    pieces: list[tuple[int, int]] = []
    piece_start: int | None = None
    piece_end = 0
    piece_len = 0
    for w_start, w_end in word_spans:
        w_len = w_end - w_start
        added = w_len if piece_start is None else w_len + 1
        if piece_start is not None and piece_len + added > chunk_size:
            pieces.append((piece_start, piece_end))
            piece_start = None
        if piece_start is None:
            piece_start = w_start
            piece_len = w_len
        else:
            piece_len += added
        piece_end = w_end
    if piece_start is not None:
        pieces.append((piece_start, piece_end))
    return pieces


def _carry_overlap(
    finished_window: list[tuple[int, int]], overlap: int
) -> tuple[list[tuple[int, int]], int]:
    """Return the trailing atoms of a finished window to seed the next one.

    A single-atom window has nothing smaller to carry without duplicating
    the whole previous chunk, so it starts the next window fresh.
    """
    if overlap == 0 or len(finished_window) <= 1:
        return [], 0

    carried: list[tuple[int, int]] = []
    carried_len = 0
    for start, end in reversed(finished_window):
        if carried and carried_len >= overlap:
            break
        length = end - start
        added = length if not carried else length + 1
        carried.insert(0, (start, end))
        carried_len += added
    return carried, carried_len


def _finalize(text: str, window: list[tuple[int, int]], index: int) -> Chunk:
    char_start = window[0][0]
    char_end = window[-1][1]
    return Chunk(
        text=text[char_start:char_end],
        index=index,
        char_start=char_start,
        char_end=char_end,
    )
