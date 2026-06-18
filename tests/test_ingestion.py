"""Tests for document text extraction and chunking."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from app.models.chunk import Chunk
from app.services.ingestion import (
    UnsupportedFileTypeError,
    assign_pages,
    chunk_text,
    extract,
    extract_text,
)

# --- extract_text ------------------------------------------------------


def test_extract_text_txt() -> None:
    assert extract_text(b"hello world", "notes.txt") == "hello world"


def test_extract_text_markdown() -> None:
    content = "# Title\n\nSome **bold** text."
    assert extract_text(content.encode("utf-8"), "notes.md") == content


def test_extract_text_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(b"binary data", "archive.zip")


def test_extract_text_pdf_smoke() -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)

    # A blank page has no text layer; this only confirms the PDF branch
    # parses successfully and returns a string instead of raising.
    result = extract_text(buffer.getvalue(), "blank.pdf")
    assert isinstance(result, str)


# --- extract: page boundaries -------------------------------------------


def _blank_pdf(num_pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_extract_pdf_records_one_page_start_per_page() -> None:
    extraction = extract(_blank_pdf(3), "doc.pdf")

    assert extraction.page_starts is not None
    # Blank pages contribute no text, so each page start advances only by the
    # 2-char "\n\n" separator: page 1 at 0, page 2 at 2, page 3 at 4.
    assert extraction.page_starts == [0, 2, 4]


def test_extract_text_has_no_pages() -> None:
    extraction = extract(b"plain text body", "notes.txt")

    assert extraction.text == "plain text body"
    assert extraction.page_starts is None


# --- assign_pages --------------------------------------------------------


def _at(char_start: int, index: int = 0) -> Chunk:
    return Chunk(text="x", index=index, char_start=char_start, char_end=char_start + 1)


def test_assign_pages_maps_offset_to_one_based_page() -> None:
    page_starts = [0, 100, 250]
    chunks = [_at(0, 0), _at(120, 1), _at(250, 2), _at(400, 3)]

    assign_pages(chunks, page_starts)

    # offset 0 -> p1, 120 -> p2, 250 (exact start) -> p3, 400 -> p3.
    assert [c.page for c in chunks] == [1, 2, 3, 3]


def test_assign_pages_leaves_page_none_for_unpaginated() -> None:
    chunks = [_at(0, 0), _at(50, 1)]

    assign_pages(chunks, None)

    assert all(c.page is None for c in chunks)


# --- chunk_text: empty / short input ------------------------------------


def test_chunk_text_empty_input() -> None:
    assert chunk_text("", chunk_size=100, overlap=20) == []


def test_chunk_text_whitespace_only_input() -> None:
    assert chunk_text("   \n\n  ", chunk_size=100, overlap=20) == []


def test_chunk_text_very_short_input_single_chunk() -> None:
    text = "Just one short sentence."
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].index == 0
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(text)


# --- chunk_text: general structure ---------------------------------------


def test_chunk_text_indices_are_sequential() -> None:
    text = " ".join(f"Sentence number {i} here." for i in range(200))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_chunk_text_respects_char_offsets() -> None:
    text = " ".join(f"Sentence number {i} here." for i in range(200))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text


# --- chunk_text: boundary handling ----------------------------------------


def test_chunk_text_never_cuts_mid_word() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=80, overlap=20)
    for chunk in chunks:
        before = text[chunk.char_start - 1] if chunk.char_start > 0 else " "
        after = text[chunk.char_end] if chunk.char_end < len(text) else " "
        assert before.isspace()
        assert after.isspace()


def test_chunk_text_oversized_atom_falls_back_to_word_split() -> None:
    # No sentence/paragraph boundaries at all, and longer than chunk_size:
    # forces the word-level fallback splitter.
    text = " ".join(f"supercalifragilisticexpialidocious{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=60, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert not chunk.text.startswith(" ")
        assert not chunk.text.endswith(" ")


# --- chunk_text: overlap ---------------------------------------------------


def test_chunk_text_overlap_between_consecutive_chunks() -> None:
    text = ". ".join(f"This is sentence {i}" for i in range(100)) + "."
    chunks = chunk_text(text, chunk_size=150, overlap=50)
    assert len(chunks) > 1
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.char_start < prev.char_end
        overlap_text = text[curr.char_start : prev.char_end]
        assert overlap_text in prev.text
        assert overlap_text in curr.text


def test_chunk_text_no_overlap_when_zero() -> None:
    text = ". ".join(f"This is sentence {i}" for i in range(100)) + "."
    chunks = chunk_text(text, chunk_size=150, overlap=0)
    assert len(chunks) > 1
    for prev, curr in zip(chunks, chunks[1:]):
        assert curr.char_start >= prev.char_end


# --- chunk_text: invalid arguments -----------------------------------------


def test_chunk_text_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=10, overlap=10)


def test_chunk_text_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=0, overlap=0)
