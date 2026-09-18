"""
Hardened-parser test suite (pytest) for core/parser.py.

Covers: oversized files, non-PDF MIME/signature, empty files, corrupted PDFs,
page-bomb caps, documents missing financial triplets (NO silent defaults!),
clean extraction, and log hygiene (document contents never logged).
"""
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from core.parser import (  # noqa: E402
    MAX_PDF_PAGES,
    DocumentParser,
    ExtractionIncompleteError,
    ExtractionResult,
    FileTooLargeError,
    InvalidDocumentError,
)


# --------------------------------------------------------------------------- #
# Minimal valid PDF builder (no external generator dependency)
# --------------------------------------------------------------------------- #
def _build_pdf(text: str = "cost: 12000", pages: int = 1) -> bytes:
    """Build a minimal but structurally valid PDF with the given text on each page."""
    n_pages = max(1, pages)
    font_id = 3 + 2 * n_pages
    page_ids = list(range(3, 3 + n_pages))
    content_ids = list(range(3 + n_pages, 3 + 2 * n_pages))

    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1", errors="replace")

    objs = {}
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objs[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
    for pid, cid in zip(page_ids, content_ids):
        objs[pid] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {cid} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode()
        objs[cid] = b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
    objs[font_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid in sorted(objs):
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + objs[oid] + b"\nendobj\n"
    xref_pos = len(out)
    count = max(objs) + 1
    out += f"xref\n0 {count}\n".encode()
    out += b"0000000000 65535 f \n"
    for oid in range(1, count):
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return bytes(out)


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


# --------------------------------------------------------------------------- #
# Clean extraction
# --------------------------------------------------------------------------- #
def test_clean_extraction(tmp_path):
    pdf = _write(tmp_path, "proposal.pdf", _build_pdf("cost: 12000"))
    result = DocumentParser(str(pdf)).parse_financial_triplets()

    assert result.is_complete
    assert result.status == ExtractionResult.COMPLETE
    assert result.extracted_fields["cost_p50"] == 12000.0
    assert result.extracted_fields["cost_p10"] == 9600.0   # 0.8x
    assert result.extracted_fields["cost_p90"] == 16800.0  # 1.4x
    assert result.missing_fields == []


def test_extract_text_returns_content(tmp_path):
    pdf = _write(tmp_path, "doc.pdf", _build_pdf("cost: 5000"))
    text = DocumentParser(str(pdf)).extract_text()
    assert "cost: 5000" in text


# --------------------------------------------------------------------------- #
# Missing triplets -> explicit partial result, ZERO fabricated values
# --------------------------------------------------------------------------- #
def test_missing_triplets_partial_no_fabricated_defaults(tmp_path):
    pdf = _write(tmp_path, "empty.pdf", _build_pdf("No financial data in this document."))
    result = DocumentParser(str(pdf)).parse_financial_triplets()

    assert not result.is_complete
    assert result.status == ExtractionResult.PARTIAL
    assert set(result.missing_fields) == {"cost_p10", "cost_p50", "cost_p90"}
    # The old parser injected 8000/10000/14000 defaults - must NEVER happen now.
    assert result.extracted_fields == {}


def test_extraction_incomplete_error_carries_fields(tmp_path):
    pdf = _write(tmp_path, "empty.pdf", _build_pdf("nothing useful"))
    result = DocumentParser(str(pdf)).parse_financial_triplets()

    with pytest.raises(ExtractionIncompleteError) as excinfo:
        if not result.is_complete:
            raise ExtractionIncompleteError(result.missing_fields, result.extracted_fields)
    assert set(excinfo.value.missing_fields) == {"cost_p10", "cost_p50", "cost_p90"}
    assert excinfo.value.partial == {}


# --------------------------------------------------------------------------- #
# Resource guardrails (DoS mitigation)
# --------------------------------------------------------------------------- #
def test_file_too_large_rejected(tmp_path):
    pdf = _write(tmp_path, "big.pdf", _build_pdf())
    parser = DocumentParser(str(pdf), max_file_size=10)  # 10-byte ceiling
    with pytest.raises(FileTooLargeError):
        parser.extract_text()


def test_non_pdf_signature_rejected(tmp_path):
    exe = _write(tmp_path, "malware.pdf", b"MZ\x90\x00" + b"A" * 64)  # PE executable header
    with pytest.raises(InvalidDocumentError):
        DocumentParser(str(exe)).extract_text()


def test_empty_file_rejected(tmp_path):
    empty = _write(tmp_path, "empty.pdf", b"")
    with pytest.raises(InvalidDocumentError):
        DocumentParser(str(empty)).extract_text()


def test_corrupted_pdf_rejected(tmp_path):
    corrupt = _write(tmp_path, "corrupt.pdf", b"%PDF-1.4 this is not a real pdf body")
    with pytest.raises(InvalidDocumentError):
        DocumentParser(str(corrupt)).extract_text()


def test_page_bomb_capped(tmp_path):
    pdf = _write(tmp_path, "bomb.pdf", _build_pdf("cost: 12000", pages=40))
    parser = DocumentParser(str(pdf), max_pages=5)
    text = parser.extract_text()
    assert text.count("cost: 12000") == 5  # only 5 of 40 pages processed


def test_default_page_cap_constant():
    assert MAX_PDF_PAGES == 30


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocumentParser(str(tmp_path / "ghost.pdf")).extract_text()


# --------------------------------------------------------------------------- #
# Sanitation & logging hygiene
# --------------------------------------------------------------------------- #
def test_sanitize_strips_control_characters():
    dirty = "cost:\x00 120\x0700\x1b[31m text\r\nmore"
    clean = DocumentParser._sanitize(dirty)
    assert "\x00" not in clean and "\x07" not in clean and "\x1b" not in clean
    assert "more" in clean


def test_document_contents_never_logged(tmp_path, caplog):
    pdf = _write(tmp_path, "secret.pdf", _build_pdf("cost: 98765"))
    with caplog.at_level("DEBUG", logger="optirnd.parser"):
        DocumentParser(str(pdf)).parse_financial_triplets()
    # The sensitive numeric payload must not appear in any log record.
    assert "98765" not in caplog.text


def test_structured_log_context_on_rejection(tmp_path, caplog):
    exe = _write(tmp_path, "bad.pdf", b"MZ" + b"B" * 32)
    with caplog.at_level("WARNING", logger="optirnd.parser"):
        with pytest.raises(InvalidDocumentError):
            DocumentParser(str(exe)).extract_text()
    assert any("invalid PDF signature" in r.message for r in caplog.records)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
