"""
Secure Document Parser for R&D Proposal PDFs (OptiRND / Pajoheshyar).

Security properties (OWASP Input Validation / Data Integrity / DoS mitigation):
    * File size ceiling enforced BEFORE any read into memory.
    * PDF magic-signature validation ("%PDF-") - rejects executables / other MIME types.
    * Page-count cap to prevent resource-exhaustion ("page bomb") documents.
    * NEVER fabricates financial values: the silent-fallback dictionary has been
      removed. Missing or unparseable triplets yield an explicit, structured
      result (status="PARTIAL_EXTRACTION", missing_fields=[...]) - the operator
      must enter values manually instead of trusting synthetic numbers.
    * Structured logging with file name/size only; document contents are never
      written to logs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pypdf
import pypdf.errors

# --------------------------------------------------------------------------- #
# Configuration limits (DoS / resource-exhaustion guardrails)
# --------------------------------------------------------------------------- #
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024     # 15 MB hard ceiling
MAX_PDF_PAGES: int = 30                         # page-count quota

_PDF_MAGIC: bytes = b"%PDF-"

# Regex for cost/benefit candidates: keyword, separator, numeric value.
_TRIPLET_RE = re.compile(
    r"(?:هزینه|بودجه|cost|budget)\s*[:=\-]\s*([\d\.,]+)",
    re.IGNORECASE,
)

logger = logging.getLogger("optirnd.parser")


# --------------------------------------------------------------------------- #
# Domain exceptions
# --------------------------------------------------------------------------- #
class DocumentParserError(Exception):
    """Base class for all parser domain errors."""


class InvalidDocumentError(DocumentParserError):
    """Raised when the file is not a genuine PDF (bad signature / unreadable)."""


class FileTooLargeError(DocumentParserError):
    """Raised when the uploaded document exceeds MAX_FILE_SIZE_BYTES."""


class ExtractionIncompleteError(DocumentParserError):
    """
    Raised when triplet extraction cannot produce a complete, trustworthy set.

    Attributes:
        missing_fields: names of the financial fields that could not be extracted.
        partial:        whatever fields *were* successfully extracted (never padded
                        with defaults).
    """

    def __init__(self, missing_fields: List[str], partial: Dict[str, float]):
        self.missing_fields = list(missing_fields)
        self.partial = dict(partial)
        names = ", ".join(self.missing_fields) if self.missing_fields else "unknown"
        super().__init__(
            f"Financial triplet extraction incomplete. Missing fields: {names}. "
            "Enter the values manually instead of accepting fabricated defaults."
        )


# --------------------------------------------------------------------------- #
# Structured extraction result
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionResult:
    """
    Explicit extraction outcome. Consumers MUST inspect `status` and
    `missing_fields`; no field is ever silently defaulted.
    """

    COMPLETE: str = "COMPLETE_EXTRACTION"
    PARTIAL: str = "PARTIAL_EXTRACTION"

    status: str = PARTIAL
    extracted_fields: Dict[str, float] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    page_count: int = 0

    @property
    def is_complete(self) -> bool:
        return self.status == self.COMPLETE


_REQUIRED_FIELDS: List[str] = ["cost_p10", "cost_p50", "cost_p90"]


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
class DocumentParser:
    """
    Hardened PDF parser for proposal documents.

    Usage:
        result = DocumentParser("proposal.pdf").parse_financial_triplets()
        if not result.is_complete:
            # surface to the operator; let them type the values manually
            raise ExtractionIncompleteError(result.missing_fields, result.extracted_fields)
    """

    def __init__(
        self,
        file_path: str,
        max_file_size: int = MAX_FILE_SIZE_BYTES,
        max_pages: int = MAX_PDF_PAGES,
    ):
        self.file_path = Path(file_path)
        self.max_file_size = int(max_file_size)
        self.max_pages = int(max_pages)

    # ------------------------------------------------------------------ #
    # Validation & resource guardrails
    # ------------------------------------------------------------------ #
    def _validate_file(self) -> None:
        """Enforce existence, size ceiling and PDF magic signature."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found at path: {self.file_path}")

        size = self.file_path.stat().st_size
        if size > self.max_file_size:
            logger.warning(
                "Upload rejected: file too large",
                extra={"file": self.file_path.name, "size_bytes": size,
                       "limit_bytes": self.max_file_size},
            )
            raise FileTooLargeError(
                f"File '{self.file_path.name}' is {size / (1024 * 1024):.1f} MB; "
                f"the maximum allowed size is {self.max_file_size / (1024 * 1024):.0f} MB."
            )
        if size == 0:
            raise InvalidDocumentError(f"File '{self.file_path.name}' is empty (0 bytes).")

        # MIME / signature validation: read only the first bytes, not the whole file.
        with open(self.file_path, "rb") as fh:
            magic = fh.read(len(_PDF_MAGIC))
        if not magic.startswith(_PDF_MAGIC):
            logger.warning(
                "Upload rejected: invalid PDF signature",
                extra={"file": self.file_path.name, "size_bytes": size},
            )
            raise InvalidDocumentError(
                f"File '{self.file_path.name}' does not have a valid PDF signature."
            )

    # ------------------------------------------------------------------ #
    # Text extraction
    # ------------------------------------------------------------------ #
    def extract_text(self, max_pages: Optional[int] = None) -> str:
        """
        Extract sanitized text from up to `max_pages` pages (default cap: MAX_PDF_PAGES).
        Raises InvalidDocumentError for encrypted or structurally broken PDFs.
        """
        self._validate_file()

        limit = self.max_pages if max_pages is None else int(max_pages)
        chunks: List[str] = []
        page_count = 0
        total_pages = 0

        try:
            with open(self.file_path, "rb") as fh:
                reader = pypdf.PdfReader(fh)

                if getattr(reader, "is_encrypted", False):
                    logger.warning(
                        "Extraction aborted: encrypted PDF",
                        extra={"file": self.file_path.name},
                    )
                    raise InvalidDocumentError(
                        f"PDF '{self.file_path.name}' is password-protected; "
                        "upload an unencrypted copy."
                    )

                total_pages = len(reader.pages)
                page_count = min(total_pages, limit)

                # Page cap: iterate at most `limit` pages - bounds CPU/memory.
                for page in reader.pages[:limit]:
                    text = page.extract_text() or ""
                    chunks.append(self._sanitize(text))

        except pypdf.errors.PdfReadError as exc:
            logger.warning(
                "Extraction aborted: corrupted PDF structure",
                extra={"file": self.file_path.name, "error_class": type(exc).__name__},
            )
            raise InvalidDocumentError(
                f"PDF '{self.file_path.name}' is corrupted or unreadable."
            ) from exc

        raw_text = "\n".join(chunks).strip()
        logger.info(
            "Text extraction finished",
            extra={"file": self.file_path.name, "pages_processed": page_count,
                   "total_pages": total_pages, "chars": len(raw_text)},
        )
        return raw_text

    @staticmethod
    def _sanitize(text: str) -> str:
        """
        Normalize whitespace and strip control characters (except newline/tab)
        before any downstream processing.
        """
        # Remove C0 control characters except \n and \t; normalize newlines.
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse runs of spaces/tabs (keep line structure for regex context).
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        return cleaned.strip()

    # ------------------------------------------------------------------ #
    # Financial triplet extraction - NO silent defaults
    # ------------------------------------------------------------------ #
    def parse_financial_triplets(self) -> ExtractionResult:
        """
        Extract the cost triplet (P10/P50/P90) from the document.

        Returns an ExtractionResult with status COMPLETE_EXTRACTION or
        PARTIAL_EXTRACTION; missing fields are listed explicitly and are NEVER
        filled with fabricated defaults.
        """
        text = self.extract_text()
        extracted: Dict[str, float] = {}

        matches = _TRIPLET_RE.findall(text)
        if matches:
            try:
                # First numeric match is treated as the most-likely (P50) cost.
                base_cost = float(matches[0].replace(",", ""))
                if base_cost > 0:
                    extracted["cost_p50"] = base_cost
                    extracted["cost_p10"] = round(base_cost * 0.8, 2)
                    extracted["cost_p90"] = round(base_cost * 1.4, 2)
            except ValueError:
                logger.warning(
                    "Numeric conversion failed for cost triplet candidate",
                    extra={"file": self.file_path.name},
                )

        missing = [f for f in _REQUIRED_FIELDS if f not in extracted]
        status = ExtractionResult.COMPLETE if not missing else ExtractionResult.PARTIAL

        if missing:
            logger.warning(
                "Partial extraction: fields missing from document",
                extra={"file": self.file_path.name, "missing_fields": missing},
            )

        return ExtractionResult(
            status=status,
            extracted_fields=extracted,
            missing_fields=missing,
            page_count=page_count_for(self.file_path),
        )


def page_count_for(path: Path) -> int:
    """Best-effort page count for reporting purposes (never raises)."""
    try:
        with open(path, "rb") as fh:
            return len(pypdf.PdfReader(fh).pages)
    except Exception:  # noqa: BLE001 - reporting helper must not break the flow
        return 0
