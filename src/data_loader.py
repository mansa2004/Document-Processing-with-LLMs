"""
Data Loading & Preprocessing
Responsible for Task 1 of the assignment:
  - Loading a subset of CUAD contracts (PDF or TXT).
  - Extracting full contract text.
  - Normalizing text (whitespace, encoding artifacts, page-break noise).

CUAD ships contracts as PDFs (`CUAD_v1/full_contract_pdf/...`) alongside a
plain-text mirror (`CUAD_v1/full_contract_txt/...`). This loader accepts both PDF and TXT contracts.

For this assignment, PDFs are processed by default to satisfy the
requirement of extracting contract text directly from PDF files.
TXT files remain supported for debugging or custom datasets.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class Contract:
    """A single loaded, normalized contract."""
    contract_id: str
    source_path: str
    text: str
    num_chars: int

# PDF text extraction

def _extract_text_from_pdf(path: Path) -> str:
    """
    Extract text from a PDF using pdfplumber, falling back to PyPDF2 if
    pdfplumber fails (some CUAD PDFs are scanned/odd-encoded and one
    library handles them better than the other).
    """
    text_parts: list[str] = []
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        text = "\n".join(text_parts)
        if text.strip():
            return text
    except Exception as e:
        logger.warning("pdfplumber failed on %s (%s); trying PyPDF2", path.name, e)
        
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        text_parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(text_parts)
    except Exception as e:
        logger.error("PyPDF2 also failed on %s (%s)", path.name, e)
        return ""


# Text normalization
def normalize_text(raw_text: str) -> str:
    """
    Clean up common PDF-extraction artifacts:
      - collapse repeated whitespace/newlines
      - remove page-number-only lines and form-feed characters
      - fix hyphenated line-wrap breaks ("confiden-\ntiality" -> "confidentiality")
      - normalize unicode quotes/dashes
      - strip non-printable control characters
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\x0c", "\n")  # form feed -> newline

    # Normalize curly quotes / dashes to plain ASCII equivalents.
    replacements = {
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
        "\xa0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Re-join words hyphenated across a line break.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Drop lines that are only a page number or "Page X of Y" style footers.
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", stripped, flags=re.IGNORECASE):
            continue
        lines.append(line)
    text = "\n".join(lines)

    # Collapse 3+ blank lines into a single blank line, and runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip remaining non-printable control characters (keep \n).
    text = re.sub(r"[^\x09\x0A\x20-\x7E]", "", text)

    return text.strip()


# Loading

def load_contract(path: Path) -> Contract:
    """
    Load and normalize a single contract file (.pdf or .txt).
    """
    logger.info("Loading contract: %s", path.name)

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw_text = _extract_text_from_pdf(path)
    elif suffix in (".txt", ".text"):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    normalized = normalize_text(raw_text)

    logger.info(
        "Loaded %s (%d characters)",
        path.name,
        len(normalized),
    )

    return Contract(
        contract_id=path.stem,
        source_path=str(path),
        text=normalized,
        num_chars=len(normalized),
    )


def load_contracts(input_dir: Path, limit: int | None = 50) -> Iterator[Contract]:
    """
    Load contracts recursively from the given directory.

    PDF files are preferred to satisfy the assignment requirement
    of extracting contract text directly from PDF files. TXT files
    remain supported as a fallback for custom datasets.

    Contracts are yielded lazily to avoid loading the entire dataset
    into memory.

    Args:
        input_dir:
        Directory containing CUAD contracts.

    limit:
        Maximum number of contracts to load.
        Default is 50, matching the assignment.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Download the CUAD dataset from https://www.atticusprojectai.org/cuad "
             "and point --input_dir to the extracted `full_contract_pdf` folder "
            "(preferred) or `full_contract_txt` folder (supported). "
            "See the README for setup instructions."
        )

    pdf_files = {p.stem: p for p in input_dir.rglob("*.pdf")}
    txt_files = {p.stem: p for p in input_dir.rglob("*.txt")}


    
    all_stems = sorted(set(pdf_files.keys()) | set(txt_files.keys()))
    if limit is not None:
        all_stems = all_stems[:limit]

    for stem in all_stems:
        path = pdf_files.get(stem) or txt_files[stem]
        try:
            yield load_contract(path)
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            continue

def extract_text_from_file(path: Path) -> str:
    """
    Extract normalized text from a PDF or TXT contract.

    Used by semantic search.
    """
    return load_contract(path).text


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Split a contract into overlapping chunks while preserving
    paragraph boundaries whenever possible.

    This prevents important legal clauses from being split across
    LLM context windows and ensures complete clause extraction.

    Args:
       text: 
            Full normalized contract text.

        max_chars:
            Maximum characters per chunk. Chosen conservatively
            to stay below the LLM context window while leaving
            room for prompts and model responses.

        overlap:
            Number of characters shared between consecutive chunks
            to reduce the risk of losing clauses near chunk boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # Start new chunk, carrying overlap from the tail of the previous one.
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{para}" if tail else para


            
            # Handle a single paragraph longer than max_chars by hard-splitting it.
            while len(current) > max_chars:
                # Prefer splitting at a paragraph break.
                split = current.rfind("\n\n", 0, max_chars)
                # Otherwise split at a single newline.
                if split == -1:
                    split = current.rfind("\n", 0, max_chars)

                # Otherwise split at a space.
                if split == -1:
                    split = current.rfind(" ", 0, max_chars)

                 # Fall back to a hard split.
                if split == -1:
                    split = max_chars

                chunks.append(current[:split].strip())
                next_start = split - overlap
                if next_start < 0:
                    next_start = split

                current = current[next_start:].strip()
            

    if current:
        chunks.append(current)

    return chunks
