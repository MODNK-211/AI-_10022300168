"""
Data Loader Module
------------------
Handles downloading, parsing, and cleaning of both knowledge-base sources:

  1. Ghana Election Results (CSV)  – tabular political/constituency data
  2. Ghana 2025 Budget Statement (PDF) – Ministry of Finance policy document

All HTTP fetches use only the `requests` library (no framework wrappers).
Downloaded files are cached in data/ so subsequent runs skip the download.

Author : [YOUR_FULL_NAME]  ([YOUR_INDEX_NUMBER])
"""

import os
import re
import logging

import requests
import pandas as pd
import PyPDF2

logger = logging.getLogger(__name__)

# ── Project-root-relative paths ───────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# ── Dataset URLs ─────────────────────────────────────────────────────────────
CSV_URL = (
    "https://github.com/GodwinDansoAcity/acitydataset/raw/main/"
    "Ghana_Election_Result.csv"
)
PDF_URL = (
    "https://mofep.gov.gh/sites/default/files/budget-statements/"
    "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
)

CSV_LOCAL = os.path.join(DATA_DIR, "Ghana_Election_Result.csv")
PDF_LOCAL = os.path.join(DATA_DIR, "2025-Budget-Statement.pdf")


# ── Generic file downloader ───────────────────────────────────────────────────

def _download(url: str, dest: str, timeout: int = 90) -> str:
    """Download *url* to *dest*; skip if already cached. Returns local path."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        logger.info("Cache hit: %s", dest)
        return dest

    logger.info("Downloading: %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (AcityBot/1.0; student project)"}
    resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()

    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)

    size_kb = os.path.getsize(dest) / 1024
    logger.info("Saved %.1f KB → %s", size_kb, dest)
    return dest


# ── CSV Loader ────────────────────────────────────────────────────────────────

def load_election_csv() -> list[dict]:
    """
    Load, clean, and convert the Ghana Election Results CSV to document dicts.

    Cleaning steps applied (manual – no pipeline):
      • Strip whitespace from column headers
      • Normalise to UPPER_SNAKE column names for consistency
      • Drop rows that are entirely empty
      • Fill NaN cells with the string 'Unknown'
      • Convert each row to a human-readable natural-language sentence

    Each row becomes ONE document so that retrieval operates at row granularity
    before chunking further splits long rows.
    """
    _download(CSV_URL, CSV_LOCAL)

    df = pd.read_csv(CSV_LOCAL, encoding="utf-8", on_bad_lines="skip")
    logger.info("CSV raw shape: %s | columns: %s", df.shape, list(df.columns))

    # ── Cleaning ──────────────────────────────────────────────────────────────
    df.columns = [
        re.sub(r"\s+", "_", c.strip().upper()) for c in df.columns
    ]
    df.dropna(how="all", inplace=True)
    df.fillna("Unknown", inplace=True)

    # Normalise all string columns: strip extra spaces, fix encoding artefacts
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    logger.info("CSV after cleaning: %d rows", len(df))

    # ── Convert rows → text documents ────────────────────────────────────────
    documents: list[dict] = []
    for idx, row in df.iterrows():
        # Build a readable sentence: "Field: Value; Field: Value …"
        fields = "; ".join(
            f"{col.replace('_', ' ').title()}: {val}"
            for col, val in row.items()
            if str(val).strip() not in ("", "Unknown")
        )
        text = f"Ghana Election Record — {fields}."

        documents.append(
            {
                "id":        f"csv_{idx}",
                "text":      text,
                "source":    "Ghana_Election_Result.csv",
                "row_index": int(idx),
            }
        )

    logger.info("CSV: produced %d documents", len(documents))
    return documents


# ── PDF Loader ────────────────────────────────────────────────────────────────

def load_budget_pdf() -> list[dict]:
    """
    Load the Ghana 2025 Budget Statement PDF, one document per page.

    Cleaning applied:
      • Collapse all whitespace runs to single spaces
      • Repair hyphenated line-breaks (word- \\n break → wordbreak)
      • Remove non-printable / non-ASCII characters
      • Skip pages with fewer than 80 characters (mostly images/headers)
    """
    try:
        _download(PDF_URL, PDF_LOCAL, timeout=120)
    except Exception as exc:
        logger.warning("PDF download failed: %s", exc)
        if not (os.path.exists(PDF_LOCAL) and os.path.getsize(PDF_LOCAL) > 0):
            raise RuntimeError(
                "Cannot fetch Budget PDF and no local cache found.\n"
                f"Manually place the PDF at: {PDF_LOCAL}"
            ) from exc
        logger.info("Using cached PDF at %s", PDF_LOCAL)

    documents: list[dict] = []
    with open(PDF_LOCAL, "rb") as fh:
        reader      = PyPDF2.PdfReader(fh)
        total_pages = len(reader.pages)
        logger.info("PDF: %d pages", total_pages)

        for page_num, page in enumerate(reader.pages, start=1):
            raw  = page.extract_text() or ""
            text = _clean_pdf_text(raw)

            if len(text) < 80:          # skip near-empty pages (images, ToC lines)
                continue

            documents.append(
                {
                    "id":          f"pdf_p{page_num}",
                    "text":        text,
                    "source":      "2025-Budget-Statement.pdf",
                    "page":        page_num,
                    "total_pages": total_pages,
                }
            )

    logger.info("PDF: produced %d page documents", len(documents))
    return documents


def _clean_pdf_text(text: str) -> str:
    """Remove common PDF extraction artefacts while preserving meaning."""
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)   # fix hyphenation
    text = re.sub(r"\s+", " ", text)                       # collapse whitespace
    text = re.sub(r"[^\x20-\x7E]", " ", text)             # strip non-ASCII
    return text.strip()


# ── Combined entry point ──────────────────────────────────────────────────────

def load_all_documents() -> list[dict]:
    """Load and return documents from both sources combined."""
    csv_docs = load_election_csv()
    pdf_docs = load_budget_pdf()
    combined = csv_docs + pdf_docs
    logger.info("Total raw documents loaded: %d", len(combined))
    return combined
