"""
Chunker Module
--------------
Implements TWO distinct chunking strategies from scratch – no framework.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY A – Fixed-Size Sliding Window
  Slides a character window of CHUNK_SIZE chars forward by (CHUNK_SIZE −
  OVERLAP) at each step, creating uniform chunks that partially overlap.

  Pros : Predictable chunk sizes; easy to tune; good for tabular/uniform text
         (CSV rows turned into prose sentences benefit from this).
  Cons : May split mid-sentence, breaking semantic continuity.

  Chosen defaults:
    CHUNK_SIZE = 500 chars  (~100 tokens in English; well within the 256-
                             token limit of all-MiniLM-L6-v2 after encoding)
    OVERLAP    = 75 chars   (~15% overlap prevents context loss at boundaries)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRATEGY B – Sentence-Aware Grouping
  Groups consecutive sentences (detected by terminal punctuation + capital
  letter) into chunks until a soft character target or a max sentence count
  is reached.

  Pros : Semantically coherent boundaries; better for narrative/policy text
         (Budget Statement paragraphs benefit from keeping sentences together).
  Cons : Variable chunk sizes can stress the embedding model; slower to build.

  Chosen defaults:
    SENT_TARGET = 450 chars  (keeps most chunks under 256 tokens)
    MAX_SENTS   = 5 sentences per chunk

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Comparative impact (documented in docs/experiment_log_partC.txt):
  • Fixed chunks retrieved slightly better for exact CSV field lookups.
  • Sentence chunks retrieved better for multi-sentence budget policy questions.
  • Hybrid search with sentence chunks produced the most coherent answers.

Author : Michael Nana Kwame Osei-Dei  (10022300168)
"""

import re
from typing import Literal

# ── Hyper-parameters ──────────────────────────────────────────────────────────
FIXED_CHUNK_SIZE  = 500   # characters per window
FIXED_OVERLAP     = 75    # characters of overlap between windows
SENT_TARGET_CHARS = 450   # soft upper limit per sentence chunk (chars)
SENT_MAX_SENTS    = 5     # hard max sentences per chunk

# Regex: sentence boundary = terminal punctuation followed by whitespace + capital
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\(])")


# ── Strategy A: Fixed-size sliding window ────────────────────────────────────

def fixed_size_chunks(
    doc: dict,
    size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_OVERLAP,
) -> list[dict]:
    """
    Chunk *doc* text with a fixed sliding character window.

    Every output dict inherits all keys from *doc* plus:
      chunk_id       – unique identifier  "<doc_id>_fixed_<n>"
      chunk_text     – the text slice for this chunk
      chunk_strategy – always "fixed"
      chunk_start    – character start offset in original text
      chunk_end      – character end offset in original text
    """
    text   = doc.get("text", "")
    chunks = []
    start  = 0
    n      = 0

    while start < len(text):
        end        = min(start + size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:                          # skip empty/whitespace-only slices
            chunks.append(
                {
                    **doc,                      # inherit all source metadata
                    "chunk_id":       f"{doc['id']}_fixed_{n}",
                    "chunk_text":     chunk_text,
                    "chunk_strategy": "fixed",
                    "chunk_start":    start,
                    "chunk_end":      end,
                }
            )
            n += 1

        start += size - overlap                 # slide forward with overlap

    return chunks


# ── Strategy B: Sentence-aware chunking ──────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Split *text* into a list of sentences using a lightweight regex.
    Falls back to fixed splitting for very long 'sentences' (run-on lines
    common in PDF extracts).
    """
    raw_sents = _SENT_RE.split(text.strip())
    sentences = []
    for s in raw_sents:
        s = s.strip()
        if not s:
            continue
        # If a 'sentence' is longer than FIXED_CHUNK_SIZE it is probably a
        # run-on PDF line – break it further at fixed size.
        if len(s) > FIXED_CHUNK_SIZE:
            for i in range(0, len(s), FIXED_CHUNK_SIZE):
                part = s[i : i + FIXED_CHUNK_SIZE].strip()
                if part:
                    sentences.append(part)
        else:
            sentences.append(s)
    return sentences


def sentence_aware_chunks(
    doc: dict,
    target: int = SENT_TARGET_CHARS,
    max_sents: int = SENT_MAX_SENTS,
) -> list[dict]:
    """
    Chunk *doc* by grouping sentences up to *target* characters or *max_sents*.

    Every output dict inherits all keys from *doc* plus:
      chunk_id        – "<doc_id>_sent_<n>"
      chunk_text      – joined sentences
      chunk_strategy  – always "sentence"
      sentence_count  – number of sentences in this chunk
    """
    sentences = _split_into_sentences(doc.get("text", ""))
    chunks    = []
    current:  list[str] = []
    cur_len   = 0
    n         = 0

    for sent in sentences:
        would_exceed = (cur_len + len(sent) > target) and bool(current)
        at_max       = len(current) >= max_sents

        if would_exceed or at_max:
            # Emit the accumulated sentences as one chunk
            joined = " ".join(current).strip()
            if joined:
                chunks.append(
                    {
                        **doc,
                        "chunk_id":       f"{doc['id']}_sent_{n}",
                        "chunk_text":     joined,
                        "chunk_strategy": "sentence",
                        "sentence_count": len(current),
                    }
                )
                n += 1
            current = []
            cur_len = 0

        current.append(sent)
        cur_len += len(sent) + 1      # +1 for the joining space

    # Flush any remaining sentences
    if current:
        joined = " ".join(current).strip()
        if joined:
            chunks.append(
                {
                    **doc,
                    "chunk_id":       f"{doc['id']}_sent_{n}",
                    "chunk_text":     joined,
                    "chunk_strategy": "sentence",
                    "sentence_count": len(current),
                }
            )

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: list[dict],
    strategy: Literal["fixed", "sentence"] = "fixed",
) -> list[dict]:
    """
    Chunk all *documents* using the chosen strategy.

    Args:
        documents : raw document dicts from data_loader
        strategy  : "fixed" (sliding window) or "sentence" (sentence groups)

    Returns:
        Flat list of chunk dicts, each with 'chunk_text' and 'chunk_id'.
    """
    all_chunks: list[dict] = []

    for doc in documents:
        if not doc.get("text", "").strip():
            continue  # skip empty documents

        if strategy == "fixed":
            all_chunks.extend(fixed_size_chunks(doc))
        elif strategy == "sentence":
            all_chunks.extend(sentence_aware_chunks(doc))
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Choose 'fixed' or 'sentence'.")

    import logging
    logging.getLogger(__name__).info(
        "Chunked %d docs → %d chunks (strategy=%s)",
        len(documents), len(all_chunks), strategy,
    )
    return all_chunks
