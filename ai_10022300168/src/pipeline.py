"""
RAG Pipeline Module
-------------------
Part D: Full pipeline – Query → Retrieval → Context Selection → Prompt → LLM → Response.

Every stage emits log entries that are:
  1. Written to logs/pipeline_<date>.log  (file handler added below)
  2. Returned in the pipeline result dict so the Streamlit UI can show them

Pipeline stages:
  [STAGE 1] Query received           – validate, log
  [STAGE 2] Hybrid retrieval         – semantic + keyword, apply feedback boosts
  [STAGE 3] Context selection        – build numbered context string, manage budget
  [STAGE 4] Prompt construction      – fill TEMPLATE_V3
  [STAGE 5] LLM inference            – HF Inference API call
  [STAGE 6] Response + query log     – append to logs/query_log.jsonl

Author : Michael Nana Kwame Osei-Dei  (10022300168)
"""

import os
import json
import logging
import datetime
from typing import Any

from .data_loader    import load_all_documents
from .chunker        import chunk_documents
from .embedder       import Embedder
from .vector_store   import VectorStore
from .retriever      import Retriever
from .prompt_builder import build_prompt
from .llm_client     import query_llm
from .feedback       import FeedbackStore

logger = logging.getLogger(__name__)

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
LOG_DIR      = os.path.join(PROJECT_ROOT, "logs")


# ── File logging setup ─────────────────────────────────────────────────────────

def _ensure_file_logging() -> None:
    """Add a date-stamped file handler to the root logger (idempotent)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"pipeline_{datetime.date.today()}.log")

    root = logging.getLogger()
    # Avoid adding duplicate handlers on Streamlit reruns
    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path)
        for h in root.handlers
    ):
        return

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
    ))
    root.addHandler(fh)
    if root.level == logging.WARNING:      # don't downgrade deliberately set levels
        root.setLevel(logging.DEBUG)


_ensure_file_logging()


# ── Knowledge-base builder ─────────────────────────────────────────────────────

def build_knowledge_base(strategy: str = "fixed") -> tuple[VectorStore, Embedder]:
    """
    Build (or load from cache) the FAISS index for the given chunking *strategy*.

    Flow:
      • If a saved index exists for *strategy* → load it (fast path)
      • Otherwise → download data, chunk, embed, index, save (slow path, one-time)
    """
    embedder = Embedder()

    if VectorStore.exists(strategy):
        logger.info("Loading cached FAISS index for strategy='%s'…", strategy)
        store = VectorStore.load(strategy)
        return store, embedder

    logger.info("Building knowledge base from scratch (strategy='%s')…", strategy)

    # STAGE 1 – Load
    logger.info("=== KB STAGE 1/4: DATA LOADING ===")
    raw_docs = load_all_documents()
    logger.info("Raw documents: %d", len(raw_docs))

    # STAGE 2 – Chunk
    logger.info("=== KB STAGE 2/4: CHUNKING ===")
    chunks = chunk_documents(raw_docs, strategy=strategy)
    logger.info("Chunks created: %d", len(chunks))

    # STAGE 3 – Embed
    logger.info("=== KB STAGE 3/4: EMBEDDING ===")
    embeddings = embedder.embed_chunks(chunks)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # STAGE 4 – Index & persist
    logger.info("=== KB STAGE 4/4: INDEXING ===")
    store = VectorStore(dim=embedder.dim)
    store.add_embeddings(embeddings, chunks)
    store.save(strategy)
    logger.info("Knowledge base ready.")

    return store, embedder


# ── Pipeline class ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Instantiated once per chunking strategy; cached by Streamlit's
    @st.cache_resource so the FAISS index and TF-IDF matrix are not rebuilt
    on every UI interaction.
    """

    def __init__(
        self,
        strategy: str   = "fixed",
        alpha:    float = 0.7,
        top_k:    int   = 5,
    ):
        logger.info(
            "RAGPipeline init: strategy=%s, α=%.2f, k=%d",
            strategy, alpha, top_k,
        )
        self.strategy = strategy
        self.store, self.embedder = build_knowledge_base(strategy)
        self.retriever = Retriever(
            self.store, self.embedder, alpha=alpha, top_k=top_k
        )
        self.feedback  = FeedbackStore()
        self.top_k     = top_k

    def run(
        self,
        query: str,
        alpha: float | None = None,
        top_k: int   | None = None,
    ) -> dict[str, Any]:
        """
        Execute the full RAG pipeline for *query*.

        Returns:
            {
              "query"       : str,
              "retrieved"   : list[dict],   # all retrieved results with scores
              "used_chunks" : list[dict],   # subset included in the prompt
              "prompt"      : str,          # exact prompt sent to LLM
              "llm_result"  : dict,         # model id, response text, error
              "pipeline_log": list[str],    # human-readable stage log
            }
        """
        pipeline_log: list[str] = []
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        def _log(msg: str) -> None:
            pipeline_log.append(msg)
            logger.info(msg)

        # ── STAGE 1: Query ────────────────────────────────────────────────────
        _log(f"[{ts}] STAGE 1 – QUERY: {query!r}")
        query = query.strip()
        if not query:
            return _empty_result("Empty query provided.")

        # ── STAGE 2: Retrieval ────────────────────────────────────────────────
        _log("STAGE 2 – RETRIEVAL")
        boosts    = self.feedback.get_boosts()
        retrieved = self.retriever.retrieve(
            query,
            alpha           = alpha,
            top_k           = top_k or self.top_k,
            feedback_boosts = boosts,
        )
        _log(f"  Retrieved {len(retrieved)} chunks")
        for r in retrieved:
            _log(
                f"  ├─ [{r['combined_score']:.4f}] "
                f"sem={r['semantic_score']:.3f} kwd={r['keyword_score']:.3f} "
                f"→ {r['chunk'].get('chunk_id','?')} "
                f"({r['chunk'].get('source','?')})"
            )

        # ── STAGE 3 + 4: Context selection & prompt ───────────────────────────
        _log("STAGE 3 – CONTEXT SELECTION & PROMPT BUILD")
        prompt, used_chunks = build_prompt(query, retrieved)
        _log(f"  Prompt: {len(prompt)} chars | {len(used_chunks)}/{len(retrieved)} snippets used")

        # ── STAGE 5: LLM inference ────────────────────────────────────────────
        _log("STAGE 5 – LLM INFERENCE")
        llm_result = query_llm(prompt)
        response   = llm_result.get("response", "")
        _log(
            f"  Model: {llm_result.get('model')} | "
            f"Response: {len(response.split())} words | "
            f"Error: {llm_result.get('error')}"
        )

        # ── STAGE 6: Persist query log ────────────────────────────────────────
        _append_query_log(
            {
                "timestamp":      ts,
                "query":          query,
                "strategy":       self.strategy,
                "n_retrieved":    len(retrieved),
                "n_used":         len(used_chunks),
                "prompt_chars":   len(prompt),
                "llm_model":      llm_result.get("model"),
                "response_words": len(response.split()),
                "error":          llm_result.get("error"),
            }
        )

        return {
            "query":        query,
            "retrieved":    retrieved,
            "used_chunks":  used_chunks,
            "prompt":       prompt,
            "llm_result":   llm_result,
            "pipeline_log": pipeline_log,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_result(reason: str) -> dict:
    return {
        "query":        "",
        "retrieved":    [],
        "used_chunks":  [],
        "prompt":       "",
        "llm_result":   {"response": reason, "model": "none", "error": reason},
        "pipeline_log": [reason],
    }


def _append_query_log(entry: dict) -> None:
    """Append a JSONL entry to logs/query_log.jsonl."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "query_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
