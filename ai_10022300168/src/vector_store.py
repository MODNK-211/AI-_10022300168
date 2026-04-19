"""
Vector Store Module
-------------------
Manual FAISS index management – no LangChain / LlamaIndex wrappers.

Index type: IndexFlatIP (exact inner-product search)
  • After L2-normalisation, inner product == cosine similarity.
  • "Flat" = brute-force exact search; fine for < 200 k vectors.
  • Deterministic: no approximate-nearest-neighbour randomness.

Persistence:
  Index binary  → data/faiss_index_{strategy}.bin
  Chunk metadata → data/chunks_meta_{strategy}.json

The strategy suffix means each chunking variant gets its own index, so the
user can switch between 'fixed' and 'sentence' in the sidebar without
contaminating cached indices.

Author : Michael Nana Kwame Osei-Dei  (10022300168)
"""

import os
import json
import logging

import numpy as np
import faiss

logger = logging.getLogger(__name__)

_HERE        = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

# Templates for strategy-aware filenames
_INDEX_TMPL = os.path.join(DATA_DIR, "faiss_index_{strategy}.bin")
_META_TMPL  = os.path.join(DATA_DIR, "chunks_meta_{strategy}.json")


def _paths(strategy: str) -> tuple[str, str]:
    return _INDEX_TMPL.format(strategy=strategy), _META_TMPL.format(strategy=strategy)


class VectorStore:
    """Wraps a FAISS IndexFlatIP with the parallel chunk metadata list."""

    def __init__(self, dim: int):
        self.dim    = dim
        # IndexFlatIP: exact inner-product (= cosine after L2 normalisation)
        self.index  = faiss.IndexFlatIP(dim)
        self.chunks: list[dict] = []          # parallel to index rows

    # ── Build ─────────────────────────────────────────────────────────────────

    def add_embeddings(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        """
        Add pre-computed embeddings to the FAISS index.

        Args:
            embeddings : float32 array of shape (N, dim), L2-normalised
            chunks     : list of N chunk dicts (metadata)
        """
        assert embeddings.ndim == 2,              "embeddings must be 2-D"
        assert embeddings.shape[1] == self.dim,   "dimension mismatch"
        assert embeddings.shape[0] == len(chunks),"count mismatch"

        self.index.add(embeddings)       # FAISS copies the vectors internally
        self.chunks.extend(chunks)
        logger.info(
            "VectorStore: %d vectors added; total = %d",
            len(chunks), self.index.ntotal,
        )

    # ── Persist / load ────────────────────────────────────────────────────────

    def save(self, strategy: str = "fixed") -> None:
        """Write index and metadata to strategy-specific files."""
        os.makedirs(DATA_DIR, exist_ok=True)
        idx_path, meta_path = _paths(strategy)

        faiss.write_index(self.index, idx_path)

        # Serialise only JSON-safe fields (drop nested dicts / numpy types)
        safe = []
        for c in self.chunks:
            entry = {
                k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in c.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            safe.append(entry)

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(safe, fh, ensure_ascii=False, indent=2)

        logger.info(
            "Saved index (%d vectors) → %s", self.index.ntotal, idx_path
        )

    @classmethod
    def load(cls, strategy: str = "fixed") -> "VectorStore":
        """Load a previously saved index from disk."""
        idx_path, meta_path = _paths(strategy)
        index  = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as fh:
            chunks = json.load(fh)

        store         = cls(dim=index.d)
        store.index   = index
        store.chunks  = chunks
        logger.info(
            "Loaded index (%d vectors) ← %s", index.ntotal, idx_path
        )
        return store

    @staticmethod
    def exists(strategy: str = "fixed") -> bool:
        """Return True if both index files for *strategy* exist on disk."""
        idx_path, meta_path = _paths(strategy)
        return os.path.exists(idx_path) and os.path.exists(meta_path)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> list[tuple[dict, float]]:
        """
        Return top-k (chunk_dict, cosine_score) pairs, highest score first.

        *query_vec* must be L2-normalised, shape (1, dim).
        Scores are in [−1, 1]; for normalised vectors expect ≈ [0, 1].
        """
        if self.index.ntotal == 0:
            logger.warning("VectorStore.search called on empty index.")
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)   # (1,k) each

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                   # FAISS uses -1 as sentinel
                continue
            results.append((self.chunks[idx], float(score)))

        return results
