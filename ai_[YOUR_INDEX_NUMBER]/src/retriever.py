"""
Retriever Module
----------------
Part B Extension: Hybrid Search (semantic + keyword).

Pure semantic search has two documented failure cases in this domain:

  Failure 1 – Acronym / abbreviation blindness
    Query: "NDC votes in Ashanti"
    Problem: 'NDC' is an uncommon token; its dense vector lands near many
             party names, so the semantic score doesn't isolate it well.
    Fix: TF-IDF keyword component rescues exact token matches.

  Failure 2 – Year-specific numerical queries
    Query: "2016 election total valid votes"
    Problem: The dense model treats '2016' and '2024' as nearly synonymous
             since both appear in similar election contexts.
    Fix: Keyword score strongly prefers chunks containing the exact year.

Hybrid formula:
    combined = α × semantic_score + (1 − α) × keyword_score

  α is controllable from the Streamlit sidebar (default 0.7).
  Higher α → more semantic; lower α → more keyword-driven.

Author : [YOUR_FULL_NAME]  ([YOUR_INDEX_NUMBER])
"""

import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .vector_store import VectorStore
from .embedder     import Embedder

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.7
DEFAULT_TOP_K = 5


class Retriever:
    def __init__(
        self,
        store:   VectorStore,
        embedder: Embedder,
        alpha:   float = DEFAULT_ALPHA,
        top_k:   int   = DEFAULT_TOP_K,
    ):
        self.store    = store
        self.embedder = embedder
        self.alpha    = alpha
        self.top_k    = top_k

        # ── Build TF-IDF index over all chunk texts ───────────────────────────
        # (Done once at init; used for keyword scoring on every query)
        corpus = [c["chunk_text"] for c in store.chunks]
        logger.info("Building TF-IDF index over %d chunks…", len(corpus))

        if corpus:
            self.tfidf = TfidfVectorizer(
                max_features=40_000,
                ngram_range=(1, 2),      # unigrams + bigrams for better matching
                stop_words="english",
                sublinear_tf=True,       # log(1+tf) dampens very frequent terms
            )
            self.tfidf_matrix = self.tfidf.fit_transform(corpus)  # sparse (N, vocab)
            logger.info("TF-IDF vocab: %d terms", len(self.tfidf.vocabulary_))
        else:
            self.tfidf        = None
            self.tfidf_matrix = None
            logger.warning("Empty corpus – TF-IDF index not built.")

        # Pre-build id→chunk lookup for O(1) access in retrieve()
        self._id_to_chunk: dict[str, dict] = {
            c["chunk_id"]: c for c in store.chunks
        }

    # ── Semantic search ───────────────────────────────────────────────────────

    def _semantic_search(self, query: str) -> dict[str, float]:
        """
        Return {chunk_id: cosine_score} for top candidates via FAISS.
        Over-fetches (3×top_k) so the hybrid re-ranking has more to work with.
        """
        q_vec = self.embedder.embed_query(query)             # (1, dim)
        raw   = self.store.search(q_vec, top_k=self.top_k * 3)
        return {chunk["chunk_id"]: score for chunk, score in raw}

    # ── Keyword search ────────────────────────────────────────────────────────

    def _keyword_search(self, query: str) -> dict[str, float]:
        """
        Return {chunk_id: tfidf_cosine_score} for top keyword-matching chunks.
        """
        if self.tfidf is None or self.tfidf_matrix is None:
            return {}

        q_vec = self.tfidf.transform([query])                # sparse (1, vocab)
        # cosine_similarity returns dense (1, N); take first row
        sims  = cosine_similarity(q_vec, self.tfidf_matrix)[0]

        # Collect only chunks with a non-trivial keyword score
        top_n    = min(self.top_k * 3, len(sims))
        top_idxs = np.argsort(sims)[::-1][:top_n]

        return {
            self.store.chunks[int(i)]["chunk_id"]: float(sims[i])
            for i in top_idxs
            if sims[i] > 1e-6           # skip zero-score chunks
        }

    # ── Hybrid retrieval ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        alpha:           float | None       = None,
        top_k:           int   | None       = None,
        feedback_boosts: dict[str, float] | None = None,
    ) -> list[dict]:
        """
        Hybrid retrieval: combine semantic and keyword scores.

        Args:
            query           : natural-language query string
            alpha           : override default semantic weight (0–1)
            top_k           : override default result count
            feedback_boosts : {chunk_id: boost_delta} from FeedbackStore

        Returns:
            List of result dicts, sorted by combined_score descending:
              {
                "chunk":          dict,   # full chunk metadata
                "semantic_score": float,
                "keyword_score":  float,
                "combined_score": float,
              }
        """
        alpha  = alpha  if alpha  is not None else self.alpha
        top_k  = top_k  if top_k  is not None else self.top_k
        boosts = feedback_boosts or {}

        sem_scores = self._semantic_search(query)
        kwd_scores = self._keyword_search(query)

        # Union of all candidate chunk IDs
        candidate_ids = set(sem_scores) | set(kwd_scores)

        results = []
        for cid in candidate_ids:
            chunk = self._id_to_chunk.get(cid)
            if chunk is None:
                continue           # safety guard against stale IDs

            sem      = sem_scores.get(cid, 0.0)
            kwd      = kwd_scores.get(cid, 0.0)
            combined = alpha * sem + (1.0 - alpha) * kwd

            # Apply feedback boost/penalty (Part G)
            boost    = boosts.get(cid, 0.0)
            combined = max(0.0, min(1.0, combined + boost))

            results.append(
                {
                    "chunk":          chunk,
                    "semantic_score": round(sem,      4),
                    "keyword_score":  round(kwd,      4),
                    "combined_score": round(combined, 4),
                }
            )

        # Sort descending by combined score, return top-k
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        top_results = results[:top_k]

        logger.info(
            "Retrieval: query=%r → %d results (α=%.2f)",
            query[:60], len(top_results), alpha,
        )
        for r in top_results:
            logger.debug(
                "  [%.4f sem=%.4f kwd=%.4f] %s",
                r["combined_score"], r["semantic_score"],
                r["keyword_score"],  r["chunk"].get("chunk_id"),
            )

        return top_results
