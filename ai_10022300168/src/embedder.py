"""
Embedder Module
---------------
Manual embedding pipeline using sentence-transformers.  No LangChain wrappers.

Model: all-MiniLM-L6-v2
  • 22 M parameters → fast CPU inference; good quality for semantic search
  • Output dimension: 384
  • Max input: 256 tokens (aligns with our ≤500-char chunks after tokenisation)
  • Licence: Apache 2.0

L2 Normalisation:
  After embedding we L2-normalise every vector so that FAISS IndexFlatIP
  (inner product) is equivalent to cosine similarity.  This lets us express
  "how similar are two texts?" as a single dot product, which FAISS computes
  very efficiently.

Author : Michael Nana Kwame Osei-Dei  (10022300168)
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64          # texts per forward pass; tune down if OOM


class Embedder:
    """Thin wrapper around SentenceTransformer with L2-normalisation."""

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info("Loading sentence-transformer model: %s", model_name)
        # Downloads model to ~/.cache/torch/sentence_transformers on first run
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        # Expose embedding dimension so vector store can initialise correctly
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dim)

    # ── Core embedding method ─────────────────────────────────────────────────

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = BATCH_SIZE,
        normalise: bool = True,
    ) -> np.ndarray:
        """
        Embed *texts* and return float32 array of shape (N, dim).

        Args:
            texts      : list of strings to embed
            batch_size : mini-batch size for encoding
            normalise  : if True, L2-normalise each vector (required for
                         cosine similarity via FAISS inner-product index)

        Returns:
            numpy float32 array, shape (len(texts), self.dim)
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        show_progress = len(texts) > 200   # only show bar for large batches
        logger.info(
            "Embedding %d texts (batch=%d, normalise=%s)…",
            len(texts), batch_size, normalise,
        )

        # sentence-transformers handles tokenisation and pooling internally
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalise,   # in-model L2 normalisation
        )
        return vecs.astype(np.float32)

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string. Returns shape (1, dim)."""
        return self.embed_texts([query], normalise=True)

    def embed_chunks(self, chunks: list[dict]) -> np.ndarray:
        """
        Embed the 'chunk_text' field of each chunk dict.
        Returns shape (len(chunks), dim).
        """
        texts = [c["chunk_text"] for c in chunks]
        return self.embed_texts(texts)
