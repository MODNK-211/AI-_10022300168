"""
Unit tests for core RAG pipeline components.

Run with:  pytest tests/ -v

These tests do NOT call the LLM API or download datasets – they only verify
the internal logic of chunking, embedding, vector storage, and prompt building.

Author : Michael Nana Kwame Osei-Dei  (10022300168)
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# ── Make src importable when running pytest from the project root ─────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunker        import fixed_size_chunks, sentence_aware_chunks, chunk_documents
from src.embedder       import Embedder
from src.vector_store   import VectorStore
from src.prompt_builder import build_prompt, build_context, TEMPLATE_V3


# ════════════════════════════════════════════════════════════════════════════════
# Chunker tests
# ════════════════════════════════════════════════════════════════════════════════

class TestFixedChunks:
    _base_doc = {"id": "d1", "text": "A" * 1200, "source": "test.csv"}

    def test_produces_multiple_chunks(self):
        chunks = fixed_size_chunks(self._base_doc, size=500, overlap=75)
        assert len(chunks) >= 2

    def test_chunk_text_key_present(self):
        chunks = fixed_size_chunks(self._base_doc)
        for c in chunks:
            assert "chunk_text" in c
            assert len(c["chunk_text"]) > 0

    def test_strategy_tag(self):
        chunks = fixed_size_chunks(self._base_doc)
        assert all(c["chunk_strategy"] == "fixed" for c in chunks)

    def test_chunk_ids_unique(self):
        chunks = fixed_size_chunks(self._base_doc)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_metadata_inherited(self):
        doc    = {"id": "x", "text": "Hello world.", "source": "elections.csv", "row_index": 42}
        chunks = fixed_size_chunks(doc)
        for c in chunks:
            assert c["source"]    == "elections.csv"
            assert c["row_index"] == 42

    def test_overlap_creates_boundary_chunks(self):
        # With overlap, a 600-char doc split at 500 with 75 overlap should give 2 chunks
        doc    = {"id": "ov", "text": "B" * 600, "source": "test.csv"}
        chunks = fixed_size_chunks(doc, size=500, overlap=75)
        assert len(chunks) == 2
        # Second chunk starts at 500-75 = 425, ends at 925 (capped to 600)
        assert chunks[1]["chunk_start"] == 425

    def test_short_text_single_chunk(self):
        doc    = {"id": "s", "text": "Short text.", "source": "t.csv"}
        chunks = fixed_size_chunks(doc, size=500, overlap=75)
        assert len(chunks) == 1


class TestSentenceChunks:
    _doc = {
        "id": "doc2",
        "source": "budget.pdf",
        "text": (
            "The government allocated GHS 10 billion for education. "
            "Health received GHS 8 billion in the 2025 budget. "
            "Infrastructure spending increased by 15 percent. "
            "Agriculture support funds rose to GHS 3 billion. "
            "The total budget is GHS 217 billion for fiscal year 2025."
        ),
    }

    def test_produces_chunks(self):
        chunks = sentence_aware_chunks(self._doc, target=150, max_sents=2)
        assert len(chunks) >= 2

    def test_strategy_tag(self):
        chunks = sentence_aware_chunks(self._doc)
        assert all(c["chunk_strategy"] == "sentence" for c in chunks)

    def test_sentence_count_within_max(self):
        max_s  = 2
        chunks = sentence_aware_chunks(self._doc, target=10000, max_sents=max_s)
        for c in chunks:
            assert c["sentence_count"] <= max_s

    def test_empty_doc_returns_empty(self):
        doc    = {"id": "empty", "text": "", "source": "t"}
        chunks = sentence_aware_chunks(doc)
        assert chunks == []


class TestChunkDocuments:
    _docs = [
        {"id": "a", "text": "Hello world sentence one. And another sentence.", "source": "s"},
        {"id": "b", "text": "Second document with some text here.", "source": "s"},
    ]

    def test_fixed_strategy(self):
        chunks = chunk_documents(self._docs, strategy="fixed")
        assert all(c["chunk_strategy"] == "fixed" for c in chunks)

    def test_sentence_strategy(self):
        chunks = chunk_documents(self._docs, strategy="sentence")
        assert all(c["chunk_strategy"] == "sentence" for c in chunks)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            chunk_documents(self._docs, strategy="invalid")

    def test_skips_empty_docs(self):
        docs = [{"id": "e", "text": "  ", "source": "s"}]
        assert chunk_documents(docs) == []


# ════════════════════════════════════════════════════════════════════════════════
# Embedder tests
# ════════════════════════════════════════════════════════════════════════════════

class TestEmbedder:
    @pytest.fixture(scope="class")
    def embedder(self):
        return Embedder()   # loads model once for the class

    def test_output_shape(self, embedder):
        vecs = embedder.embed_texts(["hello world", "Ghana election"])
        assert vecs.shape == (2, embedder.dim)

    def test_output_dtype(self, embedder):
        vecs = embedder.embed_texts(["test"])
        assert vecs.dtype == np.float32

    def test_l2_normalised(self, embedder):
        vecs  = embedder.embed_texts(["query text"], normalise=True)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_query_shape(self, embedder):
        q = embedder.embed_query("What is the budget?")
        assert q.shape == (1, embedder.dim)

    def test_empty_list_returns_empty(self, embedder):
        vecs = embedder.embed_texts([])
        assert vecs.shape == (0, embedder.dim)

    def test_embed_chunks(self, embedder):
        chunks = [
            {"chunk_text": "text one"},
            {"chunk_text": "text two"},
        ]
        vecs = embedder.embed_chunks(chunks)
        assert vecs.shape[0] == 2


# ════════════════════════════════════════════════════════════════════════════════
# VectorStore tests  (uses temp directory – no persistent side-effects)
# ════════════════════════════════════════════════════════════════════════════════

class TestVectorStore:
    DIM = 8     # use tiny dim for speed

    def _random_store(self, n: int = 5) -> tuple[VectorStore, list[dict]]:
        store  = VectorStore(self.DIM)
        vecs   = np.random.randn(n, self.DIM).astype(np.float32)
        # L2-normalise so inner product == cosine
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs   /= norms
        chunks = [{"chunk_id": f"c{i}", "chunk_text": f"text {i}", "source": "t"} for i in range(n)]
        store.add_embeddings(vecs, chunks)
        return store, chunks

    def test_add_increments_total(self):
        store, _ = self._random_store(10)
        assert store.index.ntotal == 10

    def test_search_returns_top_k(self):
        store, _ = self._random_store(20)
        q        = np.random.randn(1, self.DIM).astype(np.float32)
        q        /= np.linalg.norm(q)
        results  = store.search(q, top_k=3)
        assert len(results) == 3
        assert all(isinstance(chunk, dict) for chunk, _ in results)

    def test_scores_descending(self):
        store, _ = self._random_store(10)
        q        = np.random.randn(1, self.DIM).astype(np.float32)
        q        /= np.linalg.norm(q)
        results  = store.search(q, top_k=5)
        scores   = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_save_load_roundtrip(self, tmp_path):
        store, chunks = self._random_store(4)
        # Patch paths to use tmp_path
        strategy = "test"
        idx_path  = str(tmp_path / f"faiss_index_{strategy}.bin")
        meta_path = str(tmp_path / f"chunks_meta_{strategy}.json")

        import faiss, json
        faiss.write_index(store.index, idx_path)
        safe = [{k: v for k, v in c.items() if isinstance(v, (str, int, float, bool, type(None)))} for c in store.chunks]
        with open(meta_path, "w") as f:
            json.dump(safe, f)

        # Reload
        import faiss as _faiss
        idx2     = _faiss.read_index(idx_path)
        with open(meta_path) as f:
            loaded_chunks = json.load(f)

        assert idx2.ntotal == 4
        assert len(loaded_chunks) == 4

    def test_empty_search_returns_empty(self):
        store = VectorStore(self.DIM)   # empty index
        q     = np.random.randn(1, self.DIM).astype(np.float32)
        assert store.search(q, top_k=5) == []


# ════════════════════════════════════════════════════════════════════════════════
# Prompt builder tests
# ════════════════════════════════════════════════════════════════════════════════

def _make_result(chunk_id: str, text: str, score: float) -> dict:
    return {
        "chunk": {"chunk_id": chunk_id, "chunk_text": text, "source": "test.csv"},
        "semantic_score": score,
        "keyword_score":  score * 0.5,
        "combined_score": score,
    }


class TestPromptBuilder:
    def test_empty_retrieval_fallback(self):
        prompt, used = build_prompt("any query", [])
        assert "No relevant documents" in prompt
        assert used == []

    def test_query_in_prompt(self):
        prompt, _ = build_prompt("who won in Accra?", [_make_result("c1", "Accra results…", 0.9)])
        assert "who won in Accra?" in prompt

    def test_context_in_prompt(self):
        prompt, _ = build_prompt("test", [_make_result("c1", "unique_marker_text_xyz", 0.8)])
        assert "unique_marker_text_xyz" in prompt

    def test_used_chunks_count(self):
        results = [_make_result(f"c{i}", "x" * 100, 0.9 - i * 0.1) for i in range(6)]
        _, used  = build_prompt("query", results)
        assert 1 <= len(used) <= 6

    def test_context_respects_budget(self):
        # One giant chunk that alone exceeds MAX_CONTEXT_CHARS
        from src.prompt_builder import MAX_CONTEXT_CHARS
        big = _make_result("big", "W" * (MAX_CONTEXT_CHARS + 500), 0.9)
        ctx, used = build_context([big], max_chars=MAX_CONTEXT_CHARS)
        # The giant chunk's text is truncated inside the snippet but the
        # snippet header + partial text may still fit; at minimum used <= 1
        assert len(used) <= 1

    def test_template_v3_used_by_default(self):
        prompt, _ = build_prompt("test q", [_make_result("c", "ctx", 0.5)])
        assert "The Acity Oracle" in prompt              # V3 persona
        assert "STRICT RULES" in prompt          # V3 anti-hallucination


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
