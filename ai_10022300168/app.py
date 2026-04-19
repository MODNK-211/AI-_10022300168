"""
The Acity Oracle – Academic City Knowledge Assistant
============================================
Streamlit UI for the RAG chatbot.

Displays (Part D requirements):
  • Chat interface with conversation history
  • Retrieved context chunks with semantic, keyword, and combined scores
  • Exact prompt sent to the LLM (toggleable)
  • Pipeline execution log (toggleable)
  • User feedback buttons – 👍 / 👎 (Part G)

Run locally:
    streamlit run app.py

Author : Michael Nana Kwame Osei-Dei  (10022300168)
Course : CS4241 – Introduction to Artificial Intelligence (2026)
"""

import os
import sys
import logging

import streamlit as st

# ── Ensure src/ is importable as a package ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
)

from src.pipeline import RAGPipeline   # noqa: E402
from src.feedback import FeedbackStore # noqa: E402

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Acity Oracle – Academic City Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Retrieved chunk cards */
.chunk-card {
    background: #f0f4ff;
    border-left: 4px solid #4a6cf7;
    padding: 0.75rem 1rem;
    margin: 0.3rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.84rem;
    line-height: 1.5;
}
.score-badge {
    display: inline-block;
    background: #4a6cf7;
    color: #fff;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
/* Prompt display */
.prompt-box {
    background: #1e1e2e;
    color: #cdd6f4;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    font-family: "Courier New", monospace;
    font-size: 0.77rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 320px;
    overflow-y: auto;
}
/* Pipeline log */
.log-box {
    background: #0d1117;
    color: #7ee787;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-family: "Courier New", monospace;
    font-size: 0.73rem;
    white-space: pre-wrap;
    max-height: 220px;
    overflow-y: auto;
}
/* Source tag */
.source-tag {
    font-size: 0.72rem;
    color: #555;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 The Acity Oracle")
    st.caption("Academic City University Knowledge Assistant")
    st.divider()

    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider(
        "Top-K documents to retrieve",
        min_value=1, max_value=10, value=5,
        help="Number of context chunks shown to the LLM.",
    )
    alpha = st.slider(
        "Semantic weight α",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="α=1.0 → pure semantic search | α=0.0 → pure keyword (TF-IDF)\n"
             "Default 0.7 balances both (hybrid search).",
    )
    chunking_strategy = st.selectbox(
        "Chunking strategy",
        options=["fixed", "sentence"],
        index=0,
        help=(
            "fixed: sliding window (500 chars, 75 overlap)\n"
            "sentence: sentence-boundary grouping (~450 chars, ≤5 sentences)\n\n"
            "Changing strategy rebuilds the index on first use."
        ),
    )

    st.divider()

    st.subheader("🔍 Debug Options")
    show_prompt  = st.checkbox("Show prompt sent to LLM",   value=True)
    show_log     = st.checkbox("Show pipeline execution log", value=False)

    st.divider()

    st.subheader("📊 Feedback Stats")
    fb_store = FeedbackStore()
    stats    = fb_store.get_stats()
    c1, c2 = st.columns(2)
    c1.metric("👍 Boosted",   stats["positive"])
    c2.metric("👎 Penalised", stats["negative"])
    if st.button("🔄 Reset feedback"):
        FeedbackStore().reset()
        st.success("Feedback cleared.")
        st.rerun()

    st.divider()
    st.caption(
        "Built by **Michael Nana Kwame Osei-Dei**  \n"
        "Index: **10022300168**  \n"
        "CS4241 · AI Project 2026"
    )


# ── Cached pipeline loader ────────────────────────────────────────────────────

@st.cache_resource(
    show_spinner="⏳ Building knowledge base — first run may take a few minutes…"
)
def get_pipeline(strategy: str) -> RAGPipeline:
    """
    Cache one RAGPipeline per chunking strategy.
    Streamlit reruns reuse the cached object; retrieval parameters (α, k)
    are passed at query time rather than baked into the cached object.
    """
    return RAGPipeline(strategy=strategy)


# ── Session state initialisation ──────────────────────────────────────────────
if "messages"      not in st.session_state:
    st.session_state.messages      = []   # [{role, content}, …]
if "last_result"   not in st.session_state:
    st.session_state.last_result   = None
if "feedback_done" not in st.session_state:
    st.session_state.feedback_done = False


# ── Main page ─────────────────────────────────────────────────────────────────
st.title("🎓 The Acity Oracle – Academic City Knowledge Assistant")
st.markdown(
    "Ask me anything about **Ghana Election Results** "
    "or the **Ghana 2025 Budget Statement**."
)

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Query input ───────────────────────────────────────────────────────────────
query = st.chat_input("Type your question here…")

if query:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            pipeline = get_pipeline(chunking_strategy)
            result   = pipeline.run(query, alpha=alpha, top_k=top_k)

        st.session_state.last_result   = result
        st.session_state.feedback_done = False

        # ── Retrieved context panel ───────────────────────────────────────────
        n_retrieved = len(result["retrieved"])
        n_used      = len(result["used_chunks"])
        with st.expander(
            f"📚 Retrieved Context  ({n_used} used in prompt / {n_retrieved} retrieved)",
            expanded=True,
        ):
            if not result["retrieved"]:
                st.info("No relevant chunks found in the knowledge base.")
            else:
                for i, r in enumerate(result["retrieved"], start=1):
                    chunk  = r["chunk"]
                    text   = chunk.get("chunk_text", "")
                    source = chunk.get("source", "Unknown")
                    page   = chunk.get("page", "")
                    truncated = text[:350] + ("…" if len(text) > 350 else "")
                    in_prompt  = i <= n_used    # was this snippet sent to LLM?

                    border_col = "#4a6cf7" if in_prompt else "#aab4cc"
                    bg_col     = "#f0f4ff" if in_prompt else "#f8f9ff"
                    page_suffix = f" · page {page}" if page else ""

                    st.markdown(
                        f'<div class="chunk-card" style="border-color:{border_col};background:{bg_col};">'
                        f'<strong>#{i}</strong>&ensp;'
                        f'<span class="score-badge">Combined: {r["combined_score"]:.3f}</span>&ensp;'
                        f'<span style="font-size:0.75rem;">Sem: {r["semantic_score"]:.3f} &nbsp;|&nbsp; '
                        f'Kwd: {r["keyword_score"]:.3f}</span>'
                        f'{"&ensp;<em style=\"color:#4a6cf7;font-size:0.72rem;\">[in prompt]</em>" if in_prompt else ""}'
                        f'<br><span class="source-tag">📄 {source}{page_suffix}</span>'
                        f"<br><br>{truncated}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Prompt display ────────────────────────────────────────────────────
        if show_prompt:
            with st.expander("🔍 Exact Prompt Sent to LLM", expanded=False):
                safe_prompt = result["prompt"].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f'<div class="prompt-box">{safe_prompt}</div>',
                    unsafe_allow_html=True,
                )

        # ── Pipeline log ──────────────────────────────────────────────────────
        if show_log:
            with st.expander("🪵 Pipeline Execution Log", expanded=False):
                log_html = "\n".join(result["pipeline_log"])
                st.markdown(
                    f'<div class="log-box">{log_html}</div>',
                    unsafe_allow_html=True,
                )

        # ── LLM response ──────────────────────────────────────────────────────
        llm     = result["llm_result"]
        resp    = llm.get("response") or ""
        model   = llm.get("model", "unknown")
        err     = llm.get("error")

        if err and not resp:
            st.error(f"LLM Error: {err}")
            resp = (
                "⚠️ Could not reach the language model. "
                "Make sure **HF_TOKEN** is set in your environment or `.env` file."
            )

        st.markdown(resp)
        st.caption(f"_Model: {model}_")

        st.session_state.messages.append({"role": "assistant", "content": resp})


# ── Feedback panel (Part G) ───────────────────────────────────────────────────
if st.session_state.last_result and not st.session_state.feedback_done:
    st.divider()
    st.markdown("**Was this answer helpful?**")
    col_y, col_n, col_pad = st.columns([1, 1, 8])

    with col_y:
        if st.button("👍 Yes", key="btn_positive", use_container_width=True):
            cids = [r["chunk"].get("chunk_id", "") for r in st.session_state.last_result["retrieved"]]
            FeedbackStore().record(cids, positive=True)
            st.session_state.feedback_done = True
            st.success("Thank you! Retrieved chunks boosted for future queries.")
            st.rerun()

    with col_n:
        if st.button("👎 No", key="btn_negative", use_container_width=True):
            cids = [r["chunk"].get("chunk_id", "") for r in st.session_state.last_result["retrieved"]]
            FeedbackStore().record(cids, positive=False)
            st.session_state.feedback_done = True
            st.warning("Noted. These chunks will be ranked lower next time.")
            st.rerun()
