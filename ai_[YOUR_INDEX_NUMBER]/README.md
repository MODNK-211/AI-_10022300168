# ai_[YOUR_INDEX_NUMBER] – AcityBot: Academic City Knowledge Assistant

**Author:** [YOUR_FULL_NAME]  
**Index Number:** [YOUR_INDEX_NUMBER]  
**Course:** CS4241 – Introduction to Artificial Intelligence (2026)  
**Institution:** Academic City University, Ghana  

---

## Overview

AcityBot is a Retrieval-Augmented Generation (RAG) chatbot built **entirely from scratch** for Academic City University. It grounds every answer in one of two authoritative knowledge sources:

| Source | Type | Content |
|--------|------|---------|
| Ghana Election Results | CSV | Constituency-level voting records |
| Ghana 2025 Budget Statement | PDF | Ministry of Finance economic policy |

All core RAG components—chunking, embedding, vector indexing, hybrid retrieval, prompt construction, and LLM calling—are **manually implemented** with no LangChain, LlamaIndex, or pre-built RAG framework.

---

## Key Features

| Component | Implementation |
|-----------|---------------|
| Chunking | Two strategies: fixed sliding-window (500 chars / 75 overlap) + sentence-aware grouping |
| Embedding | `all-MiniLM-L6-v2` via sentence-transformers; L2-normalised 384-dim vectors |
| Vector Store | FAISS `IndexFlatIP` (exact cosine via inner product) |
| Retrieval | Hybrid: α × semantic + (1−α) × TF-IDF keyword; α configurable from UI |
| Prompt | Three-iteration engineered template with numbered snippets + citation requirement |
| LLM | HuggingFace Inference API (`Mistral-7B-Instruct-v0.3`, auto-fallback to Zephyr-7B) |
| Logging | Stage-by-stage logs to `logs/pipeline_<date>.log` + `logs/query_log.jsonl` |
| Novel Feature | User feedback loop: 👍/👎 adjusts chunk retrieval scores persistently |

---

## Repository Structure

```
ai_[YOUR_INDEX_NUMBER]/
├── app.py                        # Streamlit UI (Part D)
├── requirements.txt
├── .env.example                  # Copy to .env and add HF_TOKEN
├── .gitignore
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Download + clean CSV & PDF (Part A)
│   ├── chunker.py                # Two chunking strategies (Part A)
│   ├── embedder.py               # Sentence-transformer pipeline (Part B)
│   ├── vector_store.py           # FAISS IndexFlatIP store (Part B)
│   ├── retriever.py              # Hybrid search retrieval (Part B)
│   ├── prompt_builder.py         # Prompt iterations + context budget (Part C)
│   ├── llm_client.py             # HF Inference API via requests (Part D)
│   ├── pipeline.py               # Full RAG orchestration + logging (Part D)
│   └── feedback.py               # Feedback loop (Part G)
├── data/                         # Auto-populated at runtime (gitignored)
│   └── .gitkeep
├── logs/                         # Pipeline & query logs (gitignored)
│   └── .gitkeep
├── docs/
│   ├── architecture.md           # Mermaid diagram + justification (Part F)
│   ├── experiment_log_partC.txt  # Prompt iteration experiments (Part C)
│   └── experiment_log_partE.txt  # Adversarial testing log (Part E)
└── tests/
    └── test_pipeline.py          # Unit tests for core components
```

---

## Quick Start (Local)

### Prerequisites
- Python 3.10+
- A free [HuggingFace](https://huggingface.co/settings/tokens) access token

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/[YOUR_GITHUB_USERNAME]/ai_[YOUR_INDEX_NUMBER].git
cd ai_[YOUR_INDEX_NUMBER]

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your HuggingFace token
cp .env.example .env
# Edit .env and set: HF_TOKEN=hf_your_actual_token

# 5. Load env and run
export $(cat .env | xargs)        # Windows: set /p HF_TOKEN=<.env (or set HF_TOKEN=...)
streamlit run app.py
```

The first run downloads data, builds embeddings, and creates the FAISS index (~2–5 min depending on internet speed). Subsequent runs load from cache in seconds.

---

## Cloud Deployment

See **Section 6** of the full documentation in `docs/architecture.md` for Streamlit Community Cloud and Hugging Face Spaces instructions.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Collaborator Access

GitHub collaborator: **GodwinDansoAcity**  
Email your submission to the course instructor with subject:  
`CS4241_RAG_[YOUR_INDEX_NUMBER]_[YOUR_FULL_NAME]`

---

## Academic Integrity

All code in this repository was written by [YOUR_FULL_NAME] ([YOUR_INDEX_NUMBER]) as original work for CS4241, 2026. No LangChain, LlamaIndex, Haystack, or equivalent pre-built RAG pipeline was used.
