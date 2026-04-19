# The Acity Oracle: Academic City Knowledge Assistant

**Author:** Michael Nana Kwame Osei-Dei  
**Index Number:** 10022300168  
**Course:** CS4241 - Introduction to Artificial Intelligence (2026)  
**Institution:** Academic City University, Ghana  
**Repository:** [github.com/MODNK-211/AI-_10022300168](https://github.com/MODNK-211/AI-_10022300168)  
**Contact:** michael.oseidei@acity.edu.gh

---

## Overview

The Acity Oracle is a Retrieval-Augmented Generation (RAG) chatbot built from scratch for Academic City University. It answers questions using two authoritative sources:

- Ghana Election Results (CSV)
- Ghana 2025 Budget Statement (PDF)

The system manually implements chunking, embedding, vector indexing, hybrid retrieval, prompt engineering, and LLM calling, without LangChain, LlamaIndex, or other prebuilt RAG frameworks.

---

## Key Features

- Two chunking strategies: fixed sliding window and sentence-aware chunking
- Dense embeddings with `all-MiniLM-L6-v2` (384-dim, L2-normalized)
- FAISS `IndexFlatIP` vector store for exact cosine-style retrieval
- Hybrid retrieval (`alpha * semantic + (1 - alpha) * keyword`) with UI controls
- Prompt iteration framework with strict citation requirement
- Hugging Face Inference API integration with fallback model support
- Transparent logging (`logs/pipeline_<date>.log` and `logs/query_log.jsonl`)
- Persistent feedback loop (`+/-` chunk score adjustments from user votes)

---

## Project Structure

```text
ai_10022300168/
|- app.py
|- requirements.txt
|- .env.example
|- .gitignore
|- README.md
|- src/
|  |- data_loader.py
|  |- chunker.py
|  |- embedder.py
|  |- vector_store.py
|  |- retriever.py
|  |- prompt_builder.py
|  |- llm_client.py
|  |- pipeline.py
|  |- feedback.py
|- data/
|- logs/
|- docs/
|  |- architecture.md
|  |- full_documentation.md
|  |- experiment_log_partC.txt
|  |- experiment_log_partE.txt
|- tests/
   |- test_pipeline.py
```

---

## Quick Start (Windows PowerShell)

### Prerequisites

- Python 3.10 or newer
- Internet access for first-time model/data download
- A Hugging Face token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### Setup

```powershell
# Clone and enter project
git clone https://github.com/MODNK-211/AI-_10022300168.git
cd "AI-_10022300168/ai_10022300168"

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create local env file and set token
Copy-Item .env.example .env
# Edit .env and set: HF_TOKEN=hf_your_actual_token

# Run app
streamlit run app.py
```

The first run may take a few minutes while datasets are downloaded and the FAISS index is built. Later runs load cached artifacts.

---

## Quick Start (Linux/macOS)

```bash
git clone https://github.com/MODNK-211/AI-_10022300168.git
cd AI-_10022300168/ai_10022300168
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set HF_TOKEN
streamlit run app.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Documentation

- Architecture and deployment notes: `docs/architecture.md`
- Full Parts A-G write-up: `docs/full_documentation.md`
- Prompt experiments log: `docs/experiment_log_partC.txt`
- Adversarial testing log: `docs/experiment_log_partE.txt`

---

## Known Limitations

- First startup is slower due to model/data/index bootstrapping.
- Retrieval quality depends on chunking strategy and `alpha` balance.
- External API/network issues can affect LLM response reliability.
- Feedback boosts are chunk-level and may not capture query intent perfectly.

---

## Future Improvements

- Add query-type aware routing between chunking strategies.
- Introduce lightweight reranking for top retrieved chunks.
- Add automatic citation verification before final answer display.
- Add integration tests with mocked API responses for end-to-end validation.

---

## Submission Notes

- GitHub collaborator: `GodwinDansoAcity`
- Suggested email subject format:  
  `CS4241_RAG_10022300168_Michael Nana Kwame Osei-Dei`

---

## Academic Integrity

All code in this repository was written by Michael Nana Kwame Osei-Dei (10022300168) as original work for CS4241, 2026. No LangChain, LlamaIndex, Haystack, or equivalent prebuilt RAG pipeline was used.
