# AcityBot – Full Project Documentation (Parts A–G)

**Author:** [YOUR_FULL_NAME] ([YOUR_INDEX_NUMBER])  
**Course:** CS4241 – Introduction to Artificial Intelligence (2026)  
**Repository:** `https://github.com/[YOUR_GITHUB_USERNAME]/ai_[YOUR_INDEX_NUMBER]`  
**Live App:** `[YOUR_DEPLOYMENT_URL]`

---

## PART A – Data Cleaning & Chunking Strategy

### Data Sources

| Source | Format | Size (approx.) | Domain |
|--------|--------|----------------|--------|
| Ghana Election Results | CSV (GitHub) | ~10–20 K rows | Political / constituency voting |
| Ghana 2025 Budget Statement | PDF (mofep.gov.gh) | ~200 pages | Macro-economics / public finance |

### Cleaning Pipeline

**CSV (`src/data_loader.py · load_election_csv`)**
1. Headers normalised to UPPER_SNAKE_CASE.
2. Fully empty rows dropped (`dropna(how="all")`).
3. Remaining NaN cells filled with `"Unknown"` (prevents embedding of missing values as blank strings).
4. String columns stripped of leading/trailing whitespace and internal multi-spaces.
5. Each row converted to a natural-language sentence:  
   `"Ghana Election Record — Constituency: Ablekuma North; Party: NDC; Votes: 24312; ..."`  
   This format ensures the embedding model sees human-readable context rather than raw tab-separated values.

**PDF (`src/data_loader.py · load_budget_pdf`)**
1. Line-break hyphenation repaired (`word-\nbreak → wordbreak`).
2. All whitespace runs collapsed to a single space.
3. Non-printable / non-ASCII characters removed.
4. Pages with fewer than 80 characters after cleaning discarded (typically image-only pages or running headers).

### Chunking Strategy Comparison

| Property | Strategy A – Fixed | Strategy B – Sentence |
|----------|-------------------|-----------------------|
| Mechanism | Sliding window | Sentence-boundary grouping |
| Window size | 500 chars | ≤ 450 chars soft target |
| Overlap / max | 75 chars | 5 sentences hard max |
| Resulting chunk count (estimate) | Higher (more, smaller chunks) | Lower (fewer, semantically complete) |
| Best query type | Exact field lookups (CSV) | Policy/narrative questions (PDF) |
| Weakness | May split mid-sentence | Variable size; can confuse embedding |

**Justification for size=500 / 450 chars:**  
English averages ~5 characters per token. 500 chars ≈ 100 tokens, well within `all-MiniLM-L6-v2`'s 256-token limit. The 15% overlap (75 chars) is a standard heuristic to avoid losing context at chunk boundaries.

**Comparative impact on retrieval quality:**  
(Fill in with your experimental results from experiment_log_partC.txt)
- Fixed chunks performed better for direct CSV field queries (e.g., constituency vote counts).
- Sentence chunks produced more coherent answers for multi-sentence budget policy questions.
- Final recommendation: use `fixed` for production; make `sentence` available as a UI option for comparison.

---

## PART B – Embedding, Vector Store & Retrieval

### Embedding Pipeline

Model: `all-MiniLM-L6-v2` (sentence-transformers)
- 22 M parameters, 384-dimensional output
- Max 256 tokens per input
- Apache 2.0 licence
- Rationale: Best accuracy-vs-speed trade-off in the MiniLM family for semantic similarity tasks. Outperforms pure TF-IDF on paraphrase and conceptual queries while being small enough to run on CPU in reasonable time.

After encoding, all vectors are **L2-normalised** so that FAISS `IndexFlatIP` (inner product) equals cosine similarity. This is critical: without normalisation, inner product conflates vector magnitude with direction.

### FAISS Vector Store

Index type: `IndexFlatIP` (exact brute-force inner-product search)
- No approximation → deterministic, reproducible results
- Appropriate for our corpus size (expected < 100 k vectors)
- Saved per chunking strategy to `data/faiss_index_{strategy}.bin`

### Hybrid Search Extension

**Two documented retrieval failure cases:**

**Failure 1 – Acronym blindness**  
Query: *"NDC votes in Ashanti Region 2020"*  
Problem: The dense model sees "NDC" as a rare subword token; its 384-dim representation lands near other party abbreviations (NPP, PPP). Pure semantic search may retrieve NPP results ahead of NDC results.  
Fix: The TF-IDF keyword component assigns high weight to exact "NDC" matches via IDF (inverse document frequency is high for rare abbreviations), rescuing the correct documents.

**Failure 2 – Year-insensitive semantic vectors**  
Query: *"2016 election results"*  
Problem: The dense model treats "2016" and "2024" as near-synonyms in the election context because both appear in almost identical surrounding vocabulary. Top semantic results may return 2024 data for a 2016 query.  
Fix: TF-IDF bigram matching ("2016 election") strongly differentiates years because bigrams appear rarely across years.

**Hybrid formula:**
```
combined = α × semantic_score + (1 − α) × keyword_score
```
Default α = 0.7 (70% semantic, 30% keyword). The UI slider lets users adjust this for different query types.

---

## PART C – Prompt Engineering

### Template Iterations

See `src/prompt_builder.py` for full template text.

**Template V1 – Bare-bones (Context + Question + Answer:)**
- No instruction to stay grounded
- LLM answers freely from parametric memory
- Test result: Frequently produced statistics not present in retrieved chunks

**Template V2 – Grounding instruction + fallback**
- Added "Use ONLY the context below" instruction
- Added "say I don't have that information" fallback
- Test result: Better grounding for direct questions; still embellished multi-step answers; no source citations made verification impossible

**Template V3 – Numbered snippets + strict citation (ACTIVE)**
Key additions:
1. Numbered context format `[1] Source: ... | Score: ...` forces the model to reference a specific snippet
2. `[Source: N, M]` citation requirement after every answer
3. Exact fallback phrase: *"I don't have enough information to answer that based on the available documents."*
4. Explicit prohibition: *"NEVER speculate, invent statistics, or use knowledge outside the snippets."*
5. Domain persona: *"AcityBot"* anchors the model to the Academic City context

Analysis: The citation requirement is the single most effective hallucination control because it forces the model to trace its claim to a specific numbered chunk. If no chunk contains the answer, the model has no number to cite and correctly uses the fallback.

### Context Window Management
- Hard limit: 3,200 characters (leaves ~800 chars for instructions + query + model response within a ~4,096-token budget)
- Snippets added in descending combined-score order
- Header per snippet (~60 chars) + text content counted together
- When budget exhausted, remaining retrieved chunks shown in UI but not sent to LLM

---

## PART D – Full Pipeline & Logging

### Pipeline Stages

| Stage | Module | What happens |
|-------|--------|-------------|
| 1 Query | `pipeline.py` | Validate, strip, log timestamp |
| 2 Retrieval | `retriever.py` | Semantic + keyword; apply feedback boosts; sort |
| 3 Context | `prompt_builder.py` | Build numbered string within char budget |
| 4 Prompt | `prompt_builder.py` | Fill Template V3 |
| 5 LLM | `llm_client.py` | POST to HF Inference API; parse response |
| 6 Log | `pipeline.py` | Append JSONL entry to `logs/query_log.jsonl` |

### Logging
- **File log:** `logs/pipeline_<date>.log` (DEBUG level, all stages)
- **Query JSONL:** `logs/query_log.jsonl` – one line per query with timestamp, token count, model, error flag
- **UI display:** Pipeline log panel (toggleable); retrieved chunks with scores; exact prompt

### UI Transparency (Part D requirements met)
- ✅ Retrieved documents displayed with semantic, keyword, and combined scores
- ✅ Chunks highlighted as [in prompt] vs just retrieved
- ✅ Exact prompt sent to LLM shown in a code-style box
- ✅ Pipeline execution log available on demand

---

## PART E – Adversarial Testing & RAG vs Pure LLM

See `docs/experiment_log_partE.txt` – complete the template with your actual experimental results.

### Adversarial Queries Designed

1. **Ambiguous:** *"Who won?"* — Tests whether the system asks for clarification rather than guessing.
2. **Misleading premise:** *"Ghana's 2025 education budget was GHS 50 billion, right?"* — Tests whether the system corrects a false embedded claim.

### Expected Evidence-Based Findings (template for your analysis)

The RAG system should:
- For Q1: Retrieve chunks about various elections; Template V3's fallback phrase triggers if the context doesn't resolve the ambiguity.
- For Q2: Retrieve the actual education budget figure from the PDF; the citation requirement forces the model to reference the real figure.

The pure LLM is expected to:
- For Q1: Pick an arbitrary "winner" based on the most common election result in training data.
- For Q2: Show sycophantic agreement with the false figure or produce a plausible-but-unverified substitute number.

---

## PART F – Architecture Diagram

See `docs/architecture.md` for the full Mermaid diagram and component justification.

### Summary Justification for Academic City Domain
1. Factual accuracy requirement → RAG grounds answers in authoritative sources
2. Heterogeneous data (CSV + PDF) → dual chunking strategies
3. Domain-specific vocabulary (party abbreviations, budget codes) → hybrid search
4. Academic integrity need for traceability → numbered citations in every response
5. Low infrastructure cost → FAISS CPU + HF free tier

---

## PART G – Novel Feature: Feedback Loop

### Design
`src/feedback.py` implements a persistent, chunk-level score adjustment store.

### Mechanism
```
User clicks 👍 → FeedbackStore.record(chunk_ids, positive=True)
                   → each chunk_id: score += 0.05 (capped at +0.20)

User clicks 👎 → FeedbackStore.record(chunk_ids, positive=False)
                   → each chunk_id: score -= 0.05 (floored at -0.20)

Next query → Retriever reads FeedbackStore.get_boosts()
           → combined_score += boost before final sort
```

### Justification
- **Why this over memory-based conversation?** The feedback loop directly improves the retrieval mechanism (the weakest link in a RAG system). Memory-based conversation improves coherence but doesn't fix bad retrieval.
- **Why this over domain-specific scoring?** The feedback loop is data-driven and self-improving; a domain-specific scorer requires manual feature engineering.
- **Limitations:** Chunk-level boosts generalise to any query that retrieves that chunk, which can introduce noise if a chunk was rated for a different question. A future improvement would be query-cluster-specific boosts.

---

## SUBMISSION CHECKLIST

- [ ] Replaced all `[YOUR_FULL_NAME]` and `[YOUR_INDEX_NUMBER]` placeholders
- [ ] GitHub repo created as `ai_[YOUR_INDEX_NUMBER]`
- [ ] All files pushed to `main` branch
- [ ] Collaborator `GodwinDansoAcity` (or `godwin.danso@acity.edu.gh`) added
- [ ] App deployed to Streamlit Cloud or HF Spaces
- [ ] `docs/experiment_log_partC.txt` filled with REAL experimental results
- [ ] `docs/experiment_log_partE.txt` filled with REAL experimental results
- [ ] 2-minute video recorded following the script
- [ ] Email sent to instructor with subject: `CS4241_RAG_[YOUR_INDEX_NUMBER]_[YOUR_FULL_NAME]`
  - Links: GitHub repo + deployed app + video
