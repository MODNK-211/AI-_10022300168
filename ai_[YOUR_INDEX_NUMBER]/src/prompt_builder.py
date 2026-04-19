"""
Prompt Builder Module
---------------------
Part C: Manual prompt engineering with three iterations.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ITERATION 1 (V1) – Bare-bones template
  Just "Context + Question + Answer:"
  Result: LLM ignores context frequently; answers from parametric memory.

ITERATION 2 (V2) – Grounding instruction added
  Added "use ONLY the context below" and a "say I don't know" fallback.
  Result: Better grounding but no citation; LLM still embellishes statistics.

ITERATION 3 (V3) – Numbered snippets + strict citation rules (FINAL)
  Numbered context blocks force the model to cite by number.
  Explicit anti-hallucination rules + domain persona.
  Result: Most grounded and traceable answers; false statistics virtually
          eliminated because the model must cite a specific snippet number.

Hallucination controls in V3:
  1. "ONLY use the numbered context snippets" instruction
  2. Exact fallback phrase enforced ("I don't have enough information…")
  3. "[Source: X, Y]" citation requirement
  4. "Do NOT speculate or use external knowledge" prohibition

Context Window Management:
  Hard limit MAX_CONTEXT_CHARS prevents prompt overflow.
  Snippets are added in score order; addition stops when budget is full.

Author : [YOUR_FULL_NAME]  ([YOUR_INDEX_NUMBER])
"""

import logging

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 3_200    # leave headroom for instructions + query + response

# ── Template iterations ───────────────────────────────────────────────────────

TEMPLATE_V1 = """Context:
{context}

Question: {query}
Answer:"""

# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_V2 = """You are a helpful assistant for Academic City University.
Use ONLY the context below to answer. If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""

# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_V3 = """[SYSTEM]
You are AcityBot, a precise knowledge assistant for Academic City University, Ghana.
Your knowledge base contains Ghana election records and the Ghana 2025 Budget Statement.

STRICT RULES – follow every rule or your answer is invalid:
1. Answer ONLY using the numbered context snippets provided below.
2. If the answer cannot be found, respond EXACTLY:
   "I don't have enough information to answer that based on the available documents."
3. After your answer, always cite the snippet number(s) used: [Source: 2, 4]
4. NEVER speculate, invent statistics, or use knowledge outside the snippets.
5. Be concise, factual, and professional.

[CONTEXT]
{context}

[QUESTION]
{query}

[ANSWER]"""

# ─────────────────────────────────────────────────────────────────────────────

# Active template used by the pipeline (change here to compare iterations)
ACTIVE_TEMPLATE = TEMPLATE_V3


# ── Context builder ───────────────────────────────────────────────────────────

def build_context(
    retrieved:  list[dict],
    max_chars:  int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict]]:
    """
    Build a numbered context string from retrieval results.

    Snippets are added in descending score order until the character budget
    *max_chars* is exhausted.  Each snippet shows:
      [N] Source: <filename> | Score: <X.XXX>
      <text>

    Returns:
        (context_string, list_of_used_result_dicts)
    """
    snippets:    list[str]  = []
    used_chunks: list[dict] = []
    char_count = 0

    for i, result in enumerate(retrieved, start=1):
        chunk  = result["chunk"]
        score  = result["combined_score"]
        source = chunk.get("source", "Unknown")
        text   = chunk.get("chunk_text", "").strip()

        snippet = (
            f"[{i}] Source: {source} | Score: {score:.3f}\n"
            f"{text}"
        )

        if char_count + len(snippet) + 2 > max_chars:
            logger.info("Context budget exhausted at snippet %d/%d", i - 1, len(retrieved))
            break

        snippets.append(snippet)
        used_chunks.append(result)
        char_count += len(snippet) + 2     # +2 for the "\n\n" separator

    return "\n\n".join(snippets), used_chunks


# ── Prompt assembler ──────────────────────────────────────────────────────────

def build_prompt(
    query:     str,
    retrieved: list[dict],
    template:  str = ACTIVE_TEMPLATE,
) -> tuple[str, list[dict]]:
    """
    Assemble the full LLM prompt.

    Returns:
        (prompt_string, used_chunks_list)

    The used_chunks_list lets the UI display exactly which snippets were
    included in the prompt (important for Part D transparency requirement).
    """
    context_str, used_chunks = build_context(retrieved)

    if not context_str:
        context_str = (
            "No relevant documents were found in the knowledge base "
            "for this query."
        )

    prompt = template.format(context=context_str, query=query)
    logger.info(
        "Prompt built: %d chars, %d/%d snippets used",
        len(prompt), len(used_chunks), len(retrieved),
    )
    return prompt, used_chunks
