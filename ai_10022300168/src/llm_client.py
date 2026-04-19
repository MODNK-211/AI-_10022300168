"""
LLM Client Module
-----------------
Calls an LLM via the HuggingFace Inference API using only `requests`.
No OpenAI SDK, no LangChain, no Hugging Face Hub library.

Primary model  : mistralai/Mistral-7B-Instruct-v0.3
  • Free via HF Inference API with a read-only access token.
  • Instruction-tuned → respects system prompts and citation rules.

Fallback model : HuggingFaceH4/zephyr-7b-beta
  • Activated automatically if the primary model returns an error.

Environment variable required:
  HF_TOKEN  – your HuggingFace access token (read scope sufficient).
  Get one at: https://huggingface.co/settings/tokens

Author : Michael Nana Kwame Osei-Dei  (10022300168)
"""

import os
import logging

import requests

logger = logging.getLogger(__name__)

# ── Model endpoints ────────────────────────────────────────────────────────────
_HF_BASE = "https://api-inference.huggingface.co/models"

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", f"{_HF_BASE}/mistralai/Mistral-7B-Instruct-v0.3"),
    ("HuggingFaceH4/zephyr-7b-beta",        f"{_HF_BASE}/HuggingFaceH4/zephyr-7b-beta"),
]

DEFAULT_MAX_TOKENS  = 512
DEFAULT_TEMPERATURE = 0.1    # low temperature → factual / reproducible answers


# ── Token helper ───────────────────────────────────────────────────────────────

def _hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        logger.warning(
            "HF_TOKEN is not set. "
            "API calls may fail or return 503. "
            "Set it in .env or Streamlit secrets."
        )
    return token


# ── Core call ──────────────────────────────────────────────────────────────────

def query_llm(
    prompt:      str,
    max_tokens:  int   = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Send *prompt* to the HuggingFace Inference API.

    Tries the primary model first; falls back to the secondary model on any
    HTTP error.

    Returns a dict:
        {
          "response" : str,        # model output text
          "model"    : str,        # model id that succeeded
          "tokens"   : int | None, # approximate word count of response
          "error"    : str | None, # error message if all calls failed
        }
    """
    token   = _hf_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens":  max_tokens,
            "temperature":     temperature,
            "return_full_text": False,       # return only the generated part
            "do_sample":        temperature > 0,
        },
        "options": {
            "wait_for_model": True,          # wait up to 20 s if model is cold
            "use_cache":      False,         # don't reuse cached HF responses
        },
    }

    last_error: str = ""

    for model_id, url in MODELS:
        try:
            logger.info("Calling LLM: %s (max_tokens=%d)", model_id, max_tokens)
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            resp.raise_for_status()

            data = resp.json()

            # HuggingFace returns a list: [{"generated_text": "…"}]
            if isinstance(data, list) and data:
                text = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                text = data.get("generated_text") or data.get("error", "")
            else:
                text = str(data)

            # Strip any echoed prompt prefix (some models return it)
            if text.startswith(prompt):
                text = text[len(prompt):]

            text = text.strip()
            logger.info("LLM OK: %d chars received from %s", len(text), model_id)

            return {
                "response": text,
                "model":    model_id,
                "tokens":   len(text.split()),
                "error":    None,
            }

        except requests.exceptions.HTTPError as exc:
            last_error = f"HTTP {exc.response.status_code} from {model_id}: {exc}"
            logger.warning(last_error)
            # Try next model

        except requests.exceptions.Timeout:
            last_error = f"Timeout calling {model_id}"
            logger.warning(last_error)

        except Exception as exc:
            last_error = f"Unexpected error for {model_id}: {exc}"
            logger.error(last_error)

    # All models failed
    return {
        "response": (
            "⚠️ The language model could not be reached. "
            "Please verify that HF_TOKEN is set correctly and try again."
        ),
        "model":  "none",
        "tokens": None,
        "error":  last_error,
    }
