"""
LLM Client Module
-----------------
Calls an LLM via Hugging Face **Inference Providers** using only `requests`
(OpenAI-compatible chat completions on the HF router).

Legacy `https://api-inference.huggingface.co/models/...` is no longer used
(HF returns 404 for that path).

Environment variables:
  HF_TOKEN (required)
      Hugging Face access token. For Inference Providers, use a token that is
      allowed to call the router (see HF token settings / fine-grained scopes).

  HF_CHAT_MODELS (optional)
      Comma-separated model ids for the router, tried in order.
      Default:
        mistralai/Mistral-7B-Instruct-v0.3:fastest,
        HuggingFaceH4/zephyr-7b-beta:fastest

      The `:fastest` suffix follows HF docs for automatic provider selection.

Author : Michael Nana Kwame Osei-Dei  (10022300168)
"""

from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

_ROUTER_CHAT = "https://router.huggingface.co/v1/chat/completions"

_DEFAULT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3:fastest",
    "HuggingFaceH4/zephyr-7b-beta:fastest",
]

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1


def _hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        logger.warning(
            "HF_TOKEN is not set. Set it in `.env` (loaded by app.py), your shell, "
            "or Streamlit secrets before calling the LLM."
        )
    return token


def _router_models() -> list[str]:
    raw = os.getenv("HF_CHAT_MODELS", "").strip()
    if not raw:
        return list(_DEFAULT_MODELS)
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or list(_DEFAULT_MODELS)


def _parse_chat_json(data: dict) -> tuple[str, str | None]:
    """
    Returns (assistant_text, error_string_or_none).
    """
    if not isinstance(data, dict):
        return "", "Unexpected non-JSON object from router"

    err = data.get("error")
    if err is not None:
        if isinstance(err, dict):
            msg = err.get("message") or json.dumps(err)
        else:
            msg = str(err)
        return "", msg

    choices = data.get("choices") or []
    if not choices:
        return "", "No choices in router response"

    msg0 = choices[0].get("message") or {}
    text = (msg0.get("content") or "").strip()
    return text, None


def query_llm(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    Send *prompt* to Hugging Face Inference Providers (router chat completions).

    Tries each model in ``HF_CHAT_MODELS`` (or defaults) until one succeeds.

    Returns:
        {
          "response": str,
          "model": str,
          "tokens": int | None,
          "error": str | None,
        }
    """
    token = _hf_token()
    if not token:
        return {
            "response": "",
            "model": "none",
            "tokens": None,
            "error": "HF_TOKEN is not set",
        }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    last_error = ""

    for model_id in _router_models():
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            logger.info("Calling LLM (router): %s (max_tokens=%d)", model_id, max_tokens)
            resp = requests.post(_ROUTER_CHAT, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text, perr = _parse_chat_json(data)
            if perr:
                last_error = f"{model_id}: {perr}"
                logger.warning(last_error)
                continue
            if not text:
                last_error = f"{model_id}: empty assistant content"
                logger.warning(last_error)
                continue

            logger.info("LLM OK: %d chars received from %s", len(text), model_id)
            return {
                "response": text,
                "model": model_id,
                "tokens": len(text.split()),
                "error": None,
            }

        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.text[:500]
            except Exception:
                detail = ""
            last_error = f"HTTP {exc.response.status_code} from {model_id}: {exc} {detail}".strip()
            logger.warning(last_error)

        except requests.exceptions.Timeout:
            last_error = f"Timeout calling {model_id}"
            logger.warning(last_error)

        except Exception as exc:
            last_error = f"Unexpected error for {model_id}: {exc}"
            logger.error(last_error)

    return {
        "response": (
            "⚠️ The language model could not be reached. "
            "Verify **HF_TOKEN** is valid and allowed for Inference Providers, "
            "or set **HF_CHAT_MODELS** to a model you can run. "
            "See https://huggingface.co/docs/inference-providers/en/index"
        ),
        "model": "none",
        "tokens": None,
        "error": last_error,
    }
