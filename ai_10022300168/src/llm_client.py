"""
LLM Client Module
-----------------
Calls an LLM via Groq or Hugging Face using only `requests`
(OpenAI-compatible chat completions endpoints).

Legacy `https://api-inference.huggingface.co/models/...` is no longer used
(HF returns 404 for that path).

Environment variables:
  GROQ_API_KEY (optional, preferred)
      Groq API key for https://api.groq.com/openai/v1/chat/completions.

  GROQ_CHAT_MODEL (optional)
      Groq model id. Default: llama-3.1-8b-instant

  HF_TOKEN (optional fallback)
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
_GROQ_CHAT = "https://api.groq.com/openai/v1/chat/completions"

_DEFAULT_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3:fastest",
    "HuggingFaceH4/zephyr-7b-beta:fastest",
]

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1


def _groq_token() -> str:
    token = os.getenv("GROQ_API_KEY", "").strip()
    return token


def _groq_model() -> str:
    return os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"


def _hf_token() -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        logger.warning(
            "HF_TOKEN is not set. Set GROQ_API_KEY (preferred) or HF_TOKEN in `.env`, "
            "your shell, or Streamlit secrets before calling the LLM."
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
    Send *prompt* to Groq (preferred when GROQ_API_KEY exists), otherwise
    Hugging Face Inference Providers.

    Returns:
        {
          "response": str,
          "model": str,
          "tokens": int | None,
          "error": str | None,
        }
    """
    last_error = ""

    groq_token = _groq_token()
    if groq_token:
        groq_model = _groq_model()
        payload = {
            "model": groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {groq_token}",
            "Content-Type": "application/json",
        }
        try:
            logger.info("Calling LLM (groq): %s (max_tokens=%d)", groq_model, max_tokens)
            resp = requests.post(_GROQ_CHAT, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text, perr = _parse_chat_json(data)
            if perr:
                last_error = f"{groq_model}: {perr}"
                logger.warning(last_error)
            elif text:
                logger.info("LLM OK: %d chars received from %s", len(text), groq_model)
                return {
                    "response": text,
                    "model": groq_model,
                    "tokens": len(text.split()),
                    "error": None,
                }
            else:
                last_error = f"{groq_model}: empty assistant content"
                logger.warning(last_error)
        except requests.exceptions.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.text[:500]
            except Exception:
                detail = ""
            last_error = f"HTTP {exc.response.status_code} from {groq_model}: {exc} {detail}".strip()
            logger.warning(last_error)
        except requests.exceptions.Timeout:
            last_error = f"Timeout calling {groq_model}"
            logger.warning(last_error)
        except Exception as exc:
            last_error = f"Unexpected error for {groq_model}: {exc}"
            logger.error(last_error)

    token = _hf_token()
    if not token:
        return {
            "response": "",
            "model": "none",
            "tokens": None,
            "error": last_error or "Neither GROQ_API_KEY nor HF_TOKEN is set",
        }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

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
            "Verify **GROQ_API_KEY** is valid, or verify **HF_TOKEN** is valid "
            "and allowed for Inference Providers. "
            "See https://huggingface.co/docs/inference-providers/en/index"
        ),
        "model": "none",
        "tokens": None,
        "error": last_error,
    }
