"""LangChain LLM factory: Gemini via API key when available, else local Ollama.

Preference order is Gemini -> Ollama -> None. Callers must handle a ``None``
return by degrading to rule-based logic (see resume_parser, jd_parser,
chatbot).
"""
import functools
import logging

import requests

import config

logger = logging.getLogger(__name__)


def _ollama_is_reachable():
    try:
        response = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


@functools.lru_cache(maxsize=1)
def get_llm():
    """Return a cached LangChain chat model, or None if no backend is reachable."""
    if config.GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - any SDK/setup failure falls back
            logger.warning("Gemini unavailable, falling back to Ollama: %s", exc)

    if _ollama_is_reachable():
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.OLLAMA_MODEL, base_url=config.OLLAMA_HOST, temperature=0.0
        )

    logger.warning("No LLM backend reachable; falling back to rule-based logic.")
    return None


def llm_backend_name():
    """Human-readable name of the active backend, for status displays."""
    if config.GOOGLE_API_KEY:
        return f"Gemini ({config.GEMINI_MODEL})"
    if _ollama_is_reachable():
        return f"Ollama ({config.OLLAMA_MODEL})"
    return "rule-based (no LLM available)"
