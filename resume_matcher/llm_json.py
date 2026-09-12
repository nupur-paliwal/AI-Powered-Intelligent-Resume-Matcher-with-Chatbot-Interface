"""Helper for getting structured JSON out of an LLM chat call."""
import json
import logging
import re

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def ask_json(llm, system_prompt, user_content):
    """Ask the LLM to answer with JSON and parse the result.

    Returns the parsed dict, or None if the LLM is unavailable or its
    response isn't valid JSON, so callers can degrade to a rule-based path.
    """
    if llm is None:
        return None

    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
        )
        # .text normalizes across providers: some models (e.g. Gemini 3.x)
        # return `content` as a list of structured blocks rather than a
        # plain string.
        raw = response.text if hasattr(response, "text") else str(response)
    except Exception as exc:  # noqa: BLE001 - any backend failure falls back to None
        logger.warning("LLM call failed: %s", exc)
        return None

    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        logger.warning("LLM response had no JSON object: %r", raw[:200])
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON: %s", exc)
        return None
