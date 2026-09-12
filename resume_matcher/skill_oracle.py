"""LLM-judged skill relatedness: opt-in, cache-backed tail case beyond the
curated skill-cluster table in skills_taxonomy.py.

Disabled by default (config.SKILL_ORACLE_ENABLED). The local llama3.2:3b
model was empirically shown to reason unreliably about skill relatedness -
it answered "no" even when asked about a skill verbatim in the candidate's
own list - so this should only be turned on once a trusted model (Gemini,
or a larger local model you've verified yourself) is configured.

Verdicts are cached to data/skill_oracle_cache.json, tagged with the model
that produced them. Upgrading to a stronger model automatically re-judges
instead of being stuck with a weaker model's wrong answer forever.
"""
import json
import logging
import os
import threading

import config
from resume_matcher.llm_provider import get_llm, llm_backend_name

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE_FILE = os.path.join(config.DATA_DIR, "skill_oracle_cache.json")

_SYSTEM_PROMPT = (
    "You judge whether a candidate having one skill reasonably satisfies a "
    "job requirement for a different skill, because they are the same "
    "technology or domain (for example, PyTorch experience reasonably "
    "satisfies a TensorFlow requirement). Respond with ONLY the single word "
    "yes or no."
)


def _read_cache():
    if not os.path.exists(_CACHE_FILE):
        return {}
    try:
        with open(_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read skill oracle cache: %s", exc)
        return {}


def _write_cache(cache):
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def is_related(required_skill, candidate_skill):
    """True if candidate_skill is judged to satisfy required_skill.

    Returns False whenever the oracle is disabled, no LLM is available, or
    the call fails - it never claims a match it hasn't actually verified.
    """
    if not config.SKILL_ORACLE_ENABLED:
        return False

    llm = get_llm()
    if llm is None:
        return False

    key = f"{required_skill.lower().strip()}|{candidate_skill.lower().strip()}"
    model_name = llm_backend_name()

    with _LOCK:
        cached = _read_cache().get(key)
    if cached and cached.get("model") == model_name:
        return cached["related"]

    from langchain_core.messages import HumanMessage, SystemMessage

    user_content = f"Required skill: {required_skill}\nCandidate's skill: {candidate_skill}"
    try:
        response = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_content)]
        )
        # .text normalizes across providers: some models (e.g. Gemini 3.x)
        # return `content` as a list of structured blocks rather than a
        # plain string.
        related = response.text.strip().lower().startswith("y")
    except Exception as exc:  # noqa: BLE001 - degrade to "not related"
        logger.warning("Skill oracle call failed: %s", exc)
        return False

    with _LOCK:
        cache = _read_cache()
        cache[key] = {"related": related, "model": model_name}
        _write_cache(cache)

    return related
