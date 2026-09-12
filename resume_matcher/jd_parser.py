"""Extract structured requirements (skills, experience, education) from a job description."""
import logging

from resume_matcher import llm_json, skills_taxonomy
from resume_matcher.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You extract structured requirements from job descriptions. Respond "
    'with ONLY a JSON object with keys: "skills" (array of strings, '
    'lowercase, required technical skills/tools), "experience_years" '
    "(number, minimum years of experience required, or null), "
    '"education" (array of strings, required degrees/qualifications), '
    '"summary" (one-sentence plain-text summary of the role). No markdown, '
    "no explanation, just the JSON object."
)


def parse_job_description(text):
    """Return structured requirements extracted from a job description.

    Uses the configured LLM when available; falls back to keyword/regex
    extraction so scoring still works with no LLM backend at all.
    """
    llm = get_llm()
    parsed = llm_json.ask_json(llm, _SYSTEM_PROMPT, text[:8000])
    if parsed:
        parsed["skills"] = [s.lower().strip() for s in parsed.get("skills") or []]
        parsed["education"] = parsed.get("education") or []
        parsed["source"] = "llm"
        return parsed

    logger.info("Falling back to rule-based job description parsing.")
    return {
        "skills": skills_taxonomy.extract_skills(text),
        "experience_years": skills_taxonomy.extract_experience_years(text),
        "education": skills_taxonomy.extract_education(text),
        "summary": None,
        "source": "rule-based",
    }
