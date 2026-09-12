"""Extract structured fields (skills, experience, education) from resume text."""
import logging

from resume_matcher import llm_json, skills_taxonomy
from resume_matcher.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You extract structured data from resumes. Respond with ONLY a JSON "
    'object with keys: "name" (string or null), "email" (string or null), '
    '"skills" (array of strings, lowercase, technical skills/tools only), '
    '"experience_years" (number, total years of professional experience, or '
    'null), "education" (array of strings, degrees/qualifications), '
    '"summary" (one-sentence plain-text summary of the candidate). No '
    "markdown, no explanation, just the JSON object."
)


def parse_resume(text):
    """Return structured candidate data extracted from resume text.

    Uses the configured LLM when available; falls back to keyword/regex
    extraction (skills taxonomy, "N years" pattern, degree keywords) so the
    app keeps working with no LLM backend at all.
    """
    llm = get_llm()
    parsed = llm_json.ask_json(llm, _SYSTEM_PROMPT, text[:8000])
    if parsed:
        parsed["skills"] = [s.lower().strip() for s in parsed.get("skills") or []]
        parsed["education"] = parsed.get("education") or []
        parsed["source"] = "llm"
        return parsed

    logger.info("Falling back to rule-based resume parsing.")
    return {
        "name": None,
        "email": skills_taxonomy.extract_email(text),
        "skills": skills_taxonomy.extract_skills(text),
        "experience_years": skills_taxonomy.extract_experience_years(text),
        "education": skills_taxonomy.extract_education(text),
        "summary": None,
        "source": "rule-based",
    }
