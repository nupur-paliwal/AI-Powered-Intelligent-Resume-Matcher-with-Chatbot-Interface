"""RAG chatbot: retrieve relevant resume chunks, then ask the LLM to answer.

Every call is grounded in three things: the active job description, the
precomputed match scores (scoring.py) for each candidate against it, and
resume chunks retrieved from the vector store for the specific question.
Without the first two, "who is the best fit" questions have no notion of
what the role requires and the LLM ends up guessing from resume content
alone. Falls back to a plain keyword search over structured candidate data
when no LLM backend is available, so the chat box still returns something
useful offline.
"""
import logging
import re

from resume_matcher import storage, vector_store
from resume_matcher.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an assistant helping a recruiter query candidate resumes for a "
    "specific open position. You will be given the job description, "
    "precomputed match scores for each candidate against that job "
    "description, and resume excerpts retrieved for the current question. "
    "Treat the match scores as the authoritative signal for ranking/"
    "suitability questions (higher overall score = better fit for this "
    "role). If the provided context doesn't contain enough information to "
    "answer, say so instead of guessing.\n\n"
    "Default to short, direct answers: a sentence or a short list of "
    "candidate names/filenames with the one relevant fact each (a matching "
    "skill, a score), no headers, no bullet-pointed breakdowns, no "
    "restating the question. Only go longer - e.g. explaining the full "
    "score breakdown, quoting resume excerpts, comparing candidates in "
    "depth - when the question explicitly asks for that level of detail "
    "(e.g. 'explain why', 'give me details', 'break down the scores')."
)


def answer_query(query):
    """Return a chat answer for a recruiter's natural-language query."""
    chunks = vector_store.search(query, top_k=8)
    llm = get_llm()

    if llm is None:
        return _rule_based_answer(query)

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"{_build_context(chunks)}\n\nQuestion: {query}"),
    ]
    try:
        response = llm.invoke(messages)
        # .text normalizes across providers: some models (e.g. Gemini 3.x)
        # return `content` as a list of structured blocks rather than a
        # plain string.
        return response.text
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Chat LLM call failed, using rule-based fallback: %s", exc)
        return _rule_based_answer(query)


def _build_context(chunks):
    """Assemble JD + computed scores + retrieved resume excerpts for the LLM prompt."""
    sections = []

    jd_record = storage.get_job_description()
    if jd_record:
        sections.append(f"Job description for this role:\n{jd_record['text']}")

    scored_candidates = [c for c in storage.list_candidates() if c.get("score")]
    if scored_candidates:
        scored_candidates.sort(key=lambda c: c["score"]["overall"], reverse=True)
        score_lines = [
            f"- {c['filename']}: overall match {c['score']['overall']}/100 "
            f"(semantic {c['score']['semantic']}, skills {c['score']['skills']}, "
            f"experience {c['score']['experience']}, education {c['score']['education']}); "
            f"matched: {', '.join(c['score']['matched_skills']) or 'none'}; "
            f"missing: {', '.join(c['score']['missing_skills']) or 'none'}"
            for c in scored_candidates
        ]
        sections.append(
            "Precomputed match scores against the job description above "
            "(authoritative for ranking/suitability):\n" + "\n".join(score_lines)
        )

    excerpts = "\n\n".join(f"[{c['filename']}]\n{c['text']}" for c in chunks)
    sections.append(f"Relevant resume excerpts for this question:\n{excerpts or '(none found)'}")

    return "\n\n---\n\n".join(sections)


def _rule_based_answer(query):
    """Very simple keyword match over stored structured candidate fields."""
    candidates = storage.list_candidates()
    if not candidates:
        return "No resumes have been processed yet."

    keywords = set(re.findall(r"[a-zA-Z][\w.+#]*", query.lower()))

    matches = [
        c for c in candidates if {s.lower() for s in c["fields"].get("skills", [])} & keywords
    ]

    if not matches:
        return (
            "I couldn't find candidates matching that query in the extracted "
            "skills data. Try asking about a specific skill, e.g. 'python' or "
            "'django'. (No LLM backend is active, so answers are keyword-based "
            "only.)"
        )

    lines = [
        f"- {c['filename']}: skills = {', '.join(c['fields'].get('skills', []))}"
        for c in matches
    ]
    return "Candidates matching your query (keyword-based, no LLM active):\n" + "\n".join(lines)
