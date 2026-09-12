"""Combine semantic similarity and structured-field overlap into one match score.

Skill matching needs domain knowledge that a general-purpose sentence
embedding can't provide from bare jargon words alone: cosine similarity
between "tensorflow" and "rag" tops out around 0.3-0.4 with
all-MiniLM-L6-v2, well below any usable threshold. So skills are matched in
three tiers: exact (case-insensitive) match, then a small hand-curated table
of interchangeable-skill clusters (skills_taxonomy.SKILL_CLUSTERS), then -
only if config.SKILL_ORACLE_ENABLED - an LLM-judged, cached tail case
(skill_oracle.py) for anything the first two tiers miss. The oracle tier is
off by default: a small local model was shown to reason unreliably here (it
answered "no" even for a skill verbatim in the candidate's own list).
Education only needs degree-level comparison, handled deterministically.
"""
import numpy as np

import config
from resume_matcher import skill_oracle, skills_taxonomy
from resume_matcher.embeddings import get_embeddings

_DEGREE_LEVELS = {
    "phd": 4, "ph.d": 4, "doctorate": 4,
    "master": 3, "m.tech": 3, "mtech": 3, "m.sc": 3, "msc": 3,
    "mba": 3, "m.s.": 3, "ms in": 3,
    "bachelor": 2, "b.tech": 2, "btech": 2, "b.sc": 2, "bsc": 2,
    "b.e.": 2, "be in": 2,
    "diploma": 1,
}


def semantic_similarity(resume_text, jd_text):
    """Cosine similarity between whole-document embeddings, scaled to 0-1."""
    embedder = get_embeddings()
    resume_vec, jd_vec = embedder.embed_documents([resume_text, jd_text])
    resume_vec, jd_vec = np.array(resume_vec), np.array(jd_vec)
    denom = np.linalg.norm(resume_vec) * np.linalg.norm(jd_vec)
    if denom == 0:
        return 0.0
    cosine = float(np.dot(resume_vec, jd_vec) / denom)
    return max(0.0, min(1.0, (cosine + 1) / 2))


def _related_via_cluster(required_skill, resume_skills):
    """True if required_skill shares a curated skill cluster with any resume skill."""
    for cluster in skills_taxonomy.related_skill_clusters(required_skill):
        if cluster & resume_skills:
            return True
    return False


def _most_similar_resume_skills(required_skill, resume_skills, top_k):
    """Rank resume_skills by embedding similarity to required_skill.

    A cheap pre-filter so the (comparatively expensive, LLM-backed) skill
    oracle only gets asked about the few most plausible candidates instead
    of every resume skill.
    """
    if not resume_skills:
        return []
    embedder = get_embeddings()
    vectors = np.array(embedder.embed_documents([required_skill, *resume_skills]))
    required_vec, resume_vecs = vectors[0], vectors[1:]
    denom = np.clip(np.linalg.norm(required_vec) * np.linalg.norm(resume_vecs, axis=1), 1e-8, None)
    similarities = resume_vecs @ required_vec / denom
    ranked = sorted(zip(resume_skills, similarities), key=lambda pair: pair[1], reverse=True)
    return [skill for skill, _sim in ranked[:top_k]]


def skills_overlap(resume_skills, jd_skills):
    """Fraction of JD-required skills the candidate has.

    Checked in three tiers: exact (case-insensitive) match, a curated table
    of interchangeable skill clusters (e.g. "tensorflow" satisfied by
    "langchain"/"rag"), then - only if enabled - the LLM skill oracle for
    anything still unmatched.

    Returns (score, matched_skills, missing_skills).
    """
    jd_list = sorted({s.lower().strip() for s in jd_skills or [] if s.strip()})
    if not jd_list:
        return 1.0, [], []

    resume_set = {s.lower().strip() for s in resume_skills or [] if s.strip()}
    matched, missing = [], []
    for skill in jd_list:
        if skill in resume_set or _related_via_cluster(skill, resume_set):
            matched.append(skill)
            continue

        if config.SKILL_ORACLE_ENABLED:
            candidates = _most_similar_resume_skills(
                skill, sorted(resume_set), config.SKILL_ORACLE_TOP_K
            )
            if any(skill_oracle.is_related(skill, candidate) for candidate in candidates):
                matched.append(skill)
                continue

        missing.append(skill)

    return len(matched) / len(jd_list), matched, missing


def experience_score(resume_years, jd_min_years):
    """1.0 if candidate meets/exceeds required years, partial credit below that."""
    if not jd_min_years:
        return 1.0
    if resume_years is None:
        return 0.0
    if resume_years >= jd_min_years:
        return 1.0
    return max(0.0, resume_years / jd_min_years)


def _degree_level(text):
    """Highest recognized degree level mentioned in text, or 0 if none."""
    lowered = text.lower()
    return max((level for keyword, level in _DEGREE_LEVELS.items() if keyword in lowered), default=0)


def education_overlap(resume_education, jd_education):
    """1.0 if the candidate's highest degree meets the JD's required level."""
    jd_min_level = max((_degree_level(e) for e in (jd_education or [])), default=0)
    if jd_min_level == 0:
        return 1.0  # JD didn't name a recognizable degree level; don't penalize.
    resume_max_level = max((_degree_level(e) for e in (resume_education or [])), default=0)
    return 1.0 if resume_max_level >= jd_min_level else 0.0


def compute_match_score(resume_text, resume_fields, jd_text, jd_fields):
    """Return a dict with the overall score (0-100) and its component breakdown."""
    semantic = semantic_similarity(resume_text, jd_text)
    skills, matched_skills, missing_skills = skills_overlap(
        resume_fields.get("skills"), jd_fields.get("skills")
    )
    experience = experience_score(
        resume_fields.get("experience_years"), jd_fields.get("experience_years")
    )
    education = education_overlap(resume_fields.get("education"), jd_fields.get("education"))

    weights = config.SCORE_WEIGHTS
    overall = (
        semantic * weights["semantic"]
        + skills * weights["skills"]
        + experience * weights["experience"]
        + education * weights["education"]
    )

    return {
        "overall": round(overall * 100, 1),
        "semantic": round(semantic * 100, 1),
        "skills": round(skills * 100, 1),
        "experience": round(experience * 100, 1),
        "education": round(education * 100, 1),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
    }
