"""Keyword/regex fallback used when no LLM backend is available."""
import re

SKILLS_TAXONOMY = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "django", "flask", "fastapi", "spring", "react", "angular", "vue",
    "node.js", "express",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "sentence transformers", "llm", "genai", "langchain",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd",
    "git", "linux", "rest api", "graphql", "microservices",
    "html", "css", "tailwind", "bootstrap",
    "data analysis", "data engineering", "etl", "spark", "hadoop", "airflow",
    "agile", "scrum", "project management",
]

# Groups of skills treated as interchangeable for match scoring, since a JD
# asking for "tensorflow" should reasonably be satisfied by a candidate whose
# resume says "pytorch" or "langchain/rag" (same ML/AI domain) even with zero
# exact string overlap. Curated by hand rather than judged by embeddings or
# the LLM: bare skill-word embeddings don't encode this kind of domain
# relation (tensorflow/rag cosine similarity is ~0.3-0.4, well below any
# usable threshold), and a small local model demonstrably can't reason about
# it reliably either (it answered "no" even when asked about a skill that
# was verbatim in the candidate's own list).
SKILL_CLUSTERS = [
    {
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "pandas", "numpy", "sentence transformers", "llm", "llms", "genai",
        "langchain", "langgraph", "rag", "ai agents", "embeddings",
        "text classification", "sentence similarity", "prompt engineering",
        "data preprocessing", "data analysis",
    },
    {
        # Specific cloud *platforms* only. Kept separate from generic DevOps
        # tooling below: knowing Docker/CI-CD doesn't mean you've used AWS.
        "aws", "azure", "gcp", "cloud",
    },
    {
        "docker", "kubernetes", "terraform", "ci/cd", "mlops", "microservices",
    },
    {
        "react", "angular", "vue", "javascript", "typescript", "html", "css",
        "tailwind", "bootstrap",
    },
    {
        "django", "flask", "fastapi", "node.js", "express", "spring",
        "rest api", "rest api development", "graphql", "microservices",
    },
    {
        "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
        "data engineering", "etl", "spark", "hadoop", "airflow",
    },
]


def related_skill_clusters(skill):
    """Return the cluster sets a given (lowercased) skill belongs to."""
    return [cluster for cluster in SKILL_CLUSTERS if skill in cluster]


_EXPERIENCE_RE = re.compile(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", re.IGNORECASE)

_EDUCATION_KEYWORDS = [
    "phd", "ph.d", "doctorate",
    "m.tech", "mtech", "master of", "m.sc", "msc", "mba", "m.s.", "ms in",
    "b.tech", "btech", "bachelor of", "b.sc", "bsc", "b.e.", "be in",
    "diploma",
]


def extract_skills(text):
    """Return taxonomy skills that appear in text (case-insensitive)."""
    lowered = text.lower()
    found = []
    for skill in SKILLS_TAXONOMY:
        pattern = r"(?<![\w.+#])" + re.escape(skill) + r"(?![\w])"
        if re.search(pattern, lowered):
            found.append(skill)
    return found


def extract_experience_years(text):
    """Return the largest 'N years' figure mentioned, or None."""
    matches = _EXPERIENCE_RE.findall(text)
    if not matches:
        return None
    return max(float(m) for m in matches)


def extract_education(text):
    """Return education keywords found in text (original casing not preserved)."""
    lowered = text.lower()
    found = []
    for keyword in _EDUCATION_KEYWORDS:
        if keyword in lowered and keyword not in found:
            found.append(keyword)
    return found


def extract_email(text):
    """Return the first email address found in text, or None."""
    match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    return match.group(0) if match else None
