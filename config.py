"""Application configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATA_DIR = os.path.join(BASE_DIR, "data")

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")

MAX_CONTENT_LENGTH_MB = int(os.getenv("MAX_CONTENT_LENGTH_MB", "100"))
ALLOWED_EXTENSIONS = {"pdf"}
MIN_RESUMES_RECOMMENDED = 10

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
CHROMA_COLLECTION = "resumes"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

# Off by default: the curated skill-cluster table (skills_taxonomy.py) is the
# always-on baseline for related-skill matching. This oracle is a tail-case
# fallback for skills the curated table doesn't cover, judged by the LLM and
# cached - but a small local model has been shown to answer unreliably here
# (see resume_matcher/scoring.py), so it should only be enabled once you've
# configured a model you trust for this specific judgment (e.g. Gemini, or a
# larger local model you've verified yourself).
SKILL_ORACLE_ENABLED = os.getenv("SKILL_ORACLE_ENABLED", "false").strip().lower() == "true"
SKILL_ORACLE_TOP_K = int(os.getenv("SKILL_ORACLE_TOP_K", "3"))

# Weights for combining signals into the final match score (must sum to 1.0).
SCORE_WEIGHTS = {
    "semantic": 0.5,
    "skills": 0.35,
    "experience": 0.10,
    "education": 0.05,
}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
