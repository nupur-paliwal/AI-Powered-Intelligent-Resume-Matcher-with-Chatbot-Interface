"""Chroma-backed vector store of resume chunks, used for RAG retrieval."""
import functools
import sys

try:
    # Ubuntu 20.04's system sqlite3 (3.31) is older than Chroma's minimum
    # (3.35); swap in the pysqlite3-binary wheel before chromadb imports.
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from resume_matcher.embeddings import get_embeddings

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


@functools.lru_cache(maxsize=1)
def get_store():
    """Return the persistent Chroma collection of resume chunks."""
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_DIR,
    )


def index_resume(candidate_id, filename, text):
    """Split a resume's text into chunks and (re)index them for retrieval."""
    delete_resume(candidate_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return

    store = get_store()
    ids = [f"{candidate_id}::{i}" for i in range(len(chunks))]
    metadatas = [{"candidate_id": candidate_id, "filename": filename} for _ in chunks]
    store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)


def delete_resume(candidate_id):
    """Remove any previously indexed chunks for a candidate (safe on re-upload)."""
    store = get_store()
    existing = store.get(where={"candidate_id": candidate_id})
    if existing and existing.get("ids"):
        store.delete(ids=existing["ids"])


def clear_all():
    """Remove every indexed chunk for every candidate from the vector store."""
    get_store().delete_collection()
    get_store.cache_clear()


def search(query, top_k=None, candidate_id=None):
    """Return the top-k most relevant resume chunks for a query.

    If candidate_id is given, the search is restricted to that candidate's resume.
    """
    store = get_store()
    top_k = top_k or config.RAG_TOP_K
    where = {"candidate_id": candidate_id} if candidate_id else None
    results = store.similarity_search(query, k=top_k, filter=where)
    return [
        {
            "text": doc.page_content,
            "candidate_id": doc.metadata.get("candidate_id"),
            "filename": doc.metadata.get("filename"),
        }
        for doc in results
    ]
