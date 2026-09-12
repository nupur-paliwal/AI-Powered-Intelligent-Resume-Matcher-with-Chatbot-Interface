"""Sentence-transformers embeddings, wrapped for LangChain/Chroma."""
import functools

from langchain_huggingface import HuggingFaceEmbeddings

import config


@functools.lru_cache(maxsize=1)
def get_embeddings():
    """Return a cached embeddings model shared by scoring and the vector store."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
