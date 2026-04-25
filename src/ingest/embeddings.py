"""Embedding model wrapper for sentence-transformers.

Provides a consistent interface for generating embeddings using the
bge-base-en-v1.5 model, used for both document ingestion and query time.
"""

from functools import lru_cache
from typing import Optional

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model(model_name: Optional[str] = None) -> SentenceTransformer:
    """Load and cache the sentence-transformer embedding model.

    Args:
        model_name: HuggingFace model ID. Defaults to config EMBEDDING_MODEL.

    Returns:
        Loaded SentenceTransformer model.
    """
    model_name = model_name or EMBEDDING_MODEL
    print(f"  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Embedding model loaded (dim={model.get_sentence_embedding_dimension()})")
    return model


def embed_texts(texts: list[str], model: Optional[SentenceTransformer] = None) -> list[list[float]]:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        model: Pre-loaded model. If None, loads default model.

    Returns:
        List of embedding vectors (as lists of floats).
    """
    model = model or get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str, model: Optional[SentenceTransformer] = None) -> list[float]:
    """Generate embedding for a single query string.

    Args:
        query: Query text to embed.
        model: Pre-loaded model. If None, loads default model.

    Returns:
        Embedding vector as list of floats.
    """
    model = model or get_embedding_model()
    embedding = model.encode(query, normalize_embeddings=True)
    return embedding.tolist()
