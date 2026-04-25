"""Vector search using ChromaDB.

Provides both basic (raw query) and ontology enhanced vector search
over the ingested document collection.
"""

from typing import Optional

import chromadb

from src.config import CHROMA_PATH, DEFAULT_TOP_K
from src.ingest.embeddings import embed_query
from src.ingest.vector_store import get_chroma_client, get_or_create_collection, query_collection


def _format_results(raw_results: dict) -> list[dict]:
    """Convert ChromaDB results to a clean list of result dicts.

    Args:
        raw_results: Raw ChromaDB query results.

    Returns:
        List of dicts with text, metadata, and distance.
    """
    results: list[dict] = []
    if not raw_results.get("documents") or not raw_results["documents"][0]:
        return results

    for i, doc in enumerate(raw_results["documents"][0]):
        result = {
            "text": doc,
            "metadata": raw_results["metadatas"][0][i] if raw_results.get("metadatas") else {},
            "distance": raw_results["distances"][0][i] if raw_results.get("distances") else None,
            "id": raw_results["ids"][0][i] if raw_results.get("ids") else None,
        }
        results.append(result)

    return results


def naive_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection: Optional[chromadb.Collection] = None,
) -> list[dict]:
    """Perform basic vector search using the raw query.

    Args:
        query: User's original query string.
        top_k: Number of results to return.
        collection: ChromaDB collection. Created from config if None.

    Returns:
        List of result dicts with text, metadata, and distance.
    """
    if collection is None:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

    query_emb = embed_query(query)
    raw_results = query_collection(collection, query_emb, top_k=top_k)
    return _format_results(raw_results)


def enhanced_search(
    original_query: str,
    expanded_query: str,
    top_k: int = DEFAULT_TOP_K,
    collection: Optional[chromadb.Collection] = None,
) -> list[dict]:
    """Perform Ontology Enhanced vector search using the expanded query.

    The expanded query includes synonyms, related terms, and regulation
    references from the ontology, which produces better semantic matches.

    Args:
        original_query: User's original query string.
        expanded_query: Ontology-expanded query string.
        top_k: Number of results to return.
        collection: ChromaDB collection. Created from config if None.

    Returns:
        List of result dicts with text, metadata, and distance.
    """
    if collection is None:
        client = get_chroma_client()
        collection = get_or_create_collection(client)

    query_emb = embed_query(expanded_query)
    raw_results = query_collection(collection, query_emb, top_k=top_k)
    return _format_results(raw_results)
