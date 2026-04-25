"""ChromaDB vector store management.

Handles creating, populating, and querying the ChromaDB collection
used for document retrieval. Uses persistent storage.
"""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.config import CHROMA_PATH
from src.ingest.pdf_parser import DocumentChunk


def get_chroma_client(persist_dir: Optional[Path] = None) -> chromadb.ClientAPI:
    """Create a persistent ChromaDB client.

    Args:
        persist_dir: Directory for ChromaDB persistence. Defaults to config CHROMA_PATH.

    Returns:
        ChromaDB client instance.
    """
    persist_dir = persist_dir or CHROMA_PATH
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def get_or_create_collection(
    client: chromadb.ClientAPI,
    collection_name: str = "military_docs",
) -> chromadb.Collection:
    """Get or create a ChromaDB collection.

    Args:
        client: ChromaDB client.
        collection_name: Name of the collection.

    Returns:
        ChromaDB collection.
    """
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(
    chunks: list[DocumentChunk],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
    batch_size: int = 100,
) -> int:
    """Ingest document chunks with embeddings into ChromaDB.

    Args:
        chunks: List of document chunks.
        embeddings: Corresponding embedding vectors.
        collection: Target ChromaDB collection.
        batch_size: Number of chunks per batch insert.

    Returns:
        Total number of chunks ingested.
    """
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size]

        ids = [f"{c.source_doc}_chunk_{c.chunk_index}" for c in batch_chunks]
        documents = [c.text for c in batch_chunks]
        metadatas = [c.metadata for c in batch_chunks]

        collection.upsert(
            ids=ids,
            embeddings=batch_embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    return total


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 5,
    where_filter: Optional[dict] = None,
) -> dict:
    """Query the ChromaDB collection.

    Args:
        collection: ChromaDB collection to query.
        query_embedding: Query embedding vector.
        top_k: Number of results to return.
        where_filter: Optional metadata filter.

    Returns:
        ChromaDB query results dict with ids, documents, metadatas, distances.
    """
    kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }
    if where_filter:
        kwargs["where"] = where_filter

    return collection.query(**kwargs)
