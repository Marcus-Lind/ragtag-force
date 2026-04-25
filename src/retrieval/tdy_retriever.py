"""TDY Travel hybrid retriever combining vector search with per diem lookup.

Mirrors the pattern of hybrid_retriever.py for the TDY travel domain.
Uses the TDY ontology for query expansion and GSA per diem data for
structured lookups.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.ontology.tdy_expander import TDYQueryExpansion, expand_tdy_query
from src.ingest.embeddings import embed_query
from src.ingest.vector_store import get_chroma_client
from src.retrieval.gsa_client import lookup_perdiem_gsa
from src.config import DEFAULT_TOP_K


@dataclass
class TDYRetrievalResult:
    """Complete TDY retrieval result combining vector and structured data."""
    query: str
    documents: list[dict] = field(default_factory=list)
    structured_data: dict = field(default_factory=dict)
    expansion: Optional[TDYQueryExpansion] = None
    retrieval_time_ms: float = 0.0
    is_enhanced: bool = False

    @property
    def context_text(self) -> str:
        """Build context string from retrieved documents for LLM prompt."""
        parts: list[str] = []
        for i, doc in enumerate(self.documents, 1):
            source = doc.get("metadata", {}).get("source_doc", "Unknown")
            section = doc.get("metadata", {}).get("section_heading", "")
            header = f"[Source: {source}"
            if section:
                header += f" | Section: {section}"
            header += "]"
            parts.append(f"Document {i} {header}:\n{doc['text']}")

        if self.structured_data:
            parts.append("\n--- Structured Data (live from GSA Per Diem API) ---")
            for key, value in self.structured_data.items():
                if value is not None:
                    parts.append(f"{key}: {value}")

        return "\n\n".join(parts)


def _get_tdy_collection():
    """Get the TDY documents ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="tdy_documents",
        metadata={"hnsw:space": "cosine"},
    )


def _tdy_vector_search(query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Perform vector search on the TDY documents collection."""
    collection = _get_tdy_collection()
    query_emb = embed_query(query_text)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    formatted = []
    if results and results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })
    return formatted


def _gather_tdy_structured_data(expansion: Optional[TDYQueryExpansion]) -> dict:
    """Query live GSA Per Diem API based on extracted travel locations.

    Args:
        expansion: TDY query expansion with extracted entities.

    Returns:
        Dictionary of structured per diem data from the live GSA API.
    """
    if not expansion or not expansion.entities:
        return {}

    data: dict = {}
    entities = expansion.entities

    # Per diem lookup via live GSA API
    if entities.locations and expansion.location_codes:
        code = expansion.location_codes[0]
        # Parse city/state from location code (e.g., "SAN_DIEGO_CA" or "DC")
        parts = code.split("_")
        if len(parts) >= 2:
            state = parts[-1]
            city = " ".join(parts[:-1]).title()
        else:
            city = code
            state = None

        perdiem = lookup_perdiem_gsa(city, state)
        if perdiem:
            data["Per Diem Location"] = f"{perdiem['city']}, {perdiem['state'] or ''}"
            data["Lodging Rate (current month)"] = f"${perdiem['lodging_rate']}/night"
            data["M&IE Rate"] = f"${perdiem['mie_rate']}/day"
            data["Total Per Diem"] = f"${perdiem['total_perdiem']}/day"
            data["Data Source"] = perdiem.get("source", "GSA Per Diem API")
            if perdiem.get("seasonal_note"):
                data["Seasonal Info"] = perdiem["seasonal_note"]

    # Mileage rate if POV mentioned
    for uri in entities.transport_modes:
        from src.ontology.tdy_expander import EX
        if uri == EX.POV:
            data["POV Mileage Rate"] = "$0.67/mile (FY2026 GSA rate)"

    return data


def retrieve_tdy_naive(query: str, top_k: int = DEFAULT_TOP_K) -> TDYRetrievalResult:
    """Perform basic TDY retrieval using raw query only.

    No ontology expansion — just vector search on TDY documents.
    """
    start = time.time()
    documents = _tdy_vector_search(query, top_k=top_k)
    elapsed_ms = (time.time() - start) * 1000

    return TDYRetrievalResult(
        query=query,
        documents=documents,
        structured_data={},
        expansion=None,
        retrieval_time_ms=elapsed_ms,
        is_enhanced=False,
    )


def retrieve_tdy_enhanced(query: str, top_k: int = DEFAULT_TOP_K) -> TDYRetrievalResult:
    """Perform Ontology Enhanced TDY retrieval.

    Expands query with TDY ontology, then performs enhanced vector search
    plus GSA per diem structured lookup.
    """
    start = time.time()

    # Step 1: TDY ontology expansion
    expansion = expand_tdy_query(query)

    # Step 2: Enhanced vector search with expanded query
    documents = _tdy_vector_search(expansion.expanded_query, top_k=top_k)

    # Step 3: Structured per diem lookup
    structured_data = _gather_tdy_structured_data(expansion)

    elapsed_ms = (time.time() - start) * 1000

    return TDYRetrievalResult(
        query=query,
        documents=documents,
        structured_data=structured_data,
        expansion=expansion,
        retrieval_time_ms=elapsed_ms,
        is_enhanced=True,
    )
