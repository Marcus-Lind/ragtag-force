"""Federal contracts hybrid retriever combining vector search with USAspending API.

Mirrors the pattern of tdy_retriever.py for the contracts domain.
Uses the contracts ontology for query expansion and USAspending API for
structured lookups.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.ontology.contracts_expander import ContractQueryExpansion, expand_contract_query
from src.ingest.embeddings import embed_query
from src.ingest.vector_store import get_chroma_client
from src.retrieval.usaspending_client import search_contracts
from src.config import DEFAULT_TOP_K


@dataclass
class ContractRetrievalResult:
    """Complete contract retrieval result combining vector and API data."""
    query: str
    documents: list[dict] = field(default_factory=list)
    structured_data: dict = field(default_factory=dict)
    expansion: Optional[ContractQueryExpansion] = None
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
            parts.append("\n--- Structured Data (live from USAspending.gov API) ---")
            for key, value in self.structured_data.items():
                if value is not None:
                    parts.append(f"{key}: {value}")

        return "\n\n".join(parts)


def _get_contracts_collection():
    """Get the contracts documents ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="contracts_documents",
        metadata={"hnsw:space": "cosine"},
    )


def _contracts_vector_search(query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Perform vector search on the contracts documents collection."""
    collection = _get_contracts_collection()
    count = collection.count()
    if count == 0:
        return []

    query_emb = embed_query(query_text)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=min(top_k, count),
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


def _gather_contract_structured_data(expansion: Optional[ContractQueryExpansion]) -> dict:
    """Query USAspending API based on extracted entities.

    Args:
        expansion: Contract query expansion with resolved API parameters.

    Returns:
        Dictionary of structured contract data from USAspending.
    """
    if not expansion or not expansion.entities or not expansion.entities.has_entities:
        return {}

    # Build API query from ontology-resolved parameters
    keywords = None
    for uri in expansion.entities.research_domains:
        from src.ontology.contracts_expander import _get_all_labels, load_contracts_ontology
        graph = load_contracts_ontology()
        labels = _get_all_labels(graph, uri)
        if labels:
            keywords = [labels[0]]
            break

    # If no research domain but we have keywords from the original query
    if not keywords and expansion.expanded_terms:
        # Use the first few meaningful expanded terms as keywords
        domain_terms = [t for t in expansion.expanded_terms[:3] if len(t) > 3]
        if domain_terms:
            keywords = domain_terms[:2]

    agency_name = "Department of Defense"
    if expansion.agency_names:
        # Use the most specific agency
        agency_name = expansion.agency_names[0]

    naics_codes = expansion.naics_codes if expansion.naics_codes else None
    state_code = expansion.state_codes[0] if expansion.state_codes else None
    contractor_name = expansion.contractor_names[0] if expansion.contractor_names else None

    result = search_contracts(
        keywords=keywords,
        agency_name=agency_name,
        naics_codes=naics_codes,
        state_code=state_code,
        recipient_name=contractor_name,
        limit=5,
    )

    data: dict = {}
    contracts = result.get("contracts", [])

    if contracts:
        data["Data Source"] = "USAspending.gov API (live)"
        data["Contracts Found"] = str(result.get("total_found", 0))
        if result.get("has_more"):
            data["Contracts Found"] += "+"

        for i, c in enumerate(contracts[:5], 1):
            data[f"Contract {i}"] = (
                f"{c['recipient']} - {c['amount_display']} - {c['description'][:100]}"
            )
            data[f"Contract {i} Award ID"] = c["award_id"]

        # Query parameters for transparency
        params = result.get("query_params", {})
        if params.get("keywords"):
            data["Search Keywords"] = ", ".join(params["keywords"])
        if params.get("state") != "Any":
            data["State Filter"] = params["state"]
        if params.get("naics_codes"):
            data["NAICS Codes"] = ", ".join(params["naics_codes"])
    elif result.get("error"):
        data["API Error"] = result["error"]
        data["Data Source"] = "USAspending.gov API (error)"
    else:
        data["Data Source"] = "USAspending.gov API (no results)"
        data["Note"] = "No contracts matched the resolved parameters"

    return data


def retrieve_contracts_naive(query: str, top_k: int = DEFAULT_TOP_K) -> ContractRetrievalResult:
    """Perform basic contract retrieval using raw query only.

    No ontology expansion - just vector search on contract documents.
    """
    start = time.time()
    documents = _contracts_vector_search(query, top_k=top_k)
    elapsed_ms = (time.time() - start) * 1000

    return ContractRetrievalResult(
        query=query,
        documents=documents,
        structured_data={},
        expansion=None,
        retrieval_time_ms=elapsed_ms,
        is_enhanced=False,
    )


def retrieve_contracts_enhanced(query: str, top_k: int = DEFAULT_TOP_K) -> ContractRetrievalResult:
    """Perform Ontology Enhanced contract retrieval.

    Expands query with contracts ontology, then performs enhanced vector search
    plus USAspending API structured lookup.
    """
    start = time.time()

    # Step 1: Ontology expansion
    expansion = expand_contract_query(query)

    # Step 2: Enhanced vector search
    documents = _contracts_vector_search(expansion.expanded_query, top_k=top_k)

    # Step 3: USAspending API lookup
    structured_data = _gather_contract_structured_data(expansion)

    elapsed_ms = (time.time() - start) * 1000

    return ContractRetrievalResult(
        query=query,
        documents=documents,
        structured_data=structured_data,
        expansion=expansion,
        retrieval_time_ms=elapsed_ms,
        is_enhanced=True,
    )
