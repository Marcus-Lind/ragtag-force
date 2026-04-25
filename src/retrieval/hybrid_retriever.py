"""Hybrid retriever combining vector search with structured data lookup.

Orchestrates the full retrieval pipeline for both naive and enhanced paths,
combining document context from ChromaDB with structured data from SQLite.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from src.ontology.query_expander import QueryExpansion, expand_query
from src.retrieval.vector_search import naive_search, enhanced_search
from src.retrieval.structured_lookup import lookup_bah, lookup_base_pay, lookup_bas
from src.config import DEFAULT_TOP_K


@dataclass
class RetrievalResult:
    """Complete retrieval result combining vector and structured data."""

    query: str
    documents: list[dict] = field(default_factory=list)
    structured_data: dict = field(default_factory=dict)
    expansion: Optional[QueryExpansion] = None
    retrieval_time_ms: float = 0.0
    is_enhanced: bool = False

    @property
    def context_text(self) -> str:
        """Build context string from retrieved documents for LLM prompt."""
        parts: list[str] = []

        for i, doc in enumerate(self.documents, 1):
            source = doc.get("metadata", {}).get("source_doc", "Unknown")
            section = doc.get("metadata", {}).get("section_heading", "")
            page = doc.get("metadata", {}).get("page_number", "")
            header = f"[Source: {source}"
            if section:
                header += f" | Section: {section}"
            if page:
                header += f" | Page: {page}"
            header += "]"
            parts.append(f"Document {i} {header}:\n{doc['text']}")

        # Add structured data
        if self.structured_data:
            parts.append("\n--- Structured Data (from official tables) ---")
            for key, value in self.structured_data.items():
                if value is not None:
                    parts.append(f"{key}: {value}")

        return "\n\n".join(parts)


def _gather_structured_data(expansion: Optional[QueryExpansion]) -> dict:
    """Query structured data based on extracted entities.

    Args:
        expansion: Query expansion with extracted entities.

    Returns:
        Dictionary of structured data results.
    """
    if not expansion or not expansion.entities:
        return {}

    data: dict = {}
    entities = expansion.entities

    # BAH lookup: need grade + locality + dependency status
    if entities.ranks and entities.installations:
        grade = expansion.grade_notations[0] if expansion.grade_notations else None
        locality = expansion.locality_codes[0] if expansion.locality_codes else None
        dep_status = expansion.dependency_statuses[0] if expansion.dependency_statuses else "without_dependents"

        if grade and locality:
            bah = lookup_bah(grade, dep_status, locality)
            if bah:
                data["BAH Monthly Rate"] = f"${bah['monthly_rate']:,.2f}"
                data["BAH Locality"] = bah["locality"]
                data["BAH Grade"] = bah["grade"]
                data["BAH Dependency Status"] = bah["dependency_status"].replace("_", " ").title()

    # Base pay lookup: need grade
    if entities.ranks and expansion.grade_notations:
        grade = expansion.grade_notations[0]
        pay = lookup_base_pay(grade, years_of_service=0)
        if pay:
            data["Base Pay (entry level)"] = f"${pay['monthly_rate']:,.2f}/mo"

    # BAS lookup: if allowance mentions BAS
    for allowance_uri in entities.allowances:
        from src.ontology.loader import EX
        if allowance_uri == EX.BAS:
            # Determine if enlisted or officer
            if entities.ranks and expansion.grade_notations:
                grade = expansion.grade_notations[0]
                category = "Enlisted" if grade.startswith("E") else "Officer"
            else:
                category = "Enlisted"
            bas = lookup_bas(category)
            if bas:
                data["BAS Monthly Rate"] = f"${bas['monthly_rate']:,.2f}"
                data["BAS Category"] = bas["category"]

    return data


def retrieve_naive(query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
    """Perform naive retrieval using raw query only.

    No ontology expansion — just vector search + basic structured lookup.

    Args:
        query: User's original query.
        top_k: Number of vector search results.

    Returns:
        RetrievalResult with documents and any structured data.
    """
    start = time.time()

    documents = naive_search(query, top_k=top_k)

    elapsed_ms = (time.time() - start) * 1000

    return RetrievalResult(
        query=query,
        documents=documents,
        structured_data={},
        expansion=None,
        retrieval_time_ms=elapsed_ms,
        is_enhanced=False,
    )


def retrieve_enhanced(query: str, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
    """Perform ontology-enhanced retrieval.

    Expands query with SKOS ontology, then performs enhanced vector search
    plus structured data lookup using extracted entities.

    Args:
        query: User's original query.
        top_k: Number of vector search results.

    Returns:
        RetrievalResult with documents, structured data, and expansion info.
    """
    start = time.time()

    # Step 1: Ontology expansion
    expansion = expand_query(query)

    # Step 2: Enhanced vector search
    documents = enhanced_search(
        original_query=query,
        expanded_query=expansion.expanded_query,
        top_k=top_k,
    )

    # Step 3: Structured data lookup
    structured_data = _gather_structured_data(expansion)

    elapsed_ms = (time.time() - start) * 1000

    return RetrievalResult(
        query=query,
        documents=documents,
        structured_data=structured_data,
        expansion=expansion,
        retrieval_time_ms=elapsed_ms,
        is_enhanced=True,
    )
