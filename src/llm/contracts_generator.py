"""Federal contract intelligence answer generation using both basic and ontology enhanced RAG.

Mirrors the pattern from src/llm/tdy_generator.py for the contracts domain.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient
from src.retrieval.contracts_retriever import (
    ContractRetrievalResult,
    retrieve_contracts_naive,
    retrieve_contracts_enhanced,
)


CONTRACTS_SYSTEM_PROMPT = """You are a defense procurement analyst specializing in federal contract intelligence. You help users understand DoD spending, contract awards, and defense industry activity using data from USAspending.gov and federal acquisition knowledge.

Rules:
1. Keep answers to 3-5 sentences. Lead with the most important finding or number.
2. Cite your source using [Source: document name | Section: section].
3. If structured data provides real contract awards, highlight the top recipients and amounts.
4. If the context doesn't have the answer, say so in one sentence.
5. Only use information from the provided context and structured data."""

CONTRACTS_NAIVE_TEMPLATE = """Answer this federal contracting question concisely (3-5 sentences) based on the context below:

Question: {query}

Context:
{context}"""

CONTRACTS_ENHANCED_TEMPLATE = """Answer this federal contracting question concisely (3-5 sentences) using the context and structured data below. Lead with specific contract data if available.

Question: {query}

Ontology Expansion Applied: {expansion_info}

Context:
{context}"""


@dataclass
class ContractRAGAnswer:
    """Result from contract RAG answer generation."""
    answer: str
    retrieval: ContractRetrievalResult
    sources: list[str] = field(default_factory=list)
    error: Optional[str] = None
    is_enhanced: bool = False


def generate_contracts_naive_answer(
    query: str,
    client: Optional[LLMClient] = None,
) -> ContractRAGAnswer:
    """Generate a contract intelligence answer using basic RAG."""
    client = client or LLMClient()
    retrieval = retrieve_contracts_naive(query)

    user_prompt = CONTRACTS_NAIVE_TEMPLATE.format(
        query=query,
        context=retrieval.context_text,
    )

    try:
        answer = client.generate(
            system_prompt=CONTRACTS_SYSTEM_PROMPT,
            user_message=user_prompt,
        )
        sources = _extract_sources(retrieval)
        return ContractRAGAnswer(
            answer=answer,
            retrieval=retrieval,
            sources=sources,
            is_enhanced=False,
        )
    except Exception as e:
        return ContractRAGAnswer(
            answer="",
            retrieval=retrieval,
            error=f"Error: {e}",
            is_enhanced=False,
        )


def generate_contracts_enhanced_answer(
    query: str,
    client: Optional[LLMClient] = None,
) -> ContractRAGAnswer:
    """Generate a contract intelligence answer using Ontology Enhanced RAG."""
    client = client or LLMClient()
    retrieval = retrieve_contracts_enhanced(query)

    expansion_info = ""
    if retrieval.expansion:
        exp = retrieval.expansion
        parts = []
        if exp.agency_names:
            parts.append(f"Agency resolved: {', '.join(exp.agency_names)}")
        if exp.naics_codes:
            parts.append(f"NAICS codes: {', '.join(exp.naics_codes)}")
        if exp.state_codes:
            parts.append(f"Location: {', '.join(exp.state_codes)}")
        if exp.contractor_names:
            parts.append(f"Contractor: {', '.join(exp.contractor_names)}")
        if exp.synonyms:
            parts.append(f"Terms expanded: {', '.join(list(exp.synonyms.keys())[:5])}")
        expansion_info = "; ".join(parts)

    user_prompt = CONTRACTS_ENHANCED_TEMPLATE.format(
        query=query,
        expansion_info=expansion_info or "None",
        context=retrieval.context_text,
    )

    try:
        answer = client.generate(
            system_prompt=CONTRACTS_SYSTEM_PROMPT,
            user_message=user_prompt,
        )
        sources = _extract_sources(retrieval)
        return ContractRAGAnswer(
            answer=answer,
            retrieval=retrieval,
            sources=sources,
            is_enhanced=True,
        )
    except Exception as e:
        return ContractRAGAnswer(
            answer="",
            retrieval=retrieval,
            error=f"Error: {e}",
            is_enhanced=True,
        )


def _extract_sources(retrieval: ContractRetrievalResult) -> list[str]:
    """Extract unique source citations from retrieval documents."""
    sources = []
    seen = set()
    for doc in retrieval.documents:
        meta = doc.get("metadata", {})
        source = meta.get("source_doc", "Unknown")
        section = meta.get("section_heading", "")
        citation = f"{source} - {section}" if section else source
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)
    return sources
