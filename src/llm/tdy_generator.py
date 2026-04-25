"""TDY Travel answer generation using both basic and ontology enhanced RAG.

Mirrors the pattern from src/llm/generator.py for the TDY travel domain.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient
from src.retrieval.tdy_retriever import (
    TDYRetrievalResult,
    retrieve_tdy_naive,
    retrieve_tdy_enhanced,
)


TDY_SYSTEM_PROMPT = """You are a military travel advisor specializing in TDY (Temporary Duty) travel regulations, per diem rates, and transportation entitlements. You help Service members understand their travel allowances under the Joint Travel Regulations (JTR).

Rules:
1. Only use information from the provided context documents and structured data.
2. Always cite your sources using the format [Source: document name | Section: section].
3. If structured data provides exact rates (per diem, lodging, M&IE), use those exact numbers.
4. If you don't know or the context doesn't contain the answer, say so clearly.
5. Be specific about dollar amounts, regulations, and procedures."""

TDY_NAIVE_TEMPLATE = """Based on the following context documents, answer this TDY travel question:

Question: {query}

Context:
{context}"""

TDY_ENHANCED_TEMPLATE = """Based on the following context documents and structured data, answer this TDY travel question:

Question: {query}

Ontology Expansion Applied: {expansion_info}

Context:
{context}

IMPORTANT: If structured data includes per diem rates, lodging rates, or M&IE rates, include those exact dollar amounts in your answer. These come from official GSA rate tables and are authoritative."""


@dataclass
class TDYRAGAnswer:
    """Result from TDY RAG answer generation."""
    answer: str
    retrieval: TDYRetrievalResult
    sources: list[str] = field(default_factory=list)
    error: Optional[str] = None
    is_enhanced: bool = False


def generate_tdy_naive_answer(
    query: str,
    client: Optional[LLMClient] = None,
) -> TDYRAGAnswer:
    """Generate a TDY travel answer using basic RAG (no ontology enhancement)."""
    client = client or LLMClient()
    retrieval = retrieve_tdy_naive(query)

    user_prompt = TDY_NAIVE_TEMPLATE.format(
        query=query,
        context=retrieval.context_text,
    )

    try:
        answer = client.generate(
            system_prompt=TDY_SYSTEM_PROMPT,
            user_message=user_prompt,
        )
        sources = _extract_sources(retrieval)
        return TDYRAGAnswer(
            answer=answer,
            retrieval=retrieval,
            sources=sources,
            is_enhanced=False,
        )
    except Exception as e:
        return TDYRAGAnswer(
            answer="",
            retrieval=retrieval,
            error=f"Error: {e}",
            is_enhanced=False,
        )


def generate_tdy_enhanced_answer(
    query: str,
    client: Optional[LLMClient] = None,
) -> TDYRAGAnswer:
    """Generate a TDY travel answer using Ontology Enhanced RAG."""
    client = client or LLMClient()
    retrieval = retrieve_tdy_enhanced(query)

    expansion_info = ""
    if retrieval.expansion:
        exp = retrieval.expansion
        parts = []
        if exp.synonyms:
            parts.append(f"Synonyms expanded: {', '.join(list(exp.synonyms.keys())[:5])}")
        if exp.location_codes:
            parts.append(f"Location resolved: {', '.join(exp.location_codes)}")
        if exp.related_regulations:
            parts.append(f"Related regulations: {', '.join(exp.related_regulations[:3])}")
        expansion_info = "; ".join(parts)

    user_prompt = TDY_ENHANCED_TEMPLATE.format(
        query=query,
        expansion_info=expansion_info or "None",
        context=retrieval.context_text,
    )

    try:
        answer = client.generate(
            system_prompt=TDY_SYSTEM_PROMPT,
            user_message=user_prompt,
        )
        sources = _extract_sources(retrieval)
        return TDYRAGAnswer(
            answer=answer,
            retrieval=retrieval,
            sources=sources,
            is_enhanced=True,
        )
    except Exception as e:
        return TDYRAGAnswer(
            answer="",
            retrieval=retrieval,
            error=f"Error: {e}",
            is_enhanced=True,
        )


def _extract_sources(retrieval: TDYRetrievalResult) -> list[str]:
    """Extract unique source citations from retrieval documents."""
    sources = []
    seen = set()
    for doc in retrieval.documents:
        meta = doc.get("metadata", {})
        source = meta.get("source_doc", "Unknown")
        section = meta.get("section_heading", "")
        citation = f"{source} — {section}" if section else source
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)
    return sources
