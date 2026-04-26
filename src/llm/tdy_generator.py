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


TDY_SYSTEM_PROMPT = """You are a military travel advisor specializing in TDY (Temporary Duty) travel regulations, per diem rates, and transportation entitlements under the Joint Travel Regulations (JTR).

Rules:
1. Keep answers to 3-5 sentences. Lead with exact dollar amounts or the key fact.
2. Include one source citation using [Source: document name | Section: section].
3. If structured data provides rates (per diem, lodging, M&IE), state those numbers first.
4. If the context doesn't contain the answer, say so in one sentence.
5. Only use information from the provided context and structured data."""

TDY_NAIVE_TEMPLATE = """Answer this TDY travel question concisely (3-5 sentences) based on the context below:

Question: {query}

Context:
{context}"""

TDY_ENHANCED_TEMPLATE = """Answer this TDY travel question concisely (3-5 sentences) using the context and structured data below. Lead with exact dollar amounts from structured data.

Question: {query}

Ontology Expansion Applied: {expansion_info}

Context:
{context}"""


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
