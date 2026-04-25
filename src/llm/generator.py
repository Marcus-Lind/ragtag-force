"""Answer generation orchestrator.

Ties together retrieval and LLM generation for both naive and
ontology enhanced RAG pipelines.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.llm.client import LLMClient, get_llm_client
from src.llm.prompts import build_naive_prompt, build_enhanced_prompt
from src.retrieval.hybrid_retriever import (
    RetrievalResult,
    retrieve_naive,
    retrieve_enhanced,
)
from src.config import DEFAULT_TOP_K


@dataclass
class RAGAnswer:
    """Complete RAG answer with metadata for UI display."""

    query: str
    answer: str
    retrieval: RetrievalResult
    is_enhanced: bool = False
    error: Optional[str] = None

    @property
    def sources(self) -> list[str]:
        """Extract unique source documents from retrieval results."""
        sources: set[str] = set()
        for doc in self.retrieval.documents:
            source = doc.get("metadata", {}).get("source_doc", "")
            if source:
                sources.add(source)
        return sorted(sources)

    @property
    def expanded_terms_display(self) -> list[str]:
        """Get ontology expansion terms for display (enhanced only)."""
        if self.retrieval.expansion:
            return self.retrieval.expansion.expanded_terms[:20]
        return []

    @property
    def regulations_display(self) -> list[str]:
        """Get related regulations for display (enhanced only)."""
        if self.retrieval.expansion:
            return self.retrieval.expansion.related_regulations
        return []


def generate_naive_answer(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    client: Optional[LLMClient] = None,
) -> RAGAnswer:
    """Generate an answer using basic RAG (no ontology enhancement).

    Args:
        query: User's original query.
        top_k: Number of documents to retrieve.
        client: LLM client. Uses default if None.

    Returns:
        RAGAnswer with the naive response.
    """
    client = client or get_llm_client()

    # Retrieve documents
    retrieval = retrieve_naive(query, top_k=top_k)

    # Build prompt
    system_prompt, user_msg = build_naive_prompt(
        query=query,
        context=retrieval.context_text,
    )

    # Generate answer
    try:
        answer = client.generate(system_prompt, user_msg)
    except Exception as e:
        return RAGAnswer(
            query=query,
            answer="",
            retrieval=retrieval,
            is_enhanced=False,
            error=str(e),
        )

    return RAGAnswer(
        query=query,
        answer=answer,
        retrieval=retrieval,
        is_enhanced=False,
    )


def generate_enhanced_answer(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    client: Optional[LLMClient] = None,
) -> RAGAnswer:
    """Generate an answer using Ontology Enhanced RAG.

    Args:
        query: User's original query.
        top_k: Number of documents to retrieve.
        client: LLM client. Uses default if None.

    Returns:
        RAGAnswer with the enhanced response.
    """
    client = client or get_llm_client()

    # Retrieve with ontology expansion
    retrieval = retrieve_enhanced(query, top_k=top_k)

    # Build structured data summary
    structured_summary = ""
    if retrieval.structured_data:
        structured_summary = "\n".join(
            f"  {k}: {v}" for k, v in retrieval.structured_data.items()
        )

    # Build prompt
    expanded_terms = retrieval.expansion.expanded_terms if retrieval.expansion else None
    related_regs = retrieval.expansion.related_regulations if retrieval.expansion else None

    system_prompt, user_msg = build_enhanced_prompt(
        query=query,
        context=retrieval.context_text,
        expanded_terms=expanded_terms,
        related_regs=related_regs,
        structured_summary=structured_summary,
    )

    # Generate answer
    try:
        answer = client.generate(system_prompt, user_msg)
    except Exception as e:
        return RAGAnswer(
            query=query,
            answer="",
            retrieval=retrieval,
            is_enhanced=True,
            error=str(e),
        )

    return RAGAnswer(
        query=query,
        answer=answer,
        retrieval=retrieval,
        is_enhanced=True,
    )
