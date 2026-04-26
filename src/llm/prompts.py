"""Prompt templates for the RAG pipeline.

Defines system prompts and user message templates that enforce citation
format and structured answer generation.
"""

SYSTEM_PROMPT = """You are a knowledgeable military benefits and entitlements advisor for the US Army.
Answer questions about military pay, allowances (BAH, BAS, OHA), leave policies,
and regulations based ONLY on the provided context documents and data.

RULES:
1. Keep answers to 3-5 sentences. Lead with the most important fact or number.
2. Cite your source using the format [Source: document_name | Section: section_name].
3. If structured data (rates, tables) is provided, state the exact numbers up front.
4. If the context doesn't contain enough information, say so in one sentence.
5. Never make up information. Only use what's in the provided context.
6. Use standard pay grade format (e.g., E-4, O-3) and specify monthly vs annual for dollar amounts."""

NAIVE_USER_TEMPLATE = """Based on the following context documents, answer this question:

Question: {query}

Context:
{context}

Provide a concise answer (3-5 sentences) with one citation. Lead with the key fact."""

ENHANCED_USER_TEMPLATE = """Based on the following context documents and structured data, answer this question:

Question: {query}

{expansion_info}

Context:
{context}

Provide a concise answer (3-5 sentences). State exact rates/amounts first, then context. One citation is sufficient."""


def build_naive_prompt(query: str, context: str) -> tuple[str, str]:
    """Build the basic RAG prompt (no ontology expansion info).

    Args:
        query: User's original query.
        context: Retrieved document context.

    Returns:
        Tuple of (system_prompt, user_message).
    """
    user_msg = NAIVE_USER_TEMPLATE.format(query=query, context=context)
    return SYSTEM_PROMPT, user_msg


def build_enhanced_prompt(
    query: str,
    context: str,
    expanded_terms: list[str] | None = None,
    related_regs: list[str] | None = None,
    structured_summary: str = "",
) -> tuple[str, str]:
    """Build the Ontology Enhanced RAG prompt.

    Includes expansion metadata so the LLM knows what synonyms and
    regulations were used to retrieve the context.

    Args:
        query: User's original query.
        context: Retrieved document context (includes structured data).
        expanded_terms: Terms added by ontology expansion.
        related_regs: Regulations identified by ontology.
        structured_summary: Summary of structured data values.

    Returns:
        Tuple of (system_prompt, user_message).
    """
    expansion_parts: list[str] = []

    if expanded_terms:
        expansion_parts.append(
            f"Ontology Expansion: The query was enhanced with these related terms: "
            f"{', '.join(expanded_terms[:15])}"
        )

    if related_regs:
        expansion_parts.append(
            f"Related Regulations: {', '.join(related_regs)}"
        )

    if structured_summary:
        expansion_parts.append(f"Structured Data:\n{structured_summary}")

    expansion_info = "\n".join(expansion_parts) if expansion_parts else ""

    user_msg = ENHANCED_USER_TEMPLATE.format(
        query=query,
        context=context,
        expansion_info=expansion_info,
    )
    return SYSTEM_PROMPT, user_msg
