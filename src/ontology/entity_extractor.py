"""Entity extraction from user queries.

Identifies military-specific entities (ranks, installations, allowances,
dependency statuses, regulations) in natural language queries using the
SKOS ontology label index.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from rdflib import Graph, URIRef

from src.ontology.loader import (
    EX,
    build_label_index,
    get_broader,
    get_notation,
    load_ontology,
)


@dataclass
class ExtractedEntities:
    """Container for entities extracted from a query."""

    ranks: list[URIRef] = field(default_factory=list)
    installations: list[URIRef] = field(default_factory=list)
    allowances: list[URIRef] = field(default_factory=list)
    dependency_statuses: list[URIRef] = field(default_factory=list)
    regulations: list[URIRef] = field(default_factory=list)
    raw_matches: dict[str, URIRef] = field(default_factory=dict)

    @property
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return bool(
            self.ranks
            or self.installations
            or self.allowances
            or self.dependency_statuses
            or self.regulations
        )


def _categorize_entity(graph: Graph, concept_uri: URIRef) -> Optional[str]:
    """Determine which top-level category a concept belongs to.

    Walks the broader hierarchy to find the category.

    Args:
        graph: The ontology graph.
        concept_uri: URI of the concept.

    Returns:
        Category name string or None.
    """
    category_map = {
        EX.Rank: "rank",
        EX.EnlistedRank: "rank",
        EX.OfficerRank: "rank",
        EX.WarrantOfficerRank: "rank",
        EX.Allowance: "allowance",
        EX.Installation: "installation",
        EX.DependencyStatus: "dependency_status",
        EX.Regulation: "regulation",
    }

    # Check if the concept itself is a category
    if concept_uri in category_map:
        return category_map[concept_uri]

    # Walk broader hierarchy (max 3 levels to avoid infinite loops)
    current = concept_uri
    for _ in range(3):
        parents = get_broader(graph, current)
        if not parents:
            break
        for parent in parents:
            if parent in category_map:
                return category_map[parent]
        current = parents[0]

    return None


def _tokenize_for_matching(query: str) -> list[str]:
    """Generate candidate substrings from the query for entity matching.

    Generates n-grams (1 to 5 words) to match multi-word labels like
    'Fort Campbell' or 'Basic Allowance for Housing'.

    Args:
        query: The user query string.

    Returns:
        List of candidate substrings, longest first.
    """
    # Strip punctuation from individual words for matching
    cleaned = re.sub(r"[?!.,;:\"'()\[\]{}]", " ", query)
    words = cleaned.split()
    candidates: list[str] = []

    # Generate n-grams from 5-word down to 1-word
    for n in range(min(5, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            candidate = " ".join(words[i : i + n])
            candidates.append(candidate)

    return candidates


def extract_entities(
    query: str,
    graph: Optional[Graph] = None,
    label_index: Optional[dict[str, URIRef]] = None,
) -> ExtractedEntities:
    """Extract military entities from a user query.

    Uses the ontology label index to find matches in the query text.
    Handles multi-word entities and avoids duplicate matches.

    Args:
        query: User's natural language query.
        graph: The ontology graph. Loaded from default if None.
        label_index: Pre-built label index. Built from graph if None.

    Returns:
        ExtractedEntities with categorized matches.
    """
    graph = graph or load_ontology()
    label_index = label_index or build_label_index(graph)

    result = ExtractedEntities()
    matched_uris: set[URIRef] = set()
    query_lower = query.lower()

    # Also check for common patterns directly
    # E-4, O-3, etc.
    grade_pattern = re.findall(r"\b([eEoO])-?(\d{1,2})\b", query)
    for prefix, num in grade_pattern:
        normalized = f"{prefix.upper()}-{num}"
        if normalized.lower() in label_index:
            uri = label_index[normalized.lower()]
            if uri not in matched_uris:
                matched_uris.add(uri)
                result.ranks.append(uri)
                result.raw_matches[normalized] = uri

    # Check dependency status keywords
    dep_keywords = {
        "single": "without_dependents",
        "married": "with_dependents",
        "with dependents": "with_dependents",
        "without dependents": "without_dependents",
        "no dependents": "without_dependents",
        "with family": "with_dependents",
        "dependent": "with_dependents",
    }
    for keyword, _ in dep_keywords.items():
        if keyword in query_lower:
            if keyword in label_index:
                uri = label_index[keyword]
                if uri not in matched_uris:
                    matched_uris.add(uri)
                    result.dependency_statuses.append(uri)
                    result.raw_matches[keyword] = uri

    # N-gram matching for multi-word entities
    candidates = _tokenize_for_matching(query)
    for candidate in candidates:
        candidate_lower = candidate.lower()
        if candidate_lower in label_index:
            uri = label_index[candidate_lower]
            if uri not in matched_uris:
                matched_uris.add(uri)
                category = _categorize_entity(graph, uri)
                result.raw_matches[candidate] = uri

                if category == "rank":
                    result.ranks.append(uri)
                elif category == "installation":
                    result.installations.append(uri)
                elif category == "allowance":
                    result.allowances.append(uri)
                elif category == "dependency_status":
                    result.dependency_statuses.append(uri)
                elif category == "regulation":
                    result.regulations.append(uri)

    return result
