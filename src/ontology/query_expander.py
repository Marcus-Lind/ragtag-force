"""Query expansion using SKOS ontology.

Takes extracted entities and expands the query with synonyms, related
concepts, hierarchy information, and linked regulations.
"""

from dataclasses import dataclass, field
from typing import Optional

from rdflib import Graph, URIRef

from src.ontology.loader import (
    load_ontology,
    get_all_labels,
    get_notation,
    get_related,
    get_broader,
    get_narrower,
    build_label_index,
)
from src.ontology.entity_extractor import ExtractedEntities, extract_entities


@dataclass
class QueryExpansion:
    """Result of ontology-based query expansion."""

    original_query: str
    expanded_terms: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    related_regulations: list[str] = field(default_factory=list)
    locality_codes: list[str] = field(default_factory=list)
    grade_notations: list[str] = field(default_factory=list)
    dependency_statuses: list[str] = field(default_factory=list)
    entities: Optional[ExtractedEntities] = None

    @property
    def expanded_query(self) -> str:
        """Build the expanded query string combining original + expansion terms."""
        unique_terms = list(dict.fromkeys(self.expanded_terms))
        if not unique_terms:
            return self.original_query
        expansion = " ".join(unique_terms)
        return f"{self.original_query} {expansion}"


def expand_query(
    query: str,
    graph: Optional[Graph] = None,
    entities: Optional[ExtractedEntities] = None,
) -> QueryExpansion:
    """Expand a query using the SKOS ontology.

    Extracts entities (if not provided), then expands with:
    - All synonyms (altLabels) for matched concepts
    - Related concept labels
    - Regulation references
    - Locality codes for installations
    - Grade notations for ranks

    Args:
        query: Original user query.
        graph: The ontology graph. Loaded from default if None.
        entities: Pre-extracted entities. Extracted from query if None.

    Returns:
        QueryExpansion with all expansion data.
    """
    graph = graph or load_ontology()

    if entities is None:
        label_index = build_label_index(graph)
        entities = extract_entities(query, graph=graph, label_index=label_index)

    expansion = QueryExpansion(original_query=query, entities=entities)

    all_concept_uris: list[URIRef] = (
        entities.ranks
        + entities.installations
        + entities.allowances
        + entities.dependency_statuses
        + entities.regulations
    )

    seen_terms: set[str] = set()

    for uri in all_concept_uris:
        # Get all labels (synonyms)
        labels = get_all_labels(graph, uri)
        pref_label = labels[0] if labels else str(uri).split("#")[-1]
        expansion.synonyms[pref_label] = labels[1:] if len(labels) > 1 else []

        for label in labels:
            if label.lower() not in seen_terms:
                seen_terms.add(label.lower())
                expansion.expanded_terms.append(label)

        # Get notation (grade code, locality code)
        notation = get_notation(graph, uri)
        if notation:
            if uri in entities.installations:
                expansion.locality_codes.append(notation)
            elif uri in entities.ranks:
                expansion.grade_notations.append(notation)
            elif uri in entities.dependency_statuses:
                expansion.dependency_statuses.append(notation)

        # Get related concepts (especially regulations)
        related = get_related(graph, uri)
        for related_uri in related:
            related_labels = get_all_labels(graph, related_uri)
            if related_labels:
                # Check if it's a regulation
                broader_uris = get_broader(graph, related_uri)
                from src.ontology.loader import EX
                if EX.Regulation in broader_uris:
                    expansion.related_regulations.append(related_labels[0])
                # Add related labels to expansion
                for rl in related_labels:
                    if rl.lower() not in seen_terms:
                        seen_terms.add(rl.lower())
                        expansion.expanded_terms.append(rl)

    # Deduplicate
    expansion.related_regulations = list(dict.fromkeys(expansion.related_regulations))
    expansion.locality_codes = list(dict.fromkeys(expansion.locality_codes))
    expansion.grade_notations = list(dict.fromkeys(expansion.grade_notations))
    expansion.dependency_statuses = list(dict.fromkeys(expansion.dependency_statuses))

    return expansion
