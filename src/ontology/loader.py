"""Ontology loader for the SKOS military entitlements ontology.

Loads the Turtle file with RDFLib and provides helper methods for
querying concepts, labels, and relationships.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import SKOS, RDF

from src.config import ONTOLOGY_PATH

EX = Namespace("http://ragtag-force.mil/ontology#")


@lru_cache(maxsize=1)
def load_ontology(ttl_path: Optional[Path] = None) -> Graph:
    """Load and cache the SKOS ontology graph.

    Args:
        ttl_path: Path to the Turtle file. Defaults to config ONTOLOGY_PATH.

    Returns:
        Loaded RDFLib Graph.
    """
    ttl_path = ttl_path or ONTOLOGY_PATH
    g = Graph()
    g.parse(str(ttl_path), format="turtle")
    return g


def get_all_labels(graph: Graph, concept_uri: URIRef) -> list[str]:
    """Get all labels (prefLabel + altLabels) for a concept.

    Args:
        graph: The ontology graph.
        concept_uri: URI of the concept.

    Returns:
        List of all label strings.
    """
    labels: list[str] = []
    for label in graph.objects(concept_uri, SKOS.prefLabel):
        labels.append(str(label))
    for label in graph.objects(concept_uri, SKOS.altLabel):
        labels.append(str(label))
    return labels


def get_notation(graph: Graph, concept_uri: URIRef) -> Optional[str]:
    """Get the notation (code) for a concept.

    Args:
        graph: The ontology graph.
        concept_uri: URI of the concept.

    Returns:
        Notation string or None.
    """
    for notation in graph.objects(concept_uri, SKOS.notation):
        return str(notation)
    return None


def get_broader(graph: Graph, concept_uri: URIRef) -> list[URIRef]:
    """Get broader (parent) concepts.

    Args:
        graph: The ontology graph.
        concept_uri: URI of the concept.

    Returns:
        List of broader concept URIs.
    """
    return list(graph.objects(concept_uri, SKOS.broader))


def get_narrower(graph: Graph, concept_uri: URIRef) -> list[URIRef]:
    """Get narrower (child) concepts.

    Args:
        graph: The ontology graph.
        concept_uri: URI of the concept.

    Returns:
        List of narrower concept URIs.
    """
    return list(graph.objects(concept_uri, SKOS.narrower))


def get_related(graph: Graph, concept_uri: URIRef) -> list[URIRef]:
    """Get related concepts.

    Args:
        graph: The ontology graph.
        concept_uri: URI of the concept.

    Returns:
        List of related concept URIs.
    """
    return list(graph.objects(concept_uri, SKOS.related))


def find_concept_by_label(graph: Graph, label: str) -> Optional[URIRef]:
    """Find a concept URI by matching any label (case-insensitive).

    Searches prefLabel and altLabel across all concepts.

    Args:
        graph: The ontology graph.
        label: Label string to search for.

    Returns:
        Concept URI if found, None otherwise.
    """
    label_lower = label.lower().strip()

    for concept in graph.subjects(RDF.type, SKOS.Concept):
        for lbl in graph.objects(concept, SKOS.prefLabel):
            if str(lbl).lower() == label_lower:
                return concept
        for lbl in graph.objects(concept, SKOS.altLabel):
            if str(lbl).lower() == label_lower:
                return concept
    return None


def get_all_concepts_by_category(graph: Graph, category_uri: URIRef) -> list[URIRef]:
    """Get all concepts that are narrower than (children of) a category.

    Args:
        graph: The ontology graph.
        category_uri: URI of the parent category concept.

    Returns:
        List of child concept URIs.
    """
    return list(graph.subjects(SKOS.broader, category_uri))


def build_label_index(graph: Graph) -> dict[str, URIRef]:
    """Build a case-insensitive label → concept URI index for fast lookups.

    Indexes both prefLabel and altLabel values.

    Args:
        graph: The ontology graph.

    Returns:
        Dictionary mapping lowercase label strings to concept URIs.
    """
    index: dict[str, URIRef] = {}
    for concept in graph.subjects(RDF.type, SKOS.Concept):
        for label in graph.objects(concept, SKOS.prefLabel):
            index[str(label).lower()] = concept
        for label in graph.objects(concept, SKOS.altLabel):
            index[str(label).lower()] = concept
    return index
