"""Federal contracts ontology loader and query expansion.

Loads the federal contracts SKOS ontology and provides entity extraction
and query expansion for defense procurement queries. Mirrors the pattern
from tdy_expander.py for the contracts domain.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import SKOS

EX = Namespace("http://ragtag-force.mil/ontology#")

_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "federal_contracts.ttl"
_graph: Optional[Graph] = None
_label_index: Optional[dict[str, URIRef]] = None

CONTRACT_CATEGORIES = {
    EX.ServiceBranch: "service_branch",
    EX.ResearchDomain: "research_domain",
    EX.ContractLocation: "contract_location",
    EX.DefenseContractor: "defense_contractor",
    EX.ContractType: "contract_type",
    EX.AcquisitionRegulation: "acquisition_regulation",
}


def load_contracts_ontology(path: Optional[Path] = None) -> Graph:
    """Load the federal contracts ontology from Turtle file."""
    global _graph
    if _graph is not None and path is None:
        return _graph
    g = Graph()
    g.parse(str(path or _ONTOLOGY_PATH), format="turtle")
    if path is None:
        _graph = g
    return g


def build_label_index(graph: Optional[Graph] = None) -> dict[str, URIRef]:
    """Build a lowercase label -> URI index for entity matching."""
    global _label_index
    if _label_index is not None and graph is None:
        return _label_index

    graph = graph or load_contracts_ontology()
    index: dict[str, URIRef] = {}

    for s, p, o in graph.triples((None, SKOS.prefLabel, None)):
        if isinstance(o, Literal) and isinstance(s, URIRef):
            index[str(o).lower()] = s
    for s, p, o in graph.triples((None, SKOS.altLabel, None)):
        if isinstance(o, Literal) and isinstance(s, URIRef):
            index[str(o).lower()] = s

    if graph is None or graph is _graph:
        _label_index = index
    return index


def _get_all_labels(graph: Graph, uri: URIRef) -> list[str]:
    """Get all labels (pref + alt) for a concept."""
    labels = []
    for _, _, o in graph.triples((uri, SKOS.prefLabel, None)):
        labels.insert(0, str(o))
    for _, _, o in graph.triples((uri, SKOS.altLabel, None)):
        labels.append(str(o))
    return labels


def _get_notation(graph: Graph, uri: URIRef) -> Optional[str]:
    """Get the skos:notation (e.g., NAICS code, state code)."""
    for _, _, o in graph.triples((uri, SKOS.notation, None)):
        return str(o)
    return None


def _get_broader(graph: Graph, uri: URIRef) -> list[URIRef]:
    """Get broader concepts."""
    return [o for _, _, o in graph.triples((uri, SKOS.broader, None)) if isinstance(o, URIRef)]


def _get_related(graph: Graph, uri: URIRef) -> list[URIRef]:
    """Get related concepts."""
    return [o for _, _, o in graph.triples((uri, SKOS.related, None)) if isinstance(o, URIRef)]


def _categorize(graph: Graph, uri: URIRef) -> Optional[str]:
    """Determine which contract category a concept belongs to."""
    if uri in CONTRACT_CATEGORIES:
        return CONTRACT_CATEGORIES[uri]
    for _ in range(3):
        parents = _get_broader(graph, uri)
        for p in parents:
            if p in CONTRACT_CATEGORIES:
                return CONTRACT_CATEGORIES[p]
        if not parents:
            break
        uri = parents[0]
    return None


@dataclass
class ContractExtractedEntities:
    """Entities extracted from a contract intelligence query."""
    service_branches: list[URIRef] = field(default_factory=list)
    research_domains: list[URIRef] = field(default_factory=list)
    locations: list[URIRef] = field(default_factory=list)
    contractors: list[URIRef] = field(default_factory=list)
    contract_types: list[URIRef] = field(default_factory=list)
    regulations: list[URIRef] = field(default_factory=list)
    raw_matches: dict[str, URIRef] = field(default_factory=dict)

    @property
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return bool(
            self.service_branches or self.research_domains or self.locations
            or self.contractors or self.contract_types or self.regulations
        )


@dataclass
class ContractQueryExpansion:
    """Result of ontology-based query expansion for contracts."""
    original_query: str
    expanded_terms: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    related_regulations: list[str] = field(default_factory=list)
    naics_codes: list[str] = field(default_factory=list)
    state_codes: list[str] = field(default_factory=list)
    agency_names: list[str] = field(default_factory=list)
    contractor_names: list[str] = field(default_factory=list)
    entities: Optional[ContractExtractedEntities] = None

    @property
    def expanded_query(self) -> str:
        """Build the expanded query string."""
        unique_terms = list(dict.fromkeys(self.expanded_terms))
        if not unique_terms:
            return self.original_query
        return f"{self.original_query} {' '.join(unique_terms)}"


def extract_contract_entities(
    query: str,
    graph: Optional[Graph] = None,
    label_index: Optional[dict[str, URIRef]] = None,
) -> ContractExtractedEntities:
    """Extract contract-related entities from a query using the ontology."""
    graph = graph or load_contracts_ontology()
    label_index = label_index or build_label_index(graph)

    result = ContractExtractedEntities()
    matched: set[URIRef] = set()

    # N-gram matching (5-word down to 1-word)
    cleaned = re.sub(r"[?!.,;:\"'()\[\]{}]", " ", query)
    words = cleaned.split()
    for n in range(min(5, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            candidate = " ".join(words[i:i + n]).lower()
            if candidate in label_index:
                uri = label_index[candidate]
                if uri not in matched:
                    matched.add(uri)
                    cat = _categorize(graph, uri)
                    result.raw_matches[candidate] = uri
                    if cat == "service_branch":
                        result.service_branches.append(uri)
                    elif cat == "research_domain":
                        result.research_domains.append(uri)
                    elif cat == "contract_location":
                        result.locations.append(uri)
                    elif cat == "defense_contractor":
                        result.contractors.append(uri)
                    elif cat == "contract_type":
                        result.contract_types.append(uri)
                    elif cat == "acquisition_regulation":
                        result.regulations.append(uri)

    return result


def expand_contract_query(
    query: str,
    graph: Optional[Graph] = None,
    entities: Optional[ContractExtractedEntities] = None,
) -> ContractQueryExpansion:
    """Expand a contract query using the SKOS ontology.

    Resolves military jargon into USAspending API parameters:
    - Service branches -> agency names
    - Research domains -> NAICS codes + keywords
    - Locations -> state codes
    - Contractors -> canonical names
    """
    graph = graph or load_contracts_ontology()

    if entities is None:
        label_index = build_label_index(graph)
        entities = extract_contract_entities(query, graph=graph, label_index=label_index)

    expansion = ContractQueryExpansion(original_query=query, entities=entities)
    seen: set[str] = set()

    all_uris = (
        entities.service_branches + entities.research_domains
        + entities.locations + entities.contractors
        + entities.contract_types + entities.regulations
    )

    for uri in all_uris:
        labels = _get_all_labels(graph, uri)
        pref = labels[0] if labels else str(uri).split("#")[-1]
        expansion.synonyms[pref] = labels[1:] if len(labels) > 1 else []

        # Add labels as expansion terms
        for label in labels:
            if label.lower() not in seen:
                seen.add(label.lower())
                expansion.expanded_terms.append(label)

        # Extract structured parameters based on category
        notation = _get_notation(graph, uri)

        if uri in entities.service_branches and pref:
            expansion.agency_names.append(pref)

        if uri in entities.research_domains and notation:
            if notation not in expansion.naics_codes:
                expansion.naics_codes.append(notation)

        if uri in entities.locations and notation:
            if notation not in expansion.state_codes:
                expansion.state_codes.append(notation)

        if uri in entities.contractors and pref:
            expansion.contractor_names.append(pref)

        # Gather related concepts
        for rel_uri in _get_related(graph, uri):
            rel_labels = _get_all_labels(graph, rel_uri)
            for label in rel_labels[:3]:
                if label.lower() not in seen:
                    seen.add(label.lower())
                    expansion.expanded_terms.append(label)

    # Gather regulations from regulation entities
    for uri in entities.regulations:
        labels = _get_all_labels(graph, uri)
        if labels:
            expansion.related_regulations.append(labels[0])

    return expansion
