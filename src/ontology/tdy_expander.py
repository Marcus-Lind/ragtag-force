"""TDY Travel ontology loader and query expansion.

Loads the TDY travel SKOS ontology and provides entity extraction and
query expansion for travel-related queries. Mirrors the pattern from
src/ontology/ but for the TDY travel domain.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import SKOS

EX = Namespace("http://ragtag-force.mil/ontology#")

_TDY_ONTOLOGY_PATH = Path(__file__).resolve().parent.parent.parent / "ontology" / "tdy_travel.ttl"
_tdy_graph: Optional[Graph] = None
_tdy_label_index: Optional[dict[str, URIRef]] = None

# Category URIs for classification
TRAVEL_CATEGORIES = {
    EX.TravelStatus: "travel_status",
    EX.TravelLocation: "travel_location",
    EX.TravelAllowance: "travel_allowance",
    EX.TransportMode: "transport_mode",
    EX.TravelRegulation: "travel_regulation",
}


def load_tdy_ontology(path: Optional[Path] = None) -> Graph:
    """Load the TDY travel ontology from Turtle file."""
    global _tdy_graph
    if _tdy_graph is not None and path is None:
        return _tdy_graph
    g = Graph()
    g.parse(str(path or _TDY_ONTOLOGY_PATH), format="turtle")
    if path is None:
        _tdy_graph = g
    return g


def build_tdy_label_index(graph: Optional[Graph] = None) -> dict[str, URIRef]:
    """Build a lowercase label → URI index for entity matching."""
    global _tdy_label_index
    if _tdy_label_index is not None and graph is None:
        return _tdy_label_index

    graph = graph or load_tdy_ontology()
    index: dict[str, URIRef] = {}

    for s, p, o in graph.triples((None, SKOS.prefLabel, None)):
        if isinstance(o, Literal) and isinstance(s, URIRef):
            index[str(o).lower()] = s
    for s, p, o in graph.triples((None, SKOS.altLabel, None)):
        if isinstance(o, Literal) and isinstance(s, URIRef):
            index[str(o).lower()] = s

    if graph is None or graph is _tdy_graph:
        _tdy_label_index = index
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
    """Get the skos:notation for a concept (e.g., city code)."""
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
    """Determine which travel category a concept belongs to."""
    if uri in TRAVEL_CATEGORIES:
        return TRAVEL_CATEGORIES[uri]
    for _ in range(3):
        parents = _get_broader(graph, uri)
        for p in parents:
            if p in TRAVEL_CATEGORIES:
                return TRAVEL_CATEGORIES[p]
        if not parents:
            break
        uri = parents[0]
    return None


@dataclass
class TDYExtractedEntities:
    """Entities extracted from a TDY travel query."""
    travel_statuses: list[URIRef] = field(default_factory=list)
    locations: list[URIRef] = field(default_factory=list)
    allowances: list[URIRef] = field(default_factory=list)
    transport_modes: list[URIRef] = field(default_factory=list)
    regulations: list[URIRef] = field(default_factory=list)
    raw_matches: dict[str, URIRef] = field(default_factory=dict)

    @property
    def has_entities(self) -> bool:
        return bool(self.travel_statuses or self.locations or self.allowances
                     or self.transport_modes or self.regulations)


@dataclass
class TDYQueryExpansion:
    """Result of TDY ontology-based query expansion."""
    original_query: str
    expanded_terms: list[str] = field(default_factory=list)
    synonyms: dict[str, list[str]] = field(default_factory=dict)
    related_regulations: list[str] = field(default_factory=list)
    location_codes: list[str] = field(default_factory=list)
    entities: Optional[TDYExtractedEntities] = None

    @property
    def expanded_query(self) -> str:
        unique_terms = list(dict.fromkeys(self.expanded_terms))
        if not unique_terms:
            return self.original_query
        return f"{self.original_query} {' '.join(unique_terms)}"


def extract_tdy_entities(
    query: str,
    graph: Optional[Graph] = None,
    label_index: Optional[dict[str, URIRef]] = None,
) -> TDYExtractedEntities:
    """Extract TDY travel entities from a query."""
    graph = graph or load_tdy_ontology()
    label_index = label_index or build_tdy_label_index(graph)

    result = TDYExtractedEntities()
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
                    if cat == "travel_status":
                        result.travel_statuses.append(uri)
                    elif cat == "travel_location":
                        result.locations.append(uri)
                    elif cat == "travel_allowance":
                        result.allowances.append(uri)
                    elif cat == "transport_mode":
                        result.transport_modes.append(uri)
                    elif cat == "travel_regulation":
                        result.regulations.append(uri)

    return result


def expand_tdy_query(
    query: str,
    graph: Optional[Graph] = None,
    entities: Optional[TDYExtractedEntities] = None,
) -> TDYQueryExpansion:
    """Expand a TDY travel query using the SKOS ontology."""
    graph = graph or load_tdy_ontology()

    if entities is None:
        label_index = build_tdy_label_index(graph)
        entities = extract_tdy_entities(query, graph=graph, label_index=label_index)

    expansion = TDYQueryExpansion(original_query=query, entities=entities)
    seen: set[str] = set()

    all_uris = (entities.travel_statuses + entities.locations + entities.allowances
                + entities.transport_modes + entities.regulations)

    for uri in all_uris:
        labels = _get_all_labels(graph, uri)
        pref = labels[0] if labels else str(uri).split("#")[-1]
        expansion.synonyms[pref] = labels[1:] if len(labels) > 1 else []

        for label in labels:
            if label.lower() not in seen:
                seen.add(label.lower())
                expansion.expanded_terms.append(label)

        # Location code for per diem lookup
        notation = _get_notation(graph, uri)
        if notation and uri in entities.locations:
            expansion.location_codes.append(notation)

        # Related concepts (especially regulations)
        for rel_uri in _get_related(graph, uri):
            rel_labels = _get_all_labels(graph, rel_uri)
            if rel_labels:
                broader = _get_broader(graph, rel_uri)
                if EX.TravelRegulation in broader:
                    expansion.related_regulations.append(rel_labels[0])
                for rl in rel_labels:
                    if rl.lower() not in seen:
                        seen.add(rl.lower())
                        expansion.expanded_terms.append(rl)

    expansion.related_regulations = list(dict.fromkeys(expansion.related_regulations))
    expansion.location_codes = list(dict.fromkeys(expansion.location_codes))
    return expansion
