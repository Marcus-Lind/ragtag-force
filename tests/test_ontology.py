"""Tests for the SKOS ontology and ontology engine.

Tests ontology loading, entity extraction, and query expansion.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ontology.loader import (
    load_ontology,
    get_all_labels,
    get_notation,
    find_concept_by_label,
    build_label_index,
    get_related,
    EX,
)
from src.ontology.entity_extractor import extract_entities
from src.ontology.query_expander import expand_query


@pytest.fixture
def graph():
    """Load the ontology graph."""
    load_ontology.cache_clear()
    return load_ontology()


@pytest.fixture
def label_index(graph):
    """Build the label index."""
    return build_label_index(graph)


class TestOntologyLoading:
    """Test ontology file loading and basic queries."""

    def test_ontology_loads(self, graph):
        """Ontology file loads without errors and has triples."""
        assert len(graph) > 300

    def test_e4_has_synonyms(self, graph):
        """E-4 concept has all expected altLabels."""
        labels = get_all_labels(graph, EX.E4)
        label_set = {l.lower() for l in labels}
        assert "specialist" in label_set
        assert "spc" in label_set
        assert "e-4" in label_set
        assert "corporal" in label_set
        assert "cpl" in label_set

    def test_fort_campbell_locality(self, graph):
        """Fort Campbell has correct locality code."""
        notation = get_notation(graph, EX.FortCampbell)
        assert notation == "CLARKSVILLE_TN"

    def test_fort_liberty_locality(self, graph):
        """Fort Liberty has correct locality code."""
        notation = get_notation(graph, EX.FortLiberty)
        assert notation == "FAYETTEVILLE_NC"

    def test_bah_related_to_regulations(self, graph):
        """BAH concept is related to governing regulations."""
        related = get_related(graph, EX.BAH)
        related_set = set(related)
        assert EX.AR_37_104_4 in related_set
        assert EX.DoD_FMR_7A_Ch26 in related_set

    def test_find_concept_by_label(self, graph):
        """Can find concepts by various label forms."""
        assert find_concept_by_label(graph, "E-4") == EX.E4
        assert find_concept_by_label(graph, "SPC") == EX.E4
        assert find_concept_by_label(graph, "specialist") == EX.E4
        assert find_concept_by_label(graph, "Fort Campbell") == EX.FortCampbell
        assert find_concept_by_label(graph, "BAH") == EX.BAH


class TestEntityExtraction:
    """Test entity extraction from queries."""

    def test_extract_rank(self, graph, label_index):
        """Extracts rank entity from query."""
        entities = extract_entities("What is E-4 pay?", graph=graph, label_index=label_index)
        assert len(entities.ranks) >= 1

    def test_extract_installation(self, graph, label_index):
        """Extracts installation entity from query."""
        entities = extract_entities(
            "What is BAH at Fort Campbell?", graph=graph, label_index=label_index
        )
        assert len(entities.installations) >= 1

    def test_extract_allowance(self, graph, label_index):
        """Extracts allowance entity from query."""
        entities = extract_entities(
            "What is my BAH rate?", graph=graph, label_index=label_index
        )
        assert len(entities.allowances) >= 1

    def test_extract_dependency_status(self, graph, label_index):
        """Extracts dependency status from query."""
        entities = extract_entities(
            "BAH for a single E-5", graph=graph, label_index=label_index
        )
        assert len(entities.dependency_statuses) >= 1

    def test_extract_multiple_entities(self, graph, label_index):
        """Extracts multiple entity types from complex query."""
        entities = extract_entities(
            "What BAH am I entitled to as a single E-4 at Fort Campbell?",
            graph=graph,
            label_index=label_index,
        )
        assert entities.has_entities
        assert len(entities.ranks) >= 1
        assert len(entities.allowances) >= 1
        assert len(entities.dependency_statuses) >= 1
        assert len(entities.installations) >= 1

    def test_extract_leave(self, graph, label_index):
        """Extracts leave-related entities."""
        entities = extract_entities(
            "How many days of leave can I accrue?",
            graph=graph,
            label_index=label_index,
        )
        assert len(entities.allowances) >= 1


class TestQueryExpansion:
    """Test ontology-based query expansion."""

    def test_expansion_adds_synonyms(self):
        """Query expansion includes rank synonyms."""
        load_ontology.cache_clear()
        result = expand_query("What is E-4 pay?")
        assert any("specialist" in t.lower() for t in result.expanded_terms)
        assert any("spc" in t.lower() for t in result.expanded_terms)

    def test_expansion_finds_regulations(self):
        """Query expansion identifies related regulations."""
        load_ontology.cache_clear()
        result = expand_query("What BAH am I entitled to as a single E-4 at Fort Campbell?")
        assert "AR 37-104-4" in result.related_regulations

    def test_expansion_finds_locality(self):
        """Query expansion resolves installation to locality code."""
        load_ontology.cache_clear()
        result = expand_query("BAH at Fort Campbell")
        assert "CLARKSVILLE_TN" in result.locality_codes

    def test_expansion_finds_grade(self):
        """Query expansion resolves rank to grade notation."""
        load_ontology.cache_clear()
        result = expand_query("Pay for an O-3")
        assert "O-3" in result.grade_notations

    def test_expanded_query_is_longer(self):
        """Expanded query is longer than original due to added terms."""
        load_ontology.cache_clear()
        result = expand_query("What is E-4 BAH at Fort Campbell?")
        assert len(result.expanded_query) > len(result.original_query)
