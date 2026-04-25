"""Tests for the retrieval layer.

Tests structured data lookups from SQLite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.structured_loader import load_all_structured_data
from src.retrieval.structured_lookup import lookup_bah, lookup_base_pay, lookup_bas


@pytest.fixture(scope="module")
def test_db(tmp_path_factory):
    """Create a test database with all structured data."""
    db_path = tmp_path_factory.mktemp("data") / "test_entitlements.db"
    load_all_structured_data(db_path)
    return db_path


class TestStructuredLookup:
    """Test SQLite-based structured data lookups."""

    def test_bah_lookup_e4_clarksville(self, test_db):
        """Can look up BAH for E-4 at Clarksville (Fort Campbell)."""
        result = lookup_bah("E-4", "without_dependents", "CLARKSVILLE_TN", db_path=test_db)
        assert result is not None
        assert result["monthly_rate"] > 0
        assert result["grade"] == "E-4"

    def test_bah_lookup_with_dependents(self, test_db):
        """BAH with dependents is higher than without."""
        without = lookup_bah("E-5", "without_dependents", "CLARKSVILLE_TN", db_path=test_db)
        with_dep = lookup_bah("E-5", "with_dependents", "CLARKSVILLE_TN", db_path=test_db)
        assert without is not None
        assert with_dep is not None
        assert with_dep["monthly_rate"] > without["monthly_rate"]

    def test_bah_lookup_invalid_returns_none(self, test_db):
        """Invalid BAH lookup returns None."""
        result = lookup_bah("E-99", "without_dependents", "NOWHERE", db_path=test_db)
        assert result is None

    def test_base_pay_enlisted(self, test_db):
        """Can look up enlisted base pay."""
        result = lookup_base_pay("E-4", years_of_service=2, db_path=test_db)
        assert result is not None
        assert result["monthly_rate"] > 0

    def test_base_pay_officer(self, test_db):
        """Can look up officer base pay."""
        result = lookup_base_pay("O-3", years_of_service=4, db_path=test_db)
        assert result is not None
        assert result["monthly_rate"] > 0

    def test_base_pay_higher_yos_means_more(self, test_db):
        """More years of service means higher pay."""
        entry = lookup_base_pay("E-5", years_of_service=0, db_path=test_db)
        experienced = lookup_base_pay("E-5", years_of_service=8, db_path=test_db)
        assert entry is not None
        assert experienced is not None
        assert experienced["monthly_rate"] > entry["monthly_rate"]

    def test_bas_enlisted(self, test_db):
        """Can look up enlisted BAS rate."""
        result = lookup_bas("Enlisted", db_path=test_db)
        assert result is not None
        assert result["monthly_rate"] > 400

    def test_bas_officer(self, test_db):
        """Can look up officer BAS rate."""
        result = lookup_bas("Officer", db_path=test_db)
        assert result is not None
        assert result["monthly_rate"] > 200

    def test_bas_enlisted_higher_than_officer(self, test_db):
        """Enlisted BAS is higher than officer BAS."""
        enlisted = lookup_bas("Enlisted", db_path=test_db)
        officer = lookup_bas("Officer", db_path=test_db)
        assert enlisted["monthly_rate"] > officer["monthly_rate"]
