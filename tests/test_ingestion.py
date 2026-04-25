"""Tests for the data ingestion pipeline.

Tests structured data loading and PDF parsing.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.structured_loader import load_all_structured_data
from src.ingest.pdf_parser import parse_pdf, parse_all_pdfs, DocumentChunk
from src.config import DATA_RAW_PATH


class TestStructuredLoader:
    """Test CSV → SQLite loading."""

    def test_load_all_creates_tables(self, tmp_path):
        """All CSV data loads into SQLite tables."""
        db_path = tmp_path / "test.db"
        counts = load_all_structured_data(db_path)

        assert "bah_rates" in counts
        assert "enlisted_base_pay" in counts
        assert "officer_base_pay" in counts
        assert "bas_rates" in counts

        assert counts["bah_rates"] > 0
        assert counts["enlisted_base_pay"] > 0
        assert counts["officer_base_pay"] > 0
        assert counts["bas_rates"] > 0

    def test_bah_rates_queryable(self, tmp_path):
        """BAH rates are queryable after loading."""
        db_path = tmp_path / "test.db"
        load_all_structured_data(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT monthly_rate FROM bah_rates WHERE grade='E-4' AND locality_code='CLARKSVILLE_TN' AND dependency_status='without_dependents'"
        )
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] > 0

    def test_idempotent(self, tmp_path):
        """Loading twice produces same result."""
        db_path = tmp_path / "test.db"
        counts1 = load_all_structured_data(db_path)
        counts2 = load_all_structured_data(db_path)
        assert counts1 == counts2

    def test_bas_has_two_rows(self, tmp_path):
        """BAS table has exactly 2 rows (enlisted + officer)."""
        db_path = tmp_path / "test.db"
        counts = load_all_structured_data(db_path)
        assert counts["bas_rates"] == 2


class TestPDFParser:
    """Test PDF parsing."""

    @pytest.mark.skipif(
        not list(DATA_RAW_PATH.glob("*.pdf")) and not list(DATA_RAW_PATH.glob("*.PDF")),
        reason="No PDF files in data/raw/",
    )
    def test_parse_single_pdf(self):
        """Can parse a single PDF into chunks."""
        pdfs = list(DATA_RAW_PATH.glob("*.pdf")) + list(DATA_RAW_PATH.glob("*.PDF"))
        # Pick the smallest PDF for speed
        smallest = min(pdfs, key=lambda p: p.stat().st_size)
        chunks = parse_pdf(smallest)

        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)
        assert all(c.text.strip() for c in chunks)
        assert all(c.source_doc for c in chunks)

    @pytest.mark.skipif(
        not list(DATA_RAW_PATH.glob("*.pdf")) and not list(DATA_RAW_PATH.glob("*.PDF")),
        reason="No PDF files in data/raw/",
    )
    def test_chunks_have_metadata(self):
        """Parsed chunks include required metadata fields."""
        pdfs = list(DATA_RAW_PATH.glob("*.pdf")) + list(DATA_RAW_PATH.glob("*.PDF"))
        smallest = min(pdfs, key=lambda p: p.stat().st_size)
        chunks = parse_pdf(smallest)

        for chunk in chunks[:5]:
            assert "source_doc" in chunk.metadata
            assert "page_number" in chunk.metadata
            assert chunk.metadata["source_doc"] == smallest.stem
