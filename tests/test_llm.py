"""Tests for the LLM integration layer.

Tests prompt construction (does not make actual API calls).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.prompts import build_naive_prompt, build_enhanced_prompt, SYSTEM_PROMPT
from src.llm.client import LLMClient
from src.llm.generator import RAGAnswer
from src.retrieval.hybrid_retriever import RetrievalResult


class TestPrompts:
    """Test prompt template construction."""

    def test_naive_prompt_includes_query(self):
        """Naive prompt includes the user query."""
        system, user = build_naive_prompt("What is E-4 BAH?", "Some context here.")
        assert "What is E-4 BAH?" in user
        assert "Some context here." in user

    def test_naive_prompt_has_system(self):
        """Naive prompt uses the system prompt."""
        system, _ = build_naive_prompt("query", "context")
        assert "military benefits" in system.lower()
        assert "cite" in system.lower()

    def test_enhanced_prompt_includes_expansion(self):
        """Enhanced prompt includes expansion info."""
        _, user = build_enhanced_prompt(
            query="BAH for E-4",
            context="context",
            expanded_terms=["Specialist", "SPC", "E-4"],
            related_regs=["AR 37-104-4"],
            structured_summary="BAH Rate: $1,116.00",
        )
        assert "Specialist" in user
        assert "AR 37-104-4" in user
        assert "$1,116.00" in user

    def test_system_prompt_requires_citations(self):
        """System prompt instructs the model to cite sources."""
        assert "cite" in SYSTEM_PROMPT.lower() or "citation" in SYSTEM_PROMPT.lower()
        assert "source" in SYSTEM_PROMPT.lower()


class TestLLMClient:
    """Test LLM client configuration."""

    def test_unconfigured_client(self):
        """Client without API key reports not configured."""
        client = LLMClient(api_key=None)
        assert not client.is_configured

    def test_placeholder_key_not_configured(self):
        """Client with placeholder key reports not configured."""
        client = LLMClient(api_key="sk-ant-your-key-here")
        assert not client.is_configured

    def test_configured_client(self):
        """Client with real-looking key reports configured."""
        client = LLMClient(api_key="sk-ant-api03-real-key-here")
        assert client.is_configured

    def test_generate_without_key_raises(self):
        """Generating without API key raises ValueError."""
        client = LLMClient(api_key=None)
        with pytest.raises(ValueError, match="API key"):
            client.generate("system", "user")


class TestRAGAnswer:
    """Test RAGAnswer data class."""

    def test_sources_extraction(self):
        """RAGAnswer extracts unique source documents."""
        retrieval = RetrievalResult(
            query="test",
            documents=[
                {"text": "doc1", "metadata": {"source_doc": "AR_600_8_10"}},
                {"text": "doc2", "metadata": {"source_doc": "DoD_FMR_Vol7A"}},
                {"text": "doc3", "metadata": {"source_doc": "AR_600_8_10"}},
            ],
        )
        answer = RAGAnswer(query="test", answer="response", retrieval=retrieval)
        assert answer.sources == ["AR_600_8_10", "DoD_FMR_Vol7A"]

    def test_error_answer(self):
        """RAGAnswer can carry error information."""
        retrieval = RetrievalResult(query="test", documents=[])
        answer = RAGAnswer(
            query="test",
            answer="",
            retrieval=retrieval,
            error="API key not configured",
        )
        assert answer.error is not None
