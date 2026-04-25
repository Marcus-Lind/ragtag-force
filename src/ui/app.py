"""RAG-Tag Force — Streamlit UI.

Two-column comparison interface showing naive RAG vs ontology-enhanced RAG
side by side to demonstrate the value of SKOS ontology query expansion.
"""

import sys
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from src.config import CHROMA_PATH, SQLITE_PATH, ONTOLOGY_PATH, ANTHROPIC_API_KEY
from src.llm.client import LLMClient
from src.llm.generator import generate_naive_answer, generate_enhanced_answer, RAGAnswer


# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG-Tag Force",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Example Questions ───────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What BAH am I entitled to as a single E-4 at Fort Campbell?",
    "How many days of leave can I accrue per year as an O-3?",
    "What is my BAS rate as an enlisted Soldier?",
    "What housing allowance changes when I add a dependent?",
    "What regulation governs my entitlement to OHA overseas?",
]


def _check_system_status() -> dict[str, bool]:
    """Check if all system components are available."""
    status: dict[str, bool] = {}

    # ChromaDB
    try:
        from src.ingest.vector_store import get_chroma_client, get_or_create_collection
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        count = collection.count()
        status["chromadb"] = count > 0
        status["chromadb_count"] = count
    except Exception:
        status["chromadb"] = False
        status["chromadb_count"] = 0

    # SQLite
    try:
        import sqlite3
        conn = sqlite3.connect(str(SQLITE_PATH))
        cursor = conn.execute("SELECT COUNT(*) FROM bah_rates")
        count = cursor.fetchone()[0]
        conn.close()
        status["sqlite"] = count > 0
        status["sqlite_bah_count"] = count
    except Exception:
        status["sqlite"] = False
        status["sqlite_bah_count"] = 0

    # Ontology
    try:
        from src.ontology.loader import load_ontology
        g = load_ontology()
        status["ontology"] = len(g) > 0
        status["ontology_triples"] = len(g)
    except Exception:
        status["ontology"] = False
        status["ontology_triples"] = 0

    # LLM
    status["llm"] = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "sk-ant-your-key-here")

    return status


def _render_sidebar(status: dict) -> None:
    """Render the sidebar with system status and info."""
    st.sidebar.title("🪖 RAG-Tag Force")
    st.sidebar.markdown("*Ontology-Enhanced RAG for Military Benefits*")
    st.sidebar.divider()

    st.sidebar.subheader("System Status")

    # ChromaDB
    if status.get("chromadb"):
        st.sidebar.success(f"✅ ChromaDB: {status.get('chromadb_count', 0):,} chunks")
    else:
        st.sidebar.error("❌ ChromaDB: No documents ingested")

    # SQLite
    if status.get("sqlite"):
        st.sidebar.success(f"✅ SQLite: {status.get('sqlite_bah_count', 0)} BAH rates")
    else:
        st.sidebar.error("❌ SQLite: Database not populated")

    # Ontology
    if status.get("ontology"):
        st.sidebar.success(f"✅ Ontology: {status.get('ontology_triples', 0)} triples")
    else:
        st.sidebar.error("❌ Ontology: TTL not loaded")

    # LLM
    if status.get("llm"):
        st.sidebar.success("✅ Anthropic API: Configured")
    else:
        st.sidebar.error("❌ Anthropic API: Key not set")

    st.sidebar.divider()
    st.sidebar.subheader("About")
    st.sidebar.markdown(
        """
        **The Thesis**: Naive RAG fails military personnel because it doesn't
        understand that "SPC", "Specialist", "Corporal", and "E-4" are the
        same thing.

        Our **SKOS ontology layer** expands every query with synonyms, rank
        hierarchies, and installation-to-locality mappings — producing
        dramatically more accurate answers.

        *SCSP Hackathon 2026 | GenAI.mil Track*
        """
    )


def _render_answer_column(answer: RAGAnswer, label: str) -> None:
    """Render a single answer column."""
    if answer.error:
        st.error(f"Error: {answer.error}")
        return

    # Answer text
    st.markdown(answer.answer)

    # Metadata expanders
    with st.expander(f"📊 Retrieval Details ({answer.retrieval.retrieval_time_ms:.0f}ms)"):
        st.caption(f"Documents retrieved: {len(answer.retrieval.documents)}")

        if answer.retrieval.structured_data:
            st.markdown("**Structured Data (from official tables):**")
            for key, value in answer.retrieval.structured_data.items():
                st.markdown(f"- **{key}**: {value}")

        if answer.sources:
            st.markdown("**Source Documents:**")
            for source in answer.sources:
                st.markdown(f"- {source}")

    # Enhanced-only: show ontology expansion
    if answer.is_enhanced and answer.retrieval.expansion:
        expansion = answer.retrieval.expansion
        with st.expander("🧠 Ontology Expansion Details"):
            if expansion.synonyms:
                st.markdown("**Synonym Expansion:**")
                for concept, syns in expansion.synonyms.items():
                    if syns:
                        st.markdown(f"- **{concept}** → {', '.join(syns[:8])}")

            if expansion.related_regulations:
                st.markdown("**Related Regulations:**")
                for reg in expansion.related_regulations:
                    st.markdown(f"- 📋 {reg}")

            if expansion.locality_codes:
                st.markdown(f"**Locality Codes:** {', '.join(expansion.locality_codes)}")

            if expansion.grade_notations:
                st.markdown(f"**Pay Grades:** {', '.join(expansion.grade_notations)}")

            if expansion.dependency_statuses:
                st.markdown(f"**Dependency Status:** {', '.join(expansion.dependency_statuses)}")


def main() -> None:
    """Main Streamlit application."""
    # Initialize session state
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "naive_answer" not in st.session_state:
        st.session_state.naive_answer = None
    if "enhanced_answer" not in st.session_state:
        st.session_state.enhanced_answer = None

    # Check system status
    status = _check_system_status()

    # Sidebar
    _render_sidebar(status)

    # Main content
    st.title("🪖 RAG-Tag Force")
    st.markdown("### Military Benefits & Entitlements Navigator")
    st.markdown(
        "Ask a question about military pay, allowances, or regulations. "
        "Compare **naive RAG** (raw vector search) vs **ontology-enhanced RAG** "
        "(SKOS expansion + structured data)."
    )

    # Check readiness
    if not status.get("llm"):
        st.warning(
            "⚠️ Anthropic API key not configured. "
            "Add `ANTHROPIC_API_KEY=your-key` to your `.env` file."
        )

    if not status.get("chromadb"):
        st.warning(
            "⚠️ No documents in ChromaDB. Run `python scripts/ingest.py` first."
        )

    # Example questions
    st.markdown("**Try an example question:**")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        with cols[i]:
            if st.button(example[:40] + "...", key=f"example_{i}", use_container_width=True):
                st.session_state.current_query = example

    # Query input
    query = st.text_input(
        "Your question:",
        value=st.session_state.get("current_query", ""),
        placeholder="e.g., What BAH am I entitled to as a single E-4 at Fort Campbell?",
        key="query_input",
    )

    if st.button("🔍 Ask", type="primary", use_container_width=True) and query:
        if not status.get("llm"):
            st.error("Cannot generate answers without Anthropic API key configured.")
            return

        # Create LLM client
        llm_client = LLMClient()

        # Run both pipelines side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📄 Naive RAG")
            st.caption("Raw query → Vector search → LLM")
            with st.spinner("Searching with naive RAG..."):
                naive_result = generate_naive_answer(query, client=llm_client)
                st.session_state.naive_answer = naive_result

        with col2:
            st.subheader("🧠 Ontology-Enhanced RAG")
            st.caption("Query → SKOS Expansion → Enhanced search + Structured data → LLM")
            with st.spinner("Searching with ontology-enhanced RAG..."):
                enhanced_result = generate_enhanced_answer(query, client=llm_client)
                st.session_state.enhanced_answer = enhanced_result

        # Display results
        col1, col2 = st.columns(2)
        with col1:
            _render_answer_column(naive_result, "Naive")
        with col2:
            _render_answer_column(enhanced_result, "Enhanced")

        # Track history
        st.session_state.query_history.append(query)
        if "current_query" in st.session_state:
            del st.session_state.current_query

    # Show previous results if they exist
    elif st.session_state.get("naive_answer") and st.session_state.get("enhanced_answer"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Naive RAG")
            _render_answer_column(st.session_state.naive_answer, "Naive")
        with col2:
            st.subheader("🧠 Ontology-Enhanced RAG")
            _render_answer_column(st.session_state.enhanced_answer, "Enhanced")


if __name__ == "__main__":
    main()
