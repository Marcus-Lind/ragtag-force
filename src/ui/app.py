"""RAG-Tag Force — Streamlit UI.

Enterprise-grade two-column comparison interface showing naive RAG vs
ontology-enhanced RAG side by side for the SCSP Hackathon 2026 demo.
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


# -- Page Config --
st.set_page_config(
    page_title="RAG-Tag Force | Military Benefits Navigator",
    page_icon="\U0001FA96",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS --
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #E5E7EB;
}

.rtf-hero {
    background: linear-gradient(135deg, #1B2A4A 0%, #2D4A7A 60%, #3B6B9A 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.rtf-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(197,164,78,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.rtf-hero h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.02em;
}
.rtf-hero p {
    font-size: 0.95rem;
    opacity: 0.85;
    margin: 0;
    font-weight: 300;
    max-width: 600px;
}
.rtf-hero .rtf-badge {
    display: inline-block;
    background: rgba(197, 164, 78, 0.25);
    color: #E8D48B;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 20px;
    margin-bottom: 0.75rem;
    border: 1px solid rgba(197, 164, 78, 0.35);
}

.rtf-pipeline {
    background: #F8F9FB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    font-size: 0.8rem;
    color: #4B5563;
    line-height: 1.8;
}
.rtf-pipeline .rtf-pipe-label {
    font-weight: 600;
    color: #1B2A4A;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.25rem;
}
.rtf-pipeline code {
    background: #E5E7EB;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    font-size: 0.75rem;
    color: #1B2A4A;
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
}
.rtf-pipeline .rtf-arrow { color: #9CA3AF; margin: 0 0.15rem; }
.rtf-pipeline .rtf-enhanced { color: #92711A; font-weight: 600; }

div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    padding: 0.35rem 0.9rem !important;
    border: 1px solid #D1D5DB !important;
    color: #374151 !important;
    background: white !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
    border-color: #C5A44E !important;
    color: #92711A !important;
    background: #FFFDF5 !important;
    box-shadow: 0 1px 4px rgba(197,164,78,0.15) !important;
}

div[data-testid="stTextInput"] input {
    border-radius: 8px !important;
    border: 1.5px solid #D1D5DB !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #C5A44E !important;
    box-shadow: 0 0 0 2px rgba(197,164,78,0.15) !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #1B2A4A 0%, #2D4A7A 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    box-shadow: 0 4px 12px rgba(27,42,74,0.3) !important;
    transform: translateY(-1px) !important;
}

.rtf-answer-card {
    background: white;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.5rem;
    min-height: 200px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.rtf-answer-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
.rtf-answer-card.rtf-naive { border-left: 4px solid #9CA3AF; }
.rtf-answer-card.rtf-enhanced { border-left: 4px solid #C5A44E; }

.rtf-card-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #F3F4F6;
}
.rtf-card-header .rtf-card-icon {
    width: 32px; height: 32px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.rtf-card-header .rtf-card-icon.rtf-naive-icon { background: #F3F4F6; }
.rtf-card-header .rtf-card-icon.rtf-enhanced-icon { background: #FEF9E7; }
.rtf-card-header h3 { font-size: 0.95rem; font-weight: 600; color: #1B2A4A; margin: 0; }
.rtf-card-header .rtf-card-subtitle { font-size: 0.72rem; color: #9CA3AF; font-weight: 400; }

.rtf-metrics {
    display: flex; gap: 0.75rem; margin-top: 1rem;
    padding-top: 0.75rem; border-top: 1px solid #F3F4F6;
}
.rtf-metric {
    background: #F8F9FB; border-radius: 6px;
    padding: 0.4rem 0.65rem; font-size: 0.72rem; color: #6B7280;
}
.rtf-metric strong { color: #1B2A4A; font-weight: 600; }

.rtf-expansion-tag {
    display: inline-block; background: #FEF9E7; color: #92711A;
    font-size: 0.7rem; font-weight: 500; padding: 0.2rem 0.5rem;
    border-radius: 4px; margin: 0.15rem; border: 1px solid #F5E6A3;
}
.rtf-reg-tag {
    display: inline-block; background: #EFF6FF; color: #1E40AF;
    font-size: 0.7rem; font-weight: 500; padding: 0.2rem 0.5rem;
    border-radius: 4px; margin: 0.15rem; border: 1px solid #BFDBFE;
}

section[data-testid="stSidebar"] { background: #1B2A4A; color: white; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #CBD5E1; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color: white; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }

.rtf-status-item { display: flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0; font-size: 0.82rem; }
.rtf-status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.rtf-status-dot.rtf-ok { background: #34D399; }
.rtf-status-dot.rtf-err { background: #F87171; }
.rtf-status-label { color: #94A3B8; font-size: 0.78rem; }
.rtf-status-value { color: white; font-weight: 500; font-size: 0.78rem; margin-left: auto; }

details[data-testid="stExpander"] {
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    margin-top: 0.5rem !important;
}
details[data-testid="stExpander"] summary {
    font-size: 0.82rem !important; font-weight: 500 !important;
}

.rtf-section-label {
    font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #9CA3AF; margin-bottom: 0.5rem;
}

.rtf-footer {
    text-align: center; padding: 2rem 0 1rem 0; color: #9CA3AF;
    font-size: 0.72rem; border-top: 1px solid #E5E7EB; margin-top: 3rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -- Example Questions --
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

    try:
        from src.ontology.loader import load_ontology
        g = load_ontology()
        status["ontology"] = len(g) > 0
        status["ontology_triples"] = len(g)
    except Exception:
        status["ontology"] = False
        status["ontology_triples"] = 0

    status["llm"] = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "sk-ant-your-key-here")

    return status


def _render_sidebar(status: dict) -> None:
    """Render the sidebar with system status and architecture info."""
    st.sidebar.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 1.3rem; font-weight: 700; color: white; letter-spacing: -0.02em;">
                &#x1FA96; RAG-Tag Force
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 0.2rem;">
                Ontology-Enhanced RAG for Military Benefits
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        '<div class="rtf-section-label" style="color: #94A3B8;">System Status</div>',
        unsafe_allow_html=True,
    )

    def _status_row(label: str, ok: bool, value: str) -> str:
        dot = "rtf-ok" if ok else "rtf-err"
        return (
            f'<div class="rtf-status-item">'
            f'<span class="rtf-status-dot {dot}"></span>'
            f'<span class="rtf-status-label">{label}</span>'
            f'<span class="rtf-status-value">{value}</span>'
            f'</div>'
        )

    status_html = ""
    status_html += _status_row(
        "Vector Store",
        status.get("chromadb", False),
        f"{status.get('chromadb_count', 0):,} chunks" if status.get("chromadb") else "Empty",
    )
    status_html += _status_row(
        "Structured DB",
        status.get("sqlite", False),
        f"{status.get('sqlite_bah_count', 0)} rates" if status.get("sqlite") else "Empty",
    )
    status_html += _status_row(
        "SKOS Ontology",
        status.get("ontology", False),
        f"{status.get('ontology_triples', 0)} triples" if status.get("ontology") else "Not loaded",
    )
    status_html += _status_row(
        "LLM (Anthropic)",
        status.get("llm", False),
        "Connected" if status.get("llm") else "No key",
    )

    st.sidebar.markdown(
        f'<div style="margin-bottom: 1rem;">{status_html}</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        '<div class="rtf-section-label" style="color: #94A3B8;">How It Works</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div style="font-size: 0.78rem; color: #CBD5E1; line-height: 1.7;">
            <strong style="color: white;">The Problem:</strong> Naive RAG fails military
            personnel because "SPC", "Specialist", and "E-4" are the same rank &mdash; but
            vector search doesn't know that.<br/><br/>
            <strong style="color: white;">Our Solution:</strong> A SKOS ontology layer
            expands queries with synonyms, rank hierarchies, installation-to-locality
            mappings, and regulation links &mdash; then augments retrieval with authoritative
            structured data from official rate tables.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0; font-size: 0.7rem; color: #64748B;">
            <div style="font-weight: 600; color: #94A3B8; margin-bottom: 0.2rem;">
                SCSP Hackathon 2026
            </div>
            GenAI.mil Track
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_answer_card(answer: RAGAnswer, variant: str) -> None:
    """Render a styled answer card.

    Args:
        answer: The RAG answer to display.
        variant: Either 'naive' or 'enhanced'.
    """
    is_enhanced = variant == "enhanced"
    card_class = "rtf-enhanced" if is_enhanced else "rtf-naive"
    icon_class = "rtf-enhanced-icon" if is_enhanced else "rtf-naive-icon"
    icon = "&#x1F9E0;" if is_enhanced else "&#x1F4C4;"
    title = "Ontology-Enhanced RAG" if is_enhanced else "Naive RAG"
    subtitle = (
        "SKOS Expansion &rarr; Enhanced Search &rarr; Structured Data &rarr; LLM"
        if is_enhanced
        else "Raw Query &rarr; Vector Search &rarr; LLM"
    )

    st.markdown(
        f"""
        <div class="rtf-answer-card {card_class}">
            <div class="rtf-card-header">
                <div class="rtf-card-icon {icon_class}">{icon}</div>
                <div>
                    <h3>{title}</h3>
                    <div class="rtf-card-subtitle">{subtitle}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if answer.error:
        st.error(f"Error: {answer.error}")
        return

    st.markdown(answer.answer)

    doc_count = len(answer.retrieval.documents)
    time_ms = answer.retrieval.retrieval_time_ms
    metrics_html = (
        f'<div class="rtf-metrics">'
        f'<div class="rtf-metric">&#9201; <strong>{time_ms:.0f}ms</strong> retrieval</div>'
        f'<div class="rtf-metric">&#x1F4D1; <strong>{doc_count}</strong> documents</div>'
    )
    if answer.retrieval.structured_data:
        metrics_html += (
            f'<div class="rtf-metric">&#x1F4CA; <strong>'
            f'{len(answer.retrieval.structured_data)}</strong> data fields</div>'
        )
    metrics_html += "</div>"
    st.markdown(metrics_html, unsafe_allow_html=True)

    if answer.retrieval.structured_data:
        with st.expander("Structured Data (from official rate tables)"):
            for key, value in answer.retrieval.structured_data.items():
                st.markdown(f"**{key}:** {value}")

    if answer.sources:
        with st.expander(f"Source Documents ({len(answer.sources)})"):
            for source in answer.sources:
                st.markdown(f"- `{source}`")

    if is_enhanced and answer.retrieval.expansion:
        expansion = answer.retrieval.expansion
        with st.expander("Ontology Expansion Details"):
            if expansion.synonyms:
                st.markdown("**Synonym Expansion:**")
                tags = ""
                for concept, syns in expansion.synonyms.items():
                    if syns:
                        for syn in syns[:8]:
                            tags += f'<span class="rtf-expansion-tag">{syn}</span>'
                st.markdown(tags, unsafe_allow_html=True)

            if expansion.related_regulations:
                st.markdown("**Related Regulations:**")
                reg_tags = ""
                for reg in expansion.related_regulations:
                    reg_tags += f'<span class="rtf-reg-tag">{reg}</span>'
                st.markdown(reg_tags, unsafe_allow_html=True)

            cols = st.columns(3)
            if expansion.locality_codes:
                with cols[0]:
                    st.markdown(f"**Locality:** {', '.join(expansion.locality_codes)}")
            if expansion.grade_notations:
                with cols[1]:
                    st.markdown(f"**Pay Grade:** {', '.join(expansion.grade_notations)}")
            if expansion.dependency_statuses:
                with cols[2]:
                    st.markdown(f"**Dep. Status:** {', '.join(expansion.dependency_statuses)}")

def _run_query(query: str, status: dict) -> None:
    """Execute the RAG pipeline for a given query and store results."""
    if not status.get("llm"):
        st.error("Cannot generate answers without Anthropic API key configured.")
        return

    llm_client = LLMClient()

    st.markdown(
        f"""
        <div style="background: #F8F9FB; border-left: 3px solid #C5A44E; padding: 0.75rem 1rem;
                    border-radius: 0 8px 8px 0; margin-bottom: 1.5rem; font-size: 0.9rem;">
            <span style="color: #9CA3AF; font-size: 0.72rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.06em;">Question</span><br/>
            <span style="color: #1B2A4A; font-weight: 500;">{query}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.spinner("Running naive RAG pipeline..."):
            naive_result = generate_naive_answer(query, client=llm_client)
            st.session_state.naive_answer = naive_result

    with col2:
        with st.spinner("Running ontology-enhanced RAG pipeline..."):
            enhanced_result = generate_enhanced_answer(query, client=llm_client)
            st.session_state.enhanced_answer = enhanced_result

    col1, col2 = st.columns(2, gap="large")
    with col1:
        _render_answer_card(naive_result, "naive")
    with col2:
        _render_answer_card(enhanced_result, "enhanced")

    st.session_state.query_history.append(query)
    st.session_state.last_query = query


def main() -> None:
    """Main Streamlit application."""
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "naive_answer" not in st.session_state:
        st.session_state.naive_answer = None
    if "enhanced_answer" not in st.session_state:
        st.session_state.enhanced_answer = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    status = _check_system_status()
    _render_sidebar(status)

    # -- Hero Section --
    st.markdown(
        """
        <div class="rtf-hero">
            <div class="rtf-badge">SCSP Hackathon 2026 &middot; GenAI.mil</div>
            <h1>RAG-Tag Force</h1>
            <p>
                Compare naive RAG against ontology-enhanced RAG for military benefits
                and entitlements. See how SKOS knowledge graphs dramatically improve
                retrieval accuracy.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -- Pipeline Diagram --
    st.markdown(
        """
        <div class="rtf-pipeline">
            <div class="rtf-pipe-label">Pipeline Architecture</div>
            <div>
                <strong>Naive:</strong>
                <code>Query</code> <span class="rtf-arrow">&rarr;</span>
                <code>Vector Search</code> <span class="rtf-arrow">&rarr;</span>
                <code>LLM</code> <span class="rtf-arrow">&rarr;</span>
                <code>Answer</code>
            </div>
            <div>
                <strong class="rtf-enhanced">Enhanced:</strong>
                <code>Query</code> <span class="rtf-arrow">&rarr;</span>
                <code class="rtf-enhanced">Entity Extraction</code> <span class="rtf-arrow">&rarr;</span>
                <code class="rtf-enhanced">SKOS Expansion</code> <span class="rtf-arrow">&rarr;</span>
                <code>Vector Search</code> +
                <code class="rtf-enhanced">Structured Lookup</code> <span class="rtf-arrow">&rarr;</span>
                <code>LLM</code> <span class="rtf-arrow">&rarr;</span>
                <code>Answer</code>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not status.get("llm"):
        st.warning(
            "Anthropic API key not configured. "
            "Add ANTHROPIC_API_KEY=your-key to your .env file."
        )
    if not status.get("chromadb"):
        st.warning(
            "No documents in ChromaDB. Run python scripts/ingest.py first."
        )

    # -- Example Questions --
    st.markdown(
        '<div class="rtf-section-label">Try an example</div>',
        unsafe_allow_html=True,
    )

    row1_cols = st.columns(3)
    row2_cols = st.columns(3)
    example_layout = [
        (row1_cols[0], 0), (row1_cols[1], 1), (row1_cols[2], 2),
        (row2_cols[0], 3), (row2_cols[1], 4),
    ]
    for col, idx in example_layout:
        with col:
            label = EXAMPLE_QUESTIONS[idx]
            if len(label) > 50:
                label = label[:48] + "..."
            if st.button(label, key=f"example_{idx}", use_container_width=True,
                         help=EXAMPLE_QUESTIONS[idx]):
                st.session_state.pending_query = EXAMPLE_QUESTIONS[idx]

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    # -- Query Input --
    query = st.text_input(
        "Ask a question about military benefits, pay, or entitlements:",
        value=st.session_state.get("last_query", ""),
        placeholder="e.g., What BAH rate does a married E-5 at Fort Liberty receive?",
    )

    ask_clicked = st.button("Ask RAG-Tag Force", type="primary", use_container_width=True)

    # -- Execute Query --
    active_query = st.session_state.pop("pending_query", None)
    if active_query:
        _run_query(active_query, status)
    elif ask_clicked and query:
        _run_query(query, status)
    elif st.session_state.get("naive_answer") and st.session_state.get("enhanced_answer"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            _render_answer_card(st.session_state.naive_answer, "naive")
        with col2:
            _render_answer_card(st.session_state.enhanced_answer, "enhanced")

    # -- Footer --
    st.markdown(
        """
        <div class="rtf-footer">
            RAG-Tag Force &middot; SCSP Hackathon 2026 &middot; GenAI.mil Track<br/>
            Built with SKOS Ontology &middot; ChromaDB &middot; Anthropic Claude &middot; Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()