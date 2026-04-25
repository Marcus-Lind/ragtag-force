# RAG-Tag Force — Copilot Instructions

## What This Project Is
An ontology-enhanced RAG system for military benefits and entitlements navigation.
Core thesis: a SKOS ontology layer improves RAG retrieval quality over naive vector
search by expanding queries with synonyms, hierarchy, and concept-to-document mappings.
The Streamlit UI must always show BOTH answers side by side so the delta is visible.

## Stack — Use Nothing Else
- LLM: Anthropic (Claude Haiku 3.5), API key via .env
- Embeddings: sentence-transformers (bge-base-en-v1.5), fully local
- Vector store: ChromaDB, persist to ./data/chroma
- Ontology: RDFLib, SKOS/Turtle format, file at ./ontology/military_entitlements.ttl
- RAG orchestration: LlamaIndex (NOT LangChain)
- Structured data: SQLite via pandas + sqlite3, files in ./data/structured/
- PDF parsing: PyMuPDF (fitz)
- Frontend: Streamlit, single file at src/ui/app.py
- Python 3.11+

## Architecture
Query -> Entity Extraction -> Ontology Expansion -> [Vector Search + Structured Lookup] -> LLM -> Answer

## Non-Negotiable Design Rules
1. NEVER embed rate tables (BAH, pay charts, per diem). Query SQLite directly.
2. ALWAYS add source citations to LLM answers (document name + section number).
3. Streamlit UI MUST show two columns: naive RAG answer vs ontology-enhanced answer.
   The delta between these two columns is the entire demo. Never remove this.
4. All file paths use pathlib.Path. No hardcoded strings.
5. All functions have docstrings. All modules have type hints.
6. Tests go in tests/ and use pytest.
7. All scripts are idempotent — safe to run multiple times.

## Ontology Conventions
- Namespace: ex: = http://ragtag-force.mil/ontology#
- Use SKOS prefLabel, altLabel, broader, narrower, related
- Key concept classes: Rank, Allowance, Installation, DependencyStatus, Regulation
- Every rank must have ALL common synonyms as altLabels
  Example: E-4 = prefLabel "Specialist" altLabels ["SPC", "E-4", "Corporal", "CPL", "E4"]

## Key Ontology Mappings to Encode
- BAH -> governed by AR 37-104-4 and DoD FMR Vol 7A Chapter 26
- BAS -> governed by DoD FMR Vol 7A Chapter 25
- Leave accrual -> governed by AR 600-8-10
- OHA -> governed by DoD FMR Vol 7A Chapter 68 (OCONUS)
- Fort Campbell -> locality code CLARKSVILLE_TN
- Fort Bragg (Fort Liberty) -> locality code FAYETTEVILLE_NC
- Fort Hood (Fort Cavazos) -> locality code KILLEEN_TX
- Fort Lewis (JBLM) -> locality code OLYMPIA_WA

## Environment Variables
CHROMA_PATH=./data/chroma
SQLITE_PATH=./data/structured/entitlements.db
ONTOLOGY_PATH=./ontology/military_entitlements.ttl
ANTHROPIC_API_KEY=<your-api-key>
ANTHROPIC_MODEL=claude-haiku-4-20250506
