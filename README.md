# RAG-Tag Force 🪖

> Ontology Enhanced RAG for military benefits and entitlements navigation | SCSP Hackathon 2026 | GenAI.mil Track

## The Thesis
Basic RAG fails military personnel because it doesn't understand that "SPC", "Specialist", "Corporal", and "E-4" are the same thing. Our SKOS ontology layer expands every query with synonyms, rank hierarchies, and installation-to-locality mappings before hitting the vector store — producing dramatically more accurate answers.

## Architecture
```
User Question
     |
     v
Entity Extraction (rank, allowance, installation, dependency status)
     |
     v
Ontology Expansion (RDFLib + SKOS) --> synonyms, related concepts, doc filter
     |
     +------------------+
     v                  v
Vector Search       Structured Lookup
(ChromaDB)          (SQLite: BAH/pay tables)
     |                  |
     +--------+---------+
              v
       Anthropic Claude Haiku 3.5
              |
              v
    Answer + Citations + Expanded Terms
```

## Quick Start

### Prerequisites
- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)

### Setup
```bash
# 1. Clone and install dependencies
git clone https://github.com/Marcus-Lind/ragtag-force.git
cd ragtag-force
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run ingestion (parses PDFs, builds vector store + SQLite)
python scripts/ingest.py

# 4. Launch the app
cd frontend && npm install && npm run dev  # Frontend on :3000
python -m uvicorn src.api.main:app --port 8000  # API on :8000
```

Or use the all-in-one setup script:
```bash
python scripts/setup.py
```

## Example Questions
1. What BAH am I entitled to as a single E-4 at Fort Campbell?
2. How many days of leave can I accrue per year as an O-3?
3. What is my BAS rate as an enlisted Soldier?
4. What housing allowance changes when I add a dependent?
5. What regulation governs my entitlement to OHA overseas?

## How It Works

### The Demo (Two Columns)
The UI shows **two answers side by side** for every question:

| **Basic RAG** | **Ontology Enhanced RAG** |
|---|---|
| Raw query → vector search → LLM | Query → SKOS expansion → enhanced search + structured data → LLM |
| Misses synonyms and alternate terms | Expands "E-4" to include "SPC", "Specialist", "Corporal" |
| No structured data | Queries SQLite for exact BAH/BAS/pay rates |
| Generic citations | Precise regulation references (AR 37-104-4, DoD FMR chapters) |

### Ontology (SKOS)
- 46 concepts, 386 triples
- Full synonym coverage for all enlisted (E-1→E-9) and officer (O-1→O-10) ranks
- Installation-to-locality code mappings (Fort Campbell → CLARKSVILLE_TN)
- Allowance-to-regulation links (BAH → AR 37-104-4, DoD FMR Vol 7A Ch. 26)

### Data Pipeline
- **4 DoD PDFs** parsed with PyMuPDF → 9,857 chunks → ChromaDB (bge-base-en-v1.5 embeddings)
- **4 structured tables** in SQLite: BAH rates, enlisted pay, officer pay, BAS rates

## Datasets & Sources
- Army Publishing Directorate (armypubs.army.mil) — AR 37-104-4, AR 600-8-10
- DoD Financial Management Regulation Vol 7A (comptroller.defense.gov)
- Defense Travel Management Office BAH tables (defensetravel.dod.mil)
- DFAS Military Pay Charts (dfas.mil)
- Joint Travel Regulations (travel.dod.mil)

## Testing
```bash
python -m pytest tests/ -v
```

## Alternative: Ollama (Local LLM)
See [docs/OLLAMA_SETUP.md](docs/OLLAMA_SETUP.md) for running with a local LLM instead of Anthropic.

## Team
RAG-Tag Force | SCSP Hackathon 2026 | GenAI.mil Track
