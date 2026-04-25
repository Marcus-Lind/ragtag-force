# RAG-Tag Force 🪖

> Ontology-enhanced RAG for military benefits and entitlements navigation | SCSP Hackathon 2026 | GenAI.mil Track

## The Thesis
Naive RAG fails military personnel because it doesn't understand that "SPC", "Specialist", "Corporal", and "E-4" are the same thing. Our SKOS ontology layer expands every query with synonyms, rank hierarchies, and installation-to-locality mappings before hitting the vector store — producing dramatically more accurate answers.

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
         Ollama LLM (local, offline)
              |
              v
    Answer + Citations + Expanded Terms
```

## Setup (3 steps)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run setup (downloads data, builds DB, ingests docs)
bash scripts/setup.sh

# 3. Launch
streamlit run src/ui/app.py
```

## Example Questions
1. What BAH am I entitled to as a single E-4 at Fort Campbell?
2. How many days of leave can I accrue per year as an O-3?
3. What is my BAS rate as an enlisted Soldier?
4. What housing allowance changes when I add a dependent?
5. What regulation governs my entitlement to OHA overseas?

## Datasets & APIs Used
- Army Publishing Directorate (armypubs.army.mil) — AR 37-104-4, AR 600-8-10
- DoD Financial Management Regulation Vol 7A (comptroller.defense.gov)
- Defense Travel Management Office BAH tables (defensetravel.dod.mil)
- DFAS Military Pay Charts (dfas.mil)
- Joint Travel Regulations (travel.dod.mil)

## Team
RAG-Tag Force | SCSP Hackathon 2026 | GenAI.mil Track
