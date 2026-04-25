"""Ingestion pipeline orchestrator.

Runs the full data ingestion pipeline:
1. Load structured CSV data into SQLite
2. Parse PDFs into document chunks
3. Generate embeddings for all chunks
4. Ingest chunks + embeddings into ChromaDB

Idempotent — safe to run multiple times.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import DATA_RAW_PATH, CHROMA_PATH, SQLITE_PATH
from src.ingest.structured_loader import load_all_structured_data
from src.ingest.pdf_parser import parse_all_pdfs
from src.ingest.embeddings import embed_texts, get_embedding_model
from src.ingest.vector_store import get_chroma_client, get_or_create_collection, ingest_chunks


def run_ingestion() -> None:
    """Run the complete ingestion pipeline."""
    print("=" * 60)
    print("RAG-Tag Force — Data Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Structured data → SQLite
    print("\n[1/4] Loading structured data into SQLite...")
    print(f"  Database: {SQLITE_PATH}")
    counts = load_all_structured_data()
    for table, count in counts.items():
        print(f"  ✓ {table}: {count} rows")

    # Step 2: Parse PDFs
    print(f"\n[2/4] Parsing PDF documents from {DATA_RAW_PATH}...")
    chunks = parse_all_pdfs(DATA_RAW_PATH)
    print(f"  ✓ Total chunks: {len(chunks)}")

    if not chunks:
        print("  ⚠ No PDF documents found. Skipping embedding and ChromaDB ingestion.")
        print("\nDone (structured data only).")
        return

    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    model = get_embedding_model()
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, model=model)
    print(f"  ✓ Generated {len(embeddings)} embeddings")

    # Step 4: Ingest into ChromaDB
    print(f"\n[4/4] Ingesting into ChromaDB at {CHROMA_PATH}...")
    client = get_chroma_client()
    collection = get_or_create_collection(client)
    ingested = ingest_chunks(chunks, embeddings, collection)
    print(f"  ✓ Ingested {ingested} chunks into collection '{collection.name}'")

    # Validation
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)
    count = collection.count()
    print(f"  ChromaDB collection count: {count}")
    assert count > 0, "ChromaDB collection is empty after ingestion!"
    print("  ✓ All validations passed")
    print("\nIngestion complete!")


if __name__ == "__main__":
    run_ingestion()
