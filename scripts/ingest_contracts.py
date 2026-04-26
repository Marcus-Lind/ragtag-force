"""Ingest federal contracting reference content into ChromaDB.

Reads markdown documents and PDF files from data/documents/contracts/
and ingests them into a 'contracts_documents' ChromaDB collection. Idempotent.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import fitz  # PyMuPDF

from src.ingest.embeddings import embed_texts
from src.ingest.vector_store import get_chroma_client

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents" / "contracts"


def _chunk_markdown(text: str, source_name: str, max_chars: int = 1500) -> list[dict]:
    """Split markdown by headers into chunks with metadata."""
    chunks = []
    current_heading = "Introduction"
    current_text = []

    for line in text.split("\n"):
        if line.startswith("## "):
            # Flush previous section
            if current_text:
                content = "\n".join(current_text).strip()
                if len(content) > 50:
                    chunks.append({
                        "text": content,
                        "metadata": {
                            "source_doc": source_name,
                            "section_heading": current_heading,
                        },
                    })
            current_heading = line.lstrip("# ").strip()
            current_text = [line]
        else:
            current_text.append(line)

    # Flush last section
    if current_text:
        content = "\n".join(current_text).strip()
        if len(content) > 50:
            chunks.append({
                "text": content,
                "metadata": {
                    "source_doc": source_name,
                    "section_heading": current_heading,
                },
            })

    return chunks


def _chunk_pdf(pdf_path: Path, max_chars: int = 1500) -> list[dict]:
    """Extract text from PDF and split into chunks with metadata.

    Skips boilerplate navigation pages (e.g. CRS header pages) and chunks
    the real content by page, splitting large pages at paragraph breaks.
    """
    doc = fitz.open(pdf_path)
    source_name = pdf_path.stem.replace("_", " ").title()
    chunks: list[dict] = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()

        # Skip very short pages or navigation/boilerplate
        if len(text) < 100:
            continue
        # Skip CRS website wrapper pages
        if "skip to main content" in text.lower() or "enable\nJavaScript" in text:
            continue
        if text.startswith("Publication Date:") or "Download PDF" in text[:200]:
            continue

        # If page text is short enough, keep as one chunk
        if len(text) <= max_chars:
            chunks.append({
                "text": text,
                "metadata": {
                    "source_doc": source_name,
                    "section_heading": f"Page {page_num + 1}",
                },
            })
        else:
            # Split on double newlines (paragraphs)
            paragraphs = text.split("\n\n")
            current_chunk: list[str] = []
            current_len = 0
            chunk_idx = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if current_len + len(para) > max_chars and current_chunk:
                    chunks.append({
                        "text": "\n\n".join(current_chunk),
                        "metadata": {
                            "source_doc": source_name,
                            "section_heading": f"Page {page_num + 1} part {chunk_idx + 1}",
                        },
                    })
                    chunk_idx += 1
                    current_chunk = []
                    current_len = 0
                current_chunk.append(para)
                current_len += len(para)

            if current_chunk:
                chunks.append({
                    "text": "\n\n".join(current_chunk),
                    "metadata": {
                        "source_doc": source_name,
                        "section_heading": f"Page {page_num + 1} part {chunk_idx + 1}",
                    },
                })

    doc.close()
    return chunks


def ingest_contracts_content() -> int:
    """Ingest contracts reference content into ChromaDB.

    Returns:
        Number of chunks ingested.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name="contracts_documents",
        metadata={"hnsw:space": "cosine"},
    )

    # Clear existing for idempotency
    existing = collection.count()
    if existing > 0:
        collection.delete(where={"source_doc": {"$ne": ""}})
        try:
            collection = client.get_or_create_collection(
                name="contracts_documents",
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            pass

    # Read all markdown files
    all_chunks: list[dict] = []
    md_files = sorted(CONTRACTS_DIR.glob("*.md"))
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        source_name = md_file.stem.replace("_", " ").title()
        chunks = _chunk_markdown(text, source_name)
        all_chunks.extend(chunks)
        print(f"  {md_file.name}: {len(chunks)} chunks")

    # Read all PDF files
    pdf_files = sorted(CONTRACTS_DIR.glob("*.pdf"))
    for pdf_file in pdf_files:
        chunks = _chunk_pdf(pdf_file)
        all_chunks.extend(chunks)
        print(f"  {pdf_file.name}: {len(chunks)} chunks (PDF)")

    if not all_chunks:
        print("No contract documents found!")
        return 0

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    ids = [f"contracts-{i:04d}" for i in range(len(texts))]

    print(f"Embedding {len(texts)} contract document chunks...")
    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    final_count = collection.count()
    print(f"Contracts collection: {final_count} chunks ingested")
    return final_count


if __name__ == "__main__":
    count = ingest_contracts_content()
    print(f"Done: {count} contract chunks in ChromaDB")
