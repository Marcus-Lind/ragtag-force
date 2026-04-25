"""PDF document parser using PyMuPDF.

Extracts text from PDF documents at the section level, preserving metadata
for each chunk (source document, page number, section heading).
"""

from dataclasses import dataclass, field
from pathlib import Path
import re

import fitz  # PyMuPDF


@dataclass
class DocumentChunk:
    """A chunk of text extracted from a PDF document."""

    text: str
    source_doc: str
    page_number: int
    section_heading: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


def _is_section_heading(line: str) -> bool:
    """Heuristic to detect section headings in military documents.

    Looks for patterns like:
    - "Chapter 1", "Section 2", "Part III"
    - All-caps lines (likely headings)
    - Numbered sections like "1-1.", "2.3", "26.1"
    """
    line = line.strip()
    if not line or len(line) > 200:
        return False

    # Numbered section patterns common in ARs and DoD FMRs
    if re.match(r"^\d+[\-\.]\d+", line):
        return True
    # Chapter/Section/Part headers
    if re.match(r"^(Chapter|Section|Part|Article|Appendix)\s+\w+", line, re.IGNORECASE):
        return True
    # All-caps lines that are likely headings (but not too short or too long)
    if line.isupper() and 5 < len(line) < 100:
        return True

    return False


def _clean_text(text: str) -> str:
    """Clean extracted text by removing excessive whitespace."""
    # Collapse multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def parse_pdf(pdf_path: Path, chunk_max_chars: int = 2000) -> list[DocumentChunk]:
    """Parse a PDF into section-level chunks.

    Splits on detected section headings. If no headings are found,
    falls back to page-level chunking. Chunks exceeding chunk_max_chars
    are split further at paragraph boundaries.

    Args:
        pdf_path: Path to the PDF file.
        chunk_max_chars: Maximum characters per chunk before splitting.

    Returns:
        List of DocumentChunk objects with text and metadata.
    """
    doc = fitz.open(str(pdf_path))
    source_name = pdf_path.stem

    raw_sections: list[dict] = []
    current_section = {
        "heading": "Introduction",
        "text": "",
        "page": 1,
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        for line in text.split("\n"):
            if _is_section_heading(line):
                # Save current section if it has content
                if current_section["text"].strip():
                    raw_sections.append(current_section.copy())
                current_section = {
                    "heading": line.strip(),
                    "text": "",
                    "page": page_num + 1,
                }
            else:
                current_section["text"] += line + "\n"

    # Don't forget the last section
    if current_section["text"].strip():
        raw_sections.append(current_section)

    doc.close()

    # Convert to chunks, splitting oversized sections
    chunks: list[DocumentChunk] = []
    chunk_idx = 0

    for section in raw_sections:
        text = _clean_text(section["text"])
        if not text:
            continue

        if len(text) <= chunk_max_chars:
            chunks.append(DocumentChunk(
                text=text,
                source_doc=source_name,
                page_number=section["page"],
                section_heading=section["heading"],
                chunk_index=chunk_idx,
                metadata={
                    "source_doc": source_name,
                    "page_number": section["page"],
                    "section_heading": section["heading"],
                },
            ))
            chunk_idx += 1
        else:
            # Split at paragraph boundaries
            paragraphs = text.split("\n\n")
            buffer = ""
            for para in paragraphs:
                if len(buffer) + len(para) > chunk_max_chars and buffer:
                    chunks.append(DocumentChunk(
                        text=buffer.strip(),
                        source_doc=source_name,
                        page_number=section["page"],
                        section_heading=section["heading"],
                        chunk_index=chunk_idx,
                        metadata={
                            "source_doc": source_name,
                            "page_number": section["page"],
                            "section_heading": section["heading"],
                        },
                    ))
                    chunk_idx += 1
                    buffer = para + "\n\n"
                else:
                    buffer += para + "\n\n"

            if buffer.strip():
                chunks.append(DocumentChunk(
                    text=buffer.strip(),
                    source_doc=source_name,
                    page_number=section["page"],
                    section_heading=section["heading"],
                    chunk_index=chunk_idx,
                    metadata={
                        "source_doc": source_name,
                        "page_number": section["page"],
                        "section_heading": section["heading"],
                    },
                ))
                chunk_idx += 1

    return chunks


def parse_all_pdfs(pdf_dir: Path) -> list[DocumentChunk]:
    """Parse all PDF files in a directory.

    Args:
        pdf_dir: Directory containing PDF files.

    Returns:
        Combined list of DocumentChunk objects from all PDFs.
    """
    all_chunks: list[DocumentChunk] = []
    pdf_files = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
    # Deduplicate (case-insensitive match on Windows)
    seen: set[str] = set()
    unique_pdfs: list[Path] = []
    for p in pdf_files:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique_pdfs.append(p)

    for pdf_path in unique_pdfs:
        print(f"  Parsing: {pdf_path.name}")
        chunks = parse_pdf(pdf_path)
        all_chunks.extend(chunks)
        print(f"    → {len(chunks)} chunks")

    return all_chunks
