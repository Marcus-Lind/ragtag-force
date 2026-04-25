"""FastAPI REST API for RAG-Tag Force.

Exposes the existing RAG pipeline as HTTP endpoints for the Next.js frontend.
"""

import sys
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.config import ANTHROPIC_API_KEY, SQLITE_PATH
from src.llm.client import LLMClient
from src.llm.generator import generate_naive_answer, generate_enhanced_answer

app = FastAPI(
    title="RAG-Tag Force API",
    description="Ontology-enhanced RAG for military benefits and entitlements",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming query from the frontend."""
    query: str


class SourceDoc(BaseModel):
    """A source document reference."""
    name: str


class StructuredDataItem(BaseModel):
    """A key-value pair from structured data lookup."""
    key: str
    value: str


class ExpansionDetail(BaseModel):
    """Ontology expansion metadata."""
    synonyms: dict[str, list[str]] = {}
    related_regulations: list[str] = []
    locality_codes: list[str] = []
    grade_notations: list[str] = []
    dependency_statuses: list[str] = []


class AnswerResult(BaseModel):
    """A single RAG pipeline answer."""
    answer: str
    error: Optional[str] = None
    retrieval_time_ms: float = 0.0
    document_count: int = 0
    sources: list[str] = []
    structured_data: list[StructuredDataItem] = []
    expansion: Optional[ExpansionDetail] = None


class QueryResponse(BaseModel):
    """Combined response with both pipeline results."""
    query: str
    naive: AnswerResult
    enhanced: AnswerResult
    total_time_ms: float = 0.0


class StatusResponse(BaseModel):
    """System health status."""
    chromadb: bool = False
    chromadb_count: int = 0
    sqlite: bool = False
    sqlite_bah_count: int = 0
    ontology: bool = False
    ontology_triples: int = 0
    llm: bool = False


# ── Helpers ─────────────────────────────────────────────────────────────────

def _rag_answer_to_result(rag_answer: Any) -> AnswerResult:
    """Convert a RAGAnswer dataclass to an API response model."""
    structured_data = []
    if rag_answer.retrieval.structured_data:
        for k, v in rag_answer.retrieval.structured_data.items():
            structured_data.append(StructuredDataItem(key=k, value=str(v)))

    expansion = None
    if rag_answer.retrieval.expansion:
        exp = rag_answer.retrieval.expansion
        expansion = ExpansionDetail(
            synonyms=exp.synonyms or {},
            related_regulations=exp.related_regulations or [],
            locality_codes=exp.locality_codes or [],
            grade_notations=exp.grade_notations or [],
            dependency_statuses=exp.dependency_statuses or [],
        )

    return AnswerResult(
        answer=rag_answer.answer,
        error=rag_answer.error,
        retrieval_time_ms=rag_answer.retrieval.retrieval_time_ms,
        document_count=len(rag_answer.retrieval.documents),
        sources=rag_answer.sources,
        structured_data=structured_data,
        expansion=expansion,
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run both naive and enhanced RAG pipelines for a query."""
    start = time.time()
    client = LLMClient()

    naive_result = generate_naive_answer(request.query, client=client)
    enhanced_result = generate_enhanced_answer(request.query, client=client)

    total_ms = (time.time() - start) * 1000

    return QueryResponse(
        query=request.query,
        naive=_rag_answer_to_result(naive_result),
        enhanced=_rag_answer_to_result(enhanced_result),
        total_time_ms=total_ms,
    )


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Check system component health."""
    result = StatusResponse()

    try:
        from src.ingest.vector_store import get_chroma_client, get_or_create_collection
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        count = collection.count()
        result.chromadb = count > 0
        result.chromadb_count = count
    except Exception:
        pass

    try:
        import sqlite3
        conn = sqlite3.connect(str(SQLITE_PATH))
        cursor = conn.execute("SELECT COUNT(*) FROM bah_rates")
        count = cursor.fetchone()[0]
        conn.close()
        result.sqlite = count > 0
        result.sqlite_bah_count = count
    except Exception:
        pass

    try:
        from src.ontology.loader import load_ontology
        g = load_ontology()
        result.ontology = len(g) > 0
        result.ontology_triples = len(g)
    except Exception:
        pass

    result.llm = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "sk-ant-your-key-here")

    return result


EXAMPLE_QUESTIONS = [
    "What BAH am I entitled to as a single E-4 at Fort Campbell?",
    "How many days of leave can I accrue per year as an O-3?",
    "What is my BAS rate as an enlisted Soldier?",
    "What housing allowance changes when I add a dependent?",
    "What regulation governs my entitlement to OHA overseas?",
]


@app.get("/api/examples")
async def examples() -> list[str]:
    """Return example questions."""
    return EXAMPLE_QUESTIONS
