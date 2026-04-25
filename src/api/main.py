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
from src.llm.tdy_generator import generate_tdy_naive_answer, generate_tdy_enhanced_answer

app = FastAPI(
    title="RAG-Tag Force API",
    description="Ontology Enhanced RAG for military benefits and entitlements",
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


class PipelineStep(BaseModel):
    """A single step in the pipeline trace."""
    label: str
    detail: str
    highlight: bool = False


class AnswerResult(BaseModel):
    """A single RAG pipeline answer."""
    answer: str
    error: Optional[str] = None
    retrieval_time_ms: float = 0.0
    document_count: int = 0
    sources: list[str] = []
    structured_data: list[StructuredDataItem] = []
    expansion: Optional[ExpansionDetail] = None
    search_query: str = ""
    avg_distance: Optional[float] = None
    pipeline_trace: list[PipelineStep] = []


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

def _rag_answer_to_result(rag_answer: Any, original_query: str) -> AnswerResult:
    """Convert a RAGAnswer dataclass to an API response model."""
    structured_data = []
    if rag_answer.retrieval.structured_data:
        for k, v in rag_answer.retrieval.structured_data.items():
            structured_data.append(StructuredDataItem(key=k, value=str(v)))

    expansion = None
    expanded_query = ""
    if rag_answer.retrieval.expansion:
        exp = rag_answer.retrieval.expansion
        expansion = ExpansionDetail(
            synonyms=exp.synonyms or {},
            related_regulations=exp.related_regulations or [],
            locality_codes=exp.locality_codes or [],
            grade_notations=exp.grade_notations or [],
            dependency_statuses=exp.dependency_statuses or [],
        )
        expanded_query = exp.expanded_query

    # Compute average distance from retrieval results
    distances = [
        d.get("distance")
        for d in rag_answer.retrieval.documents
        if d.get("distance") is not None
    ]
    avg_dist = sum(distances) / len(distances) if distances else None

    # Build pipeline trace
    trace: list[PipelineStep] = []
    is_enhanced = rag_answer.is_enhanced

    trace.append(PipelineStep(
        label="Input Query",
        detail=original_query,
    ))

    if is_enhanced:
        trace.append(PipelineStep(
            label="Entity Extraction",
            detail=f"Found {len(rag_answer.retrieval.expansion.entities.ranks) if rag_answer.retrieval.expansion and rag_answer.retrieval.expansion.entities else 0} ranks, "
                   f"{len(rag_answer.retrieval.expansion.entities.installations) if rag_answer.retrieval.expansion and rag_answer.retrieval.expansion.entities else 0} installations, "
                   f"{len(rag_answer.retrieval.expansion.entities.allowances) if rag_answer.retrieval.expansion and rag_answer.retrieval.expansion.entities else 0} allowances",
            highlight=True,
        ))
        trace.append(PipelineStep(
            label="SKOS Expansion",
            detail=f"Query expanded with {len(rag_answer.retrieval.expansion.expanded_terms) if rag_answer.retrieval.expansion else 0} terms",
            highlight=True,
        ))

    search_q = expanded_query if is_enhanced else original_query
    trace.append(PipelineStep(
        label="Vector Search",
        detail=f"Searched {len(rag_answer.retrieval.documents)} docs (avg distance: {avg_dist:.3f})" if avg_dist else f"Retrieved {len(rag_answer.retrieval.documents)} documents",
    ))

    if is_enhanced and structured_data:
        trace.append(PipelineStep(
            label="Structured Lookup",
            detail=f"Found {len(structured_data)} fields from SQLite rate tables",
            highlight=True,
        ))
    elif not is_enhanced:
        trace.append(PipelineStep(
            label="Structured Lookup",
            detail="Skipped — basic pipeline has no entity awareness",
        ))

    trace.append(PipelineStep(
        label="LLM Generation",
        detail=f"Claude Haiku with {len(rag_answer.retrieval.documents)} context docs" +
               (f" + {len(structured_data)} data fields" if structured_data else ""),
    ))

    return AnswerResult(
        answer=rag_answer.answer,
        error=rag_answer.error,
        retrieval_time_ms=rag_answer.retrieval.retrieval_time_ms,
        document_count=len(rag_answer.retrieval.documents),
        sources=rag_answer.sources,
        structured_data=structured_data,
        expansion=expansion,
        search_query=search_q[:300],
        avg_distance=avg_dist,
        pipeline_trace=trace,
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
        naive=_rag_answer_to_result(naive_result, request.query),
        enhanced=_rag_answer_to_result(enhanced_result, request.query),
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


# ── TDY Travel Domain ──────────────────────────────────────────────────────

def _tdy_answer_to_result(tdy_answer: Any, original_query: str) -> AnswerResult:
    """Convert a TDYRAGAnswer dataclass to an API response model."""
    structured_data = []
    if tdy_answer.retrieval.structured_data:
        for k, v in tdy_answer.retrieval.structured_data.items():
            structured_data.append(StructuredDataItem(key=k, value=str(v)))

    expansion = None
    expanded_query = ""
    if tdy_answer.retrieval.expansion:
        exp = tdy_answer.retrieval.expansion
        expansion = ExpansionDetail(
            synonyms=exp.synonyms or {},
            related_regulations=exp.related_regulations or [],
            locality_codes=exp.location_codes or [],
            grade_notations=[],
            dependency_statuses=[],
        )
        expanded_query = exp.expanded_query

    distances = [
        d.get("distance")
        for d in tdy_answer.retrieval.documents
        if d.get("distance") is not None
    ]
    avg_dist = sum(distances) / len(distances) if distances else None

    # Build pipeline trace
    trace: list[PipelineStep] = []
    is_enhanced = tdy_answer.is_enhanced

    trace.append(PipelineStep(label="Input Query", detail=original_query))

    if is_enhanced and tdy_answer.retrieval.expansion:
        ent = tdy_answer.retrieval.expansion.entities
        trace.append(PipelineStep(
            label="TDY Entity Extraction",
            detail=f"Found {len(ent.locations) if ent else 0} locations, "
                   f"{len(ent.allowances) if ent else 0} allowances, "
                   f"{len(ent.transport_modes) if ent else 0} transport modes",
            highlight=True,
        ))
        trace.append(PipelineStep(
            label="SKOS Expansion",
            detail=f"Query expanded with {len(tdy_answer.retrieval.expansion.expanded_terms)} terms",
            highlight=True,
        ))

    search_q = expanded_query if is_enhanced else original_query
    trace.append(PipelineStep(
        label="Vector Search (JTR)",
        detail=f"Searched {len(tdy_answer.retrieval.documents)} docs (avg distance: {avg_dist:.3f})" if avg_dist else f"Retrieved {len(tdy_answer.retrieval.documents)} documents",
    ))

    if is_enhanced and structured_data:
        trace.append(PipelineStep(
            label="Per Diem Lookup",
            detail=f"Found {len(structured_data)} fields from GSA rate tables",
            highlight=True,
        ))
    elif not is_enhanced:
        trace.append(PipelineStep(
            label="Per Diem Lookup",
            detail="Skipped — basic pipeline has no location awareness",
        ))

    trace.append(PipelineStep(
        label="LLM Generation",
        detail=f"Claude Haiku with {len(tdy_answer.retrieval.documents)} context docs" +
               (f" + {len(structured_data)} data fields" if structured_data else ""),
    ))

    return AnswerResult(
        answer=tdy_answer.answer,
        error=tdy_answer.error,
        retrieval_time_ms=tdy_answer.retrieval.retrieval_time_ms,
        document_count=len(tdy_answer.retrieval.documents),
        sources=tdy_answer.sources,
        structured_data=structured_data,
        expansion=expansion,
        search_query=search_q[:300],
        avg_distance=avg_dist,
        pipeline_trace=trace,
    )


@app.post("/api/tdy/query", response_model=QueryResponse)
async def tdy_query(request: QueryRequest) -> QueryResponse:
    """Run both basic and Ontology Enhanced RAG pipelines for a TDY travel query."""
    start = time.time()
    client = LLMClient()

    naive_result = generate_tdy_naive_answer(request.query, client=client)
    enhanced_result = generate_tdy_enhanced_answer(request.query, client=client)

    total_ms = (time.time() - start) * 1000

    return QueryResponse(
        query=request.query,
        naive=_tdy_answer_to_result(naive_result, request.query),
        enhanced=_tdy_answer_to_result(enhanced_result, request.query),
        total_time_ms=total_ms,
    )


TDY_EXAMPLE_QUESTIONS = [
    "What is the per diem rate for TDY travel to San Diego?",
    "Can I drive my POV for TDY and get reimbursed for mileage?",
    "What are the rules for long-term TDY per diem reduction?",
    "How do I get a travel advance before my TDY trip?",
    "What is the M&IE breakdown for a travel day?",
]


@app.get("/api/tdy/examples")
async def tdy_examples() -> list[str]:
    """Return TDY travel example questions."""
    return TDY_EXAMPLE_QUESTIONS
