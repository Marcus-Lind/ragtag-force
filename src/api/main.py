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
from src.llm.contracts_generator import generate_contracts_naive_answer, generate_contracts_enhanced_answer

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


class ResolutionStep(BaseModel):
    """A single hop in an ontology resolution chain."""
    label: str
    value: str


class ResolutionChain(BaseModel):
    """A complete resolution chain for one entity from query text to structured result."""
    input_term: str
    steps: list[ResolutionStep] = []


class AnswerResult(BaseModel):
    """A single RAG pipeline answer."""
    answer: str
    error: Optional[str] = None
    retrieval_time_ms: float = 0.0
    document_count: int = 0
    sources: list[str] = []
    structured_data: list[StructuredDataItem] = []
    structured_query: str = ""
    expansion: Optional[ExpansionDetail] = None
    search_query: str = ""
    avg_distance: Optional[float] = None
    pipeline_trace: list[PipelineStep] = []
    resolution_chains: list[ResolutionChain] = []


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
    sqlite_rows: int = 0
    ontology: bool = False
    ontology_triples: int = 0
    llm: bool = False
    live_apis: int = 0


def _build_resolution_chains(rag_answer: Any, structured_data: list) -> list[ResolutionChain]:
    """Build ontology resolution chains showing how entities were resolved.

    Traces the path from raw query text through SKOS concepts to structured data.
    """
    chains: list[ResolutionChain] = []
    if not rag_answer.is_enhanced or not rag_answer.retrieval.expansion:
        return chains

    exp = rag_answer.retrieval.expansion
    entities = exp.entities
    if not entities or not hasattr(entities, "raw_matches"):
        return chains

    for input_term, uri in entities.raw_matches.items():
        concept_name = str(uri).split("#")[-1]
        steps: list[ResolutionStep] = []

        steps.append(ResolutionStep(label="Matched as", value=f"altLabel of ex:{concept_name}"))

        # Find the prefLabel (canonical name) from synonyms
        for pref, alts in exp.synonyms.items():
            if input_term in [a.lower() for a in alts] or input_term.lower() == pref.lower():
                steps.append(ResolutionStep(label="Canonical name", value=pref))
                if alts:
                    steps.append(ResolutionStep(label="Also known as", value=", ".join(alts[:4])))
                break

        # Location code resolution (TDY)
        loc_codes = getattr(exp, "location_codes", None) or getattr(exp, "locality_codes", None) or []
        if loc_codes and hasattr(entities, "locations") and uri in getattr(entities, "locations", []):
            code = loc_codes[0]
            steps.append(ResolutionStep(label="Location code", value=code))
            parts = code.split("_")
            if len(parts) >= 2:
                city = " ".join(parts[:-1]).title()
                state = parts[-1]
                steps.append(ResolutionStep(label="Resolved to", value=f"{city}, {state}"))

        # Locality code resolution (benefits)
        is_location = False
        if loc_codes and hasattr(entities, "installations") and uri in getattr(entities, "installations", []):
            steps.append(ResolutionStep(label="Locality code", value=loc_codes[0]))
            is_location = True

        # Track if this is a TDY location
        if hasattr(entities, "locations") and uri in getattr(entities, "locations", []):
            is_location = True

        # Grade notation (benefits)
        grade_notations = getattr(exp, "grade_notations", [])
        if grade_notations and hasattr(entities, "ranks") and uri in getattr(entities, "ranks", []):
            steps.append(ResolutionStep(label="Pay grade", value=grade_notations[0]))

        # Related regulations — only for the first chain to avoid repetition
        if exp.related_regulations and len(chains) == 0:
            steps.append(ResolutionStep(
                label="Governing regulation",
                value=", ".join(exp.related_regulations[:2]),
            ))

        # Final structured data result — only for location entities
        if is_location:
            sd_keys = [s.key for s in structured_data]
            if any("Per Diem" in k or "Lodging" in k or "M&IE" in k for k in sd_keys):
                for s in structured_data:
                    if "Total" in s.key:
                        steps.append(ResolutionStep(label="GSA API result", value=s.value))
                        break
            elif any("BAH" in k for k in sd_keys):
                for s in structured_data:
                    if "BAH" in s.key:
                        steps.append(ResolutionStep(label="SQLite result", value=s.value))
                        break

        chains.append(ResolutionChain(input_term=input_term, steps=steps))

    return chains


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

    # Build resolution chains
    chains = _build_resolution_chains(rag_answer, structured_data)

    # Build structured query description showing what the ontology enabled
    structured_query = ""
    if is_enhanced and rag_answer.retrieval.expansion:
        exp = rag_answer.retrieval.expansion
        parts = []
        if exp.grade_notations:
            parts.append(f"grade = '{exp.grade_notations[0]}'")
        if exp.locality_codes:
            parts.append(f"locality_code = '{exp.locality_codes[0]}'")
        if exp.dependency_statuses:
            dep = exp.dependency_statuses[0].replace("_", " ").title()
            parts.append(f"dependency = '{dep}'")
        if parts:
            structured_query = f"SELECT * FROM bah_rates WHERE {' AND '.join(parts)}"

    return AnswerResult(
        answer=rag_answer.answer,
        error=rag_answer.error,
        retrieval_time_ms=rag_answer.retrieval.retrieval_time_ms,
        document_count=len(rag_answer.retrieval.documents),
        sources=rag_answer.sources,
        structured_data=structured_data,
        structured_query=structured_query,
        expansion=expansion,
        search_query=search_q[:300],
        avg_distance=avg_dist,
        pipeline_trace=trace,
        resolution_chains=chains,
    )

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
    """Check system component health across all domains."""
    result = StatusResponse()

    # Count chunks across ALL ChromaDB collections
    try:
        from src.ingest.vector_store import get_chroma_client
        client = get_chroma_client()
        total_chunks = 0
        for col in client.list_collections():
            total_chunks += col.count()
        result.chromadb = total_chunks > 0
        result.chromadb_count = total_chunks
    except Exception:
        pass

    # Count rows across ALL SQLite tables
    try:
        import sqlite3
        total_rows = 0
        conn = sqlite3.connect(str(SQLITE_PATH))
        tables = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for table in tables:
            total_rows += conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
        conn.close()
        result.sqlite = total_rows > 0
        result.sqlite_rows = total_rows
    except Exception:
        pass

    # Count triples across ALL ontology files
    try:
        from rdflib import Graph
        total_triples = 0
        ontology_dir = Path(__file__).resolve().parent.parent.parent / "ontology"
        for ttl in ontology_dir.glob("*.ttl"):
            g = Graph()
            g.parse(str(ttl), format="turtle")
            total_triples += len(g)
        result.ontology = total_triples > 0
        result.ontology_triples = total_triples
    except Exception:
        pass

    result.llm = bool(ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != "sk-ant-your-key-here")

    # Live API connections: GSA per diem + USAspending
    result.live_apis = 2

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
            label="GSA Per Diem API",
            detail=f"Live query returned {len(structured_data)} fields from api.gsa.gov",
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

    # Build resolution chains
    chains = _build_resolution_chains(tdy_answer, structured_data)

    # Build structured query description
    structured_query = ""
    if is_enhanced and tdy_answer.retrieval.expansion:
        exp = tdy_answer.retrieval.expansion
        if exp.location_codes:
            code = exp.location_codes[0]
            parts = code.split("_")
            if len(parts) >= 2:
                state = parts[-1]
                city = " ".join(parts[:-1]).title()
            else:
                city, state = code, ""
            structured_query = f"GET api.gsa.gov/travel/perdiem/rates/city/{city}/state/{state}/year/2026"

    return AnswerResult(
        answer=tdy_answer.answer,
        error=tdy_answer.error,
        retrieval_time_ms=tdy_answer.retrieval.retrieval_time_ms,
        document_count=len(tdy_answer.retrieval.documents),
        sources=tdy_answer.sources,
        structured_data=structured_data,
        structured_query=structured_query,
        expansion=expansion,
        search_query=search_q[:300],
        avg_distance=avg_dist,
        pipeline_trace=trace,
        resolution_chains=chains,
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
    "What is the per diem rate for TDY to Fort Liberty?",
    "I'm TDY to the Pentagon for a week — what's my lodging and meal allowance?",
    "How much mileage reimbursement do I get if I drive my own car to Fort Liberty?",
    "What are the per diem rates for a TDY assignment at Redstone Arsenal?",
    "Can I rent a car during TDY at JBSA and what's the daily meal rate?",
]


@app.get("/api/tdy/examples")
async def tdy_examples() -> list[str]:
    """Return TDY travel example questions."""
    return TDY_EXAMPLE_QUESTIONS


# ── Federal Contract Intelligence Domain ───────────────────────────────────

def _contracts_answer_to_result(contracts_answer: Any, original_query: str) -> AnswerResult:
    """Convert a ContractRAGAnswer dataclass to an API response model."""
    structured_data = []
    if contracts_answer.retrieval.structured_data:
        for k, v in contracts_answer.retrieval.structured_data.items():
            structured_data.append(StructuredDataItem(key=k, value=str(v)))

    expansion = None
    expanded_query = ""
    if contracts_answer.retrieval.expansion:
        exp = contracts_answer.retrieval.expansion
        expansion = ExpansionDetail(
            synonyms=exp.synonyms or {},
            related_regulations=exp.related_regulations or [],
            locality_codes=exp.state_codes or [],
            grade_notations=[],
            dependency_statuses=[],
        )
        expanded_query = exp.expanded_query

    distances = [
        d.get("distance")
        for d in contracts_answer.retrieval.documents
        if d.get("distance") is not None
    ]
    avg_dist = sum(distances) / len(distances) if distances else None

    # Build pipeline trace
    trace: list[PipelineStep] = []
    is_enhanced = contracts_answer.is_enhanced

    trace.append(PipelineStep(label="Input Query", detail=original_query))

    if is_enhanced and contracts_answer.retrieval.expansion:
        ent = contracts_answer.retrieval.expansion.entities
        trace.append(PipelineStep(
            label="Contract Entity Extraction",
            detail=f"Found {len(ent.service_branches) if ent else 0} agencies, "
                   f"{len(ent.research_domains) if ent else 0} research domains, "
                   f"{len(ent.locations) if ent else 0} locations, "
                   f"{len(ent.contractors) if ent else 0} contractors",
            highlight=True,
        ))
        trace.append(PipelineStep(
            label="SKOS Expansion",
            detail=f"Resolved to {len(contracts_answer.retrieval.expansion.naics_codes)} NAICS codes, "
                   f"{len(contracts_answer.retrieval.expansion.state_codes)} states, "
                   f"{len(contracts_answer.retrieval.expansion.agency_names)} agencies",
            highlight=True,
        ))

    search_q = expanded_query if is_enhanced else original_query
    trace.append(PipelineStep(
        label="Vector Search (Procurement Docs)",
        detail=f"Searched {len(contracts_answer.retrieval.documents)} docs (avg distance: {avg_dist:.3f})" if avg_dist else f"Retrieved {len(contracts_answer.retrieval.documents)} documents",
    ))

    if is_enhanced and structured_data:
        trace.append(PipelineStep(
            label="USAspending.gov API",
            detail=f"Live query returned {len(structured_data)} fields from usaspending.gov",
            highlight=True,
        ))
    elif not is_enhanced:
        trace.append(PipelineStep(
            label="Contract Data Lookup",
            detail="Skipped - basic pipeline cannot resolve entities to API parameters",
        ))

    trace.append(PipelineStep(
        label="LLM Generation",
        detail=f"Claude Haiku with {len(contracts_answer.retrieval.documents)} context docs" +
               (f" + {len(structured_data)} data fields" if structured_data else ""),
    ))

    # Build resolution chains for contracts
    chains = _build_contracts_resolution_chains(contracts_answer, structured_data)

    # Build structured query description
    structured_query = ""
    if is_enhanced and contracts_answer.retrieval.expansion:
        exp = contracts_answer.retrieval.expansion
        api_parts = []
        if exp.expanded_terms:
            # Find the keywords that were actually sent to the API
            domain_terms = [t for t in exp.expanded_terms[:3] if len(t) > 3]
            if domain_terms:
                api_parts.append(f"keywords=[{', '.join(repr(t) for t in domain_terms[:2])}]")
        if exp.state_codes:
            api_parts.append(f"state_code='{exp.state_codes[0]}'")
        if exp.contractor_names:
            api_parts.append(f"recipient='{exp.contractor_names[0]}'")
        if api_parts:
            structured_query = f"POST api.usaspending.gov/api/v2/search/spending_by_award ({', '.join(api_parts)})"

    return AnswerResult(
        answer=contracts_answer.answer,
        error=contracts_answer.error,
        retrieval_time_ms=contracts_answer.retrieval.retrieval_time_ms,
        document_count=len(contracts_answer.retrieval.documents),
        sources=contracts_answer.sources,
        structured_data=structured_data,
        structured_query=structured_query,
        expansion=expansion,
        search_query=search_q[:300],
        avg_distance=avg_dist,
        pipeline_trace=trace,
        resolution_chains=chains,
    )


def _build_contracts_resolution_chains(contracts_answer: Any, structured_data: list) -> list[ResolutionChain]:
    """Build resolution chains specific to the contracts domain."""
    chains: list[ResolutionChain] = []
    if not contracts_answer.is_enhanced or not contracts_answer.retrieval.expansion:
        return chains

    exp = contracts_answer.retrieval.expansion
    entities = exp.entities
    if not entities or not hasattr(entities, "raw_matches"):
        return chains

    for input_term, uri in entities.raw_matches.items():
        concept_name = str(uri).split("#")[-1]
        steps: list[ResolutionStep] = []

        steps.append(ResolutionStep(label="Matched as", value=f"altLabel of ex:{concept_name}"))

        for pref, alts in exp.synonyms.items():
            if input_term in [a.lower() for a in alts] or input_term.lower() == pref.lower():
                steps.append(ResolutionStep(label="Canonical name", value=pref))
                if alts:
                    steps.append(ResolutionStep(label="Also known as", value=", ".join(alts[:4])))
                break

        # NAICS code resolution
        if uri in entities.research_domains and exp.naics_codes:
            from src.ontology.contracts_expander import _get_notation, load_contracts_ontology
            graph = load_contracts_ontology()
            notation = _get_notation(graph, uri)
            if notation:
                steps.append(ResolutionStep(label="NAICS code", value=notation))

        # State code resolution
        if uri in entities.locations and exp.state_codes:
            from src.ontology.contracts_expander import _get_notation, load_contracts_ontology
            graph = load_contracts_ontology()
            notation = _get_notation(graph, uri)
            if notation:
                steps.append(ResolutionStep(label="State code", value=notation))

        # Agency name resolution
        if uri in entities.service_branches and exp.agency_names:
            steps.append(ResolutionStep(label="USAspending agency", value=exp.agency_names[0]))

        # Contractor name resolution
        if uri in entities.contractors and exp.contractor_names:
            steps.append(ResolutionStep(label="Recipient search", value=exp.contractor_names[0]))

        # USAspending result for first chain
        if len(chains) == 0:
            contract_fields = [s for s in structured_data if hasattr(s, 'key') and "Contract 1" in s.key and "Award" not in s.key]
            if contract_fields:
                steps.append(ResolutionStep(label="Top contract", value=contract_fields[0].value[:100]))

        chains.append(ResolutionChain(input_term=input_term, steps=steps))

    return chains


@app.post("/api/contracts/query", response_model=QueryResponse)
async def contracts_query(request: QueryRequest) -> QueryResponse:
    """Run both basic and Ontology Enhanced RAG for a contract intelligence query."""
    start = time.time()
    client = LLMClient()

    naive_result = generate_contracts_naive_answer(request.query, client=client)
    enhanced_result = generate_contracts_enhanced_answer(request.query, client=client)

    total_ms = (time.time() - start) * 1000

    return QueryResponse(
        query=request.query,
        naive=_contracts_answer_to_result(naive_result, request.query),
        enhanced=_contracts_answer_to_result(enhanced_result, request.query),
        total_time_ms=total_ms,
    )


CONTRACTS_EXAMPLE_QUESTIONS = [
    "Find contracts for autonomous systems at Aberdeen Proving Ground",
    "What AI research contracts were awarded near Redstone Arsenal?",
    "Show me recent cybersecurity contracts from Lockheed Martin",
    "What logistics contracts were awarded near Fort Liberty?",
    "Find hypersonics research contracts in Alabama",
]


@app.get("/api/contracts/examples")
async def contracts_examples() -> list[str]:
    """Return contract intelligence example questions."""
    return CONTRACTS_EXAMPLE_QUESTIONS
