const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface StructuredDataItem {
  key: string;
  value: string;
}

export interface ExpansionDetail {
  synonyms: Record<string, string[]>;
  related_regulations: string[];
  locality_codes: string[];
  grade_notations: string[];
  dependency_statuses: string[];
}

export interface PipelineStep {
  label: string;
  detail: string;
  highlight: boolean;
}

export interface ResolutionStep {
  label: string;
  value: string;
}

export interface ResolutionChain {
  input_term: string;
  steps: ResolutionStep[];
}

export interface AnswerResult {
  answer: string;
  error: string | null;
  retrieval_time_ms: number;
  document_count: number;
  sources: string[];
  structured_data: StructuredDataItem[];
  expansion: ExpansionDetail | null;
  search_query: string;
  avg_distance: number | null;
  pipeline_trace: PipelineStep[];
  resolution_chains: ResolutionChain[];
}

export interface QueryResponse {
  query: string;
  naive: AnswerResult;
  enhanced: AnswerResult;
  total_time_ms: number;
}

export interface StatusResponse {
  chromadb: boolean;
  chromadb_count: number;
  sqlite: boolean;
  sqlite_bah_count: number;
  ontology: boolean;
  ontology_triples: number;
  llm: boolean;
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json();
}

export async function fetchExamples(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/examples`);
  if (!res.ok) throw new Error(`Examples fetch failed: ${res.status}`);
  return res.json();
}

export async function fetchTDYExamples(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/tdy/examples`);
  if (!res.ok) throw new Error(`TDY examples fetch failed: ${res.status}`);
  return res.json();
}

export async function submitQuery(query: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Query failed: ${err}`);
  }
  return res.json();
}

export async function submitTDYQuery(query: string): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/api/tdy/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`TDY query failed: ${err}`);
  }
  return res.json();
}
