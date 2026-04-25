"use client";

import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import {
  fetchStatus,
  fetchExamples,
  submitQuery,
  type StatusResponse,
  type QueryResponse,
} from "@/lib/api";
import { Hero } from "@/components/hero";
import { PipelineDiagram } from "@/components/pipeline-diagram";
import { QueryInput } from "@/components/query-input";
import { AnswerCard } from "@/components/answer-card";
import { Sidebar } from "@/components/sidebar";

export default function Home() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [examples, setExamples] = useState<string[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => setStatus(null));
    fetchExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  const handleQuery = async (query: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await submitQuery(query);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar status={status} />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <Hero />
          <PipelineDiagram />

          <QueryInput
            examples={examples}
            onSubmit={handleQuery}
            loading={loading}
          />

          {error && (
            <div className="mt-6 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
              {error}
            </div>
          )}

          {loading && (
            <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AnswerCard variant="naive" loading />
              <AnswerCard variant="enhanced" loading />
            </div>
          )}

          {result && !loading && (
            <>
              <div className="mt-4 flex items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="outline" className="text-xs">
                  Total: {result.total_time_ms.toFixed(0)}ms
                </Badge>
              </div>
              <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-6">
                <AnswerCard variant="naive" result={result.naive} />
                <AnswerCard variant="enhanced" result={result.enhanced} />
              </div>
            </>
          )}

          <footer className="mt-16 border-t pt-6 pb-8 text-center text-xs text-muted-foreground">
            RAG-Tag Force &middot; SCSP Hackathon 2026 &middot; GenAI.mil Track
            <br />
            Built with SKOS Ontology &middot; ChromaDB &middot; Anthropic Claude &middot; Next.js
          </footer>
        </div>
      </main>
    </div>
  );
}