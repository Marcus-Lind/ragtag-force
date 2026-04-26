"use client";

import { useState, useEffect, useCallback } from "react";
import { Badge } from "@/components/ui/badge";
import {
  fetchStatus,
  fetchExamples,
  fetchTDYExamples,
  fetchContractsExamples,
  submitQuery,
  submitTDYQuery,
  submitContractsQuery,
  type StatusResponse,
  type QueryResponse,
} from "@/lib/api";
import { Hero } from "@/components/hero";
import { PipelineDiagram } from "@/components/pipeline-diagram";
import { QueryInput } from "@/components/query-input";
import { AnswerCard } from "@/components/answer-card";
import { Sidebar } from "@/components/sidebar";

type Domain = "benefits" | "tdy" | "contracts";

const DOMAIN_CONFIG: Record<Domain, { label: string; icon: string; placeholder: string }> = {
  benefits: {
    label: "Benefits & Entitlements",
    icon: "🎖️",
    placeholder: "Ask about military benefits, pay, or entitlements...",
  },
  tdy: {
    label: "TDY Travel Planner",
    icon: "✈️",
    placeholder: "Ask about TDY travel, per diem rates, or transportation...",
  },
  contracts: {
    label: "Contract Intelligence",
    icon: "📊",
    placeholder: "Ask about defense contracts, spending, or procurement...",
  },
};

export default function Home() {
  const [domain, setDomain] = useState<Domain>("benefits");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [benefitsExamples, setBenefitsExamples] = useState<string[]>([]);
  const [tdyExamples, setTdyExamples] = useState<string[]>([]);
  const [contractsExamples, setContractsExamples] = useState<string[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => setStatus(null));
    fetchExamples().then(setBenefitsExamples).catch(() => setBenefitsExamples([]));
    fetchTDYExamples().then(setTdyExamples).catch(() => setTdyExamples([]));
    fetchContractsExamples().then(setContractsExamples).catch(() => setContractsExamples([]));
  }, []);

  const handleDomainChange = useCallback((d: Domain) => {
    setDomain(d);
    setResult(null);
    setError(null);
  }, []);

  const handleQuery = async (query: string) => {
    setLoading(true);
    setError(null);
    try {
      const fn = domain === "contracts" ? submitContractsQuery : domain === "tdy" ? submitTDYQuery : submitQuery;
      const res = await fn(query);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const currentExamples = domain === "contracts" ? contractsExamples : domain === "tdy" ? tdyExamples : benefitsExamples;
  const config = DOMAIN_CONFIG[domain];

  return (
    <div className="flex min-h-screen">
      <Sidebar status={status} />

      <main className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <Hero domain={domain} />

          {/* Domain Tabs */}
          <div className="flex items-center gap-1 mb-6 p-1 rounded-lg bg-muted/50 border w-fit">
            {(Object.keys(DOMAIN_CONFIG) as Domain[]).map((d) => (
              <button
                key={d}
                onClick={() => handleDomainChange(d)}
                className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-all ${
                  domain === d
                    ? "bg-white text-foreground shadow-sm border"
                    : "text-muted-foreground hover:text-foreground hover:bg-white/50"
                }`}
              >
                <span>{DOMAIN_CONFIG[d].icon}</span>
                {DOMAIN_CONFIG[d].label}
              </button>
            ))}
          </div>

          <PipelineDiagram domain={domain} />

          <QueryInput
            examples={currentExamples}
            onSubmit={handleQuery}
            loading={loading}
            placeholder={config.placeholder}
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