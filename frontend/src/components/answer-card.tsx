"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import type { AnswerResult } from "@/lib/api";

interface AnswerCardProps {
  variant: "naive" | "enhanced";
  result?: AnswerResult;
  loading?: boolean;
}

export function AnswerCard({ variant, result, loading }: AnswerCardProps) {
  const isEnhanced = variant === "enhanced";

  if (loading) {
    return (
      <Card className={`${isEnhanced ? "border-l-4 border-l-amber-500" : "border-l-4 border-l-gray-300"}`}>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className={`flex h-9 w-9 items-center justify-center rounded-lg text-lg ${
              isEnhanced ? "bg-amber-50" : "bg-gray-100"
            }`}>
              {isEnhanced ? "🧠" : "📄"}
            </div>
            <div>
              <Skeleton className="h-4 w-40" />
              <Skeleton className="mt-1 h-3 w-56" />
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-4 w-4/6" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/6" />
        </CardContent>
      </Card>
    );
  }

  if (!result) return null;

  return (
    <Card className={`${isEnhanced ? "border-l-4 border-l-amber-500" : "border-l-4 border-l-gray-300"}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-3">
          <div className={`flex h-9 w-9 items-center justify-center rounded-lg text-lg ${
            isEnhanced ? "bg-amber-50" : "bg-gray-100"
          }`}>
            {isEnhanced ? "🧠" : "📄"}
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">
              {isEnhanced ? "Ontology Enhanced RAG" : "Basic RAG"}
            </h3>
            <p className="text-xs text-muted-foreground">
              {isEnhanced
                ? "SKOS Expansion → Enhanced Search → Structured Data → LLM"
                : "Raw Query → Vector Search → LLM"}
            </p>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Pipeline Trace — always visible */}
        <PipelineTrace result={result} isEnhanced={isEnhanced} />

        {/* Answer */}
        {result.error ? (
          <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {result.error}
          </div>
        ) : (
          <div className="prose-answer text-sm leading-relaxed">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => <h3 className="text-base font-bold mt-3 mb-1.5 text-foreground">{children}</h3>,
                h2: ({ children }) => <h4 className="text-sm font-semibold mt-2.5 mb-1 text-foreground">{children}</h4>,
                h3: ({ children }) => <h5 className="text-sm font-semibold mt-2 mb-1 text-foreground">{children}</h5>,
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-5 mb-2 space-y-0.5">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 space-y-0.5">{children}</ol>,
                li: ({ children }) => <li className="text-sm">{children}</li>,
                strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
                code: ({ children }) => (
                  <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono">{children}</code>
                ),
                pre: ({ children }) => (
                  <pre className="rounded-md bg-muted p-3 text-xs font-mono overflow-x-auto mb-2">{children}</pre>
                ),
                table: ({ children }) => (
                  <div className="overflow-x-auto mb-2">
                    <table className="min-w-full text-xs border-collapse">{children}</table>
                  </div>
                ),
                th: ({ children }) => <th className="border border-border px-2 py-1 bg-muted font-semibold text-left">{children}</th>,
                td: ({ children }) => <td className="border border-border px-2 py-1">{children}</td>,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-2 border-amber-400 pl-3 italic text-muted-foreground mb-2">{children}</blockquote>
                ),
              }}
            >
              {result.answer}
            </ReactMarkdown>
          </div>
        )}

        {/* Metrics */}
        <div className="flex flex-wrap gap-2 pt-2 border-t">
          <Badge variant="secondary" className="text-xs font-normal">
            ⏱ {result.retrieval_time_ms.toFixed(0)}ms retrieval
          </Badge>
          <Badge variant="secondary" className="text-xs font-normal">
            📑 {result.document_count} documents
          </Badge>
          {result.avg_distance != null && (
            <Badge variant="secondary" className="text-xs font-normal">
              📏 {result.avg_distance.toFixed(3)} avg distance
            </Badge>
          )}
          {result.structured_data.length > 0 && (
            <Badge variant="secondary" className="text-xs font-normal">
              📊 {result.structured_data.length} data fields
            </Badge>
          )}
        </div>

        {/* Search Query Used */}
        {result.search_query && (
          <CollapsibleSection title="Actual Search Query Sent to Vector DB" defaultOpen={isEnhanced}>
            <div className="text-xs font-mono bg-muted/50 rounded p-2 break-all leading-relaxed">
              {result.search_query}
            </div>
          </CollapsibleSection>
        )}

        {/* Structured Data */}
        {result.structured_data.length > 0 && (() => {
          const dataSourceItem = result.structured_data.find(item => item.key === "Data Source");
          const isLiveGSA = dataSourceItem?.value?.includes("live");
          const title = isLiveGSA
            ? "Structured Data (live from GSA Per Diem API)"
            : "Structured Data (from official SQLite rate tables)";
          const footnote = isLiveGSA
            ? "These rates are queried live from the GSA Per Diem API (api.gsa.gov) — not generated by the LLM."
            : "These values come directly from authoritative rate tables in SQLite — not generated by the LLM.";
          return (
            <CollapsibleSection title={title} defaultOpen>
              <div className="space-y-1">
                {result.structured_data.map((item, i) => (
                  <div key={i} className="flex justify-between text-xs">
                    <span className="text-muted-foreground">{item.key}</span>
                    <span className="font-semibold text-emerald-700">{item.value}</span>
                  </div>
                ))}
              </div>
              <div className="mt-2 text-[0.65rem] text-muted-foreground italic">
                {footnote}
              </div>
            </CollapsibleSection>
          );
        })()}

        {/* Sources */}
        {result.sources.length > 0 && (
          <CollapsibleSection title={`Source Documents (${result.sources.length})`}>
            <div className="space-y-1">
              {result.sources.map((src, i) => (
                <div key={i} className="text-xs text-muted-foreground font-mono">
                  {src}
                </div>
              ))}
            </div>
          </CollapsibleSection>
        )}

        {/* Ontology Expansion */}
        {isEnhanced && result.expansion && (
          <CollapsibleSection title="Ontology Expansion Details">
            <div className="space-y-3">
              {Object.keys(result.expansion.synonyms).length > 0 && (
                <div>
                  <div className="text-xs font-medium mb-1.5">Synonym Expansion</div>
                  <div className="flex flex-wrap gap-1">
                    {Object.values(result.expansion.synonyms)
                      .flat()
                      .slice(0, 16)
                      .map((syn, i) => (
                        <span
                          key={i}
                          className="inline-block rounded bg-amber-50 border border-amber-200 px-2 py-0.5 text-[0.68rem] text-amber-800"
                        >
                          {syn}
                        </span>
                      ))}
                  </div>
                </div>
              )}

              {result.expansion.related_regulations.length > 0 && (
                <div>
                  <div className="text-xs font-medium mb-1.5">Related Regulations</div>
                  <div className="flex flex-wrap gap-1">
                    {result.expansion.related_regulations.map((reg, i) => (
                      <span
                        key={i}
                        className="inline-block rounded bg-blue-50 border border-blue-200 px-2 py-0.5 text-[0.68rem] text-blue-800"
                      >
                        📋 {reg}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="grid grid-cols-3 gap-3 text-xs">
                {result.expansion.locality_codes.length > 0 && (
                  <div>
                    <div className="font-medium mb-0.5">Locality</div>
                    <div className="text-muted-foreground">
                      {result.expansion.locality_codes.join(", ")}
                    </div>
                  </div>
                )}
                {result.expansion.grade_notations.length > 0 && (
                  <div>
                    <div className="font-medium mb-0.5">Pay Grade</div>
                    <div className="text-muted-foreground">
                      {result.expansion.grade_notations.join(", ")}
                    </div>
                  </div>
                )}
                {result.expansion.dependency_statuses.length > 0 && (
                  <div>
                    <div className="font-medium mb-0.5">Dep. Status</div>
                    <div className="text-muted-foreground">
                      {result.expansion.dependency_statuses.join(", ")}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </CollapsibleSection>
        )}
      </CardContent>
    </Card>
  );
}


function PipelineTrace({ result, isEnhanced }: { result: AnswerResult; isEnhanced: boolean }) {
  const steps = result.pipeline_trace;
  if (!steps || steps.length === 0) return null;

  return (
    <div className={`rounded-lg border p-3 ${isEnhanced ? "bg-amber-50/30 border-amber-200" : "bg-gray-50 border-gray-200"}`}>
      <div className="text-[0.68rem] font-semibold uppercase tracking-wider text-muted-foreground mb-2.5">
        Pipeline Trace
      </div>
      <div className="space-y-0">
        {steps.map((step, i) => (
          <div key={i} className="flex items-start gap-2.5">
            {/* Vertical connector line */}
            <div className="flex flex-col items-center pt-0.5">
              <div className={`h-2 w-2 rounded-full shrink-0 ${
                step.highlight
                  ? "bg-amber-500 ring-2 ring-amber-200"
                  : isEnhanced ? "bg-slate-400" : "bg-gray-300"
              }`} />
              {i < steps.length - 1 && (
                <div className={`w-px h-5 ${isEnhanced ? "bg-amber-200" : "bg-gray-200"}`} />
              )}
            </div>
            <div className="pb-1 min-w-0">
              <span className={`text-[0.68rem] font-semibold ${step.highlight ? "text-amber-700" : "text-foreground"}`}>
                {step.label}
              </span>
              <span className="text-[0.65rem] text-muted-foreground ml-1.5">
                — {step.detail}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}


function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="rounded-md border">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium hover:bg-muted/50 transition-colors"
      >
        {title}
        <span className={`text-muted-foreground transition-transform ${open ? "rotate-180" : ""}`}>
          ▾
        </span>
      </button>
      {open && (
        <div className="border-t px-3 py-2.5">
          {children}
        </div>
      )}
    </div>
  );
}
