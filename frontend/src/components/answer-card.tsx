"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
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
              {isEnhanced ? "Ontology-Enhanced RAG" : "Naive RAG"}
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
        {result.error ? (
          <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {result.error}
          </div>
        ) : (
          <div className="text-sm leading-relaxed whitespace-pre-wrap">
            {result.answer}
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
          {result.structured_data.length > 0 && (
            <Badge variant="secondary" className="text-xs font-normal">
              📊 {result.structured_data.length} data fields
            </Badge>
          )}
        </div>

        {/* Structured Data */}
        {result.structured_data.length > 0 && (
          <CollapsibleSection title="Structured Data (official rate tables)">
            <div className="space-y-1">
              {result.structured_data.map((item, i) => (
                <div key={i} className="flex justify-between text-xs">
                  <span className="text-muted-foreground">{item.key}</span>
                  <span className="font-medium">{item.value}</span>
                </div>
              ))}
            </div>
          </CollapsibleSection>
        )}

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

function CollapsibleSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);

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
