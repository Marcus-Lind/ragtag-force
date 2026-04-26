"use client";

import { useState } from "react";

interface PipelineDiagramProps {
  domain?: "benefits" | "tdy" | "contracts";
}

export function PipelineDiagram({ domain = "benefits" }: PipelineDiagramProps) {
  const [open, setOpen] = useState(false);
  const structuredLabel =
    domain === "contracts" ? "USAspending API" :
    domain === "tdy" ? "GSA Per Diem API" : "SQLite Rate Tables";

  return (
    <div className="mb-4">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        <span className={`transition-transform ${open ? "rotate-90" : ""}`}>▸</span>
        <span className="font-medium">Pipeline Architecture</span>
      </button>
      {open && (
        <div className="mt-2 rounded-lg border bg-muted/30 px-4 py-3 text-sm text-muted-foreground space-y-1.5">
          <div className="flex flex-wrap items-center gap-1">
            <span className="font-medium text-foreground w-20 text-xs">Basic:</span>
            <Step>Query</Step>
            <Arrow />
            <Step>Vector Search</Step>
            <Arrow />
            <Step>LLM</Step>
            <Arrow />
            <Step>Answer</Step>
          </div>
          <div className="flex flex-wrap items-center gap-1">
            <span className="font-medium text-amber-700 w-20 text-xs">Enhanced:</span>
            <Step>Query</Step>
            <Arrow />
            <Step accent>Entity Extraction</Step>
            <Arrow />
            <Step accent>SKOS Expansion</Step>
            <Arrow />
            <Step>Vector Search</Step>
            <span className="text-muted-foreground">+</span>
            <Step accent>{structuredLabel}</Step>
            <Arrow />
            <Step>LLM</Step>
            <Arrow />
            <Step>Answer</Step>
          </div>
        </div>
      )}
    </div>
  );
}

function Step({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <code
      className={`rounded px-2 py-0.5 text-xs font-mono ${
        accent
          ? "bg-amber-100 text-amber-800 font-semibold"
          : "bg-muted text-foreground"
      }`}
    >
      {children}
    </code>
  );
}

function Arrow() {
  return <span className="text-muted-foreground/50 text-xs">→</span>;
}
