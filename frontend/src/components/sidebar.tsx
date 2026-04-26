"use client";

import { useState } from "react";
import { Separator } from "@/components/ui/separator";
import type { StatusResponse } from "@/lib/api";

interface SidebarProps {
  status: StatusResponse | null;
}

export function Sidebar({ status }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`hidden lg:flex flex-col bg-[#1B2A4A] text-white transition-all duration-200 ${
        collapsed ? "w-14" : "w-64"
      }`}
    >
      {/* Header + collapse toggle */}
      <div className={`flex items-center ${collapsed ? "justify-center px-2 pt-4 pb-2" : "px-5 pt-6 pb-4"}`}>
        {collapsed ? (
          <button
            onClick={() => setCollapsed(false)}
            className="flex h-8 w-8 items-center justify-center rounded-md hover:bg-white/10 transition-colors text-lg"
            title="Expand sidebar"
          >
            🪖
          </button>
        ) : (
          <div className="flex items-center justify-between w-full">
            <div>
              <div className="text-lg font-bold tracking-tight">🪖 RAG-Tag Force</div>
              <div className="mt-1 text-xs text-slate-400">
                Ontology Enhanced RAG for Military Domains
              </div>
            </div>
            <button
              onClick={() => setCollapsed(true)}
              className="flex h-7 w-7 items-center justify-center rounded-md hover:bg-white/10 transition-colors text-slate-400 text-sm ml-2 shrink-0"
              title="Collapse sidebar"
            >
              ◀
            </button>
          </div>
        )}
      </div>

      {collapsed ? (
        /* Collapsed: show mini status dots */
        <div className="flex flex-col items-center gap-3 pt-4">
          <MiniDot ok={status?.chromadb ?? false} title="Vector Store" />
          <MiniDot ok={status?.sqlite ?? false} title="Structured DB" />
          <MiniDot ok={status?.ontology ?? false} title="SKOS Ontology" />
          <MiniDot ok={(status?.live_apis ?? 0) > 0} title="Live APIs" />
          <MiniDot ok={status?.llm ?? false} title="LLM" />
        </div>
      ) : (
        <>
          <div className="mx-5">
            <Separator className="bg-white/10" />
          </div>

          {/* System Status */}
          <div className="px-5 py-4">
            <div className="text-[0.65rem] font-semibold uppercase tracking-wider text-slate-400 mb-3">
              System Status
            </div>
            <div className="space-y-2.5">
              <StatusRow
                label="Vector Store"
                ok={status?.chromadb ?? false}
                value={status?.chromadb ? `${status.chromadb_count.toLocaleString()} chunks` : "Empty"}
              />
              <StatusRow
                label="Structured DB"
                ok={status?.sqlite ?? false}
                value={status?.sqlite ? `${status.sqlite_rows.toLocaleString()} rows` : "Empty"}
              />
              <StatusRow
                label="SKOS Ontology"
                ok={status?.ontology ?? false}
                value={status?.ontology ? `${status.ontology_triples.toLocaleString()} triples` : "Not loaded"}
              />
              <StatusRow
                label="Live APIs"
                ok={(status?.live_apis ?? 0) > 0}
                value={(status?.live_apis ?? 0) > 0 ? `${status!.live_apis} connected` : "None"}
              />
              <StatusRow
                label="LLM (Anthropic)"
                ok={status?.llm ?? false}
                value={status?.llm ? "Connected" : "No key"}
              />
            </div>
          </div>

          <div className="mx-5">
            <Separator className="bg-white/10" />
          </div>

          {/* Branding */}
          <div className="px-5 py-4 text-center">
            <div className="text-[0.65rem] font-semibold text-slate-400">
              SCSP Hackathon 2026
            </div>
            <div className="text-[0.6rem] text-slate-500 mt-0.5">
              GenAI.mil Track
            </div>
          </div>
        </>
      )}
    </aside>
  );
}

function StatusRow({ label, ok, value }: { label: string; ok: boolean; value: string }) {
  return (
    <div className="flex items-center gap-2.5 text-xs">
      <span
        className={`h-2 w-2 rounded-full flex-shrink-0 ${
          ok ? "bg-emerald-400" : "bg-red-400"
        }`}
      />
      <span className="text-slate-400">{label}</span>
      <span className="ml-auto text-white font-medium">{value}</span>
    </div>
  );
}

function MiniDot({ ok, title }: { ok: boolean; title: string }) {
  return (
    <span
      className={`h-2.5 w-2.5 rounded-full ${ok ? "bg-emerald-400" : "bg-red-400"}`}
      title={title}
    />
  );
}
