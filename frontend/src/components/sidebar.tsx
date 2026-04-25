import { Separator } from "@/components/ui/separator";
import type { StatusResponse } from "@/lib/api";

interface SidebarProps {
  status: StatusResponse | null;
}

export function Sidebar({ status }: SidebarProps) {
  return (
    <aside className="hidden lg:flex w-64 flex-col bg-[#1B2A4A] text-white">
      <div className="px-5 pt-6 pb-4">
        <div className="text-lg font-bold tracking-tight">🪖 RAG-Tag Force</div>
        <div className="mt-1 text-xs text-slate-400">
          Ontology-Enhanced RAG for Military Benefits
        </div>
      </div>

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
            value={status?.sqlite ? `${status.sqlite_bah_count} rates` : "Empty"}
          />
          <StatusRow
            label="SKOS Ontology"
            ok={status?.ontology ?? false}
            value={status?.ontology ? `${status.ontology_triples} triples` : "Not loaded"}
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

      {/* How It Works */}
      <div className="px-5 py-4 flex-1">
        <div className="text-[0.65rem] font-semibold uppercase tracking-wider text-slate-400 mb-3">
          How It Works
        </div>
        <div className="text-xs text-slate-300 leading-relaxed space-y-3">
          <p>
            <span className="font-semibold text-white">The Problem:</span> Naive
            RAG fails military personnel because &ldquo;SPC&rdquo;,
            &ldquo;Specialist&rdquo;, and &ldquo;E-4&rdquo; are the same rank
            &mdash; but vector search doesn&apos;t know that.
          </p>
          <p>
            <span className="font-semibold text-white">Our Solution:</span> A
            SKOS ontology layer expands queries with synonyms, rank hierarchies,
            installation-to-locality mappings, and regulation links &mdash; then
            augments retrieval with authoritative structured data from official
            rate tables.
          </p>
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
