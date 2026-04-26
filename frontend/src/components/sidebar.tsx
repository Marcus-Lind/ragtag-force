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
          Ontology Enhanced RAG for Military Domains
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
