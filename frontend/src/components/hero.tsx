import { Badge } from "@/components/ui/badge";

export function Hero() {
  return (
    <div className="relative overflow-hidden rounded-xl bg-gradient-to-br from-[#1B2A4A] via-[#2D4A7A] to-[#3B6B9A] px-8 py-10 mb-6">
      <div className="absolute -top-24 -right-16 h-72 w-72 rounded-full bg-[radial-gradient(circle,rgba(197,164,78,0.15)_0%,transparent_70%)]" />
      <div className="relative z-10">
        <Badge
          variant="outline"
          className="mb-4 border-amber-500/30 bg-amber-500/15 text-amber-200 text-[0.68rem] font-semibold uppercase tracking-wider"
        >
          SCSP Hackathon 2026 · GenAI.mil
        </Badge>
        <h1 className="text-3xl font-bold text-white tracking-tight">
          RAG-Tag Force
        </h1>
        <p className="mt-2 max-w-xl text-sm text-white/75 leading-relaxed font-light">
          Compare naive RAG against ontology-enhanced RAG for military benefits
          and entitlements. See how SKOS knowledge graphs dramatically improve
          retrieval accuracy.
        </p>
      </div>
    </div>
  );
}
