interface HeroProps {
  domain?: "benefits" | "tdy" | "contracts";
}

export function Hero({ domain = "benefits" }: HeroProps) {
  return (
    <div className="flex items-center gap-4 mb-6">
      <div className="flex items-center gap-3">
        <span className="text-2xl">🪖</span>
        <div>
          <h1 className="text-xl font-bold text-foreground tracking-tight leading-none">
            RAG-Tag Force
          </h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Ontology-Enhanced RAG · Side-by-Side Comparison
          </p>
        </div>
      </div>
      <div className="ml-auto flex items-center gap-2">
        <span className="rounded-full bg-amber-500/10 border border-amber-500/20 px-3 py-1 text-[0.65rem] font-semibold text-amber-700 uppercase tracking-wider">
          SCSP Hackathon 2026
        </span>
        <span className="rounded-full bg-slate-100 border px-3 py-1 text-[0.65rem] font-medium text-slate-500 uppercase tracking-wider">
          GenAI.mil
        </span>
      </div>
    </div>
  );
}
