export function PipelineDiagram() {
  return (
    <div className="rounded-lg border bg-muted/30 px-5 py-4 mb-6 text-sm text-muted-foreground leading-loose">
      <div className="text-xs font-semibold uppercase tracking-wider text-foreground mb-2">
        Pipeline Architecture
      </div>
      <div className="flex flex-wrap items-center gap-1">
        <span className="font-medium text-foreground">Naive:</span>
        <Step>Query</Step>
        <Arrow />
        <Step>Vector Search</Step>
        <Arrow />
        <Step>LLM</Step>
        <Arrow />
        <Step>Answer</Step>
      </div>
      <div className="flex flex-wrap items-center gap-1 mt-1">
        <span className="font-medium text-amber-700">Enhanced:</span>
        <Step>Query</Step>
        <Arrow />
        <Step accent>Entity Extraction</Step>
        <Arrow />
        <Step accent>SKOS Expansion</Step>
        <Arrow />
        <Step>Vector Search</Step>
        <span className="text-muted-foreground">+</span>
        <Step accent>Structured Lookup</Step>
        <Arrow />
        <Step>LLM</Step>
        <Arrow />
        <Step>Answer</Step>
      </div>
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
