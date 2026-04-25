"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface QueryInputProps {
  examples: string[];
  onSubmit: (query: string) => void;
  loading: boolean;
}

export function QueryInput({ examples, onSubmit, loading }: QueryInputProps) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) onSubmit(query.trim());
  };

  const handleExample = (example: string) => {
    setQuery(example);
    onSubmit(example);
  };

  return (
    <div className="space-y-4">
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Try an example
      </div>
      <div className="flex flex-wrap gap-2">
        {examples.map((ex, i) => (
          <button
            key={i}
            onClick={() => handleExample(ex)}
            disabled={loading}
            className="rounded-full border bg-white px-3.5 py-1.5 text-xs text-foreground hover:border-amber-500/50 hover:bg-amber-50 hover:text-amber-800 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {ex.length > 52 ? ex.slice(0, 50) + "…" : ex}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="flex gap-3">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about military benefits, pay, or entitlements..."
          className="flex-1"
          disabled={loading}
        />
        <Button
          type="submit"
          disabled={loading || !query.trim()}
          className="bg-gradient-to-r from-[#1B2A4A] to-[#2D4A7A] hover:from-[#243560] hover:to-[#3B6B9A] px-6"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <Spinner /> Running...
            </span>
          ) : (
            "Ask RAG-Tag Force"
          )}
        </Button>
      </form>
    </div>
  );
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
      <path
        d="M4 12a8 8 0 018-8"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        className="opacity-75"
      />
    </svg>
  );
}
