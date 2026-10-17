"use client";
import { useEffect, useState } from "react";

/** Copies a link to the current view (inputs included) to the clipboard. */
export function CopyLinkButton({ getUrl }: { getUrl: () => string }) {
  const [status, setStatus] = useState<"idle" | "copied" | "failed">("idle");

  useEffect(() => {
    if (status === "idle") return;
    const id = window.setTimeout(() => setStatus("idle"), 2000);
    return () => window.clearTimeout(id);
  }, [status]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(getUrl());
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
  };

  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex items-center gap-1.5 rounded-full border border-[var(--panel-border)] px-3 py-1.5 text-sm text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)]"
    >
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
        <path d="M10 13a5 5 0 0 0 7.07 0l3-3a5 5 0 0 0-7.07-7.07l-1.5 1.5" />
        <path d="M14 11a5 5 0 0 0-7.07 0l-3 3a5 5 0 0 0 7.07 7.07l1.5-1.5" />
      </svg>
      <span aria-live="polite">
        {status === "copied" ? "Link copied" : status === "failed" ? "Copy the address bar" : "Copy link to this view"}
      </span>
    </button>
  );
}
