"use client";
import { useEffect, useState } from "react";

export type AiStatus = "unknown" | "on" | "off";

// One request per page load, shared by every conversation the panel mounts.
let pending: Promise<AiStatus> | null = null;

function fetchStatus(): Promise<AiStatus> {
  pending ??= fetch("/api/command-panel", { cache: "no-store" })
    .then((r) => r.json())
    .then((d: { configured?: boolean }) => (d.configured ? "on" : "off") as AiStatus)
    .catch(() => "unknown" as AiStatus); // can't tell: let the chat try and report errors itself
  return pending;
}

/** Whether the chat endpoint has a model token, so the panel can say so before the first send. */
export function useAiStatus(): AiStatus {
  const [status, setStatus] = useState<AiStatus>("unknown");
  useEffect(() => {
    let live = true;
    fetchStatus().then((s) => {
      if (live) setStatus(s);
    });
    return () => {
      live = false;
    };
  }, []);
  return status;
}

/** Test-only: forget the cached status. */
export function __resetAiStatus() {
  pending = null;
}
