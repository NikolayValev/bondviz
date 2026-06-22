"use client";
import { useEffect } from "react";
import { loadSpreads } from "@/lib/spreadsCache";

/** Warms the deep spread-history cache once the home page is idle, so opening
 *  the Signal tab reuses the resolved data instead of fetching ~35 years then.
 *  Fire-and-forget: errors are swallowed here and surfaced on the Signal page
 *  itself (loadSpreads clears its cache on failure so a retry is possible). */
export function PrefetchSignal() {
  useEffect(() => {
    const warm = () => void loadSpreads().catch(() => {});
    const w = window as Window & {
      requestIdleCallback?: (cb: () => void, opts?: { timeout: number }) => number;
    };
    if (w.requestIdleCallback) {
      w.requestIdleCallback(warm, { timeout: 2500 });
    } else {
      const t = setTimeout(warm, 1200);
      return () => clearTimeout(t);
    }
  }, []);
  return null;
}
