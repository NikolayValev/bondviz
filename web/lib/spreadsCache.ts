import { SpreadPoint } from "@/lib/signal";
import { iso } from "@/lib/format";

// In-memory, per-session cache for the deep (1990→today) spread series. The App
// Router navigates client-side, so this module stays resident across route
// changes: the home page warms it on idle (see PrefetchSignal) and the Signal
// page reuses the same promise instead of refetching ~35 years of data.
let cache: Promise<SpreadPoint[]> | null = null;

/** Fetch the full spread history once and share the result. On failure the
 *  cache is cleared so a later caller can retry. */
export function loadSpreads(): Promise<SpreadPoint[]> {
  if (!cache) {
    const url = `/api/treasury/spreads?start=1990-01-01&end=${iso(new Date())}`;
    cache = fetch(url)
      .then((r) => r.json())
      .then((d) => (d.points ?? []) as SpreadPoint[])
      .catch((e) => {
        cache = null;
        throw e;
      });
  }
  return cache;
}

/** Test-only: reset the module cache between cases. */
export function __resetSpreadsCache(): void {
  cache = null;
}
