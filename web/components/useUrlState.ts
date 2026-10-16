"use client";
import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { decodeState, mergeQuery, type Schema } from "@/lib/urlState";

const subscribe = (onChange: () => void) => {
  window.addEventListener("popstate", onChange);
  return () => window.removeEventListener("popstate", onChange);
};
const clientSearch = () => window.location.search;
const serverSearch = () => "";

/**
 * Page state mirrored into the query string, so a configured view can be shared by URL.
 *
 * Until the user edits anything, state is decoded from the URL. The server snapshot is "", so
 * the prerendered HTML and the hydration pass both use the defaults, and React then re-renders
 * with the real query. Edits are kept locally and written back with history.replaceState,
 * debounced: browsers throttle replaceState, and a slider drag fires dozens of updates.
 * `schema` and `defaults` must be module-level constants.
 */
export function useUrlState<S>(schema: Schema<S>, defaults: S, delayMs = 300) {
  const search = useSyncExternalStore(subscribe, clientSearch, serverSearch);
  const fromUrl = useMemo(() => decodeState(schema, new URLSearchParams(search), defaults), [schema, search, defaults]);
  const [edited, setEdited] = useState<S | null>(null);
  const state = edited ?? fromUrl;

  useEffect(() => {
    if (edited === null) return; // nothing to write; never overwrite incoming params
    const id = window.setTimeout(() => {
      const { pathname, search: current, hash } = window.location;
      const next = mergeQuery(schema, current, edited, defaults);
      if (next !== current) window.history.replaceState(null, "", `${pathname}${next}${hash}`);
    }, delayMs);
    return () => window.clearTimeout(id);
  }, [edited, schema, defaults, delayMs]);

  const update = useCallback(
    (patch: Partial<S> | ((s: S) => Partial<S>)) =>
      setEdited((cur) => {
        const base = cur ?? fromUrl;
        return { ...base, ...(typeof patch === "function" ? patch(base) : patch) };
      }),
    [fromUrl],
  );

  /** Absolute URL of the current view, built from state (so it's right even mid-debounce). */
  const shareUrl = useCallback(() => {
    const { origin, pathname, search: current } = window.location;
    return `${origin}${pathname}${mergeQuery(schema, current, state, defaults)}`;
  }, [schema, state, defaults]);

  return [state, update, shareUrl] as const;
}
