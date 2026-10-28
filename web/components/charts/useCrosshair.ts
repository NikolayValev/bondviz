"use client";
import { useState, type KeyboardEvent } from "react";

/**
 * Crosshair state shared by the charts: which data index is active and whether the pointer or
 * the keyboard put it there (only keyboard moves are announced to screen readers). Keys, with
 * the chart focused: ←/→ step, PageUp/PageDown jump a tenth, Home/End go to the ends, Escape
 * hides.
 */
export function useCrosshair(count: number) {
  const [state, setState] = useState<{ index: number; via: "pointer" | "key" } | null>(null);
  const index = state && count > 0 ? Math.min(state.index, count - 1) : null;

  const show = (i: number, via: "pointer" | "key") => setState({ index: i, via });
  const hide = () => setState(null);

  const onKeyDown = (e: KeyboardEvent) => {
    if (count === 0) return;
    const step = Math.max(1, Math.round(count / 10));
    const at = index ?? -1;
    let next: number | null = null;
    if (e.key === "ArrowRight") next = Math.min(count - 1, at + 1);
    else if (e.key === "ArrowLeft") next = at < 0 ? count - 1 : Math.max(0, at - 1);
    else if (e.key === "PageDown") next = Math.min(count - 1, Math.max(0, at) + step);
    else if (e.key === "PageUp") next = Math.max(0, (at < 0 ? count - 1 : at) - step);
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = count - 1;
    else if (e.key === "Escape" && state) {
      e.preventDefault();
      hide();
      return;
    }
    if (next !== null) {
      e.preventDefault();
      show(next, "key");
    }
  };

  return { index, via: state?.via ?? null, show, hide, onKeyDown };
}
