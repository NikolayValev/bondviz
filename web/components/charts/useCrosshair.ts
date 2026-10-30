"use client";
import { useState, type KeyboardEvent, type PointerEvent } from "react";

/**
 * Crosshair state shared by the charts: which data index is active and whether the pointer or
 * the keyboard put it there (only keyboard moves are announced to screen readers). Keys, with
 * the chart focused: ←/→ step, PageUp/PageDown jump a tenth, Home/End go to the ends, Escape
 * hides.
 *
 * `data` is the chart's x values (or categories), compared by identity: when it changes (a new
 * lookback, another ticker) the crosshair clears instead of pointing the old index at new data.
 */
export function useCrosshair(data: readonly unknown[] | null | undefined) {
  const count = data?.length ?? 0;
  const [state, setState] = useState<{ index: number; via: "pointer" | "key"; data: unknown } | null>(null);
  const current = state && state.data === data && state.index < count ? state : null;
  const index = current?.index ?? null;
  const via = current?.via ?? null;

  const show = (i: number, how: "pointer" | "key") => setState({ index: i, via: how, data });
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
    else if (e.key === "Escape" && current) {
      e.preventDefault();
      hide();
      return;
    }
    if (next !== null) {
      e.preventDefault();
      show(next, "key");
    }
  };

  /** Props for the chart's <svg>: focusable, pointer-tracking, keyboard-driven. `indexAt` maps a
   *  pointer x, measured from the plot's left edge, to a data index. */
  const svgProps = (marginLeft: number, indexAt: (plotX: number) => number) => ({
    tabIndex: 0,
    onPointerMove: (e: PointerEvent<SVGSVGElement>) =>
      show(indexAt(e.clientX - e.currentTarget.getBoundingClientRect().left - marginLeft), "pointer"),
    onPointerLeave: () => {
      if (via === "pointer") hide();
    },
    onKeyDown,
    onBlur: () => {
      if (via === "key") hide();
    },
  });

  return { index, via, show, hide, onKeyDown, svgProps };
}
