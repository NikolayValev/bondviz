// Browser-only localStorage I/O for widget layouts; the pure grid math lives
// in lib/panelLayout.

import { parseLayouts, pruneLayouts, type WidgetLayout } from "@/lib/panelLayout";

const LAYOUT_KEY = "command-panel:layout:v1";

export function loadLayouts(): Record<string, WidgetLayout> {
  return parseLayouts(localStorage.getItem(LAYOUT_KEY));
}

/** Persist layouts, dropping entries for widgets that are no longer pinned. */
export function saveLayouts(layouts: Record<string, WidgetLayout>, pinnedIds: string[]): void {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(pruneLayouts(layouts, pinnedIds)));
  } catch {
    /* storage unavailable — sizes just won't persist */
  }
}
