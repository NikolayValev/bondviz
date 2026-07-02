// Pure layout math for the command-panel dashboard: widgets live on a
// 12-column grid with a per-widget column span and content height (px),
// persisted per widget id. No React, no storage — inputs → outputs.

export interface WidgetLayout {
  span: number; // grid columns occupied, SPAN_MIN..SPAN_MAX
  h: number; // content height in px, H_MIN..H_MAX
}

export const SPAN_MIN = 3;
export const SPAN_MAX = 12;
export const H_MIN = 200;
export const H_MAX = 800;
export const DEFAULT_LAYOUT: WidgetLayout = { span: 6, h: 320 };

export function clampLayout(l: WidgetLayout): WidgetLayout {
  return {
    span: Math.min(SPAN_MAX, Math.max(SPAN_MIN, Math.round(l.span))),
    h: Math.min(H_MAX, Math.max(H_MIN, Math.round(l.h))),
  };
}

/** Apply a pointer-drag delta to a starting layout. `dxCols` is fractional columns. */
export function resizeLayout(start: WidgetLayout, dxCols: number, dyPx: number): WidgetLayout {
  return clampLayout({ span: start.span + dxCols, h: start.h + dyPx });
}

/** Parse persisted layouts JSON; malformed input or entries degrade to {} / are dropped. */
export function parseLayouts(json: string | null): Record<string, WidgetLayout> {
  if (!json) return {};
  try {
    const raw: unknown = JSON.parse(json);
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return {};
    const out: Record<string, WidgetLayout> = {};
    for (const [id, v] of Object.entries(raw)) {
      if (
        v !== null &&
        typeof v === "object" &&
        typeof (v as WidgetLayout).span === "number" &&
        Number.isFinite((v as WidgetLayout).span) &&
        typeof (v as WidgetLayout).h === "number" &&
        Number.isFinite((v as WidgetLayout).h)
      ) {
        out[id] = clampLayout(v as WidgetLayout);
      }
    }
    return out;
  } catch {
    return {};
  }
}

/** Drop layouts for widgets that are no longer pinned. */
export function pruneLayouts(
  layouts: Record<string, WidgetLayout>,
  pinnedIds: string[],
): Record<string, WidgetLayout> {
  const keep = new Set(pinnedIds);
  return Object.fromEntries(Object.entries(layouts).filter(([id]) => keep.has(id)));
}
