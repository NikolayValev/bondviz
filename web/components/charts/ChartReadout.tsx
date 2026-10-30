import type { ReactNode } from "react";

export interface ReadoutRow {
  id: string;
  label: string;
  color?: string;
  text: string;
}

/**
 * The value box, keyboard hint and screen-reader announcement shared by every chart with a
 * crosshair. Place it inside the chart's `relative` plot box, after the <svg>. The box sits
 * `offset` px right of `x` (both measured in the plot box), or to its left past the middle.
 */
export function ChartReadout({
  hintId,
  hint = "Focus the chart and use the left and right arrow keys to read values.",
  box,
  announce,
}: {
  hintId: string;
  hint?: string;
  /** The visible readout, or null when the crosshair is hidden. */
  box: { x: number; top: number; flip: boolean; offset?: number; title: string; rows: ReadoutRow[]; footer?: ReactNode } | null;
  /** Text for the polite live region (only keyboard moves should pass one). */
  announce: string | null;
}) {
  const offset = box?.offset ?? 12;
  return (
    <>
      {box && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute z-10 min-w-32 rounded-md border border-[var(--panel-border-strong)] bg-[var(--panel-2)] px-2.5 py-2 text-xs shadow-lg"
          style={{
            top: box.top,
            left: box.x,
            transform: box.flip ? `translateX(calc(-100% - ${offset}px))` : `translateX(${offset}px)`,
          }}
        >
          <div className="tabnum mb-1 text-[var(--text)]">{box.title}</div>
          {box.rows.map((r) => (
            <div key={r.id} className="flex items-center justify-between gap-4 text-[var(--muted)]">
              <span className="inline-flex items-center gap-1.5">
                {r.color && <span className="inline-block h-2 w-2 rounded-full" style={{ background: r.color }} />}
                {r.label}
              </span>
              <span className="tabnum text-[var(--text)]">{r.text}</span>
            </div>
          ))}
          {box.footer}
        </div>
      )}
      <span id={hintId} className="sr-only">
        {hint}
      </span>
      <div className="sr-only" aria-live="polite">
        {announce ?? ""}
      </div>
    </>
  );
}
