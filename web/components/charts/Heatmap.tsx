"use client";
import { memo, useId, useMemo, useState, type KeyboardEvent, type PointerEvent } from "react";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import { HEATMAP_STOPS as STOPS } from "@/lib/chartColors";
import { ExportButtons } from "@/components/ui/ExportButtons";
import { categoryTable } from "@/lib/exportCsv";
import { ChartReadout } from "@/components/charts/ChartReadout";
import { dayLabel } from "@/lib/dates";

export function colorRamp(t: number): string {
  const tc = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0));
  const seg = tc * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(seg));
  const f = seg - i;
  const [r0, g0, b0] = STOPS[i];
  const [r1, g1, b1] = STOPS[i + 1];
  const r = Math.round(r0 + (r1 - r0) * f);
  const g = Math.round(g0 + (g1 - g0) * f);
  const b = Math.round(b0 + (b1 - b0) * f);
  return `rgb(${r}, ${g}, ${b})`;
}

export interface HeatmapProps {
  dates: string[];               // rows, old → new
  tenors: string[];              // columns
  values: (number | null)[][];   // [dateIndex][tenorIndex]
  ariaLabel: string;
  width?: number;
  height?: number;
  unit?: string;                 // appended to the legend's min/max labels
  /** Shows CSV / PNG download buttons; the title heads the image and names the files. */
  exportTitle?: string;
}

// Left margin holds the date axis; bottom holds tenor labels plus the color legend.
const M = { top: 8, right: 12, bottom: 64, left: 64 };
const LEGEND_W = 160;
const MIN_LABEL_GAP = 22; // px between date labels

/** Row indices to label: first, last, and evenly spaced rows between that fit the height. */
export function dateTickIndices(rows: number, innerHeight: number): number[] {
  if (rows <= 0) return [];
  if (rows === 1) return [0];
  const slots = Math.max(2, Math.min(6, Math.floor(innerHeight / MIN_LABEL_GAP)));
  const out = new Set<number>();
  for (let k = 0; k < slots; k++) out.add(Math.round((k * (rows - 1)) / (slots - 1)));
  return [...out].sort((a, b) => a - b);
}

const fmtDate = (d: string) =>
  new Date(`${d.slice(0, 10)}T00:00:00Z`).toLocaleDateString(undefined, {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  });

export const Heatmap = memo(function Heatmap(props: HeatmapProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 300;
  const gradientId = useId();
  const hintId = useId();

  const content = useMemo(() => {
    if (width === 0 || props.dates.length === 0 || props.tenors.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);
    const cellW = iw / props.tenors.length;
    const cellH = ih / props.dates.length;

    let min = Infinity;
    let max = -Infinity;
    for (const row of props.values) {
      for (const v of row) {
        if (v !== null && Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
    }
    const span = max > min ? max - min : 1;
    const dateTicks = dateTickIndices(props.dates.length, ih);
    const hasValues = Number.isFinite(min) && Number.isFinite(max);
    return { iw, ih, cellW, cellH, min, max, span, dateTicks, hasValues };
  }, [props.dates, props.tenors, props.values, width, height]);

  // The cells are built once per layout: the readout re-renders on every pointer move, and a
  // year of daily rows is ~3,000 rects.
  const cells = useMemo(
    () =>
      content &&
      props.values.map((row, di) =>
        row.map((v, ti) => (
          <rect
            key={`${di}-${ti}`}
            className="heatmap-cell"
            x={ti * content.cellW}
            y={di * content.cellH}
            width={content.cellW + 0.5}
            height={content.cellH + 0.5}
            fill={v === null || !Number.isFinite(v) ? "var(--grid)" : colorRamp((v - content.min) / content.span)}
          />
        )),
      ),
    [content, props.values],
  );

  // Two-dimensional readout: ←/→ change the tenor, ↑/↓ the date (PageUp/PageDown jump a tenth,
  // Home/End go to the first/last date), Escape hides; the pointer reads the cell under it. It
  // clears when the data changes.
  const rows = props.dates.length;
  const cols = props.tenors.length;
  const [cell, setCell] = useState<{ r: number; c: number; via: "pointer" | "key"; data: unknown } | null>(null);
  const active = cell && cell.data === props.values && cell.r < rows && cell.c < cols ? cell : null;
  const show = (r: number, c: number, via: "pointer" | "key") => setCell({ r, c, via, data: props.values });

  const onKeyDown = (e: KeyboardEvent) => {
    if (rows === 0 || cols === 0) return;
    const r = active?.r ?? rows - 1;
    const c = active?.c ?? 0;
    const step = Math.max(1, Math.round(rows / 10));
    const clampR = (v: number) => Math.min(rows - 1, Math.max(0, v));
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [r, Math.max(0, c - 1)],
      ArrowRight: [r, Math.min(cols - 1, c + 1)],
      ArrowUp: [clampR(r - 1), c],
      ArrowDown: [clampR(r + 1), c],
      PageUp: [clampR(r - step), c],
      PageDown: [clampR(r + step), c],
      Home: [0, c],
      End: [rows - 1, c],
    };
    if (e.key === "Escape" && active) {
      e.preventDefault();
      setCell(null);
    } else if (moves[e.key]) {
      e.preventDefault();
      // The first key press lands on the latest date's first tenor.
      const [nr, nc] = active ? moves[e.key] : [r, c];
      show(nr, nc, "key");
    }
  };

  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    if (!content) return;
    const box = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - box.left - M.left;
    const py = e.clientY - box.top - M.top;
    if (px < 0 || py < 0 || px >= content.iw || py >= content.ih) {
      if (active?.via === "pointer") setCell(null);
      return;
    }
    show(Math.floor(py / content.cellH), Math.floor(px / content.cellW), "pointer");
  };

  const readout = useMemo(() => {
    if (!content || !active) return null;
    const v = props.values[active.r]?.[active.c] ?? null;
    const text = v === null || !Number.isFinite(v) ? "no data" : `${v.toFixed(2)}${props.unit ?? ""}`;
    const title = dayLabel(props.dates[active.r]);
    return { title, tenor: props.tenors[active.c], text, announce: `${title}, ${props.tenors[active.c]}: ${text}` };
  }, [content, active, props.values, props.dates, props.tenors, props.unit]);

  return (
    <div className="w-full">
      <div ref={ref} className="relative" style={{ minHeight: height }}>
        {content && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={props.ariaLabel}
            aria-describedby={hintId}
            tabIndex={0}
            className="overflow-visible touch-pan-y"
            onPointerMove={onPointerMove}
            onPointerLeave={() => active?.via === "pointer" && setCell(null)}
            onKeyDown={onKeyDown}
            onBlur={() => active?.via === "key" && setCell(null)}
          >
            <g transform={`translate(${M.left},${M.top})`}>
              {cells}
              {active && (
                <rect
                  className="heatmap-focus"
                  x={active.c * content.cellW}
                  y={active.r * content.cellH}
                  width={content.cellW}
                  height={Math.max(2, content.cellH)}
                  fill="none"
                  stroke="var(--text)"
                  strokeWidth={1.5}
                  pointerEvents="none"
                />
              )}
              {props.tenors.map((label, ti) => (
                <text
                  key={label}
                  x={ti * content.cellW + content.cellW / 2}
                  y={content.ih + 18}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--muted)"
                  className="tabnum"
                >
                  {label}
                </text>
              ))}
              {content.dateTicks.map((di) => (
                <text
                  key={`d${di}`}
                  x={-8}
                  y={di * content.cellH + content.cellH / 2}
                  dy="0.32em"
                  textAnchor="end"
                  fontSize={11}
                  fill="var(--muted)"
                  className="tabnum heatmap-date"
                >
                  {fmtDate(props.dates[di])}
                </text>
              ))}
              {content.hasValues && (
                <g className="heatmap-legend" transform={`translate(${Math.max(0, content.iw - LEGEND_W)},${content.ih + 32})`}>
                  <defs>
                    <linearGradient id={gradientId} x1="0" x2="1" y1="0" y2="0">
                      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
                        <stop key={t} offset={t} stopColor={colorRamp(t)} />
                      ))}
                    </linearGradient>
                  </defs>
                  <rect width={Math.min(LEGEND_W, content.iw)} height={8} rx={2} fill={`url(#${gradientId})`} />
                  <text y={22} fontSize={11} fill="var(--muted)" className="tabnum">
                    {`${content.min.toFixed(2)}${props.unit ?? ""}`}
                  </text>
                  <text x={Math.min(LEGEND_W, content.iw)} y={22} textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">
                    {`${content.max.toFixed(2)}${props.unit ?? ""}`}
                  </text>
                </g>
              )}
            </g>
          </svg>
        )}
        <ChartReadout
          hintId={hintId}
          hint="Focus the chart: left and right arrow keys change the tenor, up and down the date."
          box={
            content && active && readout
              ? {
                  x: M.left + (active.c + 0.5) * content.cellW,
                  top: M.top + Math.min(active.r * content.cellH, Math.max(0, content.ih - 56)),
                  flip: (active.c + 0.5) * content.cellW > content.iw / 2,
                  offset: content.cellW / 2 + 4,
                  title: readout.title,
                  rows: [{ id: "v", label: readout.tenor, text: readout.text }],
                }
              : null
          }
          announce={readout && active?.via === "key" ? readout.announce : null}
        />
      </div>
      {props.exportTitle && (
        <ExportButtons
          className="mt-2 justify-end"
          title={props.exportTitle}
          csv={() =>
            categoryTable(
              "Date",
              props.dates,
              props.tenors.map((t, ti) => ({
                label: props.unit ? `${t} (${props.unit})` : t,
                values: props.values.map((row) => row[ti] ?? null),
              })),
            )
          }
          svg={() => ref.current?.querySelector("svg")}
        />
      )}
    </div>
  );
});
