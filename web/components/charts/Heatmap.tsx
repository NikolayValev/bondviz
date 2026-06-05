"use client";
import { useMemo } from "react";
import { useResizeObserver } from "@/components/charts/useResizeObserver";

// Three-stop ramp: deep navy (low) → teal → accent green (high). Hand-rolled
// to avoid pulling in d3-scale-chromatic.
const STOPS: [number, number, number][] = [
  [13, 27, 42],
  [27, 94, 110],
  [0, 214, 143],
];

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
}

const M = { top: 8, right: 12, bottom: 28, left: 12 };

export function Heatmap(props: HeatmapProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 260;

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
    const span = max - min || 1;
    return { iw, ih, cellW, cellH, min, max, span };
  }, [props.dates, props.tenors, props.values, width, height]);

  return (
    <div ref={ref} className="w-full">
      {content && (
        <svg width={width} height={height} role="img" aria-label={props.ariaLabel}>
          <g transform={`translate(${M.left},${M.top})`}>
            {props.values.map((row, di) =>
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
          </g>
        </svg>
      )}
    </div>
  );
}
