"use client";
import { memo, useId, useMemo } from "react";
import { scaleBand, scaleLinear } from "d3-scale";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import { ExportButtons } from "@/components/ui/ExportButtons";
import { categoryTable } from "@/lib/exportCsv";
import { nearestIndex, readoutDecimals } from "@/lib/crosshair";
import { useCrosshair } from "@/components/charts/useCrosshair";
import { ChartReadout } from "@/components/charts/ChartReadout";

export interface BarSeries {
  id: string;
  label: string;
  color: string;
  values: (number | null)[];
}

export interface CategoryBarChartProps {
  categories: string[];
  series: BarSeries[];
  ariaLabel: string;
  yUnit?: string;
  /** Tick label formatter; defaults to whole numbers plus yUnit. */
  formatY?: (v: number) => string;
  width?: number;
  height?: number;
  /** Shows CSV / PNG download buttons; the title heads the image and names the files. */
  exportTitle?: string;
  /** Header of the category column in the CSV. */
  categoryLabel?: string;
}

const M = { top: 12, right: 16, bottom: 32, left: 52 };

export const CategoryBarChart = memo(function CategoryBarChart(props: CategoryBarChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 280;
  const hintId = useId();

  const c = useMemo(() => {
    if (width === 0 || props.categories.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const val = (s: BarSeries, j: number) => s.values[j] ?? 0;

    // Diverging stack totals per category to size the y domain.
    let lo = 0;
    let hi = 0;
    props.categories.forEach((_, j) => {
      let pos = 0;
      let neg = 0;
      for (const s of props.series) {
        const v = val(s, j);
        if (v >= 0) pos += v;
        else neg += v;
      }
      hi = Math.max(hi, pos);
      lo = Math.min(lo, neg);
    });

    const x = scaleBand<string>().domain(props.categories).range([0, iw]).padding(0.25);
    const y = scaleLinear().domain([lo, hi]).range([ih, 0]).nice();

    // Pre-compute rects: stack positives upward from 0, negatives downward.
    interface Rect { key: string; x: number; y: number; w: number; h: number; fill: string; }
    const rects: Rect[] = [];
    const bw = x.bandwidth();
    props.categories.forEach((cat, j) => {
      let posAcc = 0;
      let negAcc = 0;
      for (const s of props.series) {
        const v = val(s, j);
        if (v === 0) continue;
        let y0: number;
        let y1: number;
        if (v >= 0) {
          y0 = y(posAcc);
          posAcc += v;
          y1 = y(posAcc);
        } else {
          y0 = y(negAcc);
          negAcc += v;
          y1 = y(negAcc);
        }
        const top = Math.min(y0, y1);
        rects.push({
          key: `${s.id}-${cat}`,
          x: (x(cat) ?? 0),
          y: top,
          w: bw,
          h: Math.max(1, Math.abs(y1 - y0)),
          fill: s.color,
        });
      }
    });

    const centers = props.categories.map((cat) => (x(cat) ?? 0) + bw / 2);
    // Label every nth category when the bands are narrower than a label (11px mono ≈ 6.6px a
    // character, plus a gap), so "10Y20Y30Y" doesn't run together on a phone.
    const longest = Math.max(...props.categories.map((cat) => cat.length));
    const labelEvery = Math.max(1, Math.ceil((longest * 6.6 + 8) / x.step()));
    return { iw, ih, x, y, rects, yTicks: y.ticks(5), bw, centers, span: hi - lo, labelEvery };
  }, [props.categories, props.series, width, height]);

  const fmtY = props.formatY ?? ((v: number) => `${Math.round(v)}${props.yUnit ?? ""}`);

  const cross = useCrosshair(props.categories);
  const readout = useMemo(() => {
    if (!c || cross.index === null) return null;
    const j = cross.index;
    const unit = props.yUnit ?? "";
    const decimals = /^\s*bps?$/.test(unit) ? 0 : readoutDecimals(c.span);
    const fmt = props.formatY ?? ((v: number) => `${v.toFixed(decimals)}${unit}`);
    const rows = props.series
      .map((s) => ({ id: s.id, label: s.label, color: s.color, value: s.values[j] }))
      .filter((r): r is typeof r & { value: number } => r.value !== null && r.value !== undefined)
      .map((r) => ({ ...r, text: fmt(r.value) }));
    const total = rows.length > 1 ? fmt(rows.reduce((a, r) => a + r.value, 0)) : null;
    const title = props.categories[j];
    return {
      j,
      title,
      rows,
      total,
      text: `${title}: ${rows.map((r) => `${r.label} ${r.text}`).join(", ")}${total ? `, total ${total}` : ""}`,
    };
  }, [c, cross.index, props.categories, props.series, props.formatY, props.yUnit]);

  return (
    <div className="w-full">
      {/* The plot box reserves its height before the width is measured, so the legend below
          doesn't jump down when the SVG appears (layout shift). */}
      <div ref={ref} className="relative" style={{ minHeight: height }}>
        {c && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={props.ariaLabel}
            aria-describedby={hintId}
            className="overflow-visible touch-pan-y"
            {...cross.svgProps(M.left, (px) => nearestIndex(c.centers, px))}
          >
            <g transform={`translate(${M.left},${M.top})`}>
              {readout && (
                <rect
                  className="hover-band"
                  x={c.centers[readout.j] - c.x.step() / 2}
                  y={0}
                  width={c.x.step()}
                  height={c.ih}
                  fill="var(--text)"
                  opacity={0.06}
                  pointerEvents="none"
                />
              )}
              {c.yTicks.map((t) => (
                <g key={`y${t}`} transform={`translate(0,${c.y(t)})`}>
                  <line x1={0} x2={c.iw} stroke="var(--grid)" />
                  <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">
                    {fmtY(t)}
                  </text>
                </g>
              ))}
              <line x1={0} x2={c.iw} y1={c.y(0)} y2={c.y(0)} stroke="var(--panel-border-strong)" />
              {c.rects.map((r) => (
                <rect key={r.key} className="bar" x={r.x} y={r.y} width={r.w} height={r.h} fill={r.fill} rx={1.5} />
              ))}
              {props.categories.map((cat, j) => j % c.labelEvery === 0 && (
                <text
                  key={`x${cat}`}
                  x={(c.x(cat) ?? 0) + c.bw / 2}
                  y={c.ih + 18}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--muted)"
                  className="tabnum"
                >
                  {cat}
                </text>
              ))}
            </g>
          </svg>
        )}
        <ChartReadout
          hintId={hintId}
          hint="Focus the chart and use the left and right arrow keys to read each bar."
          box={
            c && readout
              ? {
                  x: M.left + c.centers[readout.j],
                  top: M.top + 4,
                  flip: c.centers[readout.j] > c.iw / 2,
                  offset: c.x.step() / 2 + 4,
                  title: readout.title,
                  rows: readout.rows,
                  footer: readout.total ? (
                    <div className="mt-1 flex justify-between gap-4 border-t border-[var(--panel-border)] pt-1 text-[var(--muted)]">
                      Total <span className="tabnum text-[var(--text)]">{readout.total}</span>
                    </div>
                  ) : null,
                }
              : null
          }
          announce={readout && cross.via === "key" ? readout.text : null}
        />
      </div>
      {(props.series.length > 1 || props.exportTitle) && (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-[var(--muted)]">
          {props.series.length > 1 &&
            props.series.map((s) => (
              <span key={s.id} className="inline-flex items-center gap-1.5">
                <span className="inline-block h-2 w-3 rounded-sm" style={{ background: s.color }} />
                {s.label}
              </span>
            ))}
          {props.exportTitle && (
            <ExportButtons
              className="ml-auto"
              title={props.exportTitle}
              csv={() => {
                const unit = props.yUnit?.trim();
                return categoryTable(
                  props.categoryLabel ?? "Category",
                  props.categories,
                  props.series.map((s) => ({ label: unit ? `${s.label} (${unit})` : s.label, values: s.values })),
                );
              }}
              svg={() => ref.current?.querySelector("svg")}
              legend={props.series.length > 1 ? props.series : []}
            />
          )}
        </div>
      )}
    </div>
  );
});
