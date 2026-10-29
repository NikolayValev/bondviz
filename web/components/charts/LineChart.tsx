"use client";
import { memo, useId, useMemo } from "react";
import { scaleLinear, scaleSqrt, scaleUtc } from "d3-scale";
import { line } from "d3-shape";
import { extent } from "d3-array";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import { ExportButtons } from "@/components/ui/ExportButtons";
import { axisUnit, seriesTable } from "@/lib/exportCsv";
import { nearestIndex, readoutDecimals, valueAt } from "@/lib/crosshair";
import { useCrosshair } from "@/components/charts/useCrosshair";

export interface Series {
  id: string;
  label: string;
  color: string;
  points: [number, number][];
}

export interface LineChartProps {
  series: Series[];
  ariaLabel: string;
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
  xType?: "linear" | "time";
  /** "sqrt" spaces maturities by √years so the 1M–2Y front end isn't crushed against the axis;
   *  for maturity axes starting at 0. Ignored for time axes. */
  xScale?: "linear" | "sqrt";
  yUnit?: string;
  zeroBaseline?: boolean;
  /** Shows CSV / PNG download buttons; the title heads the image and names the files. */
  exportTitle?: string;
  /** Formats values in the hover / keyboard readout; defaults to a precision that suits the data. */
  formatValue?: (v: number) => string;
}

const M = { top: 12, right: 16, bottom: 36, left: 48 };

/** Tick maturities for a √ axis: dense where the tenors are, at round numbers. */
const SQRT_TICKS = [0, 1, 2, 5, 10, 20, 30, 50];

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/**
 * Labels for UTC time-axis ticks. Day ticks read "Oct 14". Month ticks read "May ’26", "Jun",
 * "Jul"…, with the year on the first tick and again at each January, so "Aug 26" can't be
 * mistaken for a date. Ticks that all fall on New Year read as plain years.
 */
export function timeTickLabels(ticks: number[]): string[] {
  const ds = ticks.map((t) => new Date(t));
  if (ds.some((d) => d.getUTCDate() !== 1)) return ds.map((d) => `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`);
  if (ds.length > 1 && ds.every((d) => d.getUTCMonth() === 0)) return ds.map((d) => String(d.getUTCFullYear()));
  return ds.map((d, i) =>
    i === 0 || d.getUTCMonth() === 0
      ? `${MONTHS[d.getUTCMonth()]} ’${String(d.getUTCFullYear()).slice(2)}`
      : MONTHS[d.getUTCMonth()],
  );
}

/** A maturity in years as a tenor: 0.0833 → "1M", 0.5 → "6M", 7.5 → "7.5Y". */
export function maturityLabel(years: number): string {
  return years < 1 ? `${Math.round(years * 12)}M` : `${+years.toFixed(2)}Y`;
}

const dateLabel = (t: number) => {
  const d = new Date(t);
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()}`;
};

export const LineChart = memo(function LineChart(props: LineChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 280;
  const hintId = useId();

  const content = useMemo(() => {
    if (width === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const allX = props.series.flatMap((s) => s.points.map((p) => p[0]));
    const allY = props.series.flatMap((s) => s.points.map((p) => p[1]));
    if (props.zeroBaseline) allY.push(0);
    const [x0, x1] = extent(allX) as [number, number];
    const [y0, y1] = extent(allY) as [number, number];

    // Time x values are epoch-ms at UTC midnight (from "YYYY-MM-DD"), so a UTC time scale puts
    // ticks on real calendar boundaries; a linear scale would tick at arbitrary milliseconds.
    const isTime = props.xType === "time";
    const isSqrt = !isTime && props.xScale === "sqrt";
    const xDomain = [x0 ?? 0, x1 ?? 1];
    const xTime = scaleUtc().domain(xDomain).range([0, iw]);
    const xLin = isSqrt
      ? scaleSqrt().domain([0, Math.max(xDomain[1], 1e-9)]).range([0, iw])
      : scaleLinear().domain(xDomain).range([0, iw]).nice();
    const x = (v: number) => (isTime ? xTime(v) : xLin(v));
    const y = scaleLinear().domain([y0 ?? 0, y1 ?? 1]).range([ih, 0]).nice();

    const gen = line<[number, number]>()
      .x((p) => x(p[0]))
      .y((p) => y(p[1]));

    // Date labels need ~55px each ("May ’26" is the widest), so a phone-width chart gets fewer time ticks.
    const xTicks = isTime
      ? xTime.ticks(Math.max(2, Math.min(6, Math.floor(iw / 56)))).map((d) => d.getTime())
      : isSqrt
        ? SQRT_TICKS.filter((t) => t <= xDomain[1] + 1e-9)
        : xLin.ticks(6);
    const yTicks = y.ticks(5);
    const timeLabels = isTime ? new Map(timeTickLabels(xTicks).map((l, i) => [xTicks[i], l])) : null;
    const fmtX = (v: number) => timeLabels?.get(v) ?? String(v);
    const fmtY = (v: number) => `${v}${props.yUnit ?? ""}`;

    // Crosshair: every distinct x across the series, and each series sorted for lookup.
    const xs = [...new Set(allX)].sort((a, b) => a - b);
    const sorted = props.series.map((s) => [...s.points].sort((a, b) => a[0] - b[0]));
    const invertX = (px: number) => (isTime ? xTime.invert(px).getTime() : xLin.invert(px));

    return { iw, ih, x, y, gen, xTicks, yTicks, fmtX, fmtY, xs, sorted, invertX, isSqrt, ySpan: (y1 ?? 0) - (y0 ?? 0) };
  }, [props.series, props.zeroBaseline, props.xType, props.xScale, props.yUnit, width, height]);

  const cross = useCrosshair(content?.xs.length ?? 0);
  const readout = useMemo(() => {
    if (!content || cross.index === null) return null;
    const xv = content.xs[cross.index];
    const xUnit = axisUnit(props.xLabel);
    const yUnit = axisUnit(props.yLabel);
    const suffix = props.yUnit || (yUnit && /^(bps?|%)$/.test(yUnit) ? (yUnit === "%" ? "%" : ` ${yUnit}`) : "");
    const decimals = /^bps?$/.test(suffix.trim()) ? 0 : readoutDecimals(content.ySpan);
    const fmt = props.formatValue ?? ((v: number) => `${v.toFixed(decimals)}${suffix}`);
    return {
      xv,
      title:
        props.xType === "time" ? dateLabel(xv) : xUnit === "years" ? maturityLabel(xv) : `${props.xLabel ?? "x"} ${+xv.toFixed(4)}`,
      rows: props.series
        .map((s, i) => ({ id: s.id, label: s.label, color: s.color, value: valueAt(content.sorted[i], xv) }))
        .filter((r): r is typeof r & { value: number } => r.value !== null)
        .map((r) => ({ ...r, text: fmt(r.value) })),
    };
  }, [content, cross.index, props.series, props.xLabel, props.yLabel, props.yUnit, props.xType, props.formatValue]);

  return (
    <div className="w-full">
      {/* The plot box reserves its height before the width is measured, so the legend below
          doesn't jump down when the SVG appears (layout shift). */}
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
            onPointerMove={(e) => {
              const left = e.currentTarget.getBoundingClientRect().left + M.left;
              cross.show(nearestIndex(content.xs, content.invertX(e.clientX - left)), "pointer");
            }}
            onPointerLeave={() => cross.via === "pointer" && cross.hide()}
            onKeyDown={cross.onKeyDown}
            onBlur={() => cross.via === "key" && cross.hide()}
          >
            <g transform={`translate(${M.left},${M.top})`}>
              {content.yTicks.map((t) => (
                <g key={`y${t}`} transform={`translate(0,${content.y(t)})`}>
                  <line x1={0} x2={content.iw} stroke="var(--grid)" />
                  <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">
                    {content.fmtY(t)}
                  </text>
                </g>
              ))}
              {content.xTicks.map((t) => (
                <g key={`x${t}`} transform={`translate(${content.x(t)},${content.ih})`}>
                  <line y1={0} y2={6} stroke="var(--muted)" />
                  <text y={20} textAnchor="middle" fontSize={11} fill="var(--muted)" className="tabnum">
                    {content.fmtX(t)}
                  </text>
                </g>
              ))}
              {props.zeroBaseline && (
                <line x1={0} x2={content.iw} y1={content.y(0)} y2={content.y(0)} stroke="var(--muted)" strokeDasharray="3 3" />
              )}
              {props.series.map((s) => (
                <path
                  key={s.id}
                  className="series-line"
                  d={content.gen(s.points) ?? ""}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                />
              ))}
              {props.xLabel && (
                <text x={content.iw / 2} y={content.ih + 32} textAnchor="middle" fontSize={12} fill="var(--muted)">
                  {content.isSqrt ? `${props.xLabel} · √ scale` : props.xLabel}
                </text>
              )}
              {props.yLabel && (
                <text transform={`translate(${-36},${content.ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--muted)">
                  {props.yLabel}
                </text>
              )}
              {readout && (
                <g className="crosshair" pointerEvents="none">
                  <line
                    x1={content.x(readout.xv)}
                    x2={content.x(readout.xv)}
                    y1={0}
                    y2={content.ih}
                    stroke="var(--muted)"
                    strokeDasharray="3 3"
                  />
                  {readout.rows.map((r, i) => (
                    <circle
                      key={r.id}
                      cx={content.x(readout.xv)}
                      cy={content.y(r.value)}
                      r={i === 0 ? 4 : 3.5}
                      fill={r.color}
                      stroke="var(--panel)"
                      strokeWidth={2}
                    />
                  ))}
                </g>
              )}
            </g>
          </svg>
        )}
        {content && readout && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute z-10 min-w-32 rounded-md border border-[var(--panel-border-strong)] bg-[var(--panel-2)] px-2.5 py-2 text-xs shadow-lg"
            style={{
              top: M.top + 4,
              left: M.left + content.x(readout.xv),
              transform: content.x(readout.xv) > content.iw / 2 ? "translateX(calc(-100% - 12px))" : "translateX(12px)",
            }}
          >
            <div className="tabnum mb-1 text-[var(--text)]">{readout.title}</div>
            {readout.rows.map((r) => (
              <div key={r.id} className="flex items-center justify-between gap-4 text-[var(--muted)]">
                <span className="inline-flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: r.color }} />
                  {r.label}
                </span>
                <span className="tabnum text-[var(--text)]">{r.text}</span>
              </div>
            ))}
          </div>
        )}
        <span id={hintId} className="sr-only">
          Focus the chart and use the left and right arrow keys to read values.
        </span>
        <div className="sr-only" aria-live="polite">
          {readout && cross.via === "key"
            ? `${readout.title}: ${readout.rows.map((r) => `${r.label} ${r.text}`).join(", ")}`
            : ""}
        </div>
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
              csv={() =>
                seriesTable(props.series, {
                  xLabel: props.xLabel ?? (props.xType === "time" ? "Date" : "x"),
                  xType: props.xType,
                  unit: axisUnit(props.yLabel),
                })
              }
              svg={() => ref.current?.querySelector("svg")}
              legend={props.series.length > 1 ? props.series : []}
            />
          )}
        </div>
      )}
    </div>
  );
});
