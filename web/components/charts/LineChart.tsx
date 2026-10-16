"use client";
import { memo, useMemo } from "react";
import { scaleLinear, scaleUtc } from "d3-scale";
import { line } from "d3-shape";
import { extent } from "d3-array";
import { useResizeObserver } from "@/components/charts/useResizeObserver";

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
  yUnit?: string;
  zeroBaseline?: boolean;
}

const M = { top: 12, right: 16, bottom: 36, left: 48 };

export const LineChart = memo(function LineChart(props: LineChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 280;

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
    const xDomain = [x0 ?? 0, x1 ?? 1];
    const xTime = scaleUtc().domain(xDomain).range([0, iw]);
    const xLin = scaleLinear().domain(xDomain).range([0, iw]).nice();
    const x = (v: number) => (isTime ? xTime(v) : xLin(v));
    const y = scaleLinear().domain([y0 ?? 0, y1 ?? 1]).range([ih, 0]).nice();

    const gen = line<[number, number]>()
      .x((p) => x(p[0]))
      .y((p) => y(p[1]));

    const xTicks = isTime ? xTime.ticks(6).map((d) => d.getTime()) : xLin.ticks(6);
    const yTicks = y.ticks(5);
    // Under ~4 months the ticks fall inside months, so label the day; beyond that, month + year.
    const shortSpan = isTime && xDomain[1] - xDomain[0] < 120 * 86_400_000;
    const timeFmt: Intl.DateTimeFormatOptions = shortSpan
      ? { month: "short", day: "numeric", timeZone: "UTC" }
      : { month: "short", year: "2-digit", timeZone: "UTC" };
    const fmtX = (v: number) => (isTime ? new Date(v).toLocaleDateString(undefined, timeFmt) : String(v));
    const fmtY = (v: number) => `${v}${props.yUnit ?? ""}`;

    return { iw, ih, x, y, gen, xTicks, yTicks, fmtX, fmtY };
  }, [props.series, props.zeroBaseline, props.xType, props.yUnit, width, height]);

  return (
    <div className="w-full">
      {/* The plot box reserves its height before the width is measured, so the legend below
          doesn't jump down when the SVG appears (layout shift). */}
      <div ref={ref} style={{ minHeight: height }}>
        {content && (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={props.ariaLabel}
            className="overflow-visible"
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
                  {props.xLabel}
                </text>
              )}
              {props.yLabel && (
                <text transform={`translate(${-36},${content.ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--muted)">
                  {props.yLabel}
                </text>
              )}
            </g>
          </svg>
        )}
      </div>
      {props.series.length > 1 && (
        <div className="mt-2 flex flex-wrap gap-4 text-xs text-[var(--muted)]">
          {props.series.map((s) => (
            <span key={s.id} className="inline-flex items-center gap-1.5">
              <span className="inline-block h-2 w-3 rounded-sm" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
});
