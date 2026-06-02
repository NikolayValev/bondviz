"use client";
import { useMemo } from "react";
import { scaleLinear } from "d3-scale";
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

export function LineChart(props: LineChartProps) {
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

    const x = scaleLinear().domain([x0 ?? 0, x1 ?? 1]).range([0, iw]).nice();
    const y = scaleLinear().domain([y0 ?? 0, y1 ?? 1]).range([ih, 0]).nice();

    const gen = line<[number, number]>()
      .x((p) => x(p[0]))
      .y((p) => y(p[1]));

    const xTicks = x.ticks(6);
    const yTicks = y.ticks(5);
    const fmtX = (v: number) =>
      props.xType === "time" ? new Date(v).toLocaleDateString(undefined, { year: "2-digit", month: "short" }) : String(v);
    const fmtY = (v: number) => `${v}${props.yUnit ?? ""}`;

    return { iw, ih, x, y, gen, xTicks, yTicks, fmtX, fmtY };
  }, [props.series, props.zeroBaseline, props.xType, props.yUnit, width, height]);

  return (
    <div ref={ref} className="w-full">
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
}
