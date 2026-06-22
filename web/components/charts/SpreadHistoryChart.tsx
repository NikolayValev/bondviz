"use client";
import { useMemo } from "react";
import { scaleTime, scaleLinear } from "d3-scale";
import { line, area } from "d3-shape";
import { extent } from "d3-array";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import type { SpreadPoint, NberRecession } from "@/lib/signal";

export interface SpreadHistoryChartProps {
  points: SpreadPoint[];
  recessions: NberRecession[];
  ariaLabel: string;
  width?: number;
  height?: number;
}

const M = { top: 14, right: 18, bottom: 30, left: 50 };

interface XY { t: number; v: number; }

export function SpreadHistoryChart(props: SpreadHistoryChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 320;

  const c = useMemo(() => {
    if (width === 0 || props.points.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const ms = (d: string) => new Date(d).getTime();
    const s10 = props.points.filter((p) => p.s10y3m !== null).map((p) => ({ t: ms(p.date), v: p.s10y3m as number * 100 }));
    const s2 = props.points.filter((p) => p.s2s10s !== null).map((p) => ({ t: ms(p.date), v: p.s2s10s as number * 100 }));

    const allT = props.points.map((p) => ms(p.date));
    const allV = [...s10.map((p) => p.v), ...s2.map((p) => p.v), 0];
    const [t0, t1] = extent(allT) as [number, number];
    const x = scaleTime().domain([t0, t1]).range([0, iw]);
    const y = scaleLinear().domain([Math.min(...allV), Math.max(...allV)]).range([ih, 0]).nice();

    const lineGen = line<XY>().x((p) => x(p.t)).y((p) => y(p.v));

    // Inversion fill: area between the 10y3m line and zero, clipped to negatives.
    const zeroY = y(0);
    const invArea = area<XY>()
      .x((p) => x(p.t))
      .y0(() => zeroY)
      .y1((p) => (p.v < 0 ? y(p.v) : zeroY));

    const recBands = props.recessions
      .map((r) => {
        const rx0 = x(ms(r.start));
        const rx1 = x(ms(r.end));
        return { key: r.start, x: Math.min(rx0, rx1), w: Math.max(1, Math.abs(rx1 - rx0)) };
      })
      .filter((b) => b.x + b.w >= 0 && b.x <= iw);

    return {
      iw, ih, x, y, zeroY, s10, s2, lineGen, invArea, recBands,
      xTicks: x.ticks(8), yTicks: y.ticks(5),
    };
  }, [props.points, props.recessions, width, height]);

  return (
    <div ref={ref} className="w-full">
      {c && (
        <svg width={width} height={height} role="img" aria-label={props.ariaLabel} className="overflow-visible">
          <g transform={`translate(${M.left},${M.top})`}>
            {/* recession bands (behind everything) */}
            {c.recBands.map((b) => (
              <rect key={b.key} className="recession" x={b.x} y={0} width={b.w} height={c.ih} fill="var(--muted)" opacity={0.12} />
            ))}
            {/* y grid + labels */}
            {c.yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${c.y(t)})`}>
                <line x1={0} x2={c.iw} stroke="var(--grid)" />
                <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">{t}</text>
              </g>
            ))}
            {/* x ticks */}
            {c.xTicks.map((t) => (
              <g key={`x${+t}`} transform={`translate(${c.x(t)},${c.ih})`}>
                <line y1={0} y2={6} stroke="var(--faint)" />
                <text y={20} textAnchor="middle" fontSize={11} fill="var(--muted)" className="tabnum">
                  {new Date(+t).getUTCFullYear()}
                </text>
              </g>
            ))}
            {/* inversion fill (red, below zero) */}
            <path className="inversion-fill" d={c.invArea(c.s10) ?? ""} fill="var(--neg)" opacity={0.18} />
            {/* zero baseline */}
            <line x1={0} x2={c.iw} y1={c.zeroY} y2={c.zeroY} stroke="var(--panel-border-strong)" strokeDasharray="3 3" />
            {/* series */}
            <path className="series-2s10s" d={c.lineGen(c.s2) ?? ""} fill="none" stroke="#5b8def" strokeWidth={1.5} opacity={0.85} />
            <path className="series-10y3m" d={c.lineGen(c.s10) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2} />
          </g>
        </svg>
      )}
      <div className="mt-2 flex flex-wrap gap-4 text-xs text-[var(--muted)]">
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded bg-[var(--accent)]" /> 10y–3m</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded" style={{ background: "#5b8def" }} /> 2s10s</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[var(--neg)] opacity-30" /> inverted</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[var(--muted)] opacity-30" /> NBER recession</span>
      </div>
    </div>
  );
}
