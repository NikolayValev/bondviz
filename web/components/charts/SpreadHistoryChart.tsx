"use client";
import { useId, useMemo } from "react";
import { scaleUtc, scaleLinear } from "d3-scale";
import { line, area } from "d3-shape";
import { extent, min, max } from "d3-array";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import { ExportButtons } from "@/components/ui/ExportButtons";
import { useCrosshair } from "@/components/charts/useCrosshair";
import { nearestIndex } from "@/lib/crosshair";
import { SERIES } from "@/lib/chartColors";
import type { SpreadPoint, NberRecession } from "@/lib/signal";

export interface SpreadHistoryChartProps {
  points: SpreadPoint[];
  recessions: NberRecession[];
  ariaLabel: string;
  width?: number;
  height?: number;
  /** Shows CSV / PNG download buttons; the title heads the image and names the files. */
  exportTitle?: string;
}

const M = { top: 14, right: 18, bottom: 30, left: 50 };

const bps = (v: number | null) => (v === null ? null : v * 100);
const fmtBps = (v: number | null) => (v === null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(0)} bps`);
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const dateLabel = (iso: string) => `${MONTHS[Number(iso.slice(5, 7)) - 1]} ${Number(iso.slice(8, 10))}, ${iso.slice(0, 4)}`;

interface XY { t: number; v: number; }

/** Insert a point at each linearly interpolated sign change, so a fill clipped at zero meets
 *  the baseline exactly where the line crosses it rather than at the neighbouring sample. */
export function withZeroCrossings(points: XY[]): XY[] {
  const out: XY[] = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const prev = points[i - 1];
    if (prev && ((prev.v < 0 && p.v > 0) || (prev.v > 0 && p.v < 0))) {
      const f = prev.v / (prev.v - p.v);
      out.push({ t: prev.t + f * (p.t - prev.t), v: 0 });
    }
    out.push(p);
  }
  return out;
}

export function SpreadHistoryChart(props: SpreadHistoryChartProps) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = props.width ?? measured;
  const height = props.height ?? 320;
  const hintId = useId();

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
    // UTC, like the dates: a local-time scale ticks at local midnight on 1 January, which is still
    // 31 December in UTC east of Greenwich, so the year labels came out one year early there.
    const x = scaleUtc().domain([t0, t1]).range([0, iw]);
    const y = scaleLinear().domain([min(allV) ?? 0, max(allV) ?? 0]).range([ih, 0]).nice();

    const lineGen = line<XY>().x((p) => x(p.t)).y((p) => y(p.v));

    // Inversion fill: area between the 10y3m line and zero, clipped to negatives (drawn over
    // withZeroCrossings(s10) so the clip lands on the exact crossing).
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
      iw, ih, x, y, zeroY, s10, s2, lineGen, invArea, recBands, s10Fill: withZeroCrossings(s10),
      xTicks: x.ticks(8), yTicks: y.ticks(5), ts: allT,
    };
  }, [props.points, props.recessions, width, height]);

  const cross = useCrosshair(c?.ts.length ?? 0);
  const readout = useMemo(() => {
    if (!c || cross.index === null) return null;
    const p = props.points[cross.index];
    const s10 = bps(p.s10y3m);
    const s2 = bps(p.s2s10s);
    const inRecession = props.recessions.some((r) => r.start <= p.date && p.date <= r.end);
    return {
      t: c.ts[cross.index],
      title: dateLabel(p.date),
      s10,
      s2,
      inRecession,
      text: `${dateLabel(p.date)}: 10y–3m ${fmtBps(s10)}, 2s10s ${fmtBps(s2)}${inRecession ? ", NBER recession" : ""}`,
    };
  }, [c, cross.index, props.points, props.recessions]);

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
            tabIndex={0}
            className="overflow-visible touch-pan-y"
            onPointerMove={(e) => {
              const left = e.currentTarget.getBoundingClientRect().left + M.left;
              cross.show(nearestIndex(c.ts, c.x.invert(e.clientX - left).getTime()), "pointer");
            }}
            onPointerLeave={() => cross.via === "pointer" && cross.hide()}
            onKeyDown={cross.onKeyDown}
            onBlur={() => cross.via === "key" && cross.hide()}
          >
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
              <path className="inversion-fill" d={c.invArea(c.s10Fill) ?? ""} fill="var(--neg)" opacity={0.18} />
              {/* zero baseline */}
              <line x1={0} x2={c.iw} y1={c.zeroY} y2={c.zeroY} stroke="var(--panel-border-strong)" strokeDasharray="3 3" />
              {/* series */}
              <path className="series-2s10s" d={c.lineGen(c.s2) ?? ""} fill="none" stroke={SERIES[1]} strokeWidth={1.5} opacity={0.85} />
              <path className="series-10y3m" d={c.lineGen(c.s10) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2} />
              {readout && (
                <g className="crosshair" pointerEvents="none">
                  <line x1={c.x(readout.t)} x2={c.x(readout.t)} y1={0} y2={c.ih} stroke="var(--muted)" strokeDasharray="3 3" />
                  {readout.s2 !== null && (
                    <circle cx={c.x(readout.t)} cy={c.y(readout.s2)} r={3.5} fill={SERIES[1]} stroke="var(--panel)" strokeWidth={2} />
                  )}
                  {readout.s10 !== null && (
                    <circle cx={c.x(readout.t)} cy={c.y(readout.s10)} r={4} fill="var(--accent)" stroke="var(--panel)" strokeWidth={2} />
                  )}
                </g>
              )}
            </g>
          </svg>
        )}
        {c && readout && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute z-10 min-w-36 rounded-md border border-[var(--panel-border-strong)] bg-[var(--panel-2)] px-2.5 py-2 text-xs shadow-lg"
            style={{
              top: M.top + 4,
              left: M.left + c.x(readout.t),
              transform: c.x(readout.t) > c.iw / 2 ? "translateX(calc(-100% - 12px))" : "translateX(12px)",
            }}
          >
            <div className="tabnum mb-1 text-[var(--text)]">{readout.title}</div>
            <div className="flex justify-between gap-4 text-[var(--muted)]">
              10y–3m <span className="tabnum text-[var(--text)]">{fmtBps(readout.s10)}</span>
            </div>
            <div className="flex justify-between gap-4 text-[var(--muted)]">
              2s10s <span className="tabnum text-[var(--text)]">{fmtBps(readout.s2)}</span>
            </div>
            {readout.inRecession && <div className="mt-1 text-[var(--muted)]">NBER recession</div>}
          </div>
        )}
        <span id={hintId} className="sr-only">
          Focus the chart and use the left and right arrow keys to read values; Page Up and Page Down jump further.
        </span>
        <div className="sr-only" aria-live="polite">
          {readout && cross.via === "key" ? readout.text : ""}
        </div>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-[var(--muted)]">
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded bg-[var(--accent)]" /> 10y–3m</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded" style={{ background: SERIES[1] }} /> 2s10s</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[var(--neg)] opacity-30" /> inverted</span>
        <span className="inline-flex items-center gap-1.5"><span className="inline-block h-2 w-3 rounded-sm bg-[var(--muted)] opacity-30" /> NBER recession</span>
        {props.exportTitle && (
          <ExportButtons
            className="ml-auto"
            title={props.exportTitle}
            csv={() => ({
              columns: ["Date", "10y–3m (bps)", "2s10s (bps)"],
              rows: props.points.map((p) => [p.date, bps(p.s10y3m), bps(p.s2s10s)]),
            })}
            svg={() => ref.current?.querySelector("svg")}
            legend={[
              { label: "10y–3m", color: "var(--accent)" },
              { label: "2s10s", color: SERIES[1] },
              { label: "inverted", color: "var(--neg)", opacity: 0.3 },
              { label: "NBER recession", color: "var(--muted)", opacity: 0.3 },
            ]}
          />
        )}
      </div>
    </div>
  );
}
