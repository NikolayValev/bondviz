"use client";
import { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line, area } from "d3-shape";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import type { ScenarioRow } from "@/lib/finance";

const M = { top: 16, right: 18, bottom: 40, left: 64 };

interface Point {
  bps: number;
  actual: number;
  approx: number;
}

export function ScenarioChart({ rows, basePrice }: { rows: ScenarioRow[]; basePrice: number }) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = measured;
  const height = 300;
  const [hover, setHover] = useState<number | null>(null);

  const c = useMemo(() => {
    if (width === 0 || rows.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const pts: Point[] = rows.map((r) => ({
      bps: r.shiftBps,
      actual: r.newPrice,
      approx: basePrice * (1 + r.approxPctChange),
    }));

    const allY = pts.flatMap((p) => [p.actual, p.approx]);
    const x = scaleLinear().domain([pts[0].bps, pts[pts.length - 1].bps]).range([0, iw]);
    const y = scaleLinear().domain([Math.min(...allY), Math.max(...allY)]).range([ih, 0]).nice();

    const lineActual = line<Point>().x((p) => x(p.bps)).y((p) => y(p.actual));
    const lineApprox = line<Point>().x((p) => x(p.bps)).y((p) => y(p.approx));
    const gap = area<Point>().x((p) => x(p.bps)).y0((p) => y(p.actual)).y1((p) => y(p.approx));

    return { iw, ih, x, y, pts, lineActual, lineApprox, gap, xTicks: x.ticks(7), yTicks: y.ticks(5) };
  }, [rows, basePrice, width]);

  const fmtPrice = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  const hovered = c && hover != null ? c.pts[hover] : null;

  return (
    <div ref={ref} className="w-full">
      {c && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Bond price versus parallel yield shift: actual reprice and duration plus convexity approximation"
          className="overflow-visible"
          onMouseLeave={() => setHover(null)}
          onMouseMove={(e) => {
            const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const mx = e.clientX - rect.left - M.left;
            let nearest = 0;
            let best = Infinity;
            c.pts.forEach((p, i) => {
              const d = Math.abs(c.x(p.bps) - mx);
              if (d < best) { best = d; nearest = i; }
            });
            setHover(nearest);
          }}
        >
          <g transform={`translate(${M.left},${M.top})`}>
            {/* y grid + labels */}
            {c.yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${c.y(t)})`}>
                <line x1={0} x2={c.iw} stroke="var(--grid)" />
                <text x={-10} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">
                  {fmtPrice(t)}
                </text>
              </g>
            ))}
            {/* x ticks */}
            {c.xTicks.map((t) => (
              <g key={`x${t}`} transform={`translate(${c.x(t)},${c.ih})`}>
                <line y1={0} y2={6} stroke="var(--faint)" />
                <text y={22} textAnchor="middle" fontSize={11} fill="var(--muted)" className="tabnum">
                  {t > 0 ? `+${t}` : t}
                </text>
              </g>
            ))}
            {/* zero baseline */}
            <line x1={c.x(0)} x2={c.x(0)} y1={0} y2={c.ih} stroke="var(--panel-border-strong)" strokeDasharray="3 4" />

            {/* convexity gap */}
            <path d={c.gap(c.pts) ?? ""} fill="var(--warn)" opacity={0.12} />

            {/* approximation (dashed amber) */}
            <path d={c.lineApprox(c.pts) ?? ""} fill="none" stroke="var(--warn)" strokeWidth={1.75} strokeDasharray="5 4" />
            {/* actual reprice (accent) */}
            <path d={c.lineActual(c.pts) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2.25} />
            {c.pts.map((p) => (
              <circle key={p.bps} cx={c.x(p.bps)} cy={c.y(p.actual)} r={2.75} fill="var(--accent)" />
            ))}

            {/* hover crosshair */}
            {hovered && (
              <g>
                <line x1={c.x(hovered.bps)} x2={c.x(hovered.bps)} y1={0} y2={c.ih} stroke="var(--panel-border-strong)" />
                <circle cx={c.x(hovered.bps)} cy={c.y(hovered.actual)} r={4.5} fill="var(--accent)" stroke="var(--bg)" strokeWidth={2} />
                <circle cx={c.x(hovered.bps)} cy={c.y(hovered.approx)} r={4} fill="var(--warn)" stroke="var(--bg)" strokeWidth={2} />
              </g>
            )}

            {/* axis titles */}
            <text x={c.iw / 2} y={c.ih + 36} textAnchor="middle" fontSize={12} fill="var(--muted)">
              Parallel yield shift (bps)
            </text>
            <text transform={`translate(${-50},${c.ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--muted)">
              Price
            </text>
          </g>
        </svg>
      )}

      {/* Legend + readout */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap gap-4 text-[var(--muted)]">
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4 rounded bg-[var(--accent)]" /> Actual reprice
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-0.5 w-4 rounded bg-[var(--warn)]" style={{ borderTop: "1.75px dashed var(--warn)", background: "transparent" }} /> Duration + convexity
          </span>
        </div>
        {hovered && (
          <div className="tabnum text-[var(--muted)]">
            <span className="text-[var(--text)]">{hovered.bps > 0 ? `+${hovered.bps}` : hovered.bps} bps</span>
            {" · actual "}
            <span className="text-[var(--accent)]">{hovered.actual.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            {" · approx "}
            <span className="text-[var(--warn)]">{hovered.approx.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          </div>
        )}
      </div>
    </div>
  );
}
