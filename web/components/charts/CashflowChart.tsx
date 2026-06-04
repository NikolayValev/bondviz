"use client";
import { useMemo, useState } from "react";
import { scaleLinear } from "d3-scale";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import type { DiscountedCashflow } from "@/lib/finance";

const M = { top: 16, right: 18, bottom: 40, left: 64 };

/** Discounted cashflows over time as bars, with the Macaulay duration shown as
 *  the present-value "center of mass" — where the bond's value sits in time. */
export function CashflowChart({ data, duration }: { data: DiscountedCashflow[]; duration: number }) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = measured;
  const height = 280;
  const [hover, setHover] = useState<number | null>(null);

  const c = useMemo(() => {
    if (width === 0 || data.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const maxT = Math.max(...data.map((d) => d.t));
    const maxPv = Math.max(...data.map((d) => d.pv));
    const x = scaleLinear().domain([0, maxT]).range([0, iw]);
    const y = scaleLinear().domain([0, maxPv]).range([ih, 0]).nice();
    const bw = Math.max(2, (iw / data.length) * 0.6);

    return { iw, ih, x, y, bw, xTicks: x.ticks(6), yTicks: y.ticks(4) };
  }, [data, width]);

  const fmt = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  const hovered = c && hover != null ? data[hover] : null;

  return (
    <div ref={ref} className="w-full">
      {c && (
        <svg width={width} height={height} role="img" aria-label="Discounted cashflows over time with the duration center of mass" className="overflow-visible" onMouseLeave={() => setHover(null)}>
          <g transform={`translate(${M.left},${M.top})`}>
            {c.yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${c.y(t)})`}>
                <line x1={0} x2={c.iw} stroke="var(--grid)" />
                <text x={-10} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">{fmt(t)}</text>
              </g>
            ))}
            {c.xTicks.map((t) => (
              <g key={`x${t}`} transform={`translate(${c.x(t)},${c.ih})`}>
                <line y1={0} y2={6} stroke="var(--faint)" />
                <text y={22} textAnchor="middle" fontSize={11} fill="var(--muted)" className="tabnum">{t}y</text>
              </g>
            ))}

            {data.map((d, i) => (
              <rect
                key={i}
                x={c.x(d.t) - c.bw / 2}
                y={c.y(d.pv)}
                width={c.bw}
                height={c.ih - c.y(d.pv)}
                rx={1.5}
                fill={hover === i ? "var(--accent)" : "color-mix(in srgb, var(--accent) 55%, transparent)"}
                onMouseEnter={() => setHover(i)}
              />
            ))}

            {/* duration center-of-mass marker */}
            <g transform={`translate(${c.x(duration)},0)`}>
              <line y1={0} y2={c.ih} stroke="var(--warn)" strokeWidth={1.5} strokeDasharray="4 4" />
              <text y={-4} textAnchor="middle" fontSize={11} fill="var(--warn)" className="tabnum">D {duration.toFixed(1)}y</text>
            </g>

            <text x={c.iw / 2} y={c.ih + 36} textAnchor="middle" fontSize={12} fill="var(--muted)">Time (years)</text>
            <text transform={`translate(${-50},${c.ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--muted)">Present value</text>
          </g>
        </svg>
      )}
      <div className="mt-3 flex items-center justify-between text-xs text-[var(--muted)]">
        <span>Bars: PV of each coupon · the tall final bar carries the face value.</span>
        {hovered && (
          <span className="tabnum">
            <span className="text-[var(--text)]">{hovered.t}y</span>{" · cf "}
            <span className="text-[var(--text)]">{hovered.cf.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>{" · pv "}
            <span className="text-[var(--accent)]">{hovered.pv.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
          </span>
        )}
      </div>
    </div>
  );
}
