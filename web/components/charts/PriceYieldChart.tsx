"use client";
import { useId, useMemo } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { useResizeObserver } from "@/components/charts/useResizeObserver";
import { useCrosshair } from "@/components/charts/useCrosshair";
import { ChartReadout } from "@/components/charts/ChartReadout";
import { ExportButtons } from "@/components/ui/ExportButtons";
import { closestIndex } from "@/lib/crosshair";

const M = { top: 16, right: 18, bottom: 40, left: 64 };

/** Price–yield relationship (the convex "bowl") with the duration tangent at
 *  the current yield, so the user sees where the linear estimate breaks down. */
export function PriceYieldChart({
  curve,
  currentYield,
  currentPrice,
  dv01,
  exportTitle,
}: {
  curve: { yield: number; price: number }[];
  currentYield: number;
  currentPrice: number;
  dv01: number;
  /** Shows CSV / PNG download buttons; the title heads the image and names the files. */
  exportTitle?: string;
}) {
  const { ref, width: measured } = useResizeObserver<HTMLDivElement>();
  const width = measured;
  const height = 300;
  const hintId = useId();
  const cross = useCrosshair(curve);
  const hover = cross.index;

  const c = useMemo(() => {
    if (width === 0 || curve.length === 0) return null;
    const iw = Math.max(1, width - M.left - M.right);
    const ih = Math.max(1, height - M.top - M.bottom);

    const xs = curve.map((p) => p.yield);
    const ys = curve.map((p) => p.price);
    const x = scaleLinear().domain([Math.min(...xs), Math.max(...xs)]).range([0, iw]);
    const y = scaleLinear().domain([Math.min(...ys), Math.max(...ys)]).range([ih, 0]).nice();

    const gen = line<{ yield: number; price: number }>().x((p) => x(p.yield)).y((p) => y(p.price));

    // Duration tangent: slope dP/dy = −DV01·1e4 (per unit yield). Draw across domain.
    const slope = -dv01 * 1e4;
    const tangent = [Math.min(...xs), Math.max(...xs)].map((yy) => ({
      yield: yy,
      price: currentPrice + slope * (yy - currentYield),
    }));

    return { iw, ih, x, y, gen, xTicks: x.ticks(6), yTicks: y.ticks(5), tangent, slope, px: xs.map((v) => x(v)) };
  }, [curve, currentYield, currentPrice, dv01, width]);

  const fmtPrice = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  const hovered = c && hover != null ? curve[hover] : null;

  return (
    <div ref={ref} className="w-full" style={{ minHeight: height }}>
      {c && (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label="Bond price as a function of yield, with the duration tangent line"
          aria-describedby={hintId}
          className="overflow-visible touch-pan-y"
          {...cross.svgProps(M.left, (px) => closestIndex(c.px, px))}
        >
          <g transform={`translate(${M.left},${M.top})`}>
            {c.yTicks.map((t) => (
              <g key={`y${t}`} transform={`translate(0,${c.y(t)})`}>
                <line x1={0} x2={c.iw} stroke="var(--grid)" />
                <text x={-10} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--muted)" className="tabnum">{fmtPrice(t)}</text>
              </g>
            ))}
            {c.xTicks.map((t) => (
              <g key={`x${t}`} transform={`translate(${c.x(t)},${c.ih})`}>
                <line y1={0} y2={6} stroke="var(--faint)" />
                <text y={22} textAnchor="middle" fontSize={11} fill="var(--muted)" className="tabnum">{(t * 100).toFixed(1)}%</text>
              </g>
            ))}

            {/* duration tangent (dashed) */}
            <path d={c.gen(c.tangent) ?? ""} fill="none" stroke="var(--warn)" strokeWidth={1.5} strokeDasharray="5 4" />
            {/* price–yield curve */}
            <path d={c.gen(curve) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2.25} />

            {/* current point */}
            <line x1={c.x(currentYield)} x2={c.x(currentYield)} y1={c.y(currentPrice)} y2={c.ih} stroke="var(--panel-border-strong)" strokeDasharray="3 4" />
            <circle cx={c.x(currentYield)} cy={c.y(currentPrice)} r={5} fill="var(--accent)" stroke="var(--bg)" strokeWidth={2} />

            {hovered && (
              <circle cx={c.x(hovered.yield)} cy={c.y(hovered.price)} r={4} fill="var(--text)" stroke="var(--bg)" strokeWidth={2} />
            )}

            <text x={c.iw / 2} y={c.ih + 36} textAnchor="middle" fontSize={12} fill="var(--muted)">Yield</text>
            <text transform={`translate(${-50},${c.ih / 2}) rotate(-90)`} textAnchor="middle" fontSize={12} fill="var(--muted)">Price</text>
          </g>
        </svg>
      )}
      <ChartReadout
        hintId={hintId}
        box={null}
        announce={
          hovered && cross.via === "key"
            ? `Yield ${(hovered.yield * 100).toFixed(2)}%: price ${hovered.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
            : null
        }
      />
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap gap-4 text-[var(--muted)]">
          <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded bg-[var(--accent)]" /> Price–yield curve</span>
          <span className="inline-flex items-center gap-1.5"><span className="inline-block h-0.5 w-4 rounded" style={{ borderTop: "1.5px dashed var(--warn)" }} /> Duration (linear estimate)</span>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {hovered && (
            <div className="tabnum text-[var(--muted)]">
              <span className="text-[var(--text)]">{(hovered.yield * 100).toFixed(2)}%</span>
              {" → "}
              <span className="text-[var(--accent)]">{hovered.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            </div>
          )}
          {exportTitle && c && (
            <ExportButtons
              title={exportTitle}
              csv={() => ({
                columns: ["Yield (%)", "Price", "Duration estimate"],
                rows: curve.map((p) => [p.yield * 100, p.price, currentPrice + c.slope * (p.yield - currentYield)]),
              })}
              svg={() => ref.current?.querySelector("svg")}
              legend={[
                { label: "Price–yield curve", color: "var(--accent)" },
                { label: "Duration (linear estimate)", color: "var(--warn)" },
              ]}
            />
          )}
        </div>
      </div>
    </div>
  );
}
