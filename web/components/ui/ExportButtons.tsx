"use client";
import { useEffect, useState } from "react";
import { exportFilename, toCsv, type CsvTable } from "@/lib/exportCsv";
import { chartPng, type LegendItem } from "@/components/charts/exportPng";

/** Save a blob under a file name via a temporary link. */
export function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const BUTTON =
  "inline-flex min-h-7 items-center gap-1 rounded-full border border-[var(--panel-border)] px-2.5 text-xs text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)] disabled:opacity-60";

function DownloadIcon() {
  return (
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 4v11" />
      <path d="m7 10 5 5 5-5" />
      <path d="M5 20h14" />
    </svg>
  );
}

/**
 * "CSV" and "PNG" download buttons for a chart or table. Each is shown only when its source is
 * given: `csv` builds the table on click; `svg` returns the chart's SVG element, rendered with
 * `title` and `legend` into the image.
 */
export function ExportButtons({
  title,
  csv,
  svg,
  legend,
  className = "",
}: {
  title: string;
  csv?: () => CsvTable;
  svg?: () => SVGSVGElement | null | undefined;
  legend?: LegendItem[];
  className?: string;
}) {
  const [png, setPng] = useState<"idle" | "busy" | "failed">("idle");

  useEffect(() => {
    if (png !== "failed") return;
    const id = window.setTimeout(() => setPng("idle"), 2500);
    return () => window.clearTimeout(id);
  }, [png]);

  const saveCsv = () => {
    if (!csv) return;
    // The BOM makes Excel read the file as UTF-8 (labels use "−", "Δ" and the like).
    download(new Blob(["﻿", toCsv(csv())], { type: "text/csv;charset=utf-8" }), exportFilename(title, "csv"));
  };

  const savePng = async () => {
    const el = svg?.();
    if (!el) return setPng("failed");
    setPng("busy");
    try {
      download(await chartPng(el, title, legend), exportFilename(title, "png"));
      setPng("idle");
    } catch {
      setPng("failed");
    }
  };

  return (
    <div className={`flex items-center gap-1.5 ${className}`}>
      {png === "failed" && (
        <span role="status" className="text-xs text-[var(--muted)]">
          Image export failed
        </span>
      )}
      {csv && (
        <button type="button" onClick={saveCsv} aria-label={`Download ${title} data as CSV`} className={BUTTON}>
          <DownloadIcon />
          CSV
        </button>
      )}
      {svg && (
        <button
          type="button"
          onClick={savePng}
          disabled={png === "busy"}
          aria-label={`Download ${title} chart as PNG`}
          className={BUTTON}
        >
          <DownloadIcon />
          PNG
        </button>
      )}
    </div>
  );
}
