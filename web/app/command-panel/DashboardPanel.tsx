"use client";

import * as React from "react";
import {
  WidgetErrorBoundary,
  WidgetRenderer,
  usePinnedStore,
  type ComponentRegistry,
  type PinnedWidget,
} from "@nikolayvalev/command-panel";
import { DEFAULT_LAYOUT, resizeLayout, type WidgetLayout } from "@/lib/panelLayout";
import { loadLayouts, saveLayouts } from "./layoutStorage";
import { ghostBtn } from "./panelChrome";

// Replaces the package's DashboardGrid: 12-column grid with per-widget column
// span + height (persisted), header drag-to-reorder (↑/↓ buttons kept for
// keyboard), a corner resize handle, and its own scroll containment so tall
// widgets can't stretch the page.
export function DashboardPanel({
  registry,
  onImprove,
}: {
  registry: ComponentRegistry;
  onImprove?: (widget: PinnedWidget) => void;
}) {
  const { widgets, unpin, reorder } = usePinnedStore();
  const [layouts, setLayouts] = React.useState<Record<string, WidgetLayout>>(loadLayouts);
  const [dragIdx, setDragIdx] = React.useState<number | null>(null);
  const [overIdx, setOverIdx] = React.useState<number | null>(null);
  const gridRef = React.useRef<HTMLDivElement | null>(null);

  const onResizeStart = (e: React.PointerEvent, id: string) => {
    e.preventDefault();
    const startX = e.clientX;
    const startY = e.clientY;
    const start = layouts[id] ?? DEFAULT_LAYOUT;
    const colWidth = (gridRef.current?.clientWidth ?? 960) / 12;
    const pinnedIds = widgets.map((w) => w.id);
    const apply = (ev: PointerEvent, persist: boolean) => {
      const next = resizeLayout(start, (ev.clientX - startX) / colWidth, ev.clientY - startY);
      setLayouts((prev) => {
        const merged = { ...prev, [id]: next };
        if (persist) saveLayouts(merged, pinnedIds);
        return merged;
      });
    };
    const move = (ev: PointerEvent) => apply(ev, false);
    const up = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      apply(ev, true);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const endDrag = () => {
    setDragIdx(null);
    setOverIdx(null);
  };

  if (widgets.length === 0) {
    return (
      <div className="grid min-h-[16rem] place-items-center rounded-lg border border-dashed border-[var(--panel-border-strong)] p-8 text-center text-sm text-[var(--muted)] lg:h-full">
        No pinned widgets yet — ask the chat for a chart, then pin it here.
      </div>
    );
  }

  return (
    <div
      ref={gridRef}
      role="list"
      aria-label="Pinned widgets"
      className="cp-grid lg:min-h-0 lg:overflow-y-auto lg:pr-1"
    >
      {widgets.map((widget, i) => {
        const layout = layouts[widget.id] ?? DEFAULT_LAYOUT;
        const isDropTarget = overIdx === i && dragIdx !== null && dragIdx !== i;
        return (
          <section
            key={widget.id}
            role="listitem"
            aria-label={widget.title}
            style={{ "--cp-span": layout.span } as React.CSSProperties}
            className={`cp-item relative flex min-w-0 flex-col overflow-hidden rounded-lg border bg-[var(--panel)] transition-[border-color,opacity] ${
              isDropTarget ? "border-[var(--accent)]" : "border-[var(--panel-border)]"
            } ${dragIdx === i ? "opacity-50" : ""}`}
            onDragOver={(e) => {
              if (dragIdx === null) return;
              e.preventDefault();
              e.dataTransfer.dropEffect = "move";
              if (overIdx !== i) setOverIdx(i);
            }}
            onDrop={(e) => {
              e.preventDefault();
              if (dragIdx !== null && dragIdx !== i) reorder(dragIdx, i);
              endDrag();
            }}
          >
            <header
              draggable
              title="Drag to reorder"
              onDragStart={(e) => {
                setDragIdx(i);
                e.dataTransfer.effectAllowed = "move";
                e.dataTransfer.setData("text/plain", widget.id);
              }}
              onDragEnd={endDrag}
              className="flex cursor-grab items-center gap-2 border-b border-[var(--panel-border)] px-3 py-2 active:cursor-grabbing"
            >
              <span aria-hidden="true" className="select-none text-xs text-[var(--faint)]">
                ⠿
              </span>
              <h3 className="min-w-0 flex-1 truncate text-sm font-medium">{widget.title}</h3>
              {onImprove && (
                <button
                  type="button"
                  aria-label={`Improve ${widget.title} in chat`}
                  title="Improve in chat"
                  onClick={() => onImprove(widget)}
                  className={`${ghostBtn} hover:text-[var(--accent)]`}
                >
                  ✦
                </button>
              )}
              <button
                type="button"
                aria-label="Move up"
                disabled={i === 0}
                onClick={() => reorder(i, i - 1)}
                className={`${ghostBtn} hover:text-[var(--text)]`}
              >
                ↑
              </button>
              <button
                type="button"
                aria-label="Move down"
                disabled={i === widgets.length - 1}
                onClick={() => reorder(i, i + 1)}
                className={`${ghostBtn} hover:text-[var(--text)]`}
              >
                ↓
              </button>
              <button
                type="button"
                aria-label={`Remove ${widget.title}`}
                onClick={() => unpin(widget.id)}
                className={`${ghostBtn} hover:text-[var(--neg)]`}
              >
                ✕
              </button>
            </header>

            <div className="min-h-0 overflow-auto p-3" style={{ height: layout.h }}>
              <WidgetErrorBoundary
                resetKey={widget.jsx}
                fallback={(message) => (
                  <p className="text-sm text-[var(--neg)]">Widget failed: {message}</p>
                )}
              >
                <WidgetRenderer source={widget.jsx} registry={registry} />
              </WidgetErrorBoundary>
            </div>

            <button
              type="button"
              aria-label={`Resize ${widget.title} (drag)`}
              onPointerDown={(e) => onResizeStart(e, widget.id)}
              className="absolute bottom-0 right-0 flex h-5 w-5 cursor-nwse-resize items-end justify-end p-0.5 text-[var(--faint)] transition-colors hover:text-[var(--accent)]"
            >
              <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
                <path d="M9 1v8H1" fill="none" stroke="currentColor" strokeWidth="1.5" />
              </svg>
            </button>
          </section>
        );
      })}
    </div>
  );
}
