"use client";

import dynamic from "next/dynamic";

const Panel = dynamic(() => import("./PanelMount"), {
  ssr: false,
  loading: () => (
    <p className="py-10 text-center text-sm text-[var(--muted)]">Loading command panel…</p>
  ),
});

export function CommandPanelClient() {
  return (
    <div className="rise" style={{ height: "calc(100dvh - 12rem)", minHeight: "480px" }}>
      <Panel />
    </div>
  );
}
