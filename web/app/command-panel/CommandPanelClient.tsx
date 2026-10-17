"use client";

import dynamic from "next/dynamic";

const Panel = dynamic(() => import("./PanelMount"), {
  ssr: false,
  loading: () => (
    <p className="py-10 text-center text-sm text-[var(--muted)]">Loading command panel…</p>
  ),
});

export function CommandPanelClient() {
  // Fixed height (viewport minus nav/title/footer chrome) on desktop so the chat and
  // dashboard scroll independently; natural page flow on small screens.
  return (
    <div className="space-y-4">
      <header className="flex flex-col gap-1 sm:flex-row sm:items-baseline sm:gap-4">
        <h1 className="text-2xl">Command Panel</h1>
        <p className="text-sm text-[var(--muted)]">
          Ask for a chart in plain English; pin the widgets you want to keep.
        </p>
      </header>
      <div className="rise lg:h-[calc(100dvh-14rem)] lg:min-h-[480px]">
        <Panel />
      </div>
    </div>
  );
}
