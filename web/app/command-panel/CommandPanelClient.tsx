"use client";

import dynamic from "next/dynamic";

const Panel = dynamic(() => import("./PanelMount"), {
  ssr: false,
  loading: () => (
    <p className="py-10 text-center text-sm text-[var(--muted)]">Loading command panel…</p>
  ),
});

export function CommandPanelClient() {
  // Fixed height (viewport minus nav/footer chrome) on desktop so the chat and
  // dashboard scroll independently; natural page flow on small screens.
  return (
    <div className="rise lg:h-[calc(100dvh-11rem)] lg:min-h-[480px]">
      <Panel />
    </div>
  );
}
