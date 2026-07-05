"use client";

import * as React from "react";
import {
  DataResolverProvider,
  PinnedStoreProvider,
  createHttpDataResolver,
  defaultComponentRegistry,
  type PinnedWidget,
} from "@nikolayvalev/command-panel";
import { ChatPanel } from "./ChatPanel";
import { DashboardPanel } from "./DashboardPanel";
import type { ImproveRequest } from "./types";

// Loaded only in the browser (dynamic, ssr:false — see CommandPanelClient):
// the panel is chat + localStorage-pinned widgets, nothing to server-render,
// and the package is server-externalized for the API routes, which would make
// SSR evaluate it against Node's own React copy.
//
// Composes the package's providers with our own panes instead of the stock
// <CommandPanel>: ChatPanel persists multiple conversations, DashboardPanel
// adds drag-reorder + resize, and the two are linked by the "improve in chat"
// flow (a pinned widget attached to the next chat message). Module-scope
// resolver and registry keep both references stable across renders.
const resolver = createHttpDataResolver("/api/command-panel/data");

export default function PanelMount() {
  const [improve, setImprove] = React.useState<ImproveRequest | null>(null);
  const onImprove = React.useCallback(
    (widget: PinnedWidget) => setImprove({ widget, nonce: Date.now() }),
    [],
  );

  return (
    <DataResolverProvider resolver={resolver}>
      <PinnedStoreProvider>
        <div className="grid grid-cols-1 gap-4 lg:h-full lg:min-h-0 lg:grid-cols-[380px_minmax(0,1fr)]">
          <ChatPanel registry={defaultComponentRegistry} improve={improve} />
          <DashboardPanel registry={defaultComponentRegistry} onImprove={onImprove} />
        </div>
      </PinnedStoreProvider>
    </DataResolverProvider>
  );
}
