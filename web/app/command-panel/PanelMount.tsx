"use client";

import {
  DataResolverProvider,
  PinnedStoreProvider,
  createHttpDataResolver,
  defaultComponentRegistry,
} from "@nikolayvalev/command-panel";
import { ChatPanel } from "./ChatPanel";
import { DashboardPanel } from "./DashboardPanel";

// Loaded only in the browser (dynamic, ssr:false — see CommandPanelClient):
// the panel is chat + localStorage-pinned widgets, nothing to server-render,
// and the package is server-externalized for the API routes, which would make
// SSR evaluate it against Node's own React copy.
//
// Composes the package's providers with our own panes instead of the stock
// <CommandPanel>: ChatPanel persists the conversation, DashboardPanel adds
// drag-reorder + resize and scroll containment. Module-scope resolver and
// registry keep both references stable across renders.
const resolver = createHttpDataResolver("/api/command-panel/data");

export default function PanelMount() {
  return (
    <DataResolverProvider resolver={resolver}>
      <PinnedStoreProvider>
        <div className="grid grid-cols-1 gap-4 lg:h-full lg:min-h-0 lg:grid-cols-[380px_minmax(0,1fr)]">
          <ChatPanel registry={defaultComponentRegistry} />
          <DashboardPanel registry={defaultComponentRegistry} />
        </div>
      </PinnedStoreProvider>
    </DataResolverProvider>
  );
}
