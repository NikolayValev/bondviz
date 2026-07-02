"use client";

import { CommandPanel, defaultComponentRegistry } from "@nikolayvalev/command-panel";

// Loaded only in the browser (dynamic, ssr:false — see CommandPanelClient):
// the panel is chat + localStorage-pinned widgets, nothing to server-render,
// and the package is server-externalized for the API routes, which would make
// SSR evaluate it against Node's own React copy. defaultComponentRegistry is
// module-scope, so the registry reference stays stable across renders.
export default function PanelMount() {
  return <CommandPanel registry={defaultComponentRegistry} />;
}
