# Decision log — command-panel integration (2026-07-02)

Integration of `@nikolayvalev/command-panel` (generative-UI chat panel) into `web/`.
Every deviation from, or interpretation of, the task spec is recorded here.

| # | Decision | Why |
| --- | --- | --- |
| 1 | **npm, not pnpm.** Installed with `npm install` even though the task said `pnpm add`. | `web/` has a `package-lock.json` and no pnpm lockfile; Vercel builds from the npm lockfile. Introducing `pnpm-lock.yaml` would create dual lockfiles and break deploy reproducibility. Same version ranges as specified. |
| 2 | **`load()` calls `web/lib/*` directly — no HTTP hop.** The spec's example fetched `http://localhost:8000/api/yield-curve`. | The registry's `load()` runs inside the Next.js server itself, so it can call the same pure functions (`fetchTreasuryYear`, `computeCurveKpis`, `bootstrapZeros`, `carryRollDown`, `toSpreadPoints`) the app's own API routes use. No localhost URL that breaks on Vercel, no double serialization, same caching path. There is no separate Python backend in production — the Python app is a frozen reference (docs/ARCHITECTURE.md). |
| 3 | **Five real metrics registered** (see `web/app/command-panel/data.ts`): latest par curve, curve KPIs, bootstrapped zero curve, carry/roll-down per tenor, 2s10s spread history (2y). | Covers every chartable dataset the app actually computes; all chart-shaped sources return `{ label, value }[]` as required. Parameterless `load()`s only — keeps the model contract simple and mirrors what the app's own pages show. |
| 4 | **Page follows repo convention: server `page.tsx` wrapper + `CommandPanelClient.tsx`** instead of the spec's single client `page.tsx`. | Every route in `web/app` is a thin server wrapper (sets `metadata.title`) rendering one `"use client"` component (docs/ARCHITECTURE.md recipe step 3–4). Identical runtime behavior; consistent codebase. |
| 5 | **Panel height is `calc(100dvh - 12rem)` inside the app shell**, not the spec's full-viewport `100dvh` `<main>`. | The root layout already wraps pages in a sticky `<Nav>` + padded `<main>`; a nested `100dvh` main would double-scroll. The panel gets the remaining viewport under the nav. |
| 6 | **Default model `openai/gpt-4o` via `GITHUB_MODELS_MODEL` env override** — exactly as specified. | — |
| 7 | **Added `/command-panel` to the Nav `LINKS`** (label "Command"). | Repo recipe step 6: every page gets a nav entry. Not in the spec, but the page would otherwise be unreachable except by URL. |
| 8 | **Styling wired in `globals.css`:** `@import '@nikolayvalev/design-system/styles/editorial.css'` placed with the other top-of-file `@import` (CSS requires imports before rules), `@source '../node_modules/@nikolayvalev/design-system/dist'` after it. | Per package README; `@source` path is relative to `web/app/globals.css`. |
| 9 | **`GITHUB_MODELS_TOKEN` handling:** read only in `web/app/api/command-panel/route.ts` (server). See the Env section below for how the value was provisioned. | Spec constraint: never client-reachable. `web/.env.local` is gitignored (verified with `git check-ignore`). |
| 10 | **Branch/finish flow:** work on `feat/command-panel`, verify (`npm test`, `eslint`, `build`, dev smoke), then merge `--no-ff` to `main` and push, delete branch. | Standing repo convention (docs/ARCHITECTURE.md “Process conventions”; recorded user preference — no PRs). |
| 11 | **`@source` also scans `@nikolayvalev/command-panel/dist`**, not just the design-system dist the README lists. | The panel's own chrome (chat pane, dashboard grid, preview cards) emits Tailwind utility classes too; without scanning its dist those classes would be purged. Superset, harmless. |
| 12 | **`serverExternalPackages: ["@nikolayvalev/command-panel"]` in `next.config.ts`.** | First build failed: the chat/data routes import the package's client entry (for `defaultComponentRegistry` / `createDataRouteHandler`), which bundles `@ai-sdk/react` → `swr`; under Turbopack's `react-server` condition swr has no default export. Externalizing defers to Node's runtime resolution (normal `import` conditions), where it does. |
| 13 | **Panel mounts client-only** (`next/dynamic`, `ssr: false`, via `PanelMount.tsx`) with a loading placeholder. | Second build failure: with the package server-externalized, prerendering `/command-panel` evaluated it against Node's own React copy → dual-React hook crash (`useMemo` of null). The panel is chat + localStorage-pinned widgets — zero SSR value — so it renders only in the browser; the page shell still prerenders statically. |
| 14 | **Carry metric uses the 3-month horizon (0.25y)**; spread history is sampled weekly (every 5th trading day + latest close). | 0.25 is the Carry page's default (`CarryClient.tsx`), so panel numbers match the app. Weekly sampling keeps the 2-year series ~100 points — plenty for a line chart. |
| 15 | **`docs/ARCHITECTURE.md` updated** (pages table, API-route map, next.config note). | It is the agent briefing surface; a new page + two new routes must be discoverable there. |

## Env

- `GITHUB_MODELS_TOKEN` — **placeholder appended to `web/.env.local`; value still needed.** No GitHub
  token exists on this machine (no `gh` CLI, no env var, no stored git credential — origin uses SSH).
  Create a fine-grained PAT with the **models: read** permission at
  <https://github.com/settings/personal-access-tokens>, paste it into `web/.env.local`, and the chat
  route works with no code change. Until then chat returns a streamed error event; everything else
  (page, widgets' data channel, pinning) is live.
- `GITHUB_MODELS_MODEL` — optional override, defaults to `openai/gpt-4o` in the chat route.
- For the Vercel deployment, add `GITHUB_MODELS_TOKEN` to the project's env vars as well.

## Verification results (2026-07-02)

- `npm test` — **109/109 pass** (19 files); `npx eslint .` — clean; `npm run build` — green
  (18 routes; `/command-panel` static shell, both `/api/command-panel*` routes dynamic).
- Dev smoke (`npm run dev`):
  - `GET /command-panel` → 200, client-only shell with loading placeholder.
  - `POST /api/command-panel/data` → 200 with **live** data for all five ids (par curve, KPIs,
    zero curve, carry/roll-down, 2s10s history); unknown id → **404** (registry rejects it).
  - `POST /api/command-panel` (chat) → streams `start` → `error` → `[DONE]` — correct wiring,
    erroring only at the GitHub Models call because the token placeholder is empty.
  - Observed 10Y = 4.55 then 4.49 across two calls: today's Treasury row (2026-07-02) published
    mid-test; re-query confirmed both code paths read the same row. Data freshness, not a bug.
- **Not verified end-to-end:** an actual LLM widget proposal + pin (blocked on the token above).
  Once the token is set: open `/command-panel`, send “chart the treasury yield curve”, pin the
  widget, reload — the dashboard restores from `localStorage` (`command-panel:pinned`).

---

## Iteration 2 (2026-07-02) — UX fixes: persistence, layout, drag/resize, palette

User feedback after trying the deployed panel: (a) chat disappears on refresh, (b) widgets run
the full length of the page, (c) wants drag-and-drop, resizable widgets, and the page's palette.

| # | Decision | Why |
| --- | --- | --- |
| 16 | **Diagnosis.** Stock `ChatPane` holds messages only in `useChat` memory — any refresh wipes them. Stock `DashboardGrid` uses `auto-fill minmax(280px,1fr)` with **no scroll containment** — one widget fills the whole row and tall widgets stretch the page. `editorial.css` is a **light** theme (white `--vde-color-surface` etc. on our dark page). | Read from the package dist (`dist/index.js`) directly. |
| 17 | **Recomposed the panel from the package's exported parts** (`DataResolverProvider`, `PinnedStoreProvider`, `WidgetRenderer`, `WidgetErrorBoundary`, `WidgetPreviewCard`, `usePinnedStore`) instead of the stock `<CommandPanel>`; endpoints unchanged. | The stock component can't be configured for any of the three requests; the package explicitly exports the pieces for custom composition. |
| 18 | **Chat persists** in `localStorage` (`command-panel:chat:v1`): seeded via `useChat({ messages })`, saved when a turn settles (`status` = ready/error), capped at 40 messages, plus a Clear button. Added `ai` + `@ai-sdk/react` as direct deps (were transitive). | Fixes bug (a) without a server database; matches how pinned widgets already persist. Cap avoids storage-quota creep. |
| 19 | **Custom dashboard grid**: 12 columns (single column < 640px), each widget `{ span: 3–12, h: 200–800px }` persisted in `command-panel:layout:v1` (pruned to pinned ids on save). Header = HTML5 drag handle reordering via the package store's `reorder`; ↑/↓ buttons kept for keyboard access; bottom-right pointer-drag handle resizes, snapping spans to columns. Pure math in `web/lib/panelLayout.ts` (+7 Vitest tests). | Fixes bug (b) (spans + own `overflow-y` container) and delivers drag + resize with zero new dependencies, matching the repo's hand-rolled-D3 ethos and the pure-&-tested `lib/` rule. |
| 20 | **Palette**: `globals.css` now overrides all three token families the widgets read — `--vde-color-*`, the shadcn-style aliases (`--card`, `--muted-foreground`, …), and `--chart-1..5 → --series-1..5`. **`--muted` and `--accent` are deliberately not remapped** — BondViz already owns those names app-wide (text gray / brand green); the design-system's meanings are covered via the `*-foreground` and vde equivalents. | Fixes (c) “take the palette of the page”: dark panels, IBM Plex, brand green accent, BondViz chart series colors — theme swaps still flow through automatically. |
| 21 | **Two-pane layout**: desktop = fixed-height `380px / 1fr` grid columns (viewport minus chrome), each pane scrolling independently; mobile stacks with the chat at 65dvh. | Chat stays visible regardless of dashboard height — the other half of bug (a)'s “not displayed”. |
| 22 | **Resize is pointer-only** for now (reorder and remove are keyboard-accessible). | Known gap, noted; keyboard resize would need span/height arrow-key handling — add if asked. |

### Verification (iteration 2)

- `npm test` — **116/116 pass** (7 new for `panelLayout`); `npx eslint .` — clean; `npm run build` — green.
- Dev smoke: `/command-panel` → 200; compiled dev CSS contains `.cp-grid`, the
  `--vde-color-background: var(--bg)` override, and `--chart-1: var(--series-1)`; data route
  returns live KPIs.
- **Still not verified locally:** a live LLM turn — `GITHUB_MODELS_TOKEN` remains empty in
  `web/.env.local` (the deployed Vercel site has it, which is presumably where the feedback came
  from). Drag/resize/persistence need a real browser session: reload after chatting (history
  restores), pin two widgets, drag one by its header, resize via the corner handle, reload again.
