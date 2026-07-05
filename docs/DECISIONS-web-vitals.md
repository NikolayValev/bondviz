# Decision log — Core Web Vitals pass (2026-07-05)

User request: “fix core web vitals and separate the code properly.” Companion log:
[DECISIONS-command-panel.md](DECISIONS-command-panel.md) (iteration 4 covers the code separation).

## Measurements (before)

- `/` was **`force-dynamic`**: every visit server-rendered behind an upstream Treasury XML fetch —
  TTFB (and therefore LCP) paid the feed's latency on each request.
- All **7 chart components rendered `null` until the ResizeObserver reported a width**, then popped
  in at full height (280–320px) — guaranteed layout shift on every chart page. The data pages also
  swapped a one-line “Loading…” paragraph for a full page of cards when data arrived (more CLS).
- `body { background-attachment: fixed }` — full-page repaint on scroll, ignored on iOS.
- Built CSS: one global file, **92.5 KB raw** (~15 KB gzipped).
- Built JS: largest chunk **1,098 KB** (command-panel JSX sandbox: sucrase + @babel/parser + AI SDK).

## Decisions

| # | Decision | Why |
| --- | --- | --- |
| 1 | **Home → ISR** (`revalidate = 3600`, replacing `force-dynamic`). | The KPI snapshot is daily data on an hourly feed cache; ISR serves a CDN-static page (route table now shows `○ / Revalidate 1h`) and re-renders in the background at most hourly. Largest single LCP win available. |
| 2 | **Every chart wrapper reserves its height** (`style={{ minHeight: height }}` on the measured `<div>`) in all 7 chart components. | The chart renders nothing for the first frame(s) until width is measured; the reservation makes that flash layout-neutral (CLS → 0 for chart mount). |
| 3 | **`Skeleton` primitive** (`components/ui/Skeleton.tsx`) replaces the one-line “Loading…” text in the 5 data-fetching clients (yield-curve, carry, signal, pca, stocks), sized to approximate the real content; `role="status"` keeps it announced to screen readers. | Data arriving no longer reshapes the page (footer/content below stops jumping); perceived performance improves. Pricing/portfolio are pure client-side (no fetch) and need none. |
| 4 | **Backdrop gradients moved from `background-attachment: fixed` to a `body::before` fixed layer** (z-index −1). | Visually identical, but the browser composites a fixed layer instead of repainting the page background on every scroll; also restores the intended look on iOS, which ignores `background-attachment: fixed`. |
| 5 | **The 1.1 MB command-panel chunk is left as-is.** Verified with `next start`: it is absent from every route's initial `<script>` set (Home 622 KB / pricing 673 KB / command-panel 625 KB raw initial JS, all shared runtime) and loads only via the `dynamic(ssr:false)` import after `/command-panel` paints. | It cannot affect LCP/FCP of any page; its weight is inherent to the JSX-sandbox approach (transpiler in the browser). Slimming it would mean changing the upstream package. |
| 6 | **CSS left at 92 KB raw.** | ~15 KB gzipped is not a meaningful CWV lever; de-duplicating the design-system's embedded Tailwind import would require forking vendor CSS or scoping tricks with real cascade risk (the widget palette overrides depend on declaration order). Revisit only if CSS ever dominates a trace. |

## Verification

- `npm test` — **130/130**; `npx eslint .` — clean; `npm run build` — green, `/` now `○ (Static) Revalidate 1h`, all other pages static.
- `next start` script audit — per-route initial JS as above; 1.1 MB chunk lazy-only.
- Not measured here: field CWV (needs real browsers — check Vercel Speed Insights after deploy).
