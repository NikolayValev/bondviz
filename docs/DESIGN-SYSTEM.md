# BondViz — Design System: state, tokens, and a swap-in playbook

> You said you're building a design system to drop into this app. This doc is the map of what's
> here today, every place a color/spacing/font decision is made, and the concrete steps (and
> blockers) to replace it. Last reviewed: 2026-06-28.

## Current system in one paragraph

The look is a **dark "quant terminal"**: near-black layered background with radial glows and a
faint engineering grid, IBM Plex Sans/Mono, a single green accent (`#00d68f`), and a handful of
flat panels with a left accent border. There is **no component library** (no shadcn, no Radix) —
just CSS variables + Tailwind v4 utilities + four hand-written primitives. It's small and clean,
which makes it easy to restyle, but the tokens are **partial**: colors are centralized, while
chart series colors, spacing, radii, and fonts are partly hardcoded.

## Where every design decision lives

| Concern | Source of truth | Notes |
| --- | --- | --- |
| **Color tokens** | `web/app/globals.css` `:root` | `--bg`, `--panel`, `--accent`, `--pos/neg/warn`, `--text/muted/faint`, `--grid`, etc. The real palette. |
| **Fonts** | `web/app/layout.tsx` | `IBM_Plex_Sans` + `IBM_Plex_Mono` via `next/font/google`, exposed as `--font-plex-sans/mono`, aliased to `--font-sans/mono` in globals.css. |
| **Background / grid / scrollbars / focus / animations** | `web/app/globals.css` | `.gridlines::before`, `body` radial gradients, `.rise`/`.menu-drop` keyframes, `::selection`, reduced-motion. |
| **Radius** | `--radius: 14px` in globals.css **and** ad-hoc `rounded-lg`/`rounded-full` in components | Not fully tokenized. |
| **UI primitives** | `web/components/ui/` | `Card`, `Metric`, `Kpi`, `Segmented`. |
| **Chart styling** | `web/components/charts/*` | Axes/grid use `var(--*)`; **series colors are hardcoded hex** (the main gap). |
| **Tailwind** | `@import "tailwindcss"` in globals.css; `postcss.config.mjs` | Tailwind **v4**, no `tailwind.config.js`. CSS vars are **not** registered as Tailwind theme tokens. |

### The token palette (today)

```
Surfaces : --bg #070a10  --bg-2 #0a0e16  --panel #111620  --panel-2 #0d121b
Borders  : --panel-border rgba(255,255,255,.08)  --panel-border-strong rgba(255,255,255,.16)
Type     : --text #e8ecf2  --muted #8b95a7  --faint #5b6577
Brand    : --accent #00d68f  --accent-dim rgba(0,214,143,.14)
Semantic : --pos #2fd98a  --neg #ff6b81  --warn #f2b65a
Chart    : --grid rgba(255,255,255,.07)
Shape    : --radius 14px
```

### UI primitives (the whole kit)

- **`Card`** — panel with left accent border (`border-l-[3px] border-l-[var(--accent)]`).
- **`Metric`** — labelled stat with `tone` (`accent|pos|neg|neutral`) and `size` (`md|lg`). The
  workhorse for KPIs.
- **`Kpi`** — simpler label/value pair (older; `Metric` largely supersedes it).
- **`Segmented`** — pill segmented control, single- or multi-select; used for every toggle
  (frequency, horizon, lookback, shift magnitudes, active ticker).

That's the entire component inventory. Everything else is inline Tailwind utilities.

## The two things that will fight your design-system swap

### 1. Hardcoded chart series colors (highest-friction)

Axis lines, gridlines, and fills already use CSS vars, but **line/series colors are raw hex**
sprinkled across charts and clients (20 occurrences in 8 files). A token/theme change will **not**
reach them. Inventory:

| File | Hardcoded colors |
| --- | --- |
| `app/yield-curve/YieldCurveClient.tsx` | compare palette `#5b8def #f5a623 #e5484d #9b59b6`, series `#00d68f #5b8def #f5a623` |
| `app/pca/PcaClient.tsx` | `COMPONENT_COLORS = ["#00d68f","#5b8def","#f5a623"]` |
| `app/carry/CarryClient.tsx`, `app/stocks/StocksClient.tsx` | `#5b8def`, `#00d68f` |
| `components/charts/SpreadHistoryChart.tsx` | `#5b8def` (2s10s line) |
| `components/charts/Heatmap.tsx` | `STOPS` ramp `[13,27,42]→[27,94,110]→[0,214,143]` (RGB triples) |

**Fix shape (do this before restyling):** create `web/lib/chartColors.ts` exporting a named
palette (`SERIES = ["var(--series-1)", …]`, `ACCENT`, `COMPARE`, plus the heatmap ramp) sourced
from CSS vars, add `--series-1..6` (and ramp stops) to `:root`, and replace every literal. Then a
theme swap is a token edit. Tracked as **FEAT-2 / BUG-7** in the backlog.

### 2. Tailwind v4 vars aren't theme tokens

Because the CSS vars aren't registered with Tailwind's `@theme`, components must write
`bg-[var(--panel)]`, `text-[var(--muted)]`, `border-[var(--panel-border)]` everywhere (verbose,
easy to typo, invisible to tooling). Mapping them once:

```css
/* globals.css */
@import "tailwindcss";
@theme {
  --color-bg: var(--bg);
  --color-panel: var(--panel);
  --color-accent: var(--accent);
  --color-muted: var(--muted);
  /* … */
}
```

…lets agents write `bg-panel text-muted border-accent` and makes the whole surface restyle from
the token block. Tracked as **FEAT-3**.

## Playbook: drop in a new design system

Ordered so each step is independently shippable and low-risk.

1. **Land the token plumbing first** (FEAT-2 + FEAT-3 above): chart-color module + Tailwind
   `@theme` mapping, with **no visual change** (map new tokens to the current hex). This is the
   enabling refactor — after it, restyling is editing one block.
2. **Swap the palette** in `:root` (and `@theme`). The background treatment (`body`/`.gridlines`)
   is separate CSS — restyle it there.
3. **Swap fonts** in `layout.tsx` (`next/font/google` or `next/font/local`) + the `--font-*`
   aliases. Keep `tabnum`/monospace for numerics — it's load-bearing for the "terminal" feel.
4. **Restyle the four primitives** (`ui/Card|Metric|Kpi|Segmented`) — this propagates to every
   page because they're reused everywhere.
5. **Sweep the charts** for spacing/stroke widths/radii once series colors are tokenized.
6. **Verify**: `npm run build` + `npx eslint .` + a manual pass of all 8 routes (light/dark if you
   add theming). Add visual coverage if desired (see FEAT-9, Playwright/Storybook).

### If the design system is an external package

- **shadcn/ui** is the path of least resistance on this stack (Next 16 + Tailwind v4 + React 19);
  there's a `vercel:shadcn` skill available. It would replace `ui/*` and add Radix primitives.
  Decision needed: adopt shadcn vs. keep the hand-rolled kit (it's currently dependency-free,
  which is itself a selling point for a portfolio piece). Tracked as **FEAT-10**.
- Keep charts hand-rolled regardless — they are the explicit "front-end skill" signal of this
  project (see the Vercel front-end spec). A component library should style *around* them.

## Gaps worth knowing

- **No theming infrastructure** (single hardcoded dark `:root`; no light mode, no
  `prefers-color-scheme`, no theme toggle). Add only if the new system needs it (FEAT-11).
- **No spacing/typography scale tokens** — spacing is Tailwind defaults inline; type sizes are
  ad-hoc (`text-2xl`, `text-5xl`). Fine, but not centrally tunable.
- **`Kpi` vs `Metric` overlap** — consolidate to one primitive during the restyle.
- **No component catalog** — there's no Storybook/preview page to see primitives in isolation;
  agents must read source. A `/styleguide` route would speed design iteration (FEAT-9).
