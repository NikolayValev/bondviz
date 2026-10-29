# BondViz — Portfolio-readiness plan

> Goal: make BondViz something you can put on a portfolio and have it hold up at three depths —
> the **30-second glance** (GitHub repo page, live site, link preview), the **5-minute tour** (every
> tool works, charts read correctly, the math is explained), and the **code review** (CI, tests,
> clean architecture). Written 2026-10-14. IDs continue the `FEAT-*` / `BUG-*` scheme from
> [BACKLOG.md](BACKLOG.md) and [BUGS.md](BUGS.md); new items start at **FEAT-15**.

## Progress

**2026-10-14:** Phases 1 and 2 are done on branch `ccr-7b1cc429-4khf5e` (one commit per item, CI
green). Beyond the plan, a screenshot pass at 390/800/1024/1440px found and fixed: link color
utilities silently overridden by an unlayered `a {}` rule, nav wrapping at tablet widths, the
pricing frequency control collapsing, and a Signal footnote that misstated the 24-month rule.
Still open: the **decisions** below, **Phase 3** (pick 2–3) and the parts of **Phase 4** that
need you (Lighthouse on production, the portfolio write-up, pinning the repo).

**2026-10-16:** Phase 3 done in full rather than a pick of 2–3: FEAT-8 (curve shocks), FEAT-14
(key-rate durations), FEAT-4 (shareable URLs) and FEAT-13 (spread z-scores), plus FEAT-7 (⌘K
palette). Phase 4's QA is done too: zero axe violations on every route; Lighthouse a11y / best
practices / SEO 100 and performance 95–99 (local production build). What's left needs you:
production Lighthouse, the portfolio write-up, pinning the repo, and the ETL-branch decision.

**2026-10-20:** dependency advisories cleared (Next 16.4.2; BUG-14) with an audit gate in CI; the
command panel says upfront when the AI demo is off; MIT `LICENSE`; the forward-curve spike at 20Y
fixed with monotone cubic par interpolation (BUG-15).

**2026-10-27:** FEAT-5 (CSV / PNG export on every shared chart and the key tables) and FEAT-12
(multi-ticker comparison on Stocks) done, plus clearer time-axis labels (BUG-16). The portfolio
write-up is drafted in [PORTFOLIO-ENTRY.md](PORTFOLIO-ENTRY.md). Still needs you: production
Lighthouse, repo description / website / pin, and the ETL-branch decision.

**2026-10-28:** a fresh critical pass over every page. The recession signal is scored by inversion
period with lead times (BUG-18; it was counting one-day flickers, 42 "episodes" for five
inversions). Every shared chart has a hover / touch / keyboard readout. Home leads with a live
chart and month moves. Curve charts use a √ maturity axis, and bar labels no longer collide on
phones (BUG-20). Spread-history years were a year early east of UTC (BUG-19). Zod advisory
patched (BUG-17). README screenshots refreshed.

## Where things stood (audit, 2026-10-14)

| Check | Result |
| --- | --- |
| Production | `bondviz.nikolayvalev.com` serving `main` @ `c44bfe0`. Live Treasury snapshot current (as of 2026-10-13). |
| Stocks | `POLYGON_API_KEY` is set in prod; `/api/stocks/aggregates` returns bars. |
| Web health | `npm test` **130 pass / 21 files** (docs still say 105) · `npx eslint .` clean · `npm run build` green (19 routes). |
| CI | None on `main`. The unmerged ETL branch adds a **Python-only** workflow (ruff, mypy, pytest). |
| Open bugs | BUG-1, 3, 4, 5, 6, 8, 10, 11, 12, 13 all still present in code. BUG-9 half done (CLAUDE.md yes, README no). |
| Unmerged work | `claude/bondviz-etl-postgres-savvl4`: Python ETL → Postgres (Alembic, FastAPI, scheduled ingest workflow, Dockerfile, README rewrite), 41 files / +1.9k. `feat/auth-clerk` (referenced in BACKLOG) is **not** on the remote. |
| Linear | Couldn't reach it from the planning session (network policy blocks `mcp.linear.app`), so this plan uses the repo trackers. Reconcile with Linear using the table at the bottom. |

### What a visitor sees today, and what's missing

- **GitHub repo page:** the root README describes the old Streamlit app (Python project tree, venv
  setup). There's no screenshot, no live link, and nothing on the Next.js app. This is the biggest gap.
- **Link preview (LinkedIn, portfolio site, Slack):** there's no Open Graph image or description
  beyond `"Fixed-income & markets research terminal"`. The link card shows up bare.
- **Home page:** the cards cover only 4 of the 8 tools (Portfolio, PCA, Stocks, and Command are
  missing), and the hero copy mentions only the curve and pricing.
- **Footer:** has no author name, source link, or way back to your portfolio.
- **404 / errors:** an unknown URL shows Next's default unstyled 404, which breaks the theme. There
  are no `loading.tsx` or `error.tsx` boundaries (BUG-12).
- **Methodology:** nowhere in the UI explains the conventions (continuous compounding, semiannual
  bootstrap, carry, PCA, the NBER signal). For a quant portfolio, this is where the credibility comes from (FEAT-6).
- **Command panel:** the public LLM endpoint has no rate limit and no "token missing / quota hit"
  state. A visitor can burn through the GitHub Models quota and break the demo for the next person.

---

## Decisions needed first (these change the work below)

1. **ETL/Postgres branch: merge, rework, or park?** It's a solid data-engineering story (raw →
   curated → operational layers, idempotent upserts, an audit log, scheduled ingest), but it lives
   on the "frozen" Python side, the web app doesn't read from it, and it rewrites the root README.
   *Recommendation:* merge it as a separate **"Data pipeline"** section of the portfolio story,
   after FEAT-15 (README) so the two README rewrites don't fight. Add the web job to its `ci.yml`
   (FEAT-1). Its `keepalive.yml` and the Postgres host need a running database, so confirm you're
   happy to keep one up.
2. **Legacy Streamlit code at the repo root.** `setup.py`, `pyproject.toml`, `src/`, and `app/` sit
   at the top level while the product lives in `web/`. *Recommendation:* don't move files (that
   conflicts with the ETL branch and breaks paths). Instead, reframe it in the README as **"Python
   reference implementation + pybind11 C++ extension, numerically matched to the TS lib."** That's
   a selling point, not clutter.
3. **Command panel exposure.** Keep it in the main nav as the AI showcase (with FEAT-18
   guardrails), or move it behind a "Labs" label? *Recommendation:* keep it, and add the guardrails
   plus suggested prompts.
4. **Scope of the showcase phase.** Pick **2–3** depth features rather than all 10. Suggested
   picks are in Phase 3.

---

## Phase 1 — First impression (do first, all S, ~2–3 days total)

### FEAT-15 · Portfolio-grade root README — **S** *(closes BUG-9, BUG-10)*

- **Goal:** the GitHub landing page sells the project in one screen.
- **Files:** `README.md` (root), `web/README.md`, new `docs/img/` (screenshots).
- **Content:** a one-line pitch, a **live link**, a hero screenshot or short GIF, the 8 tools in a
  table (one line each), the tech stack (Next 16 · React 19 · TS strict · Tailwind v4 · hand-rolled
  D3 · Vitest), the architecture diagram from ARCHITECTURE.md, a "the math" section pointing to
  `/about`, CI and test-count badges, and local dev commands. Legacy Python and the ETL go in a
  short section at the bottom (see Decision 2).
- **Acceptance:** a stranger can tell what it is, see it, and open the live app without scrolling
  past the first screen.

### FEAT-16 · Open Graph / social card + per-page metadata — **S**

- **Goal:** rich link previews on LinkedIn, X, Slack, and portfolio sites.
- **Files:** `web/app/layout.tsx` (`metadataBase`, `openGraph`, `twitter`, title template
  `"%s · BondViz"`), new `web/app/opengraph-image.tsx` (`next/og` `ImageResponse`: wordmark plus a
  small stylized curve, with no live fetch so it stays static), `web/app/robots.ts`, and
  `web/app/sitemap.ts`. Give each route's `metadata` a `description`.
- **Acceptance:** the card validator / `curl` shows the og:image, title, and description on `/`
  and on one tool route. Build stays green.

### FEAT-17 · Home page covers all tools + stronger hero — **S**

- **Goal:** Home works as a guided tour of the whole app, not half of it.
- **Files:** `web/app/page.tsx`.
- **Approach:** group the cards as **Rates** (Yield Curve, Carry, Signal, PCA) · **Risk** (Pricing,
  Portfolio) · **Markets & AI** (Stocks, Command). Rewrite the hero line to cover the full scope.
  Reuse `Card`; the `LINKS` array in `Nav.tsx` can become the single source for both.
- **Acceptance:** all 8 tools reachable from Home. Build, lint, and tests green.

### FEAT-6 · Methodology / "About the math" page — **S** *(existing brief)*

- Already specced in BACKLOG. Add an author blurb, a source link, and a link to `/styleguide` as
  "design system". Highest signal per hour of anything here.

### FEAT-19 · Branded footer + 404 + route boundaries — **S** *(closes BUG-12)*

- **Files:** `web/app/layout.tsx` (footer: your name, GitHub source, portfolio link, data credits),
  new `web/app/not-found.tsx`, `web/app/error.tsx`, and a shared `loading.tsx` skeleton (reuse
  `ui/Skeleton`).
- **Acceptance:** `/does-not-exist` renders in the app theme with a way home, and a thrown render
  error shows the boundary instead of a blank page.

### FEAT-1 · CI (test + lint + build) — **S** *(existing brief)*

- If the ETL branch merges, extend its `ci.yml` with a `web` job (Node 20, `npm ci`, `npm test`,
  `npx eslint .`, `npm run build`, with `working-directory: web`). Otherwise create the workflow as
  briefed. Add the badge to the README (FEAT-15).

---

## Phase 2 — Correctness & polish (things a careful reviewer would catch, ~2 days)

In priority order. Each is already briefed in [BUGS.md](BUGS.md):

1. **BUG-1** (P1): timezone-dependent Treasury date parsing. A small fix plus a `TZ=Asia/Tokyo` test.
2. **BUG-3**: `LineChart` uses a linear scale for time axes, so month ticks land oddly on the
   Yield Curve, PCA, and Stocks charts. This is the most visible chart defect.
3. **BUG-4**: the heatmap has no date axis or color legend, and its blurb describes an axis that
   isn't drawn.
4. **FEAT-18 · Command-panel guardrails — S (new):** in `web/app/api/command-panel/route.ts`,
   return a typed "demo unavailable" response when `GITHUB_MODELS_TOKEN` is missing or upstream
   returns 429, add a simple per-IP rate limit, and show 3–4 suggested prompts in the empty chat
   (e.g. "Chart 2s10s over the last year", "What's the carry on the 5Y?"). *Accept:* with the token
   unset, the page shows a friendly message instead of an error, and the rate limit is covered by a
   route test.
5. **BUG-8**: parallelize `treasury/range` (faster PCA 5-year load; a one-line pattern copy).
6. **BUG-5, BUG-6**: small React correctness cleanups.
7. **BUG-11**: fold `Kpi` into `Metric` (only `/styleguide` still uses `Kpi`).
8. **Docs refresh:** update the health baselines (130 tests) in `docs/README.md` and
   `ARCHITECTURE.md`, and fix the stale "No CI yet" line once FEAT-1 lands. BUG-13 is optional.

---

## Phase 3 — Showcase depth (pick 2–3, ~1–2 days each)

Ranked by portfolio payoff per effort:

| Pick | Item | Why it lands on a portfolio |
| --- | --- | --- |
| ★ | **FEAT-4** URL-encoded state | Lets you deep-link a configured view from your portfolio or résumé ("here's a 10Y bond under a +100bp shock"). |
| ★ | **FEAT-8** Curve shock builder | Interactive and visual. Bull/bear steepener and twist show you know rates. |
| ★ | **FEAT-14** Key-rate durations | Real desk-level risk analytics; KRDs summing to total duration is a nice testable invariant. |
| | FEAT-13 Spread z-scores | Cheap reuse of cached history; good for the Signal page. |
| | FEAT-5 CSV/PNG export | Practical, but less signal. |
| | FEAT-7 ⌘K palette | Polish; no global palette exists yet (the command panel's palette is page-local). |
| | FEAT-12 Multi-ticker overlay | Nice, but Stocks is the least differentiated page. |

**Deprioritized for portfolio purposes:** FEAT-10 (shadcn spike: the zero-dependency kit is the
better story), FEAT-11 (light mode), auth (no user value for a public demo).

---

## Phase 4 — Ship & present

- A **final visual QA pass** on desktop and mobile widths for every route (run
  `design:accessibility-review` on the key pages). Check contrast, focus states, and chart
  `aria-label`s.
- **Lighthouse** on `/` and one heavy route. The web-vitals pass already landed; confirm the
  numbers and quote them in the README.
- **Portfolio entry:** a 2–3 sentence summary, 3 screenshots (curve, pricing scenario, command
  panel), stack tags, and links to the live app and repo. Optionally a 30–45s screen recording.
- Pin the repo on your GitHub profile and set the repo description and website field to the live URL.

---

## Ticket list (sync to Linear)

| ID | Title | Size | Phase | Status |
| --- | --- | --- | --- | --- |
| — | Decide: ETL branch / legacy framing / command panel / Phase-3 picks | — | 0 | needs you |
| FEAT-15 | Portfolio-grade root README (+BUG-9, BUG-10) | S | 1 | ✅ done |
| FEAT-16 | Open Graph image + per-page metadata, robots, sitemap | S | 1 | ✅ done |
| FEAT-17 | Home covers all 8 tools + hero rewrite | S | 1 | ✅ done |
| FEAT-6 | `/about` methodology page | S | 1 | ✅ done |
| FEAT-19 | Footer, branded 404, error/loading boundaries (+BUG-12) | S | 1 | ✅ done |
| FEAT-1 | CI: web + Python jobs | S | 1 | ✅ done |
| BUG-1 | Timezone-safe Treasury date parsing | S | 2 | ✅ done |
| BUG-3 | Time scale for `LineChart` time axes | S | 2 | ✅ done |
| BUG-4 | Heatmap date axis + color legend | S | 2 | ✅ done |
| FEAT-18 | Command-panel guardrails + suggested prompts | S | 2 | ✅ done |
| BUG-8 | Parallel fetch in `treasury/range` | S | 2 | ✅ done |
| BUG-5 / BUG-6 | Portfolio `indexOf`, Stocks effect deps | S | 2 | ✅ done |
| BUG-11 | Merge `Kpi` into `Metric` (+ BUG-13 zero-crossing fill) | S | 2 | ✅ done |
| FEAT-4 | URL-encoded shareable state (4a–4d) | M | 3 | ✅ done |
| FEAT-8 | Curve shock builder (8a–8c) | M | 3 | ✅ done |
| FEAT-14 | Key-rate durations (14a–14c) | M | 3 | ✅ done |
| FEAT-13 | Spread z-scores / percentiles on /signal | M | 3 | ✅ done |
| FEAT-7 | ⌘K command palette | S/M | 3 | ✅ done |
| — | Accessibility (axe) + Lighthouse pass | S | 4 | ✅ done |
| BUG-14 / BUG-15 / BUG-16 | Dependency advisories, forward-curve spike, time-axis labels | S | — | ✅ done |
| FEAT-5 | CSV / PNG export for charts and tables | M | 3 | ✅ done |
| FEAT-12 | Multi-ticker overlay on Stocks | S/M | 3 | ✅ done |
| — | Portfolio write-up | S | 4 | ✅ drafted ([PORTFOLIO-ENTRY.md](PORTFOLIO-ENTRY.md)) |
| — | Lighthouse on prod, repo description / website, pin repo | S | 4 | needs you |

Done already (no action): FEAT-2, FEAT-3, FEAT-9, BUG-7.
