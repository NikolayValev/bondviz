# BondViz: portfolio entry

Copy for a portfolio site, résumé or LinkedIn project. Every claim here can be checked in the repo
or on the live site. Last updated 2026-10-27.

## One line

A fixed-income research terminal on live U.S. Treasury data: curve analytics, bond and portfolio
risk, and a recession signal, with hand-rolled D3 charts on unit-tested bond math.

## Short (2–3 sentences)

BondViz turns the U.S. Treasury's daily par-yield feed into a research terminal. You can explore
and stress the yield curve, price bonds and portfolios down to key-rate DV01s, track the inversion
recession signal back to 1990, or ask for a chart in plain English. It's built with Next.js and
TypeScript on a pure, tested math library ported from an earlier Python/C++ version, and every
formula is written up on the site.

## Highlights (pick 3–5)

- **Real rates analytics:** par → zero → forward bootstrap (monotone cubic par interpolation, so
  forwards don't spike at tenor kinks), named curve shocks (bull/bear steepeners and flatteners,
  twist, butterfly) that reprice benchmark bonds, and key-rate durations that sum to the parallel
  duration.
- **Market signal with honest framing:** every daily 10y–3m and 2s10s spread since 1990 against
  NBER recessions, each inversion episode, and today's spread as a z-score and percentile of its
  own history.
- **Engineering:** TypeScript strict; the math and parsing live in pure `lib/` modules, each with
  a Vitest suite beside it (291 web tests). CI runs tests, lint, typecheck, build and a
  dependency-audit gate, plus the original Python reference's pytest suite.
- **No charting library:** D3 computes scales and paths, React renders the SVG, colors come from
  design tokens; the charts and data tables download as PNG or CSV.
- **Shareable and accessible:** every input lives in the URL, so any configured bond, portfolio or
  scenario is a link; zero axe-core (WCAG 2.1 AA) violations on every route at desktop and mobile
  widths.
- **AI with guardrails:** the command panel turns a plain-English request into a chart, but the
  model can only pick from a fixed registry of server-side metrics, and requests are rate-limited
  per visitor.

## Stack tags

Next.js 16 · React 19 · TypeScript · Tailwind CSS v4 · D3 · Vitest · GitHub Actions · Vercel ·
Python · pybind11 (C++) · U.S. Treasury XML · Polygon.io · GitHub Models

## Links

- Live: https://bondviz.nikolayvalev.com
- Source: https://github.com/NikolayValev/bondviz
- Methodology (every formula and convention): https://bondviz.nikolayvalev.com/about
- Design system: https://bondviz.nikolayvalev.com/styleguide

Deep links that show the depth in one click:

- [A 100 bp bear steepener and its effect on 2Y–30Y bonds](https://bondviz.nikolayvalev.com/yield-curve?shock=bear-steepener&bp=100)
- [A 30-year 3% bond priced at a 4.5% yield, with its key-rate risk](https://bondviz.nikolayvalev.com/pricing?coupon=3&years=30&yield=4.5)
- [The recession signal since 1990](https://bondviz.nikolayvalev.com/signal)

## Screenshots

In [`docs/img/`](img/): `yield-curve.png` (the lead image), `shock.png`, `krd.png`, `pricing.png`,
`signal.png`, `pca.png`, `home.png`. For three, use the curve, the shock builder and the
recession signal.

## Still to do (needs you)

- Set the GitHub repo **description** to the one-line summary and the **website** field to
  https://bondviz.nikolayvalev.com, then pin the repo on your profile.
- Run Lighthouse against production and update the README's numbers, which come from a local
  production build (performance 95–99; accessibility, best practices and SEO 100).
- Optional: a 30–45 s screen recording (curve shock → pricing key rates → signal).
