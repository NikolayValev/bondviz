import Link from "next/link";
import type { ReactNode } from "react";
import { Card } from "@/components/ui/Card";
import { AUTHOR, SOURCE_URL, pageMetadata } from "@/lib/site";

export const metadata = pageMetadata({
  title: "Methodology",
  description:
    "How BondViz works: data sources, compounding conventions, and the formulas behind pricing, the curve bootstrap, carry, the recession signal and PCA.",
  path: "/about",
});

const SECTIONS = [
  { id: "conventions", title: "Conventions" },
  { id: "data", title: "Data" },
  { id: "pricing", title: "Bond pricing & risk" },
  { id: "krd", title: "Key-rate durations" },
  { id: "bootstrap", title: "Curve bootstrap" },
  { id: "shocks", title: "Curve shocks" },
  { id: "carry", title: "Carry & roll-down" },
  { id: "signal", title: "Recession signal" },
  { id: "relative", title: "Today vs history" },
  { id: "pca", title: "Yield-curve PCA" },
  { id: "portfolio", title: "Portfolio" },
  { id: "command", title: "Command panel" },
  { id: "sharing", title: "Shareable links" },
  { id: "engineering", title: "How it's built" },
  { id: "limits", title: "Simplifications" },
];

function Formula({ children }: { children: ReactNode }) {
  return (
    <div className="tabnum my-3 overflow-x-auto rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--panel-2)] px-4 py-3 text-sm text-[var(--text)]">
      {children}
    </div>
  );
}

function Section({ id, title, tool, children }: { id: string; title: string; tool?: string; children: ReactNode }) {
  return (
    <section id={id} className="scroll-mt-24">
      <Card>
        <div className="flex items-baseline justify-between gap-4">
          <h2 className="text-xl">{title}</h2>
          {tool && (
            <Link href={tool} className="shrink-0 text-sm text-[var(--accent)] hover:underline">
              Open tool →
            </Link>
          )}
        </div>
        <div className="mt-3 space-y-3 text-[var(--muted)] [&_strong]:text-[var(--text)]">{children}</div>
      </Card>
    </section>
  );
}

export default function AboutPage() {
  return (
    <div className="space-y-8">
      <header className="rise">
        <div className="eyebrow">Methodology</div>
        <h1 className="mt-2 text-3xl sm:text-4xl">About the math</h1>
        <p className="mt-3 max-w-2xl text-lg text-[var(--muted)]">
          BondViz turns the U.S. Treasury&apos;s daily par-yield curve into a set of fixed-income
          tools. This page documents the data, the conventions and every formula the app uses, so
          each number on screen can be traced back to its definition.
        </p>
        <p className="mt-3 text-sm text-[var(--muted)]">
          Built by{" "}
          <a href={AUTHOR.url} className="text-[var(--accent)] underline decoration-[var(--accent-ring)] underline-offset-4 hover:decoration-current">
            {AUTHOR.name}
          </a>{" "}
          ·{" "}
          <a href={SOURCE_URL} target="_blank" rel="noreferrer" className="text-[var(--accent)] underline decoration-[var(--accent-ring)] underline-offset-4 hover:decoration-current">
            Source on GitHub
          </a>{" "}
          ·{" "}
          <Link href="/styleguide" className="text-[var(--accent)] underline decoration-[var(--accent-ring)] underline-offset-4 hover:decoration-current">
            Design system
          </Link>
        </p>
      </header>

      <nav aria-label="On this page" className="flex flex-wrap gap-2 text-sm">
        {SECTIONS.map((s) => (
          <a
            key={s.id}
            href={`#${s.id}`}
            className="rounded-full border border-[var(--panel-border)] px-3 py-1 text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)]"
          >
            {s.title}
          </a>
        ))}
      </nav>

      <Section id="conventions" title="Conventions">
        <p>
          <strong>Pricing and risk use continuous compounding.</strong> A cashflow due in t years is
          discounted by e<sup>−y·t</sup>. That keeps duration and convexity in their cleanest form
          and matches the original Python model and its C++ extension.
        </p>
        <p>
          <strong>The curve bootstrap is the one exception.</strong> Treasury par yields are quoted as
          semiannual-coupon bond yields, so the bootstrap treats them as such and reports
          annual-compounded zero and forward rates. The two conventions are never mixed in one
          calculation.
        </p>
        <p>
          Yields are shown in percent, spreads and shocks in basis points (1 bp = 0.01%), and money in
          dollars per the face value you enter.
        </p>
      </Section>

      <Section id="data" title="Data">
        <p>
          <strong>U.S. Treasury daily par yield curve</strong> (home.treasury.gov, no API key): 1M to
          30Y, one XML feed per calendar year. Server routes fetch, parse and cache it: the current
          year refreshes hourly and past years weekly. Multi-year windows fetch years in parallel, and
          a failed year degrades to a gap rather than an error. Dates are read as calendar dates,
          independent of the server&apos;s timezone.
        </p>
        <p>
          <strong>Polygon.io</strong> supplies daily stock bars. The API key lives only in a server
          route and never reaches the browser.
        </p>
        <p>
          Tenor labels (1M … 30Y) map to maturities in years through one table shared by every page,
          so the curve, carry, PCA and the command panel all agree on what &quot;5Y&quot; means.
        </p>
      </Section>

      <Section id="pricing" title="Bond pricing & risk" tool="/pricing">
        <p>
          A bond with face F, annual coupon rate c, maturity T and payment frequency m pays c·F/m at
          times 1/m, 2/m, … T, plus F at T. Every analytic below is computed from that one cashflow
          schedule, so price and risk always agree.
        </p>
        <Formula>
          P = Σ CF<sub>i</sub> · e<sup>−y·t<sub>i</sub></sup>
        </Formula>
        <Formula>
          D<sub>Mac</sub> = Σ t<sub>i</sub> · CF<sub>i</sub> · e<sup>−y·t<sub>i</sub></sup> / P &nbsp;&nbsp;
          (= modified duration under continuous compounding)
        </Formula>
        <Formula>
          C = Σ t<sub>i</sub><sup>2</sup> · CF<sub>i</sub> · e<sup>−y·t<sub>i</sub></sup> / P &nbsp;&nbsp;·&nbsp;&nbsp;
          DV01 ≈ D · P · 0.0001
        </Formula>
        <p>
          <strong>Scenarios</strong> reprice the bond exactly at each parallel shift and compare it
          with the second-order approximation, which shows how convexity bends the line:
        </p>
        <Formula>
          ΔP / P ≈ −D · Δy + ½ · C · Δy<sup>2</sup>
        </Formula>
        <p>
          <strong>Yield from price</strong> is solved by bisection, which is safe because price falls
          monotonically as yield rises. <strong>Holding-period return</strong> collects coupons to
          the horizon (held, not reinvested, to keep the breakdown transparent) and revalues the
          remaining bond at the shifted yield.
        </p>
      </Section>

      <Section id="krd" title="Key-rate durations" tool="/pricing">
        <p>
          Duration says how much a bond moves for a <em>parallel</em> shift. Key-rate durations split
          that into curve buckets: how much it moves when only one key tenor (3M, 6M, 1Y, 2Y, 3Y, 5Y,
          7Y, 10Y, 20Y, 30Y) moves. Each key k gets a &quot;tent&quot; weight w<sub>k</sub>(t) that is 1
          at its tenor and falls linearly to 0 at the neighbouring keys:
        </p>
        <Formula>
          KRD<sub>k</sub> = Σ t<sub>i</sub> · w<sub>k</sub>(t<sub>i</sub>) · CF<sub>i</sub> · e<sup>−y·t<sub>i</sub></sup> / P
          &nbsp;&nbsp;·&nbsp;&nbsp; Σ<sub>k</sub> w<sub>k</sub>(t) = 1 ⇒ Σ<sub>k</sub> KRD<sub>k</sub> = D
        </Formula>
        <p>
          That is the exact limit of bumping one key rate by 1bp and repricing (the tests check it
          against a ±1bp reprice). Because the tents sum to one, the buckets add back up to the
          bond&apos;s duration, and their dollar DV01s to its DV01; the page shows that check under the
          chart. A portfolio&apos;s key-rate DV01s are the holdings&apos; dollars added together.
        </p>
      </Section>

      <Section id="bootstrap" title="Curve bootstrap" tool="/yield-curve">
        <p>
          Par yields are filled in onto a half-year grid (flat beyond the ends). Each node is a par
          bond with semiannual coupon c = r/2 priced at 1, which pins down its discount factor given
          all earlier ones:
        </p>
        <Formula>
          DF<sub>n</sub> = (1 − c · Σ<sub>k&lt;n</sub> DF<sub>k</sub>) / (1 + c)
        </Formula>
        <Formula>
          zero(T) = DF(T)<sup>−1/T</sup> − 1 &nbsp;&nbsp;·&nbsp;&nbsp; fwd<sub>6m</sub> = (DF<sub>n−1</sub> / DF<sub>n</sub>)<sup>2</sup> − 1
        </Formula>
        <p>
          How the gaps between quoted tenors are filled matters for the forward curve. Since
          fwd ≈ zero + T · dzero/dT, any kink in the par curve becomes a jump in the forwards, scaled
          up by maturity. Linear interpolation, which the Python reference uses, kinks at every tenor
          and puts a visible spike in the forwards at 20Y. The explorer therefore defaults to a
          monotone cubic (Fritsch–Carlson PCHIP): it passes through every quoted yield, keeps a
          continuous slope, and never overshoots the data. Linear stays one click away for comparison.
        </p>
        <p>
          The explorer also shows the latest curve against 1, 3, 6 and 12 months ago, the 2s10s and
          3m10y spreads over time, and a date × tenor heatmap of the whole window.
        </p>
      </Section>

      <Section id="shocks" title="Curve shocks" tool="/yield-curve">
        <p>
          The scenario builder applies named shapes to today&apos;s curve. Every shape is defined on
          one coordinate, u(t): maturity on a log scale, −1 at 1M, 0 at a 5Y pivot and +1 at 30Y, so the
          short tenors are spread out the way the curve is quoted. With size s in bps:
        </p>
        <Formula>
          parallel ±s &nbsp;·&nbsp; bull steepener −s·(1−u)/2 &nbsp;·&nbsp; bear steepener +s·(1+u)/2
          &nbsp;·&nbsp; bull flattener −s·(1+u)/2 &nbsp;·&nbsp; bear flattener +s·(1−u)/2 &nbsp;·&nbsp;
          twist s·u &nbsp;·&nbsp; butterfly s·(2|u| − 1)
        </Formula>
        <p>
          Benchmark 2Y, 5Y, 10Y and 30Y bonds (coupon equal to today&apos;s yield) are repriced exactly
          at their own tenor&apos;s shifted yield, next to the duration-convexity estimate.
        </p>
      </Section>

      <Section id="carry" title="Carry & roll-down" tool="/carry">
        <p>
          For a holding horizon h and a <strong>static curve</strong>, a bond bought at maturity T
          earns its yield and, as it ages to T − h, rolls down to a lower yield and gains price:
        </p>
        <Formula>
          carry = y(T) · h &nbsp;&nbsp;·&nbsp;&nbsp; roll = D(T−h) · (y(T) − y(T−h)) &nbsp;&nbsp;·&nbsp;&nbsp;
          breakeven = (carry + roll) / D(T−h)
        </Formula>
        <p>
          D(T−h) is the duration of a par bond at the rolled maturity. Breakeven is how far yields can
          rise over the horizon before the trade returns zero. Tenors shorter than the horizon, or
          that would roll below the shortest point on the curve, are excluded rather than shown with
          a meaningless zero.
        </p>
      </Section>

      <Section id="signal" title="Recession signal" tool="/signal">
        <p>
          The curve is <strong>inverted</strong> when the 10-year yield is at or below the 3-month
          yield (10y − 3m ≤ 0). Daily data breaks one inversion into many short runs: the spread dips
          below zero, recovers for a few days and dips again. Counting each run as a signal would turn
          five inversions since 1990 into more than forty, so runs less than six months apart are
          merged into one <strong>period</strong>, with its inverted days, number of runs and deepest
          point recorded.
        </p>
        <p>
          A recession <strong>follows</strong> a period if an NBER-dated recession overlaps it or
          starts within 24 months of its end. Measuring from the end rather than the start credits
          long inversions that run right up to a downturn; the <strong>lead</strong> is counted from
          the first inverted day. A period whose 24 months haven&apos;t passed yet is reported as too
          early to tell rather than as a miss, and recessions without two years of data before them
          (1990) aren&apos;t scored. The NBER peak-to-trough dates are maintained by hand, and the chart
          shades them behind every daily spread since 1990.
        </p>
      </Section>

      <Section id="relative" title="Today vs history" tool="/signal">
        <p>
          For each spread, today&apos;s value is scored against every trading day in the chosen window
          (5Y, 10Y or since 1990):
        </p>
        <Formula>
          z = (x<sub>today</sub> − mean) / σ &nbsp;&nbsp;·&nbsp;&nbsp; percentile = share of days with x ≤ x<sub>today</sub>
        </Formula>
        <p>
          σ is the population standard deviation. Beyond ±1σ the page calls the curve steeper or
          flatter than usual. Daily observations are heavily autocorrelated, so treat this as a
          description of where today sits, not a significance test.
        </p>
      </Section>

      <Section id="pca" title="Yield-curve PCA" tool="/pca">
        <p>
          Daily yields for every tenor with complete data in the window are standardized (z-scores
          with population standard deviation), so the covariance matrix is the correlation matrix. A
          Jacobi eigensolver, written from scratch with no linear-algebra dependency, returns the
          eigenvalues and eigenvectors in descending order. Each vector is sign-normalized so its
          largest component is positive, which keeps the charts stable between runs.
        </p>
        <Formula>
          explained<sub>k</sub> = λ<sub>k</sub> / Σ λ &nbsp;&nbsp;·&nbsp;&nbsp; score<sub>k</sub>(t) = z(t) · v<sub>k</sub>
        </Formula>
        <p>
          On Treasury data the first three components almost always read as the curve&apos;s
          <strong> level</strong>, <strong>slope</strong> and <strong>curvature</strong>, and together
          they explain nearly all daily variation.
        </p>
      </Section>

      <Section id="portfolio" title="Portfolio" tool="/portfolio">
        <p>
          Each holding is priced with the single-bond functions above. Portfolio duration, convexity,
          yield and maturity are <strong>market-value-weighted</strong> averages, and DV01 is summed
          because it is already in dollars. The stress test moves every holding&apos;s yield by the
          same shift and reprices the whole book exactly, alongside the duration-convexity estimate.
        </p>
      </Section>

      <Section id="command" title="Command panel" tool="/command-panel">
        <p>
          Ask for a chart in plain English and a language model (via GitHub Models) proposes a widget
          for it. The model can&apos;t fetch anything on its own: widgets can only read from a fixed
          registry of five server-side metrics (par curve, curve KPIs, zero curve, carry and spread
          history), which call the same tested functions as the rest of the app. Pinned widgets are
          saved in your browser. Because the model runs on one shared quota, the endpoint is
          rate-limited per visitor and caps the conversation length.
        </p>
      </Section>

      <Section id="sharing" title="Shareable links">
        <p>
          Every configurable view keeps its inputs in the URL: a bond on the pricing page, a whole
          portfolio, a curve shock, a lookback window. Only values that differ from the defaults are
          written, and a missing or invalid value falls back to its default, so a link can be shared,
          bookmarked or hand-edited. Press <kbd className="tabnum">⌘K</kbd> /{" "}
          <kbd className="tabnum">Ctrl-K</kbd> anywhere to jump between tools.
        </p>
      </Section>

      <Section id="engineering" title="How it's built">
        <p>
          <strong>Next.js (App Router) · React · TypeScript (strict) · Tailwind CSS.</strong> All the
          math lives in pure, unit-tested modules, and every one has a Vitest suite. External I/O is
          isolated in server routes that fail soft. Charts are hand-rolled: D3 computes scales and
          paths, and React renders the SVG. There&apos;s no charting library, and colors come from
          design tokens shared with the{" "}
          <Link href="/styleguide" className="text-[var(--accent)] underline decoration-[var(--accent-ring)] underline-offset-4 hover:decoration-current">
            styleguide
          </Link>
          .
        </p>
        <p>
          The TypeScript library is a port of an earlier Python/Streamlit version, whose pricing core
          also has an optional pybind11 C++ extension. The Python tests were carried over with the
          port, and the Python code stays in the repository as the numerical reference.
        </p>
      </Section>

      <Section id="limits" title="Simplifications">
        <ul className="list-disc space-y-1.5 pl-5">
          <li>A single flat yield per bond: pricing does not discount each cashflow off the zero curve.</li>
          <li>Carry and roll-down assume the curve doesn&apos;t move over the horizon.</li>
          <li>Holding-period returns hold coupons as cash rather than reinvesting them.</li>
          <li>The bootstrap interpolates par yields between tenors rather than fitting a Nelson–Siegel or smoothing-spline model, so it reproduces every quote exactly, noise included.</li>
          <li>No day-count conventions, settlement lags or accrued interest; times are in exact years.</li>
          <li>The inversion signal is descriptive history, not a forecast.</li>
          <li>Curve shocks are stylized shapes, not estimated from historical moves.</li>
          <li>Key-rate durations bump a flat yield with tent weights rather than a fitted zero curve.</li>
        </ul>
      </Section>
    </div>
  );
}
