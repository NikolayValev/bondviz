// The one list of tools in the app. Nav, Home, the sitemap and per-page metadata all read it,
// so adding a page is one entry here (plus the route itself).

export type ToolGroup = "Rates" | "Risk" | "Markets & AI";

export interface Tool {
  href: string;
  /** Short label for the nav pill. */
  nav: string;
  /** Full name, used for Home cards and the page <title>. */
  title: string;
  /** One sentence: Home card blurb and the page's meta description. */
  description: string;
  group: ToolGroup;
}

export const TOOLS: Tool[] = [
  {
    href: "/yield-curve",
    nav: "Yield Curve",
    title: "Yield Curve Explorer",
    description: "The live Treasury curve, shifts versus the past year, key spreads, a heatmap and a bootstrapped zero/forward curve.",
    group: "Rates",
  },
  {
    href: "/carry",
    nav: "Carry",
    title: "Carry & Roll-Down",
    description: "Horizon carry, roll-down and the breakeven sell-off cushion for every tenor on the curve.",
    group: "Rates",
  },
  {
    href: "/signal",
    nav: "Signal",
    title: "Inversion & Recession Signal",
    description: "Every daily 10y–3m and 2s10s spread since 1990 against NBER recessions, with each inversion episode.",
    group: "Rates",
  },
  {
    href: "/pca",
    nav: "PCA",
    title: "Yield Curve PCA",
    description: "Principal components of daily yields: the level, slope and curvature factors that drive the curve.",
    group: "Rates",
  },
  {
    href: "/pricing",
    nav: "Pricing",
    title: "Bond Pricing & Risk Lab",
    description: "Price, duration, convexity, DV01, rate-shock scenarios and holding-period returns, all live as you type.",
    group: "Risk",
  },
  {
    href: "/portfolio",
    nav: "Portfolio",
    title: "Portfolio Analyzer",
    description: "Build a book of bonds and see value-weighted duration, convexity and DV01, then stress it with a rate shift.",
    group: "Risk",
  },
  {
    href: "/stocks",
    nav: "Stocks",
    title: "Stocks",
    description: "Daily price history and returns for any ticker, from Polygon.io.",
    group: "Markets & AI",
  },
  {
    href: "/command-panel",
    nav: "Command",
    title: "Command Panel",
    description: "Ask for a chart in plain English and an LLM builds a live widget over the app's curve metrics.",
    group: "Markets & AI",
  },
];

export const TOOL_GROUPS: ToolGroup[] = ["Rates", "Risk", "Markets & AI"];

/** Nav menu: every tool, then the methodology page. */
export const NAV_LINKS: { href: string; label: string }[] = [
  ...TOOLS.map((t) => ({ href: t.href, label: t.nav })),
  { href: "/about", label: "About" },
];

/** Next.js `metadata` for a tool's page wrapper. The root layout adds " · BondViz". */
export function toolMetadata(href: string): { title: string; description: string } {
  const tool = TOOLS.find((t) => t.href === href);
  if (!tool) throw new Error(`No tool registered for ${href}`);
  return { title: tool.title, description: tool.description };
}
