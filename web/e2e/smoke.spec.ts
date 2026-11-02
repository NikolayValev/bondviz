import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { mockApis, walkPage } from "./mocks";

// Every route, at desktop and phone width (the two projects in playwright.config.ts): it renders
// its heading, logs no errors, and has no axe-core WCAG 2.1 AA or best-practice violations with
// all of its charts mounted. This is what backs the README's accessibility claim.
const ROUTES: [path: string, heading: string][] = [
  ["/", "BONDVIZ"],
  ["/yield-curve", "Yield Curve Explorer"],
  ["/carry", "Carry & Roll-Down"],
  ["/signal", "Inversion & Recession Signal"],
  ["/pca", "Yield Curve PCA"],
  ["/pricing", "Bond Pricing & Risk Lab"],
  ["/portfolio", "Portfolio Analyzer"],
  ["/stocks", "Stocks"],
  ["/command-panel", "Command Panel"],
  ["/about", "About the math"],
  ["/styleguide", "Styleguide"],
  ["/no-such-page", "This page is off the curve."],
];

for (const [path, heading] of ROUTES) {
  test(`${path} renders cleanly and passes axe`, async ({ page }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      // The 404 route's own status line is expected.
      if (m.type() === "error" && !(path === "/no-such-page" && m.text().includes("404"))) errors.push(m.text());
    });
    await mockApis(page);
    await page.goto(path);
    await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
    await page.waitForLoadState("networkidle");
    await walkPage(page);

    const { violations } = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"])
      .analyze();
    expect(violations.map((v) => `${v.id} (${v.nodes.length}): ${v.help}`)).toEqual([]);
    expect(errors).toEqual([]);
  });
}
