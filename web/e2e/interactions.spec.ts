import { test, expect, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { mockApis } from "./mocks";

// The interactions the README promises, end to end in a real browser. Keyboard-driven, so they
// run once, on the desktop project.
test.skip(({ isMobile }) => isMobile, "keyboard flows run on desktop");

test.beforeEach(async ({ page }) => {
  await mockApis(page);
});

/** The chart <svg> inside the card titled `heading`, scrolled into view and mounted. */
async function chartIn(page: Page, heading: string) {
  const card = page.locator("div.rounded-lg", { has: page.getByRole("heading", { name: heading, exact: true }) }).first();
  await card.scrollIntoViewIfNeeded();
  const svg = card.locator("svg[role=img]").first();
  await expect(svg).toBeVisible();
  return { card, svg };
}

test("⌘K / Ctrl-K jumps to a tool", async ({ page }) => {
  await page.goto("/");
  await page.keyboard.press("Control+k");
  await page.getByRole("combobox", { name: "Search pages" }).fill("carry");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/carry$/);
  await expect(page.getByRole("heading", { level: 1, name: "Carry & Roll-Down" })).toBeVisible();
});

test("a shared link restores the curve shock", async ({ page }) => {
  await page.goto("/yield-curve?shock=twist&bp=100");
  await expect(page.getByRole("radio", { name: "Twist" })).toBeChecked();
  await expect(page.getByRole("radio", { name: "100 bp" })).toBeChecked();
  await page.getByRole("radio", { name: "Butterfly" }).click();
  await expect(page).toHaveURL(/shock=butterfly/);
});

test("pricing inputs live in the URL", async ({ page }) => {
  await page.goto("/pricing?coupon=3&years=30&yield=4.5");
  await expect(page.getByRole("spinbutton", { name: /coupon/i }).first()).toHaveValue("3");
  await expect(page.getByRole("spinbutton", { name: /maturity/i }).first()).toHaveValue("30");
});

test("a chart reads its values by keyboard", async ({ page }) => {
  await page.goto("/yield-curve");
  const { card, svg } = await chartIn(page, "Curve shifts vs the past");
  await svg.focus();
  await page.keyboard.press("End");
  await expect(card.locator("[aria-live]")).toHaveText(/^30Y: .+%, 1M ago .+%/);
  await page.keyboard.press("Escape");
  await expect(card.locator(".crosshair")).toHaveCount(0);
});

test("the heatmap reads date × tenor by keyboard", async ({ page }) => {
  await page.goto("/yield-curve");
  const { card, svg } = await chartIn(page, "Yield heatmap (time × tenor)");
  await svg.focus();
  await page.keyboard.press("ArrowRight");
  await expect(card.locator("[aria-live]")).toHaveText(/^Oct 13, 2026, 1M: \d+\.\d{2}%$/);
});

test("a table downloads as CSV", async ({ page }) => {
  await page.goto("/carry");
  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download Carry and roll-down 3M data as CSV" }).click(),
  ]);
  expect(download.suggestedFilename()).toMatch(/^bondviz-carry-and-roll-down-3m-\d{4}-\d{2}-\d{2}\.csv$/);
  const text = readFileSync(await download.path(), "utf-8");
  expect(text.startsWith("﻿Tenor,Yield (%),Carry (bps),Roll (bps),Total (bps),Breakeven (bps),Return (%)\r\n")).toBe(true);
});

test("the recession signal lists inversion periods with what followed", async ({ page }) => {
  await page.goto("/signal");
  await expect(page.getByText(/has inverted in \d+ distinct periods/)).toBeVisible();
  const table = page.getByRole("table").filter({ hasText: "What followed" });
  await expect(table.getByText(/Recession from Dec 2007/)).toBeVisible();
  await expect(table.getByText(/Too early to tell/)).toBeVisible();
});

test("stocks compares tickers rebased to 100", async ({ page }) => {
  await page.goto("/stocks?tickers=AAPL,MSFT,GOOGL");
  const table = page.getByRole("table", { name: /Return over 3M/ });
  await expect(table.getByRole("row")).toHaveCount(4);
});
