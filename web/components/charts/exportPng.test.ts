import { describe, it, expect, afterEach } from "vitest";
import { resolveColor, standaloneSvg } from "@/components/charts/exportPng";

afterEach(() => {
  document.body.innerHTML = "";
  document.documentElement.style.removeProperty("--series-1");
});

function chart() {
  document.body.innerHTML = `
    <svg width="300" height="120" class="overflow-visible" role="img" aria-label="Test chart">
      <defs><linearGradient id="g"><stop offset="0" stop-color="red" /></linearGradient></defs>
      <rect width="10" height="10" fill="url(#g)" />
      <path d="M0 0L10 10" stroke="blue" stroke-width="2" fill="none" />
      <text font-size="11">4%</text>
    </svg>`;
  return document.querySelector("svg") as SVGSVGElement;
}

describe("standaloneSvg", () => {
  it("returns namespaced markup at the chart's size", () => {
    const { markup, width, height } = standaloneSvg(chart());
    expect(width).toBe(300);
    expect(height).toBe(120);
    expect(markup).toContain('xmlns="http://www.w3.org/2000/svg"');
    expect(markup).not.toContain("overflow-visible");
  });

  it("inlines class-based styles on the copy and leaves the original alone", () => {
    const svg = chart();
    const style = document.createElement("style");
    style.textContent = ".series { stroke: rgb(0, 0, 255); }";
    document.head.appendChild(style);
    svg.querySelector("path")?.removeAttribute("stroke");
    svg.querySelector("path")?.setAttribute("class", "series");
    const { markup } = standaloneSvg(svg);
    const copy = new DOMParser().parseFromString(markup, "image/svg+xml");
    expect(copy.querySelector("path")?.getAttribute("style")).toContain("stroke: rgb(0, 0, 255)");
    expect(svg.querySelector("path")?.getAttribute("style")).toBeNull();
    style.remove();
  });

  it("leaves plain presentation attributes as they are", () => {
    const { markup } = standaloneSvg(chart());
    const path = new DOMParser().parseFromString(markup, "image/svg+xml").querySelector("path");
    expect(path?.getAttribute("stroke")).toBe("blue");
    expect(path?.getAttribute("style") ?? "").not.toContain("stroke:");
  });

  it("keeps local gradient references", () => {
    const { markup } = standaloneSvg(chart());
    const rect = new DOMParser().parseFromString(markup, "image/svg+xml").querySelector("rect");
    expect(rect?.getAttribute("fill")).toBe("url(#g)");
    expect(rect?.getAttribute("style") ?? "").not.toContain("url(");
  });
});

describe("resolveColor", () => {
  it("reads a token from :root and passes other colors through", () => {
    document.documentElement.style.setProperty("--series-1", "#00d68f");
    expect(resolveColor("var(--series-1)")).toBe("#00d68f");
    expect(resolveColor("rgb(1, 2, 3)")).toBe("rgb(1, 2, 3)");
  });

  it("returns the input when the token is undefined", () => {
    expect(resolveColor("var(--nope)")).toBe("var(--nope)");
  });
});
