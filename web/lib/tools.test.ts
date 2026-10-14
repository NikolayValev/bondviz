import { describe, it, expect } from "vitest";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { TOOLS, TOOL_GROUPS, NAV_LINKS, toolMetadata } from "@/lib/tools";

const appDir = fileURLToPath(new URL("../app", import.meta.url));

describe("tool registry", () => {
  it("has unique hrefs, each backed by a real route", () => {
    const hrefs = TOOLS.map((t) => t.href);
    expect(new Set(hrefs).size).toBe(hrefs.length);
    for (const h of hrefs) expect(existsSync(`${appDir}${h}/page.tsx`), h).toBe(true);
  });

  it("puts every tool in a known group and leaves no group empty", () => {
    for (const t of TOOLS) expect(TOOL_GROUPS).toContain(t.group);
    for (const g of TOOL_GROUPS) expect(TOOLS.some((t) => t.group === g)).toBe(true);
  });

  it("links every tool from the nav", () => {
    expect(NAV_LINKS.map((l) => l.href)).toEqual(TOOLS.map((t) => t.href));
  });

  it("builds page metadata and rejects unknown routes", () => {
    expect(toolMetadata("/pricing").title).toBe("Bond Pricing & Risk Lab");
    expect(() => toolMetadata("/nope")).toThrow();
  });
});
