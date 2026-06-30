import { describe, it, expect } from "vitest";
import { SERIES, ACCENT, HEATMAP_STOPS } from "@/lib/chartColors";

describe("chartColors", () => {
  it("exposes six series colors, all CSS vars", () => {
    expect(SERIES).toHaveLength(6);
    for (const c of SERIES) expect(c).toMatch(/^var\(--series-\d\)$/);
  });

  it("ACCENT is the accent CSS var", () => {
    expect(ACCENT).toBe("var(--accent)");
  });

  it("has no raw hex literals in the series palette", () => {
    for (const c of SERIES) expect(c).not.toMatch(/#[0-9a-fA-F]{6}/);
  });

  it("exposes a three-stop heatmap ramp of RGB triples", () => {
    expect(HEATMAP_STOPS).toHaveLength(3);
    for (const stop of HEATMAP_STOPS) {
      expect(stop).toHaveLength(3);
      for (const channel of stop) expect(channel).toBeGreaterThanOrEqual(0);
    }
  });
});
