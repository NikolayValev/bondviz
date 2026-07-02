import { describe, expect, it } from "vitest";
import {
  DEFAULT_LAYOUT,
  H_MAX,
  H_MIN,
  SPAN_MAX,
  SPAN_MIN,
  clampLayout,
  parseLayouts,
  pruneLayouts,
  resizeLayout,
} from "./panelLayout";

describe("clampLayout", () => {
  it("rounds and clamps span and height to bounds", () => {
    expect(clampLayout({ span: 5.6, h: 333.4 })).toEqual({ span: 6, h: 333 });
    expect(clampLayout({ span: 0, h: 0 })).toEqual({ span: SPAN_MIN, h: H_MIN });
    expect(clampLayout({ span: 99, h: 9999 })).toEqual({ span: SPAN_MAX, h: H_MAX });
  });
});

describe("resizeLayout", () => {
  it("adds fractional column and pixel deltas, then clamps", () => {
    expect(resizeLayout({ span: 6, h: 320 }, 2.4, -40)).toEqual({ span: 8, h: 280 });
    expect(resizeLayout({ span: 6, h: 320 }, -20, 0).span).toBe(SPAN_MIN);
    expect(resizeLayout({ span: 6, h: 320 }, 0, 10_000).h).toBe(H_MAX);
  });

  it("is a no-op for zero deltas on an in-bounds layout", () => {
    expect(resizeLayout(DEFAULT_LAYOUT, 0, 0)).toEqual(DEFAULT_LAYOUT);
  });
});

describe("parseLayouts", () => {
  it("round-trips a valid record", () => {
    const layouts = { a: { span: 4, h: 250 }, b: { span: 12, h: 500 } };
    expect(parseLayouts(JSON.stringify(layouts))).toEqual(layouts);
  });

  it("returns {} for null, garbage, or non-object JSON", () => {
    expect(parseLayouts(null)).toEqual({});
    expect(parseLayouts("not json")).toEqual({});
    expect(parseLayouts("[1,2]")).toEqual({});
    expect(parseLayouts("42")).toEqual({});
  });

  it("drops malformed entries and clamps out-of-range ones", () => {
    const json = JSON.stringify({
      ok: { span: 6, h: 320 },
      tooBig: { span: 50, h: 5000 },
      missing: { span: 6 },
      wrongType: { span: "6", h: 320 },
      nullEntry: null,
    });
    expect(parseLayouts(json)).toEqual({
      ok: { span: 6, h: 320 },
      tooBig: { span: SPAN_MAX, h: H_MAX },
    });
  });
});

describe("pruneLayouts", () => {
  it("keeps only layouts whose id is still pinned", () => {
    const layouts = { a: DEFAULT_LAYOUT, b: DEFAULT_LAYOUT };
    expect(pruneLayouts(layouts, ["b", "c"])).toEqual({ b: DEFAULT_LAYOUT });
    expect(pruneLayouts(layouts, [])).toEqual({});
  });
});
