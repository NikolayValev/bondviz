import { describe, it, expect } from "vitest";
import { scoreItem, searchPalette, type PaletteItem } from "@/lib/paletteSearch";

const items: PaletteItem[] = [
  { id: "yc", title: "Yield Curve Explorer", hint: "Rates", keywords: "treasury par zero forward heatmap", href: "/yield-curve" },
  { id: "pricing", title: "Bond Pricing & Risk Lab", hint: "Risk", keywords: "duration convexity dv01", href: "/pricing" },
  { id: "pca", title: "Yield Curve PCA", hint: "Rates", keywords: "level slope curvature factors", href: "/pca" },
  { id: "carry", title: "Carry & Roll-Down", hint: "Rates", keywords: "breakeven horizon", href: "/carry" },
];

describe("searchPalette", () => {
  it("returns everything, in order, for an empty query", () => {
    expect(searchPalette(items, "  ").map((i) => i.id)).toEqual(["yc", "pricing", "pca", "carry"]);
  });

  it("ranks title prefix > title word > keyword", () => {
    expect(searchPalette(items, "yield").map((i) => i.id)).toEqual(["yc", "pca"]);
    expect(searchPalette(items, "pca").map((i) => i.id)).toEqual(["pca"]);
    expect(searchPalette(items, "roll")[0].id).toBe("carry"); // title word
    expect(searchPalette(items, "convexity").map((i) => i.id)).toEqual(["pricing"]); // keyword only
  });

  it("requires every word to match", () => {
    expect(searchPalette(items, "yield pca").map((i) => i.id)).toEqual(["pca"]);
    expect(searchPalette(items, "yield banana")).toEqual([]);
  });

  it("is case-insensitive and matches inside words as a last resort", () => {
    expect(searchPalette(items, "DV01").map((i) => i.id)).toEqual(["pricing"]);
    expect(scoreItem(items[0], "xplor")).toBe(2); // substring of a title word
  });
});
