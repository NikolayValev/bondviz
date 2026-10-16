import { describe, it, expect } from "vitest";
import { decodeState, encodeState, mergeQuery, num, numSet, oneOf, records, text, type Schema } from "@/lib/urlState";

interface S {
  face: number;
  mode: "price" | "yield";
  freq: 1 | 2 | 4;
  mags: number[];
}
const schema: Schema<S> = {
  face: num({ min: 0, max: 1e9 }),
  mode: oneOf(["price", "yield"] as const),
  freq: oneOf([1, 2, 4] as const),
  mags: numSet([25, 50, 100, 200]),
};
const defaults: S = { face: 1000, mode: "price", freq: 2, mags: [25, 50, 100] };

describe("encodeState / decodeState", () => {
  it("omits defaults, so the default view has a clean URL", () => {
    expect(encodeState(schema, defaults, defaults).toString()).toBe("");
  });

  it("round-trips a changed state", () => {
    const state: S = { face: 2500.5, mode: "yield", freq: 4, mags: [50, 200] };
    const qs = encodeState(schema, state, defaults);
    expect(qs.toString()).toBe("face=2500.5&mode=yield&freq=4&mags=50%2C200");
    expect(decodeState(schema, new URLSearchParams(qs.toString()), defaults)).toEqual(state);
  });

  it("falls back to defaults field by field for missing or invalid values", () => {
    const got = decodeState(schema, new URLSearchParams("face=abc&mode=sideways&freq=3&mags=50,999,50"), defaults);
    expect(got).toEqual({ face: 1000, mode: "price", freq: 2, mags: [50] });
  });

  it("clamps numbers into range", () => {
    expect(decodeState(schema, new URLSearchParams("face=-5"), defaults).face).toBe(0);
  });

  it("an empty numSet is a valid choice, distinct from the default", () => {
    const qs = encodeState(schema, { ...defaults, mags: [] }, defaults);
    expect(decodeState(schema, new URLSearchParams(qs.toString()), defaults).mags).toEqual([]);
  });
});

describe("records", () => {
  const book = records({ label: text(40), face: num({ min: 0 }), freq: oneOf([1, 2, 4] as const) });

  it("round-trips labels containing separators", () => {
    const items = [
      { label: "10Y, on-the-run ~ test", face: 50_000, freq: 2 as const },
      { label: "Zero", face: 1_000, freq: 1 as const },
    ];
    const raw = book.encode(items);
    expect(book.decode(raw)).toEqual(items);
    // Survives a trip through URLSearchParams too.
    const p = new URLSearchParams();
    p.set("b", raw);
    expect(book.decode(new URLSearchParams(p.toString()).get("b")!)).toEqual(items);
  });

  it("drops malformed items and keeps the rest", () => {
    expect(book.decode("A,100,2~B,notanumber,2~C,5~%E0%A4%A,1,2~D,300,4")).toEqual([
      { label: "A", face: 100, freq: 2 },
      { label: "D", face: 300, freq: 4 },
    ]);
  });

  it("caps the number of items read", () => {
    const capped = records({ x: num() }, 3);
    expect(capped.decode("1~2~3~4~5")).toHaveLength(3);
  });
});

describe("mergeQuery", () => {
  it("replaces this page's params and keeps unrelated ones", () => {
    expect(mergeQuery(schema, "?utm_source=cv&face=5", { ...defaults, mode: "yield" }, defaults)).toBe(
      "?utm_source=cv&mode=yield",
    );
    expect(mergeQuery(schema, "?face=5", defaults, defaults)).toBe("");
  });
});
