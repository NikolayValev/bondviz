import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useUrlState } from "@/components/useUrlState";
import { num, oneOf, type Schema } from "@/lib/urlState";

interface S { face: number; mode: "price" | "yield" }
const schema: Schema<S> = { face: num({ min: 0 }), mode: oneOf(["price", "yield"] as const) };
const defaults: S = { face: 1000, mode: "price" };

beforeEach(() => {
  vi.useFakeTimers();
  window.history.replaceState(null, "", "/pricing");
});
afterEach(() => vi.useRealTimers());

describe("useUrlState", () => {
  it("adopts values from the URL after mount", () => {
    window.history.replaceState(null, "", "/pricing?face=2500&ref=cv");
    const { result } = renderHook(() => useUrlState(schema, defaults));
    expect(result.current[0]).toEqual({ face: 2500, mode: "price" });
  });

  it("does not rewrite incoming params before hydrating", () => {
    window.history.replaceState(null, "", "/pricing?face=2500");
    renderHook(() => useUrlState(schema, defaults));
    act(() => vi.advanceTimersByTime(1000));
    expect(window.location.search).toBe("?face=2500");
  });

  it("writes changes back, debounced, keeping unrelated params", () => {
    window.history.replaceState(null, "", "/pricing?ref=cv");
    const spy = vi.spyOn(window.history, "replaceState");
    const { result } = renderHook(() => useUrlState(schema, defaults));
    act(() => {
      result.current[1]({ face: 1100 });
      result.current[1]({ face: 1200 });
      result.current[1]((s) => ({ face: s.face + 5, mode: "yield" }));
    });
    expect(window.location.search).toBe("?ref=cv"); // nothing yet
    act(() => vi.advanceTimersByTime(300));
    expect(window.location.search).toBe("?ref=cv&face=1205&mode=yield");
    expect(spy).toHaveBeenCalledTimes(1); // one write for the burst
  });

  it("returning to the defaults cleans the URL", () => {
    window.history.replaceState(null, "", "/pricing?face=5");
    const { result } = renderHook(() => useUrlState(schema, defaults));
    act(() => result.current[1]({ face: 1000 }));
    act(() => vi.advanceTimersByTime(300));
    expect(window.location.search).toBe("");
  });
});
