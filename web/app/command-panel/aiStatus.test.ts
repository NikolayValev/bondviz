import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useAiStatus, __resetAiStatus } from "@/app/command-panel/aiStatus";

beforeEach(() => __resetAiStatus());
afterEach(() => vi.unstubAllGlobals());

describe("useAiStatus", () => {
  it.each([
    [{ configured: true }, "on"],
    [{ configured: false }, "off"],
  ])("maps %j to %s", async (body, expected) => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(body))));
    const { result } = renderHook(() => useAiStatus());
    expect(result.current).toBe("unknown");
    await waitFor(() => expect(result.current).toBe(expected));
  });

  it("stays unknown when the check fails, so the chat can still try", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("offline"); }));
    const { result } = renderHook(() => useAiStatus());
    await new Promise((r) => setTimeout(r, 10));
    expect(result.current).toBe("unknown");
  });

  it("asks the server once per page load", async () => {
    const f = vi.fn(async () => new Response(JSON.stringify({ configured: true })));
    vi.stubGlobal("fetch", f);
    const a = renderHook(() => useAiStatus());
    const b = renderHook(() => useAiStatus());
    await waitFor(() => expect(a.result.current).toBe("on"));
    await waitFor(() => expect(b.result.current).toBe("on"));
    expect(f).toHaveBeenCalledTimes(1);
  });
});
