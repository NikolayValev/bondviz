import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { LazyMount } from "@/components/LazyMount";

afterEach(() => vi.unstubAllGlobals());

describe("LazyMount", () => {
  it("holds a placeholder until the element nears the viewport", () => {
    let fire: (hit: boolean) => void = () => {};
    vi.stubGlobal(
      "IntersectionObserver",
      class {
        constructor(cb: (e: { isIntersecting: boolean }[]) => void) {
          fire = (hit) => cb([{ isIntersecting: hit }]);
        }
        observe() {}
        disconnect() {}
      },
    );
    const { container } = render(<LazyMount minHeight={300}><p>chart</p></LazyMount>);
    expect(screen.queryByText("chart")).toBeNull();
    expect((container.firstChild as HTMLElement).style.minHeight).toBe("300px");
    act(() => fire(false));
    expect(screen.queryByText("chart")).toBeNull();
    act(() => fire(true));
    expect(screen.getByText("chart")).toBeInTheDocument();
  });

  it("falls back to rendering when IntersectionObserver is missing", async () => {
    vi.stubGlobal("IntersectionObserver", undefined);
    render(<LazyMount minHeight={100}><p>chart</p></LazyMount>);
    expect(await screen.findByText("chart")).toBeInTheDocument();
  });
});
