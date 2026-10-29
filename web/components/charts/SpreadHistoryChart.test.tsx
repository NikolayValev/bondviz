import { describe, it, expect } from "vitest";
import { render, fireEvent } from "@testing-library/react";
import { SpreadHistoryChart, withZeroCrossings } from "@/components/charts/SpreadHistoryChart";

const points = [
  { date: "2019-01-01", s10y3m: 0.3, s2s10s: 0.2 },
  { date: "2019-06-01", s10y3m: -0.1, s2s10s: 0.05 },
  { date: "2020-01-01", s10y3m: -0.2, s2s10s: -0.1 },
  { date: "2020-06-01", s10y3m: 0.5, s2s10s: 0.4 },
];

const recessions = [{ start: "2020-02-01", end: "2020-04-01" }];

describe("SpreadHistoryChart", () => {
  it("renders both series, a recession band, and the inversion fill", () => {
    const { container, getByRole } = render(
      <SpreadHistoryChart width={800} ariaLabel="spread history" points={points} recessions={recessions} />,
    );
    expect(getByRole("img", { name: "spread history" })).toBeTruthy();
    expect(container.querySelector("path.series-10y3m")).toBeTruthy();
    expect(container.querySelector("path.series-2s10s")).toBeTruthy();
    expect(container.querySelectorAll("rect.recession").length).toBe(1);
    expect(container.querySelector("path.inversion-fill")).toBeTruthy();
  });

  it("renders nothing fatal for empty points", () => {
    const { container } = render(
      <SpreadHistoryChart width={800} ariaLabel="empty" points={[]} recessions={recessions} />,
    );
    expect(container.querySelector("svg")).toBeNull();
  });
});

describe("withZeroCrossings", () => {
  it("inserts an interpolated zero at each sign change", () => {
    expect(withZeroCrossings([{ t: 0, v: 1 }, { t: 10, v: -3 }, { t: 20, v: -1 }, { t: 30, v: 1 }])).toEqual([
      { t: 0, v: 1 },
      { t: 2.5, v: 0 },
      { t: 10, v: -3 },
      { t: 20, v: -1 },
      { t: 25, v: 0 },
      { t: 30, v: 1 },
    ]);
  });
  it("leaves runs without a strict sign change alone", () => {
    const pts = [{ t: 0, v: 1 }, { t: 1, v: 0 }, { t: 2, v: -1 }];
    expect(withZeroCrossings(pts)).toEqual(pts);
    expect(withZeroCrossings([])).toEqual([]);
  });
});

describe("SpreadHistoryChart readout", () => {
  it("reads each day's spreads by keyboard, flagging recessions", () => {
    const { container } = render(
      <SpreadHistoryChart width={800} ariaLabel="spread history" points={points} recessions={recessions} />,
    );
    const svg = container.querySelector("svg") as SVGSVGElement;
    const live = container.querySelector("[aria-live]") as HTMLElement;
    fireEvent.keyDown(svg, { key: "Home" });
    expect(live.textContent).toBe("Jan 1, 2019: 10y–3m +30 bps, 2s10s +20 bps");
    fireEvent.keyDown(svg, { key: "ArrowRight" });
    fireEvent.keyDown(svg, { key: "ArrowRight" });
    expect(live.textContent).toBe("Jan 1, 2020: 10y–3m −20 bps, 2s10s −10 bps");
  });

  it("labels years correctly east of UTC", () => {
    const tz = process.env.TZ;
    process.env.TZ = "Europe/Sofia"; // UTC+2: local midnight on 1 January is 31 December in UTC
    try {
      const long = Array.from({ length: 26 }, (_, i) => ({ date: `${1995 + i}-06-01`, s10y3m: 1, s2s10s: 0.5 }));
      const { container } = render(
        <SpreadHistoryChart width={800} ariaLabel="long" points={long} recessions={[]} />,
      );
      const labels = Array.from(container.querySelectorAll("text")).map((t) => t.textContent);
      expect(labels).toContain("2000");
      expect(labels).not.toContain("1994");
    } finally {
      process.env.TZ = tz;
    }
  });
});

