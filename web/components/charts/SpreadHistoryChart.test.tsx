import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { SpreadHistoryChart } from "@/components/charts/SpreadHistoryChart";

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
    expect(container).toBeTruthy();
  });
});
