import { describe, it, expect } from "vitest";
import {
  toSpreadPoints,
  currentStatus,
  inversionEpisodes,
  NBER_RECESSIONS,
} from "@/lib/signal";

const row = (date: string, y3m: number | null, y2: number | null, y10: number | null) => ({
  date,
  BC_3MONTH: y3m,
  BC_2YEAR: y2,
  BC_10YEAR: y10,
});

describe("toSpreadPoints", () => {
  it("computes 10y3m and 2s10s spreads", () => {
    const pts = toSpreadPoints([row("2025-01-02", 5.0, 4.0, 4.5)]);
    expect(pts[0].s10y3m).toBeCloseTo(-0.5, 9); // 4.5 - 5.0
    expect(pts[0].s2s10s).toBeCloseTo(0.5, 9); // 4.5 - 4.0
  });

  it("yields null when a leg is missing", () => {
    const pts = toSpreadPoints([row("2025-01-02", null, 4.0, 4.5)]);
    expect(pts[0].s10y3m).toBeNull();
    expect(pts[0].s2s10s).toBeCloseTo(0.5, 9);
  });
});

describe("currentStatus", () => {
  it("zeroes out for empty input", () => {
    expect(currentStatus([])).toEqual({
      date: null, s10y3m: null, s2s10s: null, inverted: false, streakDays: 0,
    });
  });

  it("counts the trailing inverted streak and stops at a positive point", () => {
    const pts: { date: string; s10y3m: number | null; s2s10s: number | null }[] = [
      { date: "d1", s10y3m: 0.5, s2s10s: 0.2 },
      { date: "d2", s10y3m: 0.1, s2s10s: 0.1 },
      { date: "d3", s10y3m: -0.2, s2s10s: -0.1 },
      { date: "d4", s10y3m: -0.3, s2s10s: -0.1 },
    ];
    const s = currentStatus(pts);
    expect(s.inverted).toBe(true);
    expect(s.streakDays).toBe(2);
    expect(s.date).toBe("d4");
  });

  it("reports not inverted when the latest point is positive", () => {
    const s = currentStatus([{ date: "d1", s10y3m: 0.3, s2s10s: 0.1 }]);
    expect(s.inverted).toBe(false);
    expect(s.streakDays).toBe(0);
  });
});

describe("inversionEpisodes", () => {
  const mk = (vals: (number | null)[], start = 2020): { date: string; s10y3m: number | null; s2s10s: number | null }[] =>
    vals.map((v, i) => ({ date: `${start}-01-${String(i + 1).padStart(2, "0")}`, s10y3m: v, s2s10s: 0 }));

  it("returns [] for an all-positive series", () => {
    expect(inversionEpisodes(mk([0.1, 0.2, 0.3]))).toEqual([]);
  });

  it("detects a single inverted day as a 1-day episode", () => {
    const eps = inversionEpisodes(mk([0.1, -0.2, 0.3]));
    expect(eps).toHaveLength(1);
    expect(eps[0].days).toBe(1);
    expect(eps[0].start).toBe("2020-01-02");
    expect(eps[0].end).toBe("2020-01-02");
    expect(eps[0].maxDepthBps).toBeCloseTo(-20, 6);
  });

  it("detects a multi-day run with correct boundaries and depth", () => {
    const eps = inversionEpisodes(mk([0.1, -0.2, -0.5, -0.3, 0.2]));
    expect(eps).toHaveLength(1);
    expect(eps[0].start).toBe("2020-01-02");
    expect(eps[0].end).toBe("2020-01-04");
    expect(eps[0].days).toBe(3);
    expect(eps[0].maxDepthBps).toBeCloseTo(-50, 6);
  });

  it("closes an episode that is still inverted at the last point", () => {
    const eps = inversionEpisodes(mk([0.1, -0.2, -0.3]));
    expect(eps).toHaveLength(1);
    expect(eps[0].end).toBe("2020-01-03");
    expect(eps[0].days).toBe(2);
  });

  it("splits two runs separated by a positive (or null) point", () => {
    const eps = inversionEpisodes(mk([-0.1, 0.2, null, -0.3]));
    expect(eps).toHaveLength(2);
    expect(eps[0].start).toBe("2020-01-01");
    expect(eps[1].start).toBe("2020-01-04");
  });

  it("flags recessionFollowed when an NBER recession starts within 24 months", () => {
    // 2007-12 recession start; an inversion starting 2006-06 is ~18 months prior.
    const eps = inversionEpisodes([
      { date: "2006-06-01", s10y3m: -0.1, s2s10s: 0 },
      { date: "2006-06-02", s10y3m: -0.2, s2s10s: 0 },
    ]);
    expect(eps[0].recessionFollowed).toBe(true);
  });

  it("does not flag recessionFollowed when no recession is within 24 months", () => {
    // 2013 inversion: nearest NBER start (2020-02) is > 24 months away.
    const eps = inversionEpisodes([
      { date: "2013-01-01", s10y3m: -0.1, s2s10s: 0 },
      { date: "2013-01-02", s10y3m: -0.2, s2s10s: 0 },
    ]);
    expect(eps[0].recessionFollowed).toBe(false);
  });
});

describe("NBER_RECESSIONS", () => {
  it("lists the four recessions since 1990", () => {
    expect(NBER_RECESSIONS).toHaveLength(4);
    expect(NBER_RECESSIONS[0].start).toBe("1990-07-01");
    expect(NBER_RECESSIONS[3].end).toBe("2020-04-01");
  });
});
