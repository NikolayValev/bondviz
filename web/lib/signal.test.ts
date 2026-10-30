import { describe, it, expect } from "vitest";
import {
  toSpreadPoints,
  currentStatus,
  inversionEpisodes,
  inversionPeriods,
  signalRecord,
  describeRecord,
  NBER_RECESSIONS,
  type InversionEpisode,
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
});

describe("NBER_RECESSIONS", () => {
  it("lists the four recessions since 1990", () => {
    expect(NBER_RECESSIONS).toHaveLength(4);
    expect(NBER_RECESSIONS[0].start).toBe("1990-07-01");
    expect(NBER_RECESSIONS[3].end).toBe("2020-04-01");
  });
});

const run = (start: string, end: string, days: number, maxDepthBps: number): InversionEpisode => ({
  start,
  end,
  days,
  maxDepthBps,
});

describe("inversionPeriods", () => {
  // The shape of the real 1990–2026 record: a 1998 blip, then inversions ahead of 2001, 2008 and
  // 2020, each broken into several runs, and the long 2022 inversion.
  const runs = [
    run("1998-09-10", "1998-09-11", 2, -2),
    run("1998-10-05", "1998-10-05", 1, -7),
    run("2000-07-03", "2000-07-05", 2, 0),
    run("2000-07-18", "2001-01-19", 128, -95),
    run("2006-01-17", "2006-03-20", 30, -7),
    run("2006-07-17", "2007-05-29", 219, -64), // 119 days after the last run: same period
    run("2019-05-13", "2019-10-10", 100, -52),
    run("2020-01-30", "2020-03-02", 20, -20),
    run("2022-10-18", "2024-12-12", 534, -189),
    run("2025-02-25", "2025-03-21", 19, -19),
  ];
  const periods = inversionPeriods(runs, "2026-10-20");

  it("merges runs under six months apart into one period", () => {
    expect(periods.map((p) => [p.start, p.end, p.runs])).toEqual([
      ["1998-09-10", "1998-10-05", 2],
      ["2000-07-03", "2001-01-19", 2],
      ["2006-01-17", "2007-05-29", 2],
      ["2019-05-13", "2020-03-02", 2],
      ["2022-10-18", "2025-03-21", 2],
    ]);
  });

  it("totals inverted days and keeps the deepest point", () => {
    expect(periods[1].invertedDays).toBe(130);
    expect(periods[4].maxDepthBps).toBe(-189);
  });

  it("names the recession that followed and its lead from the first inverted day", () => {
    expect(periods[1].outcome).toEqual({ kind: "recession", recessionStart: "2001-03-01", leadMonths: 8 });
    expect(periods[2].outcome).toEqual({ kind: "recession", recessionStart: "2007-12-01", leadMonths: 23 });
    expect(periods[3].outcome).toEqual({ kind: "recession", recessionStart: "2020-02-01", leadMonths: 9 });
  });

  it("calls a period with no recession inside the window a miss", () => {
    // Ended 1998-10; the 2001 recession began 29 months later.
    expect(periods[0].outcome).toEqual({ kind: "none" });
  });

  it("leaves a period pending while its window is open", () => {
    expect(periods[4].outcome).toEqual({ kind: "pending", windowEnds: "2027-03-21" });
  });

  it("doesn't credit a recession already under way when the curve inverted", () => {
    // Inverts in March 2008, inside the Dec 2007 – Jun 2009 recession; nothing starts within
    // 24 months after it ends.
    const [p] = inversionPeriods([run("2008-03-03", "2008-03-04", 2, -5)], "2026-10-20");
    expect(p.outcome).toEqual({ kind: "none" });
  });

  it("measures the window from the end, so a long inversion still counts", () => {
    // Starts 30 months before the Dec 2007 recession but runs to 6 months before it.
    const [p] = inversionPeriods([run("2005-06-01", "2007-06-01", 300, -40)], "2026-10-20");
    expect(p.outcome).toEqual({ kind: "recession", recessionStart: "2007-12-01", leadMonths: 30 });
  });

  it("keeps runs apart when the gap is long enough", () => {
    expect(inversionPeriods(runs.slice(0, 2), "2026-10-20", 10)).toHaveLength(2);
  });

  it("handles no inversions", () => {
    expect(inversionPeriods([], "2026-10-20")).toEqual([]);
  });

  it("scores the record, leaving out recessions without two years of data before them", () => {
    expect(signalRecord(periods, "1990-01-02")).toEqual({
      periods: 5,
      testableFrom: "1992-01-02",
      recessionsTestable: 3, // 1990-07 starts six months into the data
      recessionsPreceded: 3,
      leadMonthsMin: 8,
      leadMonthsMax: 23,
      falseAlarms: 1,
      pending: 1,
    });
  });
});

describe("describeRecord", () => {
  const base = {
    periods: 5,
    testableFrom: "1992-01-02",
    recessionsTestable: 3,
    recessionsPreceded: 3,
    leadMonthsMin: 8,
    leadMonthsMax: 23,
    falseAlarms: 1,
    pending: 1,
  };

  it("states the record in plain words", () => {
    expect(describeRecord(base, "1990-01-02")).toBe(
      "Since 1990, 10y–3m has inverted in 5 distinct periods (runs under six months apart count as one). " +
        "Each of the 3 recessions since 1992 came after one, 8–23 months after it began. " +
        "1 period had no recession within 24 months, and 1 is still inside that window.",
    );
  });

  it("handles a partial record and nothing left over", () => {
    expect(
      describeRecord({ ...base, periods: 1, recessionsPreceded: 2, leadMonthsMin: 12, leadMonthsMax: 12, falseAlarms: 0, pending: 0 }, "1990-01-02"),
    ).toBe(
      "Since 1990, 10y–3m has inverted in 1 distinct period (runs under six months apart count as one). " +
        "2 of the 3 recessions since 1992 came after one, 12 months after it began.",
    );
  });
});

