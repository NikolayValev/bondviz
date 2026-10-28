export interface SpreadPoint {
  date: string;
  s10y3m: number | null; // 10Y − 3M, percentage points
  s2s10s: number | null; // 10Y − 2Y, percentage points
}

export interface SignalStatus {
  date: string | null;
  s10y3m: number | null;
  s2s10s: number | null;
  inverted: boolean;
  streakDays: number;
}

export interface InversionEpisode {
  start: string;
  end: string;
  days: number;
  maxDepthBps: number; // most negative s10y3m over the run, in bps (<= 0)
  recessionFollowed: boolean;
}

export interface NberRecession {
  start: string;
  end: string;
}

// Source: NBER US business-cycle peaks→troughs (nber.org/research/business-cycle-dating).
// Last updated 2026-06; extend when a new recession is dated.
export const NBER_RECESSIONS: NberRecession[] = [
  { start: "1990-07-01", end: "1991-03-01" },
  { start: "2001-03-01", end: "2001-11-01" },
  { start: "2007-12-01", end: "2009-06-01" },
  { start: "2020-02-01", end: "2020-04-01" },
];

function num(v: unknown): number | null {
  return typeof v === "number" && !Number.isNaN(v) ? v : null;
}

function spread(a: number | null, b: number | null): number | null {
  return a !== null && b !== null ? a - b : null;
}

/** Map raw yield rows (BC_* columns) to slim spread points, sorted by date. */
export function toSpreadPoints(rows: Record<string, unknown>[]): SpreadPoint[] {
  return rows
    .map((r) => {
      const y10 = num(r.BC_10YEAR);
      const y3m = num(r.BC_3MONTH);
      const y2 = num(r.BC_2YEAR);
      return {
        date: String(r.date),
        s10y3m: spread(y10, y3m),
        s2s10s: spread(y10, y2),
      };
    })
    .sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
}

const isInverted = (v: number | null): v is number => v !== null && v <= 0;

/** Latest values plus the trailing consecutive-inverted streak. */
export function currentStatus(points: SpreadPoint[]): SignalStatus {
  if (points.length === 0) {
    return { date: null, s10y3m: null, s2s10s: null, inverted: false, streakDays: 0 };
  }
  const last = points[points.length - 1];
  let streak = 0;
  for (let i = points.length - 1; i >= 0; i--) {
    if (isInverted(points[i].s10y3m)) streak++;
    else break;
  }
  return {
    date: last.date,
    s10y3m: last.s10y3m,
    s2s10s: last.s2s10s,
    inverted: isInverted(last.s10y3m),
    streakDays: streak,
  };
}

/** Months between two ISO dates (approximate, calendar-based). */
function monthsBetween(fromIso: string, toIso: string): number {
  const a = new Date(fromIso);
  const b = new Date(toIso);
  return (b.getUTCFullYear() - a.getUTCFullYear()) * 12 + (b.getUTCMonth() - a.getUTCMonth());
}

/** A recession "follows" an inversion episode if it overlaps the episode or its
 *  start falls within 24 months after the episode ends. Measuring lead time from
 *  the episode END (not start) correctly flags long multi-year inversions that
 *  begin well over 24 months before the recession but run right up to it. The
 *  month difference is calendar-month-coarse (day-of-month is ignored). */
function recessionFollows(episodeStart: string, episodeEnd: string): boolean {
  return NBER_RECESSIONS.some((r) => {
    const overlaps = r.start <= episodeEnd && r.end >= episodeStart;
    const lead = monthsBetween(episodeEnd, r.start);
    return overlaps || (lead >= 0 && lead <= 24);
  });
}

/** Maximal consecutive runs of inverted (s10y3m <= 0) points. */
export function inversionEpisodes(points: SpreadPoint[]): InversionEpisode[] {
  const episodes: InversionEpisode[] = [];
  let runStart = -1;
  let minVal = Infinity;

  const close = (endIdx: number) => {
    const start = points[runStart].date;
    episodes.push({
      start,
      end: points[endIdx].date,
      days: endIdx - runStart + 1,
      maxDepthBps: minVal * 100,
      recessionFollowed: recessionFollows(start, points[endIdx].date),
    });
    runStart = -1;
    minVal = Infinity;
  };

  for (let i = 0; i < points.length; i++) {
    const v = points[i].s10y3m;
    if (isInverted(v)) {
      if (runStart === -1) runStart = i;
      if (v < minVal) minVal = v;
    } else if (runStart !== -1) {
      close(i - 1);
    }
  }
  if (runStart !== -1) close(points.length - 1);
  return episodes;
}

// --- Inversion periods -------------------------------------------------------
// Daily data turns one inversion into many runs: the spread dips below zero, pops back for a
// few days, dips again. Counting each run as its own "signal" inflates the record (1990–2026
// has 42 runs but only five distinct inversions), so runs closer than six months are merged
// into one period, and each period is judged once.

/** Runs of inversion separated by fewer calendar days than this are one period. */
export const PERIOD_MERGE_DAYS = 182;
/** A recession starting within this many months after a period ends counts as following it. */
export const SIGNAL_WINDOW_MONTHS = 24;

export type PeriodOutcome =
  | { kind: "recession"; recessionStart: string; leadMonths: number }
  | { kind: "none" }
  | { kind: "pending"; windowEnds: string };

export interface InversionPeriod {
  start: string;
  end: string;
  /** Separate runs of inverted days merged into this period. */
  runs: number;
  /** Trading days with 10y–3m ≤ 0. */
  invertedDays: number;
  maxDepthBps: number;
  outcome: PeriodOutcome;
}

const DAY_MS = 86_400_000;

function addMonths(iso: string, months: number): string {
  const d = new Date(iso);
  d.setUTCMonth(d.getUTCMonth() + months);
  return d.toISOString().slice(0, 10);
}

/** What followed a period: the first NBER recession overlapping it or starting within the
 *  window after it ends (lead counted from the period's first inverted day); otherwise none, or
 *  "pending" while the window is still open as of the last data date. */
function periodOutcome(start: string, end: string, asOf: string): PeriodOutcome {
  const windowEnds = addMonths(end, SIGNAL_WINDOW_MONTHS);
  const r = NBER_RECESSIONS.find((r) => (r.start <= end && r.end >= start) || (r.start >= end && r.start <= windowEnds));
  if (r) return { kind: "recession", recessionStart: r.start, leadMonths: Math.max(0, monthsBetween(start, r.start)) };
  return windowEnds > asOf ? { kind: "pending", windowEnds } : { kind: "none" };
}

/** Merge inversion runs (chronological) into periods; `asOf` is the last data date. */
export function inversionPeriods(
  episodes: InversionEpisode[],
  asOf: string,
  mergeGapDays = PERIOD_MERGE_DAYS,
): InversionPeriod[] {
  const groups: InversionEpisode[][] = [];
  for (const e of episodes) {
    const last = groups[groups.length - 1];
    const prevEnd = last?.[last.length - 1].end;
    if (prevEnd && (Date.parse(e.start) - Date.parse(prevEnd)) / DAY_MS < mergeGapDays) last.push(e);
    else groups.push([e]);
  }
  return groups.map((g) => {
    const start = g[0].start;
    const end = g[g.length - 1].end;
    return {
      start,
      end,
      runs: g.length,
      invertedDays: g.reduce((n, e) => n + e.days, 0),
      maxDepthBps: Math.min(...g.map((e) => e.maxDepthBps)),
      outcome: periodOutcome(start, end, asOf),
    };
  });
}

export interface SignalRecord {
  periods: number;
  /** Recessions starting before this date have too little data before them to judge. */
  testableFrom: string;
  /** Recessions with at least a full signal window of data before them (others can't be judged). */
  recessionsTestable: number;
  recessionsPreceded: number;
  leadMonthsMin: number | null;
  leadMonthsMax: number | null;
  /** Periods with no recession inside the window. */
  falseAlarms: number;
  /** Periods whose window is still open. */
  pending: number;
}

/** The scorecard behind the page's summary sentence. `firstDate` is the first data date. */
export function signalRecord(periods: InversionPeriod[], firstDate: string): SignalRecord {
  const testableFrom = addMonths(firstDate, SIGNAL_WINDOW_MONTHS);
  const testable = NBER_RECESSIONS.filter((r) => r.start >= testableFrom);
  const hits = periods.flatMap((p) => (p.outcome.kind === "recession" ? [p.outcome] : []));
  const preceded = testable.filter((r) => hits.some((h) => h.recessionStart === r.start));
  const leads = hits.filter((h) => preceded.some((r) => r.start === h.recessionStart)).map((h) => h.leadMonths);
  return {
    periods: periods.length,
    testableFrom,
    recessionsTestable: testable.length,
    recessionsPreceded: preceded.length,
    leadMonthsMin: leads.length ? Math.min(...leads) : null,
    leadMonthsMax: leads.length ? Math.max(...leads) : null,
    falseAlarms: periods.filter((p) => p.outcome.kind === "none").length,
    pending: periods.filter((p) => p.outcome.kind === "pending").length,
  };
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** The record in one paragraph, e.g. "Since 1990, 10y–3m has inverted in 5 distinct periods…". */
export function describeRecord(r: SignalRecord, firstDate: string): string {
  const parts = [
    `Since ${firstDate.slice(0, 4)}, 10y–3m has inverted in ${plural(r.periods, "distinct period")} (runs under six months apart count as one).`,
  ];
  if (r.recessionsTestable > 0) {
    const lead =
      r.leadMonthsMin === null
        ? ""
        : r.leadMonthsMin === r.leadMonthsMax
          ? `, ${r.leadMonthsMin} months after it began`
          : `, ${r.leadMonthsMin}–${r.leadMonthsMax} months after it began`;
    const since = `since ${r.testableFrom.slice(0, 4)}`;
    parts.push(
      r.recessionsPreceded === r.recessionsTestable
        ? `Each of the ${plural(r.recessionsTestable, "recession")} ${since} came after one${lead}.`
        : `${r.recessionsPreceded} of the ${plural(r.recessionsTestable, "recession")} ${since} came after one${lead}.`,
    );
  }
  const rest: string[] = [];
  if (r.falseAlarms) rest.push(`${plural(r.falseAlarms, "period")} had no recession within ${SIGNAL_WINDOW_MONTHS} months`);
  if (r.pending) rest.push(`${r.pending} ${r.pending === 1 ? "is" : "are"} still inside that window`);
  if (rest.length) parts.push(rest.join(", and ").replace(/^./, (c) => c.toUpperCase()) + ".");
  return parts.join(" ");
}

