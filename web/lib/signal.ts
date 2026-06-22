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

const isInverted = (v: number | null): boolean => v !== null && v <= 0;

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

function recessionWithin24m(episodeStart: string): boolean {
  return NBER_RECESSIONS.some((r) => {
    const m = monthsBetween(episodeStart, r.start);
    return m >= 0 && m <= 24;
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
      recessionFollowed: recessionWithin24m(start),
    });
    runStart = -1;
    minVal = Infinity;
  };

  for (let i = 0; i < points.length; i++) {
    const v = points[i].s10y3m;
    if (isInverted(v)) {
      if (runStart === -1) runStart = i;
      if ((v as number) < minVal) minVal = v as number;
    } else if (runStart !== -1) {
      close(i - 1);
    }
  }
  if (runStart !== -1) close(points.length - 1);
  return episodes;
}
