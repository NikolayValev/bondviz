// Portfolio-level fixed-income analytics. Pure functions built on the
// single-bond primitives in finance.ts so a portfolio stays consistent with
// the individual bond views. Everything is continuous compounding.
import {
  bondCashflows,
  convexity,
  dv01,
  modifiedDuration,
  priceFromCashflows,
  type BondParams,
  type ScenarioRow,
} from "@/lib/finance";

export interface HoldingMetrics {
  value: number; // market value (present value of the position)
  weight: number; // share of total portfolio value
  duration: number; // modified duration (yrs)
  convexity: number;
  dv01: number; // dollar value of 1 bp for the position
  yield: number;
}

export interface PortfolioMetrics {
  totalValue: number;
  weightedDuration: number;
  weightedConvexity: number;
  weightedYield: number;
  weightedMaturity: number;
  dv01: number;
  weights: number[];
  holdings: HoldingMetrics[];
}

function analyse(h: BondParams) {
  const freq = h.freq ?? 2;
  const { cashflows, times } = bondCashflows(h.face, h.coupon, h.years, freq);
  return {
    value: priceFromCashflows(cashflows, times, h.yield_),
    duration: modifiedDuration(cashflows, times, h.yield_),
    convexity: convexity(cashflows, times, h.yield_),
    dv01: dv01(cashflows, times, h.yield_),
    yield: h.yield_,
    years: h.years,
  };
}

export function portfolioMetrics(holdings: BondParams[]): PortfolioMetrics {
  const rows = holdings.map(analyse);
  const totalValue = rows.reduce((a, r) => a + r.value, 0);

  const empty: PortfolioMetrics = {
    totalValue: 0,
    weightedDuration: 0,
    weightedConvexity: 0,
    weightedYield: 0,
    weightedMaturity: 0,
    dv01: 0,
    weights: [],
    holdings: [],
  };
  if (totalValue === 0) return rows.length === 0 ? empty : { ...empty, weights: rows.map(() => 0), holdings: rows.map((r) => ({ ...r, weight: 0 })) };

  const weights = rows.map((r) => r.value / totalValue);
  const wsum = (pick: (r: (typeof rows)[number]) => number) =>
    rows.reduce((a, r, i) => a + weights[i] * pick(r), 0);

  return {
    totalValue,
    weightedDuration: wsum((r) => r.duration),
    weightedConvexity: wsum((r) => r.convexity),
    weightedYield: wsum((r) => r.yield),
    weightedMaturity: wsum((r) => r.years),
    dv01: rows.reduce((a, r) => a + r.dv01, 0), // additive dollar measure
    weights,
    holdings: rows.map((r, i) => ({
      value: r.value,
      weight: weights[i],
      duration: r.duration,
      convexity: r.convexity,
      dv01: r.dv01,
      yield: r.yield,
    })),
  };
}

/** Re-price the whole portfolio across parallel yield shifts (every holding's
 *  yield moves by the same amount). Returns rows shaped like the single-bond
 *  ScenarioRow so the existing chart and tables can render them, where
 *  `newPrice` is total portfolio value and `newYield` is the value-weighted
 *  yield after the shift. */
export function portfolioScenario(holdings: BondParams[], shiftsBps: number[]): ScenarioRow[] {
  const prepared = holdings.map((h) => {
    const freq = h.freq ?? 2;
    const { cashflows, times } = bondCashflows(h.face, h.coupon, h.years, freq);
    return {
      cashflows,
      times,
      yield_: h.yield_,
      value: priceFromCashflows(cashflows, times, h.yield_),
      duration: modifiedDuration(cashflows, times, h.yield_),
      convexity: convexity(cashflows, times, h.yield_),
    };
  });
  const base = prepared.reduce((a, p) => a + p.value, 0);
  const baseYield = base > 0 ? prepared.reduce((a, p) => a + (p.value / base) * p.yield_, 0) : 0;

  return shiftsBps.map((shiftBps) => {
    const dy = shiftBps / 10_000;
    let newPrice = 0;
    let approxDollar = 0;
    for (const p of prepared) {
      newPrice += priceFromCashflows(p.cashflows, p.times, p.yield_ + dy);
      approxDollar += p.value * (-p.duration * dy + 0.5 * p.convexity * dy * dy);
    }
    const dollarChange = newPrice - base;
    return {
      shiftBps,
      newYield: baseYield + dy,
      newPrice,
      dollarChange,
      pctChange: base > 0 ? dollarChange / base : 0,
      approxPctChange: base > 0 ? approxDollar / base : 0,
    };
  });
}
