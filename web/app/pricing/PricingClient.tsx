"use client";
import { useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Kpi } from "@/components/ui/Kpi";
import { LineChart } from "@/components/charts/LineChart";
import { pvContinuous, discountFactors } from "@/lib/finance";

function NumberField({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="block">
      <span className="text-sm text-[var(--muted)]">{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="tabnum mt-1 w-full rounded border border-[var(--panel-border)] bg-[var(--bg)] px-3 py-2 text-[var(--text)]"
      />
    </label>
  );
}

export function PricingClient() {
  const [face, setFace] = useState(1000);
  const [coupon, setCoupon] = useState(0.05);
  const [ytm, setYtm] = useState(0.04);
  const [years, setYears] = useState(10);

  const { pv, dfPoints } = useMemo(() => {
    const safe = (n: number) => (Number.isFinite(n) ? n : 0);
    const pv = pvContinuous(safe(face), safe(coupon), safe(ytm), safe(years));
    const grid = Array.from({ length: 11 }, (_, i) => (safe(years) * i) / 10);
    const dfPoints = discountFactors(safe(ytm), grid).map((d) => [d.t, d.df] as [number, number]);
    return { pv, dfPoints };
  }, [face, coupon, ytm, years]);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl">Bond Pricing</h1>
      <p className="text-[var(--muted)]">Present value of a fixed-coupon bond under continuous compounding.</p>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card>
          <h2 className="mb-3 text-lg">Inputs</h2>
          <div className="grid grid-cols-2 gap-4">
            <NumberField label="Face" value={face} step={100} onChange={setFace} />
            <NumberField label="Coupon rate" value={coupon} step={0.005} onChange={setCoupon} />
            <NumberField label="Continuous yield" value={ytm} step={0.005} onChange={setYtm} />
            <NumberField label="Years to maturity" value={years} step={1} onChange={setYears} />
          </div>
          <div className="mt-5">
            <Kpi label="Present Value" value={pv.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} />
          </div>
        </Card>

        <Card>
          <h2 className="mb-2 text-lg">Discount factors</h2>
          <LineChart
            ariaLabel="Continuous-compounding discount factor by maturity"
            series={[{ id: "df", label: "Discount factor", color: "#00d68f", points: dfPoints }]}
            xLabel="Years"
            yLabel="Discount factor"
          />
        </Card>
      </div>
    </div>
  );
}
