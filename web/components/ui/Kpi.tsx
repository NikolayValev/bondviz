export function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-sm text-[var(--muted)]">{label}</div>
      <div className="tabnum text-2xl text-[var(--accent)]">{value}</div>
    </div>
  );
}
