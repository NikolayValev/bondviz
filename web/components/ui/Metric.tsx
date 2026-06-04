import { ReactNode } from "react";

type Tone = "accent" | "pos" | "neg" | "neutral";

const TONE: Record<Tone, string> = {
  accent: "text-[var(--accent)]",
  pos: "text-[var(--pos)]",
  neg: "text-[var(--neg)]",
  neutral: "text-[var(--text)]",
};

export function Metric({
  label,
  value,
  sub,
  tone = "neutral",
  size = "md",
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
  size?: "md" | "lg";
}) {
  return (
    <div className="rounded-[var(--radius)] border border-[var(--panel-border)] bg-[var(--panel-2)] px-4 py-3.5">
      <div className="eyebrow">{label}</div>
      <div className={`tabnum mt-1.5 font-semibold tracking-tight ${TONE[tone]} ${size === "lg" ? "text-3xl sm:text-4xl" : "text-2xl"}`}>
        {value}
      </div>
      {sub != null && <div className="mt-1 text-xs text-[var(--muted)]">{sub}</div>}
    </div>
  );
}
