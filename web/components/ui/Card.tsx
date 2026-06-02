import { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-lg border border-[var(--panel-border)] border-l-[3px] border-l-[var(--accent)] bg-[var(--panel)] p-5 ${className}`}
    >
      {children}
    </div>
  );
}
