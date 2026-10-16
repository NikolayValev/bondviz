import { ReactNode } from "react";

/** Horizontally scrollable wrapper (wide tables on small screens). It is focusable and
 *  labelled so keyboard users can reach and scroll it (WCAG 2.1.1). */
export function ScrollRegion({ label, className = "", children }: { label: string; className?: string; children: ReactNode }) {
  return (
    <div
      role="region"
      aria-label={label}
      tabIndex={0}
      className={`overflow-x-auto rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)] ${className}`}
    >
      {children}
    </div>
  );
}
