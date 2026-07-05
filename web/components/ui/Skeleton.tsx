/** Loading placeholder that reserves the space of the content it stands in
 *  for, so data arriving doesn't shift the layout (CLS). Size it with height
 *  utilities matching the real content (e.g. className="h-80"). */
export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded-lg border border-[var(--panel-border)] bg-[var(--panel)] ${className}`}
    />
  );
}
