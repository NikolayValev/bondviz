"use client";
import Link from "next/link";
import { useEffect } from "react";

// Route-level error boundary: catches render errors in any page under the root layout, so a
// failure shows this panel (with nav and footer intact) instead of a blank screen.
export default function Error({
  error,
  unstable_retry,
}: {
  error: Error & { digest?: string };
  unstable_retry: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div role="alert" className="space-y-4">
      <div className="eyebrow">Error</div>
      <h1 className="text-3xl">Something went wrong on this page.</h1>
      <p className="max-w-xl text-[var(--muted)]">
        The rest of BondViz is fine. Data feeds are occasionally slow; trying again usually works.
      </p>
      <div className="flex flex-wrap gap-3 text-sm">
        <button
          type="button"
          onClick={() => unstable_retry()}
          className="rounded-full bg-[var(--accent)] px-4 py-2 font-medium text-[var(--on-accent)] transition-opacity hover:opacity-90"
        >
          Try again
        </button>
        <Link
          href="/"
          className="rounded-full border border-[var(--panel-border)] px-4 py-2 text-[var(--muted)] transition-colors hover:bg-white/5 hover:text-[var(--text)]"
        >
          Back to home
        </Link>
      </div>
      {error.digest && <p className="tabnum text-xs text-[var(--faint)]">Reference: {error.digest}</p>}
    </div>
  );
}
