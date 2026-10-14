"use client";
import "./globals.css";

// Last-resort boundary for errors in the root layout itself. It replaces the layout, so it
// brings its own <html>/<body> and keeps to plain markup.
export default function GlobalError({ unstable_retry }: { error: Error & { digest?: string }; unstable_retry: () => void }) {
  return (
    <html lang="en">
      <body>
        <title>Something went wrong · BondViz</title>
        <main role="alert" className="mx-auto max-w-xl space-y-4 px-5 py-20">
          <div className="eyebrow">BondViz</div>
          <h1 className="text-3xl">Something went wrong.</h1>
          <p className="text-[var(--muted)]">An unexpected error stopped the app from loading.</p>
          <button
            type="button"
            onClick={() => unstable_retry()}
            className="rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[var(--on-accent)]"
          >
            Try again
          </button>
        </main>
      </body>
    </html>
  );
}
