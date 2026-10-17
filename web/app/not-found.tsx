import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { TOOLS } from "@/lib/tools";

export const metadata = { title: "Page not found" };

export default function NotFound() {
  return (
    <div className="space-y-8">
      <section>
        <div className="eyebrow">404</div>
        <h1 className="mt-2 text-3xl sm:text-4xl">This page is off the curve.</h1>
        <p className="mt-3 max-w-xl text-[var(--muted)]">
          There&apos;s nothing at this address. Head back home, or jump straight into one of the tools.
        </p>
        <Link
          href="/"
          className="mt-5 inline-block rounded-full bg-[var(--accent)] px-4 py-2 text-sm font-medium text-[var(--on-accent)] transition-opacity hover:opacity-90"
        >
          Back to home
        </Link>
      </section>
      <Card>
        <h2 className="eyebrow mb-3">Tools</h2>
        <ul className="grid grid-cols-1 gap-2 text-sm sm:grid-cols-2">
          {TOOLS.map((t) => (
            <li key={t.href}>
              <Link href={t.href} className="text-[var(--accent)] underline decoration-[var(--accent-ring)] underline-offset-4 hover:decoration-current">
                {t.title}
              </Link>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
