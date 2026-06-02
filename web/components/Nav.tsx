import Link from "next/link";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/yield-curve", label: "Yield Curve" },
  { href: "/pricing", label: "Bond Pricing" },
];

export function Nav() {
  return (
    <header className="border-b border-[var(--panel-border)]">
      <nav className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
        <Link href="/" className="font-semibold tracking-wide text-[var(--accent)]">
          BONDVIZ
        </Link>
        <div className="flex gap-5 text-sm text-[var(--muted)]">
          {LINKS.slice(1).map((l) => (
            <Link key={l.href} href={l.href} className="hover:text-[var(--text)]">
              {l.label}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}
