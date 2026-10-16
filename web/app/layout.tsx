import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { AUTHOR, SOURCE_URL, rootMetadata } from "@/lib/site";

const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-plex-sans",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = rootMetadata;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${plexSans.variable} ${plexMono.variable}`}>
      <body className="gridlines">
        <div className="relative z-10">
          <Nav />
          <main className="mx-auto max-w-6xl px-5 py-8 sm:px-6 sm:py-10">{children}</main>
          <footer className="mx-auto flex max-w-6xl flex-col gap-2 border-t border-[var(--panel-border)] px-5 pb-10 pt-6 text-xs text-faint sm:flex-row sm:items-center sm:justify-between sm:px-6">
            <p>
              Built by{" "}
              <a href={AUTHOR.url} className="text-[var(--muted)] underline decoration-[var(--panel-border-strong)] underline-offset-4 hover:text-[var(--text)] hover:decoration-current">
                {AUTHOR.name}
              </a>{" "}
              · Data: U.S. Treasury, Polygon.io
            </p>
            <a href={SOURCE_URL} target="_blank" rel="noreferrer" className="text-[var(--muted)] hover:text-[var(--text)]">
              Source on GitHub ↗
            </a>
          </footer>
        </div>
      </body>
    </html>
  );
}
