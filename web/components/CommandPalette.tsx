"use client";
import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { TOOLS } from "@/lib/tools";
import { SOURCE_URL } from "@/lib/site";
import { searchPalette, type PaletteItem } from "@/lib/paletteSearch";

/** Window event that opens the palette (dispatched by the nav button). */
export const OPEN_PALETTE_EVENT = "bondviz:open-palette";

const ITEMS: PaletteItem[] = [
  { id: "home", title: "Home", hint: "Overview", keywords: "market snapshot kpis start", href: "/" },
  ...TOOLS.map((t) => ({ id: t.href, title: t.title, hint: t.group, keywords: `${t.nav} ${t.description}`, href: t.href })),
  { id: "about", title: "Methodology", hint: "About", keywords: "about math formulas conventions how it works", href: "/about" },
  { id: "styleguide", title: "Design system", hint: "About", keywords: "styleguide tokens components charts", href: "/styleguide" },
  { id: "source", title: "Source on GitHub", hint: "External", keywords: "code repository repo", href: SOURCE_URL, external: true },
];

/**
 * ⌘K / Ctrl-K palette for jumping between tools. Hand-rolled (no dependency): a modal dialog
 * holding a combobox input and a listbox, navigated with ↑/↓/Enter and closed with Escape or a
 * click outside. Focus stays in the input while open and returns to where it was on close.
 */
export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  const ids = useId();

  const results = useMemo(() => searchPalette(ITEMS, query), [query]);
  const activeIndex = Math.min(active, Math.max(0, results.length - 1));

  const show = useCallback(() => {
    returnFocus.current = document.activeElement as HTMLElement | null;
    setQuery("");
    setActive(0);
    setOpen(true);
  }, []);
  const close = useCallback(() => {
    setOpen(false);
    returnFocus.current?.focus?.();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (open) close();
        else show();
      }
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener(OPEN_PALETTE_EVENT, show);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener(OPEN_PALETTE_EVENT, show);
    };
  }, [open, show, close]);

  useEffect(() => {
    if (!open) return;
    inputRef.current?.focus();
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  const go = (item: PaletteItem | undefined) => {
    if (!item) return;
    close();
    if (item.external) window.open(item.href, "_blank", "noopener,noreferrer");
    else router.push(item.href);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((activeIndex + 1) % Math.max(1, results.length));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((activeIndex - 1 + results.length) % Math.max(1, results.length));
    } else if (e.key === "Enter") {
      e.preventDefault();
      go(results[activeIndex]);
    } else if (e.key === "Escape") {
      e.preventDefault();
      close();
    } else if (e.key === "Tab") {
      e.preventDefault(); // the input is the only tab stop; keep focus inside the dialog
    }
  };

  if (!open) return null;

  const listId = `${ids}-list`;
  const optionId = (i: number) => `${ids}-opt-${i}`;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 px-4 pt-[12vh] backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) close();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Go to a page"
        className="w-full max-w-lg overflow-hidden rounded-[var(--radius)] border border-[var(--panel-border-strong)] bg-[var(--panel)] shadow-2xl"
      >
        <div className="flex items-center gap-2 border-b border-[var(--panel-border)] px-4">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="shrink-0 text-[var(--faint)]" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="m20 20-3.5-3.5" />
          </svg>
          <input
            ref={inputRef}
            role="combobox"
            aria-expanded="true"
            aria-controls={listId}
            aria-autocomplete="list"
            aria-activedescendant={results.length ? optionId(activeIndex) : undefined}
            aria-label="Search pages"
            placeholder="Jump to a tool…"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(0);
            }}
            onKeyDown={onKeyDown}
            className="w-full bg-transparent py-3.5 text-[var(--text)] outline-none placeholder:text-[var(--faint)]"
          />
          <kbd className="tabnum rounded border border-[var(--panel-border)] px-1.5 py-0.5 text-[0.65rem] text-[var(--faint)]">esc</kbd>
        </div>
        <ul id={listId} role="listbox" aria-label="Pages" className="max-h-[50vh] overflow-y-auto p-1.5">
          {results.length === 0 && <li className="px-3 py-6 text-center text-sm text-[var(--muted)]">No matching page.</li>}
          {results.map((item, i) => (
            <li
              key={item.id}
              id={optionId(i)}
              role="option"
              aria-selected={i === activeIndex}
              onMouseMove={() => setActive(i)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => go(item)}
              className={`flex cursor-pointer items-center justify-between gap-3 rounded-lg px-3 py-2.5 text-sm ${
                i === activeIndex ? "bg-[var(--accent-dim)] text-[var(--text)]" : "text-[var(--muted)]"
              }`}
            >
              <span>{item.title}</span>
              <span className="eyebrow text-[var(--muted)]">{item.hint}</span>
            </li>
          ))}
        </ul>
        <div className="flex gap-4 border-t border-[var(--panel-border)] px-4 py-2 text-xs text-[var(--faint)]">
          <span>↑↓ to move</span>
          <span>↵ to open</span>
          <span className="ml-auto">⌘K / Ctrl-K anywhere</span>
        </div>
      </div>
    </div>
  );
}
