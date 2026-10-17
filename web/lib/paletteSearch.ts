// Ranking for the ⌘K command palette: which destinations match a query, best first.
// Pure so it can be tested without a DOM.

export interface PaletteItem {
  id: string;
  title: string;
  /** Short label or group shown beside the title. */
  hint?: string;
  /** Extra words that should match (synonyms, description). */
  keywords?: string;
  href: string;
  external?: boolean;
}

/** Lower is better; null means no match. Every query word must match somewhere. */
export function scoreItem(item: PaletteItem, query: string): number | null {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length === 0) return 0;
  const title = item.title.toLowerCase();
  const titleWords = title.split(/[^a-z0-9]+/).filter(Boolean);
  const rest = `${item.hint ?? ""} ${item.keywords ?? ""}`.toLowerCase();
  let score = 0;
  for (const w of words) {
    if (title.startsWith(w)) score += 0;
    else if (titleWords.some((t) => t.startsWith(w))) score += 1;
    else if (title.includes(w)) score += 2;
    else if (rest.split(/[^a-z0-9]+/).some((t) => t.startsWith(w))) score += 3;
    else if (rest.includes(w)) score += 4;
    else return null;
  }
  return score;
}

/** Items matching `query`, best first; ties keep their original order. */
export function searchPalette(items: PaletteItem[], query: string): PaletteItem[] {
  return items
    .map((item, i) => ({ item, i, s: scoreItem(item, query) }))
    .filter((r): r is { item: PaletteItem; i: number; s: number } => r.s !== null)
    .sort((a, b) => a.s - b.s || a.i - b.i)
    .map((r) => r.item);
}
