// Literal values of the :root color tokens in app/globals.css, for renderers that can't read
// CSS variables (the Open Graph image is drawn by Satori, outside the page). Everything that
// renders in the browser keeps using var(--…); brand.test.ts fails if these drift.

export const BRAND_HEX = {
  bg: "#070a10",
  panel: "#111620",
  text: "#e8ecf2",
  muted: "#8b95a7",
  faint: "#5b6577",
  accent: "#00d68f",
  series2: "#5b8def",
} as const;
