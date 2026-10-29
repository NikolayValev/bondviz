import { SITE_URL } from "@/lib/site";

// Chart → PNG, in the browser, no dependencies. Chart colors are CSS variables and text styles
// come from classes, neither of which exists once the SVG is drawn as a standalone image, so the
// computed styles are inlined first. The PNG adds what lives outside the SVG on screen: a title,
// the legend, and a source line.

const SVG_NS = "http://www.w3.org/2000/svg";

const INLINE_PROPS = [
  "fill",
  "fill-opacity",
  "stroke",
  "stroke-width",
  "stroke-opacity",
  "stroke-dasharray",
  "opacity",
  "font-family",
  "font-size",
  "font-weight",
  "font-variant-numeric",
  "text-anchor",
  "dominant-baseline",
  "visibility",
] as const;

/**
 * A self-contained copy of a chart's SVG with computed styles inlined. The original is untouched.
 * A property is written only where the copy would otherwise get it wrong: not where a plain
 * presentation attribute already says it, and not where the element would inherit the same value
 * (that keeps a heatmap's thousands of cells small).
 */
export function standaloneSvg(svg: SVGSVGElement): { markup: string; width: number; height: number } {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  const from = [svg, ...svg.querySelectorAll("*")];
  const to = [clone, ...clone.querySelectorAll("*")];
  const computed = new Map<Element, CSSStyleDeclaration>();
  const styleOf = (el: Element) => {
    let cs = computed.get(el);
    if (!cs) computed.set(el, (cs = getComputedStyle(el)));
    return cs;
  };

  from.forEach((el, i) => {
    const cs = styleOf(el);
    const parent = i === 0 ? null : el.parentElement && styleOf(el.parentElement);
    const style = (to[i] as SVGElement).style;
    for (const p of INLINE_PROPS) {
      const attr = el.getAttribute(p);
      if (attr && !attr.includes("var(")) continue;
      const v = cs.getPropertyValue(p);
      // A computed url() paint (gradients) is absolute to the page; the attribute's local "#id"
      // reference already resolves inside the copy.
      if (!v || v.includes("url(")) continue;
      const inherited = p === "opacity" ? v === "1" : !attr && parent?.getPropertyValue(p) === v;
      if (!inherited) style.setProperty(p, v);
    }
  });

  clone.removeAttribute("class");
  let markup = new XMLSerializer().serializeToString(clone);
  // Elements React created are already in the SVG namespace and serialize with xmlns.
  if (!/^<svg[^>]*\sxmlns=/.test(markup)) markup = markup.replace(/^<svg/, `<svg xmlns="${SVG_NS}"`);
  const width = Number(svg.getAttribute("width")) || svg.getBoundingClientRect().width;
  const height = Number(svg.getAttribute("height")) || svg.getBoundingClientRect().height;
  return { markup, width, height };
}

/** "var(--series-1)" → that token's value on :root; any other color passes through. */
export function resolveColor(color: string, root: Element = document.documentElement): string {
  const m = /^var\((--[\w-]+)\)$/.exec(color.trim());
  return m ? getComputedStyle(root).getPropertyValue(m[1]).trim() || color : color;
}

export interface LegendItem {
  label: string;
  color: string;
  /** For swatches of translucent marks (bands, fills). */
  opacity?: number;
}

const SCALE = 2; // device pixels per CSS pixel, for a crisp image
const PAD = 24;
const TITLE_H = 34;
const LEGEND_ROW = 20;
const FOOTER_H = 26;

/** Render a chart to a PNG blob: title, the chart, its legend and a source line on the panel color. */
export async function chartPng(svg: SVGSVGElement, title: string, legend: LegendItem[] = []): Promise<Blob> {
  const { markup, width, height } = standaloneSvg(svg);
  const img = new Image();
  img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;
  await img.decode();

  const font = getComputedStyle(document.body).fontFamily;
  const colors = {
    bg: resolveColor("var(--panel)"),
    text: resolveColor("var(--text)"),
    muted: resolveColor("var(--muted)"),
  };

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is unavailable");

  // Lay the legend out in rows that fit the chart width before sizing the canvas.
  ctx.font = `12px ${font}`;
  const rows: { x: number; item: LegendItem }[][] = [];
  let x = 0;
  for (const item of legend) {
    const w = 18 + ctx.measureText(item.label).width;
    if (rows.length === 0 || (x > 0 && x + w > width)) {
      rows.push([]);
      x = 0;
    }
    rows[rows.length - 1].push({ x, item });
    x += w + 16;
  }

  const cssW = width + PAD * 2;
  const cssH = PAD + TITLE_H + height + rows.length * LEGEND_ROW + FOOTER_H + PAD / 2;
  canvas.width = Math.round(cssW * SCALE);
  canvas.height = Math.round(cssH * SCALE);
  ctx.scale(SCALE, SCALE);

  ctx.fillStyle = colors.bg;
  ctx.fillRect(0, 0, cssW, cssH);

  ctx.textBaseline = "middle";
  ctx.fillStyle = colors.text;
  ctx.font = `600 16px ${font}`;
  ctx.fillText(title, PAD, PAD + 8);

  let y = PAD + TITLE_H;
  ctx.drawImage(img, PAD, y, width, height);
  y += height;

  ctx.font = `12px ${font}`;
  for (const row of rows) {
    const cy = y + LEGEND_ROW / 2 + 2;
    for (const { x: lx, item } of row) {
      ctx.globalAlpha = item.opacity ?? 1;
      ctx.fillStyle = resolveColor(item.color);
      ctx.fillRect(PAD + lx, cy - 4, 12, 8);
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors.muted;
      ctx.fillText(item.label, PAD + lx + 18, cy);
    }
    y += LEGEND_ROW;
  }

  ctx.font = `11px ${font}`;
  ctx.fillStyle = colors.muted;
  ctx.fillText(new URL(SITE_URL).host, PAD, y + FOOTER_H / 2 + 2);
  ctx.textAlign = "right";
  ctx.fillText(new Date().toISOString().slice(0, 10), PAD + width, y + FOOTER_H / 2 + 2);

  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png"),
  );
}
