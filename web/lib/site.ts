// Site-wide identity: canonical URL, author and links used by metadata, the footer and Home.

export const SITE_URL = "https://bondviz.nikolayvalev.com";
export const SITE_NAME = "BondViz";
export const SITE_TAGLINE = "Fixed-income research terminal on live U.S. Treasury data";
export const SITE_DESCRIPTION =
  "Explore the live U.S. Treasury yield curve, price and stress-test bonds and portfolios, " +
  "decompose the curve with PCA and track the inversion recession signal. Hand-rolled D3 on " +
  "unit-tested bond math.";
export const AUTHOR = { name: "Nikolay Valev", url: "https://nikolayvalev.com" };
export const SOURCE_URL = "https://github.com/NikolayValev/bondviz";

export const OG_IMAGE = {
  url: "/opengraph-image",
  width: 1200,
  height: 630,
  alt: "BondViz: a fixed-income research terminal on live U.S. Treasury data",
};

// A page that sets `openGraph` stops inheriting the file-based image from app/opengraph-image,
// so the image is listed explicitly in the shared fields.
const SHARED_OG = { siteName: SITE_NAME, type: "website" as const, locale: "en_US", images: [OG_IMAGE] };

/**
 * Metadata for one page. Next shallow-merges metadata per segment, so a page that sets
 * `openGraph` replaces the layout's whole object; build every page's from the same shared
 * fields so link previews carry the page's own title and description (and the shared image
 * drawn by app/opengraph-image.tsx). `title` gets " · BondViz" from the root layout's template.
 */
export function pageMetadata({ title, description, path }: { title: string; description: string; path: string }) {
  const full = `${title} · ${SITE_NAME}`;
  return {
    title,
    description,
    alternates: { canonical: path },
    openGraph: { ...SHARED_OG, title: full, description, url: path },
    twitter: { card: "summary_large_image" as const, title: full, description, images: [OG_IMAGE] },
  };
}

/** Root-layout metadata: base URL, title template and the home page's own preview. */
export const rootMetadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: `${SITE_NAME} · ${SITE_TAGLINE}`, template: `%s · ${SITE_NAME}` },
  description: SITE_DESCRIPTION,
  applicationName: SITE_NAME,
  authors: [{ name: AUTHOR.name, url: AUTHOR.url }],
  creator: AUTHOR.name,
  keywords: [
    "yield curve", "Treasury", "bond pricing", "duration", "convexity", "carry", "roll-down",
    "PCA", "recession signal", "fixed income", "D3", "Next.js",
  ],
  alternates: { canonical: "/" },
  openGraph: { ...SHARED_OG, title: `${SITE_NAME} · ${SITE_TAGLINE}`, description: SITE_DESCRIPTION, url: "/" },
  twitter: {
    card: "summary_large_image" as const,
    title: `${SITE_NAME} · ${SITE_TAGLINE}`,
    description: SITE_DESCRIPTION,
    images: [OG_IMAGE],
  },
};
