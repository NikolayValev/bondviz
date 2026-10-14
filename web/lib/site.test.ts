import { describe, it, expect } from "vitest";
import { SITE_URL, SOURCE_URL, AUTHOR, pageMetadata, rootMetadata } from "@/lib/site";

describe("site constants", () => {
  // metadataBase, the sitemap and robots concatenate paths onto these.
  it.each([SITE_URL, SOURCE_URL, AUTHOR.url])("%s is absolute https with no trailing slash", (u) => {
    expect(u).toMatch(/^https:\/\/[^\s]+[^/]$/);
  });
});

describe("pageMetadata", () => {
  it("gives each page its own title, description, canonical and preview fields", () => {
    const m = pageMetadata({ title: "Carry & Roll-Down", description: "d", path: "/carry" });
    expect(m.title).toBe("Carry & Roll-Down");
    expect(m.alternates.canonical).toBe("/carry");
    expect(m.openGraph).toMatchObject({ title: "Carry & Roll-Down · BondViz", description: "d", url: "/carry", siteName: "BondViz" });
    expect(m.twitter).toMatchObject({ card: "summary_large_image", title: "Carry & Roll-Down · BondViz" });
    // Setting openGraph drops the inherited file-based image, so it must be listed explicitly.
    expect(m.openGraph.images[0].url).toBe("/opengraph-image");
    expect(m.twitter.images[0].url).toBe("/opengraph-image");
  });
  it("root metadata sets the base URL and a title template", () => {
    expect(rootMetadata.metadataBase.href).toBe(`${SITE_URL}/`);
    expect(rootMetadata.title.template).toBe("%s · BondViz");
  });
});
