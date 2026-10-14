import { describe, it, expect } from "vitest";
import { SITE_URL, SOURCE_URL, AUTHOR } from "@/lib/site";

describe("site constants", () => {
  // metadataBase, the sitemap and robots concatenate paths onto these.
  it.each([SITE_URL, SOURCE_URL, AUTHOR.url])("%s is absolute https with no trailing slash", (u) => {
    expect(u).toMatch(/^https:\/\/[^\s]+[^/]$/);
  });
});
