import { describe, it, expect } from "vitest";
import robots from "@/app/robots";
import sitemap from "@/app/sitemap";
import { SITE_URL } from "@/lib/site";
import { TOOLS } from "@/lib/tools";

describe("robots", () => {
  it("allows the site, keeps crawlers off the API, and points at the sitemap", () => {
    const r = robots();
    expect(r.rules).toMatchObject({ allow: "/", disallow: "/api/" });
    expect(r.sitemap).toBe(`${SITE_URL}/sitemap.xml`);
  });
});

describe("sitemap", () => {
  it("lists home, every tool and the methodology page as absolute URLs", () => {
    const urls = sitemap().map((e) => e.url);
    expect(urls).toEqual([SITE_URL, ...TOOLS.map((t) => `${SITE_URL}${t.href}`), `${SITE_URL}/about`]);
  });
});
