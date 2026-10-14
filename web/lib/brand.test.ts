import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { BRAND_HEX } from "@/lib/brand";

const css = readFileSync(fileURLToPath(new URL("../app/globals.css", import.meta.url)), "utf-8");
const TOKEN: Record<keyof typeof BRAND_HEX, string> = {
  bg: "--bg",
  panel: "--panel",
  text: "--text",
  muted: "--muted",
  faint: "--faint",
  accent: "--accent",
  series2: "--series-2",
};

describe("BRAND_HEX", () => {
  it.each(Object.entries(TOKEN))("%s matches %s in globals.css", (key, token) => {
    const m = new RegExp(`${token}:\\s*(#[0-9a-fA-F]{6})`).exec(css);
    expect(m?.[1].toLowerCase()).toBe(BRAND_HEX[key as keyof typeof BRAND_HEX]);
  });
});
