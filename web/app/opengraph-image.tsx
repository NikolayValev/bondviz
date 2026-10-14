import { ImageResponse } from "next/og";
import { BRAND_HEX as C } from "@/lib/brand";
import { OG_IMAGE, SITE_URL } from "@/lib/site";

// Social preview card (also used for X via summary_large_image). Static: no data fetch, so it
// renders once at build. The curves are illustrative shapes, not live data.

export const alt = OG_IMAGE.alt;
export const size = { width: OG_IMAGE.width, height: OG_IMAGE.height };
export const contentType = "image/png";

const W = 440;
const H = 280;
const TENORS = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30];

function path(yields: number[]): string {
  const x = (t: number) => (Math.sqrt(t) / Math.sqrt(30)) * (W - 20) + 10;
  const y = (v: number) => H - 20 - ((v - 3.4) / (5.6 - 3.4)) * (H - 40);
  return TENORS.map((t, i) => `${i === 0 ? "M" : "L"}${x(t).toFixed(1)},${y(yields[i]).toFixed(1)}`).join(" ");
}

const NOW = [4.05, 4.0, 3.95, 3.9, 3.98, 4.15, 4.4, 4.62, 5.1, 5.2];
const YEAR_AGO = [5.35, 5.3, 5.05, 4.6, 4.4, 4.25, 4.3, 4.38, 4.7, 4.6];

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "64px 72px",
          background: C.bg,
          backgroundImage: `radial-gradient(circle at 12% 0%, ${C.accent}2e, transparent 55%), radial-gradient(circle at 100% 0%, ${C.series2}22, transparent 50%)`,
          color: C.text,
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div style={{ display: "flex", flexDirection: "column", flex: 1, marginRight: 40 }}>
            <div style={{ display: "flex", fontSize: 18, letterSpacing: 4, color: C.faint, textTransform: "uppercase" }}>
              Fixed-income research terminal
            </div>
            <div style={{ display: "flex", fontSize: 128, fontWeight: 700, letterSpacing: -2, marginTop: 12 }}>
              BOND<span style={{ color: C.accent }}>VIZ</span>
            </div>
            <div style={{ display: "flex", fontSize: 30, lineHeight: 1.35, color: C.muted, marginTop: 16 }}>
              Live U.S. Treasury curve, bond pricing &amp; risk, carry, PCA and the inversion
              recession signal.
            </div>
          </div>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              flexShrink: 0,
              width: W + 40,
              marginTop: 24,
              padding: 20,
              borderRadius: 18,
              border: "1px solid rgba(255,255,255,0.08)",
              background: C.panel,
            }}
          >
            <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
              {[0, 1, 2, 3].map((i) => (
                <line key={i} x1={0} x2={W} y1={20 + i * ((H - 40) / 3)} y2={20 + i * ((H - 40) / 3)} stroke="rgba(255,255,255,0.07)" />
              ))}
              <path d={path(YEAR_AGO)} fill="none" stroke={C.series2} strokeWidth={4} strokeDasharray="10 8" />
              <path d={path(NOW)} fill="none" stroke={C.accent} strokeWidth={6} />
            </svg>
            <div style={{ display: "flex", gap: 28, marginTop: 10, fontSize: 18, color: C.muted }}>
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 22, height: 6, background: C.accent, borderRadius: 3 }} /> Today
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 22, height: 6, background: C.series2, borderRadius: 3 }} /> 1 year ago
              </span>
            </div>
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 22, color: C.muted }}>
          <span>Hand-rolled D3 · unit-tested bond math · Next.js</span>
          <span style={{ color: C.accent }}>{SITE_URL.replace("https://", "")}</span>
        </div>
      </div>
    ),
    size,
  );
}
