# BondViz Web (Next.js front-end)

Native front-end for BondViz: Home, Yield Curve explorer, and Bond Pricing, built with
Next.js (App Router), TypeScript, Tailwind, and hand-rolled D3 charts. Data comes from the
keyless U.S. Treasury par-yield XML feed via server route handlers (cached hourly).

## Develop

```bash
cd web
npm install
npm run dev      # http://localhost:3000
npm test         # Vitest unit tests
npm run build    # production build
```

## Stocks (Polygon)

The Stocks page reads daily price data from [Polygon.io](https://polygon.io). Provide an API key:

- **Local:** create `web/.env.local` with `POLYGON_API_KEY=your_key_here` (already git-ignored).
- **Vercel:** add `POLYGON_API_KEY` as a Project Environment Variable.

The key is read only in the server route (`/api/stocks/aggregates`) and is never sent to the
browser. Without a key, the Stocks page shows a "not configured" message and the rest of the
app is unaffected.

## Deploy to Vercel (at bondviz.nikolayvalev.com)

1. Import the GitHub repo in Vercel. Set **Root Directory = `web/`** (framework preset: Next.js). No required env vars (optionally set `POLYGON_API_KEY` for the Stocks page).
2. Deploy. Vercel builds `web/` and serves the app on a `*.vercel.app` URL.
3. Add the custom domain in **Project → Settings → Domains**: `bondviz.nikolayvalev.com`.
   Vercel shows a target like `cname.vercel-dns.com`.
4. In **Cloudflare DNS** for `nikolayvalev.com`, add:
   - Type `CNAME`, Name `bondviz`, Target `cname.vercel-dns.com`, Proxy status **DNS only** (grey cloud).

   Vercel then provisions the TLS certificate; the app is live at the custom domain.
