# BondViz Web (Next.js front-end)

The BondViz web app: Home, Yield Curve Explorer, Carry & Roll-Down, Inversion & Recession
Signal, Yield Curve PCA, Bond Pricing & Risk Lab, Portfolio Analyzer, Stocks, Command Panel and a
Methodology page, built with Next.js (App Router), TypeScript, Tailwind and hand-rolled D3 charts.
Treasury data comes from the keyless U.S. Treasury par-yield XML feed via server route handlers
(current year cached hourly, past years weekly). The tool list lives in `lib/tools.ts`.

## Develop

```bash
cd web
npm install
npm run dev      # http://localhost:3000
npm test         # Vitest unit tests
npx eslint .     # lint (`next lint` was removed in Next 16)
npm run build    # production build
```

## Stocks (Polygon)

The Stocks page reads daily price data from [Polygon.io](https://polygon.io). Provide an API key:

- **Local:** create `web/.env.local` with `POLYGON_API_KEY=your_key_here` (already git-ignored).
- **Vercel:** add `POLYGON_API_KEY` as a Project Environment Variable.

The key is read only in the server route (`/api/stocks/aggregates`) and is never sent to the
browser. Without a key, the Stocks page shows a "not configured" message and the rest of the
app is unaffected.

Comparing tickers costs one Polygon call per ticker and window. Polygon's free tier allows 5 calls
a minute, so the page waits for typing to pause before fetching, and the server caches each
response for an hour; a rate-limited ticker shows a plain "rate limited" note.

## Command panel (GitHub Models)

The Command page sends chat to GitHub Models. Set `GITHUB_MODELS_TOKEN` (a fine-grained PAT with
**models: read**) in `web/.env.local` or as a Vercel environment variable; `GITHUB_MODELS_MODEL`
optionally overrides the default `openai/gpt-4o`. Without a token the page explains that the AI
demo is off and the rest of the app is unaffected. The route rate-limits each client (5 requests
a minute, 20 a day) because every visitor shares the token's quota.

## Deploy to Vercel (at bondviz.nikolayvalev.com)

1. Import the GitHub repo in Vercel. Set **Root Directory = `web/`** (framework preset: Next.js). No required env vars (optionally set `POLYGON_API_KEY` for the Stocks page).
2. Deploy. Vercel builds `web/` and serves the app on a `*.vercel.app` URL.
3. Add the custom domain in **Project → Settings → Domains**: `bondviz.nikolayvalev.com`.
   Vercel shows a target like `cname.vercel-dns.com`.
4. In **Cloudflare DNS** for `nikolayvalev.com`, add:
   - Type `CNAME`, Name `bondviz`, Target `cname.vercel-dns.com`, Proxy status **DNS only** (grey cloud).

   Vercel then provisions the TLS certificate; the app is live at the custom domain.
