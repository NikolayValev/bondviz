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

## Deploy to Vercel (at bondviz.nikolayvalev.com)

1. Import the GitHub repo in Vercel. Set **Root Directory = `web/`** (framework preset: Next.js). No env vars.
2. Deploy. Vercel builds `web/` and serves the app on a `*.vercel.app` URL.
3. Add the custom domain in **Project → Settings → Domains**: `bondviz.nikolayvalev.com`.
   Vercel shows a target like `cname.vercel-dns.com`.
4. In **Cloudflare DNS** for `nikolayvalev.com`, add:
   - Type `CNAME`, Name `bondviz`, Target `cname.vercel-dns.com`, Proxy status **DNS only** (grey cloud).

   Vercel then provisions the TLS certificate; the app is live at the custom domain.
