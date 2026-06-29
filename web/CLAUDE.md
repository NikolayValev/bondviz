@AGENTS.md

# BondViz web app

This `web/` Next.js app is the **active product** (the Python/Streamlit code at the repo root is a
frozen reference). Before starting, read the orientation docs — they're written to give full
context in few tokens:

- **../docs/ARCHITECTURE.md** — mental model, file map, the "add a page" recipe, conventions, data
  contracts, testing. Read this first.
- **../docs/DESIGN-SYSTEM.md** — tokens, UI primitives, hardcoded-color inventory, design-system
  swap playbook. Read when touching look-and-feel or charts.
- **../docs/BACKLOG.md** (`FEAT-*`) and **../docs/BUGS.md** (`BUG-*`) — the work queue.

Quick checks (keep all green): `npm test` · `npx eslint .` · `npm run build`. Note: `next lint` is
removed in Next 16 — use `npm run lint` / `npx eslint .`.
