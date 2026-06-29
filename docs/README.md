# BondViz docs

Start here. These four documents are written to be **token-efficient**: an agent should be able to
read one or two of them plus the file it's editing and have full context.

| Doc | Read it when you… |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | …are new to the repo or starting any task. The mental model, file map, the "add a page" recipe, conventions, data contracts, testing. **The default entry point.** |
| [DESIGN-SYSTEM.md](DESIGN-SYSTEM.md) | …are touching look-and-feel, tokens, the design-system swap, or charts. Inventory of every styling decision + a migration playbook. |
| [BACKLOG.md](BACKLOG.md) | …are picking up a feature/improvement. Agent-ready briefs (`FEAT-*`) with files, approach, effort, acceptance. |
| [BUGS.md](BUGS.md) | …are fixing something. Tiered bugs & tech debt (`BUG-*`) with location + fix sketch. |

**Most important orientation fact:** the active product is the **Next.js app in `web/`**. The
Python/Streamlit code (`src/bondviz/`, `app/`, root `README.md`/`CLAUDE.md`) is a frozen reference
that the TS lib was ported from — don't take it as the source of truth for current work.

Process artifacts (design specs + numbered implementation plans for past features) live under
[superpowers/specs/](superpowers/specs/) and [superpowers/plans/](superpowers/plans/) — the two
most recent are the best templates for building a new tool the way this codebase does.

> Health baseline at last review (2026-06-28): web `npm test` 105 pass · `npm run build` green ·
> `npx eslint .` clean; Python `pytest` 18 pass.
