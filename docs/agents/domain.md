# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`GLOSSARY-MAP.md`** at the repo root: it points at one `GLOSSARY.md` per context. Read each one relevant to the topic.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in (system-wide decisions).
- **`<context>/docs/adr/`**: context-scoped decisions, next to that context's `GLOSSARY.md`.
- **`docs-for-dev/`** is this repo's own developer documentation (`ARCHITECTURE.md`, `MODULES.md`, `TOOLS_AND_CAPABILITIES.md`, `DEVELOPMENT.md`, …). Read it to learn how something works — it is *not* domain language or decided trade-offs, so it does not replace the glossary or ADRs.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grilling --docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

This is a **multi-context** repo: root `GLOSSARY-MAP.md` indexes per-context `GLOSSARY.md` files.

```
/
├── GLOSSARY-MAP.md                    ← index of contexts
├── docs/adr/                          ← system-wide decisions
├── deeptutor/                         ← context: core runtime (Python)
│   ├── GLOSSARY.md
│   └── docs/adr/
├── web/                               ← context: Next.js frontend
│   ├── GLOSSARY.md
│   └── docs/adr/
├── packages/chat-engine/              ← context: shared TS chat engine
│   ├── GLOSSARY.md
│   └── docs/adr/
└── apps/desktop/                      ← context: Electron desktop shell
    ├── GLOSSARY.md
    └── docs/adr/
```

The contexts, and what belongs to each:

| Context                | Root                  | Scope                                                                                                    |
| ---------------------- | --------------------- | -------------------------------------------------------------------------------------------------------- |
| Core runtime           | `deeptutor/`          | turn application service, turn engine, tool and capability registries, built-in tools/capabilities, services. `deeptutor_cli/` and `deeptutor_web/` are thin entry shims and belong to this context. |
| Web frontend           | `web/`                | Next.js app: routes, components, player, i18n, generated contracts.                                        |
| Chat engine            | `packages/chat-engine/` | Shared TypeScript chat engine consumed by the web frontend and desktop shell.                            |
| Desktop shell          | `apps/desktop/`       | Electron shell plus per-platform packaging under `platforms/`.                                            |

The upstream skill template writes contexts as `src/<context>/`. This repo has no
`src/` directory, so the context roots above are the real top-level package
directories. A decision that spans entries (CLI ↔ WebSocket ↔ SDK) belongs in
`docs/adr/`; a decision inside one context belongs in that context's `docs/adr/`.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the relevant context's `GLOSSARY.md`. Don't drift to synonyms the glossary explicitly avoids. When two contexts name the same thing differently, prefer the glossary of the context you are changing and flag the divergence.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
