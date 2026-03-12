# plshelp

# plshelp

Local documentation search for your terminal. Point it at any docs site — or your own Markdown files — and search it instantly without touching the web.

## What it does

`plshelp add` crawls and indexes a documentation site. After that, `plshelp query` searches it locally using hybrid BM25 + semantic search, with no network required. Everything lives in a SQLite database on your machine.
```sh
plshelp add nextjs https://nextjs.org/docs
plshelp query nextjs "how does the app router work"
```

You can index multiple libraries and search across all of them at once, merge related libraries together, or point it at local Markdown files instead of a URL.

## Why it exists

Docs sites are slow to search, require a browser, and disappear offline. Coding agents like Claude Code and Codex can call `plshelp` directly — run `plshelp init` at your project root and it writes the instruction files that tell the agent how to use it. The `--json` flag makes output machine-readable.

## Install

**macOS / Linux**
```sh
curl -fsSL https://plshelp.run/install.sh | sh
```

**Windows**
```powershell
irm https://plshelp.run/install.ps1 | iex
```

## The basics
```sh
# Index a docs site
plshelp add rust https://doc.rust-lang.org/book/

# Query it
plshelp rust "what are the rules for borrowing"

# Index a local file
plshelp index notes --file ./notes/architecture.md

# Search across everything
plshelp ask "how do I handle async errors"

# Wire up to your coding agent
plshelp init
```

## Full docs

[plshelp.run/docs](https://plshelp.run/docs)