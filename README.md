# plshelp

# plshelp

Turn documentation websites, web pages, Markdown, and plain text files into a local searchable corpus for coding assistants and RAG workflows. Crawl, clean, chunk, and index — so your agent has accurate context without hitting the internet at query time. 

## What it does

`plshelp add` crawls and indexes a documentation site. After that, `plshelp query` searches it locally using hybrid BM25 + semantic search, completely locally. Everything lives in a SQLite database on your machine.
```sh
plshelp add nextjs https://nextjs.org/docs
plshelp query nextjs "how does the app router work"
```

You can index multiple libraries and search across all of them at once, merge related libraries together, or index local Markdown files instead of a website.

## Why it exists

All current context augmentation tools require calling an API, moving your data off your machine and to a remote server. plshelp is completely local, both human and agent-friendly, and open-source, giving you control over which embedding models to use and the ability to tune it to your own machine's capabilities.

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