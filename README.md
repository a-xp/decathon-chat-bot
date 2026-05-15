# Decathlon Product Expert

A chat consultant for the Decathlon (KZ) catalog. Ask it about sports gear in
plain language and it answers like a knowledgeable in-store advisor —
grounded in real catalog products via vector search.

## How it works

Each chat turn goes through a small retrieval pipeline before the assistant
replies:

1. **Router** — a tool-capable LLM decides whether the turn needs catalog
   search. If so it rewrites the message into a clean product query, strips
   the gender/audience into a separate facet, and may pick a category.
2. **Search** — the query is embedded (bge-m3) and matched against the
   product catalog in an embedded Chroma store, filtered by gender and softly
   biased by category.
3. **Answer** — the retrieved products are injected as context; the chat
   model recommends from them and the UI renders the cited ones as cards.

Retrieval is best-effort: if anything in the pipeline fails, the assistant
falls back to plain chat instead of erroring.

Everything (chat, router, embeddings) talks to a single **OpenAI-compatible
endpoint** — LM Studio by default, but any compatible server works.

> For architecture and the non-obvious design decisions, see
> [`CLAUDE.md`](CLAUDE.md).

### Project layout

```
decathlon/
  app.py              # FastAPI app: chat API + static UI
  router.py           # routes a turn: search-or-not, query rewrite, facets
  search.py           # vector search over the product catalog
  catalog.py          # category list shown to the router
  embeddings.py       # bge-m3 embeddings via the OpenAI endpoint
  vectordb.py         # embedded Chroma helpers
  index_vectors.py    # build the vector store from products.db
  query_vectors.py    # CLI to debug retrieval without the LLM
  scrapers/           # catalog scrapers -> products.db
  static/index.html   # chat UI
products.db           # scraped catalog (git-ignored)
chroma_data/          # vector store (git-ignored)
```

## Setup

You need an OpenAI-compatible endpoint serving a chat model (e.g. `phi-4`)
and an embedding model (`bge-m3`). LM Studio works out of the box.

```sh
cp .env.example .env   # adjust OPENAI_BASE_URL / model names if needed
```

Build the catalog and vector store (scrapes the catalog, then embeds it):

```sh
mise run reindex
```

This is the one command for a fresh catalog — it runs the category and
product scrapers and rebuilds the vector index in order. (The individual
steps `fetch-categories`, `fetch-products`, `index-vectors` still exist if
you need to run just one; `index-vectors` alone re-embeds without re-scraping.)

## Run

```sh
mise run ui
```

Open http://localhost:8000/.

## Configuration

Settings are read from the environment; a local `.env` is loaded
automatically (real env vars / Replit Secrets take precedence). See
[`.env.example`](.env.example) for the full list. Most-used:

| Var                  | Default                     | Purpose                          |
|----------------------|-----------------------------|----------------------------------|
| `OPENAI_BASE_URL`    | `http://localhost:1234/v1`  | OpenAI-compatible endpoint       |
| `OPENAI_MODEL`       | `microsoft/phi-4`           | Chat model                       |
| `OPENAI_ROUTER_MODEL`| = `OPENAI_MODEL`            | Routing model (keep = chat model)|
| `OPENAI_EMBED_MODEL` | `bge-m3`                    | Embedding model                  |
| `CHROMA_PATH`        | `./chroma_data`             | Vector store location            |
| `PRODUCT_SEARCH_N`   | `8`                         | Products injected per grounded turn |
| `CHAT_LANGUAGE`      | `auto`                      | Force reply language, or match user |
| `PORT`               | `8000`                      | HTTP port for the chat UI        |

## Debugging retrieval

Query the vector store directly, without the LLM:

```sh
mise run query-vectors "носки детские"
mise run query-vectors "палатка" --collection categories -n 5
```
