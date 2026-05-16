# Decathlon Product Expert

A chat consultant for the Decathlon (KZ) catalog. Ask it about sports gear in
plain language and it answers like a knowledgeable in-store advisor —
grounded in real catalog products via retrieval.

## How it works

Each chat turn runs an agentic tool-use loop before the assistant replies:

1. **Category discovery** — the model calls `find_categories` to look up valid
   category paths when it wants to narrow the search to a section.
2. **Search** — the query is embedded (bge-m3) and matched against the product
   catalog in a local Chroma store, optionally filtered by gender, category, and
   brand.
3. **Product detail** — the model can call `get_product` for full specs
   (composition, sizes, benefits) before a detailed comparison.
4. **Answer** — the assistant recommends from retrieved products; the UI renders
   cited ones as cards.

There is **no separate router model**: the chat model drives the whole loop via
the system prompt. Everything (chat, embeddings) talks to a single
**OpenAI-compatible endpoint** — LM Studio by default, but any compatible
server works.

> For architecture decisions and non-obvious design choices, see
> [`CLAUDE.md`](CLAUDE.md).

## Project layout

```
decathlon/
  app/
    main.py         # FastAPI app: chat API (/api/chat) + static UI
    agent.py        # tool-use loop: tool schemas, executors, run_agent()
    search.py       # vector search over the products collection
    productdb.py    # request-time SQLite reader for get_product
    catalog.py      # category display paths + id<->display maps
    static/
      index.html    # chat UI
  core/
    embeddings.py   # bge-m3 embeddings via the OpenAI endpoint
    vectordb.py     # Chroma client + collection helpers
    documents.py    # shared helpers for indexed document format
  indexing/
    index.py        # build vector store from products.db
    query.py        # CLI to debug retrieval without the LLM
  scrapers/
    categories.py   # scrape category tree -> products.db
    products.py     # scrape product catalog -> products.db
products.db         # scraped catalog (git-ignored, built by scrapers)
chroma_data/        # vector store (git-ignored, built by index-vectors)
```

## Setup

You need an OpenAI-compatible endpoint serving a **tool-capable chat model**
(e.g. `google/gemma-4-31b`) and an **embedding model** (`bge-m3`). LM Studio
works out of the box.

```sh
cp .env.example .env   # adjust OPENAI_BASE_URL / model names if needed
uv sync                # install dependencies
```

Build the catalog and vector store (scrapes the catalog, then embeds it):

```sh
mise run reindex
```

This is the one command for a fresh catalog — it runs the category and product
scrapers and rebuilds the vector index in order. The individual steps
`fetch-categories`, `fetch-products`, and `index-vectors` still exist if you
need to run just one.

## Run

```sh
mise run ui
```

Open http://localhost:8000/.

## Configuration

Settings are read from the environment; a local `.env` is loaded automatically
(real env vars take precedence). See [`.env.example`](.env.example) for the
full list. Most-used:

| Var                  | Default                     | Purpose                               |
|----------------------|-----------------------------|---------------------------------------|
| `OPENAI_BASE_URL`    | `http://localhost:1234/v1`  | OpenAI-compatible endpoint            |
| `OPENAI_MODEL`       | `google/gemma-4-31b`        | Chat + tool-use model                 |
| `OPENAI_EMBED_MODEL` | `bge-m3`                    | Embedding model                       |
| `OPENAI_API_KEY`     | `lm-studio`                 | API key (any string for LM Studio)    |
| `OPENAI_TIMEOUT`     | `120`                       | Request timeout in seconds            |
| `CHROMA_PATH`        | `./chroma_data`             | Vector store location                 |
| `PRODUCT_SEARCH_N`   | `10`                        | Products retrieved per search call    |
| `MAX_TOOL_ROUNDS`    | `6`                         | Max tool-call rounds per turn         |
| `CHAT_LANGUAGE`      | `auto`                      | Force reply language, or match user   |
| `PORT`               | `8000`                      | HTTP port for the chat UI             |

## Debugging retrieval

Query the vector store directly, without the LLM:

```sh
mise run query-vectors "носки детские"
mise run query-vectors "палатка" --collection categories -n 5
mise run query-vectors "обувь" --ancestor 583
```
