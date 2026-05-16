# Decathlon Product Expert

A chat consultant for the Decathlon (KZ) catalog: the user asks for sports
gear in natural language and a local LLM answers like an in-store advisor,
grounded in real catalog products via retrieval.

## Architecture

```
user turn
  -> agent.run_agent()     # tool-use loop on the chat model
       |-> find_categories  # keyword + semantic lookup of valid category paths
       |-> search_products  # embed query, gender pre-filter, hard category filter
       |-> get_facets       # aggregate colors/brands/sizes/prices for a query slice
       |-> get_product      # full characteristics from products.db
  -> chat model reply       # cites products; UI renders cited ones as cards
```

- `app/main.py` — FastAPI: chat API (`/api/chat`) + static UI. Builds the
  system prompt and runs the agentic loop. Tool failures are best-effort
  (fed back to the model); only model/transport failure is a 502.
- `app/agent.py` — the tool-use loop: tool schemas, executors, and
  `run_agent()`. The chat model decides whether/which tools to call
  (`tool_choice="auto"`); the system prompt restricts tool use to genuine
  product questions.
- `app/search.py` — vector search over the `products` Chroma collection
  (`search_products` and `get_facets` tools).
- `app/productdb.py` — runtime read of `products.db` for the `get_product`
  tool (the only request-time SQLite reader).
- `app/catalog.py` — the level-1+2 category list + id<->display maps.
- `core/embeddings.py` / `core/vectordb.py` — shared bge-m3 + Chroma helpers.
- `indexing/index.py` — builds the `products` and `categories` collections
  from `products.db`. `indexing/query.py` — CLI for ad-hoc retrieval debugging.
- `scrapers/` — populate `products.db` (run rarely; not part of the app).

Everything (chat, embeddings) talks to one **OpenAI-compatible endpoint**
(LM Studio by default). `products.db` and `chroma_data/` are git-ignored
build artifacts.

## Core domain decisions

These are the non-obvious choices; honour them or change them deliberately.

1. **Local-first, single endpoint.** Chat + embeddings use `OPENAI_BASE_URL`.
   One tool-capable chat model (`OPENAI_MODEL`, default `google/gemma-4-31b`)
   runs the whole agentic loop — it must support OpenAI tool calling and have
   `n_ctx` comfortably above the tool-schema + history size. The system prompt
   gates tool use so the model only calls tools for genuine product questions.

2. **`app/main.py` and `app/agent.py` both call `load_dotenv()`** — `main.py`
   must load `.env` *before* importing `agent` (which reads `OPENAI_*` at
   import). `AsyncOpenAI` needs an explicit `httpx.AsyncClient`; its default
   transport fails on the link-local LM Studio host.

3. **The category tree is non-linear and over-broad.** `products.category_id`
   is always a root Shopify collection id (useless for the path). Real
   sections come from `category:<n>` tags (`<n>` = numeric prefix of
   `categories.slug`). A product spans multiple leaf branches; `path_ids` is
   the union of every category id across all branches. Level-0 roots carry no
   signal and are stripped from all paths/docs. Upstream tags are
   deliberately over-broad — unrelated branches appearing is **not a bug**.
   The agent picks categories deliberately (via `find_categories`), so the
   filter is hard but always relaxes rather than collapsing results (see #5).

4. **`query` should be a pure product description.** The system prompt tells
   the model to keep gender/age/audience words *out* of the `search_products`
   `query` ("носочки для девочки" -> query `носки`, `gender=girls`): those
   words otherwise dominate the embedding and drag retrieval toward clothing.
   Product nouns are kept verbatim — never translated ("удочка" stays
   "удочка").

5. **Gender and category are both hard filters with a relax fallback.**
   - `gender` (`search.GENDERS` = men/women/boys/girls/kids) maps to a set of
     `gender_id` metadata values (`search.GENDER_IDS`) and is applied as a
     near-hard Chroma pre-filter, with a relax-to-unfiltered fallback when it
     starves results. `gender_id` is reliable; high-leverage facet.
   - `categories` is a list of full display paths the model obtained from
     `find_categories`. `search_products` keeps a result only if it falls
     under **any** of them (`path_ids` ancestor match), but if that leaves
     `< n` it relaxes to the raw semantic ranking — an over-specific pick can
     never collapse results.
   - `brand`, `size`, `color` are case-insensitive substring post-filters,
     same relax rule. `size` and `color` match against `\x1f`-delimited
     variant lists stored in Chroma metadata.
   - Age has no clean field and is not a facet.

6. **Category paths are NOT enumerated in the tool schema.** Inlining all
   ~204 L1+L2 paths as a JSON-schema enum costs ~6k tokens and balloons
   `n_ctx`. Instead the model calls `find_categories` to discover valid paths,
   then passes them to `search_products`. Displays are full `root / L1 / L2`
   paths because stripping the root collides (men's vs women's "Обувь");
   `find_categories` matches by substring + semantic search over the
   `categories` collection and maps hits back via `catalog.ID_TO_DISPLAY`.

## Working on this

- Run the UI: `mise run ui` (http://localhost:8000).
- Rebuild the vector store after catalog/index changes:
  `mise run index-vectors`.
- Debug retrieval without the LLM: `mise run query-vectors "<q>"
  [--collection products|categories] [--ancestor ID] [-n]`.
- Refresh catalog (rare): `mise run fetch-categories`, `mise run
  fetch-products`.
- Config is env-driven (`.env` auto-loaded; real env wins). Key vars:
  `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EMBED_MODEL`, `CHROMA_PATH`,
  `PRODUCT_SEARCH_N`, `MAX_TOOL_ROUNDS`, `CHAT_LANGUAGE`.
- When tuning retrieval, always check a *control* query (e.g. "кроссовки
  мужские", "палатка") alongside the one you're fixing — facet/category
  changes regress easily. Verify with `query-vectors` and via the real
  `agent.run_agent()` / `search_products()` path, not just one of them.
