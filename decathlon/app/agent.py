"""Agentic tool-use loop for the Decathlon product expert.

The chat model (OPENAI_MODEL, tool-capable, e.g. google/gemma-4-31b) drives
catalog retrieval itself: turn by turn it decides whether to call a tool and
which one. There is no separate router stage — the system prompt (built in
`main.py`) tells the model to use these tools only when the user is asking
about products.

Tools:
  - find_categories(query)            -> valid full category paths to filter by
  - search_products(query, gender?, categories?, brand?) -> up to 10 products
  - get_product(product_id)           -> full characteristics of one product

The loop is best-effort: a failing tool returns its error to the model (which
can recover or apologise) rather than crashing the request.
"""

import json
import logging
import os

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from decathlon.app import search as search_mod
from decathlon.app.catalog import CATEGORY_DISPLAYS, ID_TO_DISPLAY, PRODUCT_URL
from decathlon.app.productdb import get_product_details
from decathlon.core.embeddings import embed_texts
from decathlon.core.vectordb import CATEGORIES, get_client, get_collection

logger = logging.getLogger(__name__)

# app.py imports this before its own load_dotenv(); load here too.
load_dotenv()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "http://localhost:1234/v1"
).rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "lm-studio")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemma-4-31b")
REQUEST_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "120"))

# Hard cap on tool rounds so a confused model can't loop forever.
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "6"))
PRODUCT_SEARCH_N = int(os.getenv("PRODUCT_SEARCH_N", "10"))
FIND_CATEGORIES_N = 15


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_categories",
            "description": (
                "Look up valid Decathlon catalog category paths matching a "
                "keyword or concept. Call this BEFORE search_products when you "
                "want to constrain a search to a section, to discover the "
                "exact full path strings to pass as `categories`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A category keyword or concept in the user's "
                            "language, e.g. 'кроссовки для бега', 'палатки', "
                            "'детская обувь'."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Search the Decathlon catalog and return up to 10 matching "
                "products. Use this when the user is asking for, comparing, or "
                "shopping for sports gear."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The product description in the user's language — "
                            "the thing being shopped for and its defining "
                            "attributes (type, sport, material, colour, "
                            "season). Keep the user's product nouns verbatim; "
                            "never translate them. Do NOT put the audience / "
                            "gender / age here — use `gender` instead."
                        ),
                    },
                    "gender": {
                        "type": "string",
                        "enum": search_mod.GENDERS,
                        "description": (
                            "Who the product is for, if stated or clearly "
                            "implied ('для девочки'->girls, 'мужские'->men, "
                            "'детские'->kids). Omit if unspecified or "
                            "irrelevant (a tent, a ball)."
                        ),
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional. Full category path strings exactly as "
                            "returned by find_categories, to restrict the "
                            "search. A product matching ANY of them is kept."
                        ),
                    },
                    "brand": {
                        "type": "string",
                        "description": (
                            "Optional brand to filter by, e.g. 'Quechua', "
                            "'Kipsta'. Matched case-insensitively."
                        ),
                    },
                    "size": {
                        "type": "string",
                        "description": (
                            "Optional size to filter by, as the user stated it "
                            "(e.g. 'M', 'XL', 'EU42', '42'). Only products "
                            "that have a variant of approximately this size are "
                            "returned. Omit when the user has not mentioned a size."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": (
                "Get full characteristics of one product (description, "
                "composition, technical specs, benefits, available "
                "sizes/colours) for detailed exploration or comparison. Use "
                "the product `id` from a search_products result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product id from search_products.",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
]


_client_singleton: AsyncOpenAI | None = None


def _client() -> AsyncOpenAI:
    # Reuse one client. An explicit httpx client is required: the SDK's
    # default transport fails to connect to the (link-local) LM Studio host.
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = AsyncOpenAI(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            http_client=httpx.AsyncClient(timeout=REQUEST_TIMEOUT),
        )
    return _client_singleton


def _find_categories(query: str) -> list[str]:
    """Keyword-substring + semantic lookup over the catalog categories."""
    q = (query or "").strip()
    if not q:
        return []
    q_lc = q.lower()
    out: list[str] = [d for d in CATEGORY_DISPLAYS if q_lc in d.lower()]
    seen = set(out)

    try:
        vec = embed_texts([q])[0]
        res = get_collection(get_client(), CATEGORIES).query(
            query_embeddings=[vec],
            n_results=FIND_CATEGORIES_N,
            include=[],
        )
        for cat_id in res["ids"][0]:
            try:
                cid = int(str(cat_id).removeprefix("cat-"))
            except ValueError:
                continue
            disp = ID_TO_DISPLAY.get(cid)
            if disp and disp not in seen:
                seen.add(disp)
                out.append(disp)
    except Exception as e:  # noqa: BLE001 - degrade to substring matches
        logger.warning("Semantic category lookup failed: %s", e)

    return out[:FIND_CATEGORIES_N]


def _exec_tool(name: str, args: dict, pool: list[dict]) -> object:
    """Run one tool call. Accumulates renderable products into `pool`."""
    if name == "find_categories":
        cats = _find_categories(args.get("query", ""))
        return {"categories": cats} if cats else {
            "categories": [],
            "note": "No matching categories; search without a category filter.",
        }

    if name == "search_products":
        products = search_mod.search_products(
            args.get("query", ""),
            categories=args.get("categories"),
            n=PRODUCT_SEARCH_N,
            gender=args.get("gender"),
            brand=args.get("brand"),
            size=args.get("size"),
        )
        pool.extend(products)
        return [
            {
                "id": p["id"],
                "title": p["title"],
                "url": PRODUCT_URL.format(handle=p["handle"]) if p.get("handle") else None,
                "category": p.get("section_path", "").replace("\n", " | "),
                "price": p.get("price"),
                "brand": p.get("brand"),
            }
            for p in products
        ] or {"note": "No products found for that query/filters."}

    if name == "get_product":
        details = get_product_details(str(args.get("product_id", "")))
        if details:
            pool.append(details)
        return details or {"error": "No product with that id."}

    return {"error": f"Unknown tool {name!r}."}


async def run_agent(messages: list[dict]) -> tuple[str, list[dict]]:
    """Drive the tool-use loop. Returns (reply, products seen during the run).

    `messages` is the full chat history including the leading system prompt.
    Raises on model/transport failure (the endpoint maps that to a 502); tool
    failures are fed back to the model instead of raising.
    """
    convo = list(messages)
    pool: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        resp = await _client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=convo,
            tools=TOOLS,
            tool_choice="auto",
            stream=False,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            return (msg.content or "").strip(), pool

        convo.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = _exec_tool(tc.function.name, args, pool)
            except Exception as e:  # noqa: BLE001 - feed errors back to model
                logger.warning("Tool %s failed: %s", tc.function.name, e)
                result = {"error": f"Tool failed: {e}"}
            convo.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # Round budget exhausted: ask for a final answer with no more tools.
    resp = await _client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=convo,
        stream=False,
    )
    return (resp.choices[0].message.content or "").strip(), pool
