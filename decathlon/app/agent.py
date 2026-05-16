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
from typing import Annotated, Literal, Optional

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from decathlon.app import search as search_mod
from decathlon.app.catalog import CATEGORY_DISPLAYS, ID_TO_DISPLAY
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


# --- Tool parameter models ---------------------------------------------------

class FindCategoriesArgs(BaseModel):
    query: Annotated[
        str,
        Field(description=(
            "A category keyword or concept in the user's language, "
            "e.g. 'кроссовки для бега', 'палатки', 'детская обувь'."
        )),
    ]


class SearchProductsArgs(BaseModel):
    query: Annotated[
        str,
        Field(description=(
            "The product description in the user's language — the thing being "
            "shopped for and its defining attributes (type, sport, material, "
            "season). Normalize informal or diminutive Russian forms to "
            "standard catalog terms (e.g. 'штанишки'/'штаны' → 'брюки', "
            "'шортики' → 'шорты', 'маечка' → 'майка/футболка'). Never translate "
            "product nouns to another language. Do NOT put colour, audience, "
            "gender, or age here — use the dedicated fields instead."
        )),
    ]
    gender: Annotated[
        Optional[Literal[tuple(search_mod.GENDERS)]],  # type: ignore[valid-type]
        Field(None, description=(
            "Who the product is for, if stated or clearly implied "
            "('для девочки'->girls, 'мужские'->men, 'детские'->kids). "
            "Omit if unspecified or irrelevant (a tent, a ball)."
        )),
    ] = None
    categories: Annotated[
        Optional[list[str]],
        Field(None, description=(
            "Full category path strings exactly as returned by find_categories, "
            "to restrict the search. A product matching ANY of them is kept."
        )),
    ] = None
    brand: Annotated[
        Optional[str],
        Field(None, description=(
            "Brand to filter by, e.g. 'Quechua', 'Kipsta'. "
            "Matched case-insensitively."
        )),
    ] = None
    size: Annotated[
        Optional[str],
        Field(None, description=(
            "Size to filter by, as the user stated it (e.g. 'M', 'XL', "
            "'EU42', '42'). Only products that have a variant of approximately "
            "this size are returned. Omit when the user has not mentioned a size."
        )),
    ] = None
    color: Annotated[
        Optional[str],
        Field(None, description=(
            "Color to filter by in Russian, as the user stated it "
            "(e.g. 'желтый', 'синий', 'красный'). Only products that have a "
            "variant of this color are returned. Omit when the user has not "
            "mentioned a color."
        )),
    ] = None


class GetFacetsArgs(BaseModel):
    query: Annotated[
        str,
        Field(description=(
            "The product concept to explore, same normalisation rules as "
            "search_products (e.g. 'брюки', 'кроссовки для бега')."
        )),
    ]
    gender: Annotated[
        Optional[Literal[tuple(search_mod.GENDERS)]],  # type: ignore[valid-type]
        Field(None, description="Same as search_products `gender`."),
    ] = None
    categories: Annotated[
        Optional[list[str]],
        Field(None, description=(
            "Category paths from find_categories to narrow the slice."
        )),
    ] = None


class GetProductArgs(BaseModel):
    product_id: Annotated[
        str,
        Field(description="The product id from search_products."),
    ]


def _tool_schema(name: str, description: str, model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


TOOLS = [
    _tool_schema(
        "get_facets",
        (
            "Explore what is actually available for a product type: returns "
            "colors, brands, sizes, and price range aggregated over semantically "
            "relevant products. Call this (1) when the user asks what colors/brands/"
            "options exist for something, or (2) after search_products returns "
            "nothing, to discover alternatives to suggest."
        ),
        GetFacetsArgs,
    ),
    _tool_schema(
        "find_categories",
        (
            "Look up valid Decathlon catalog category paths matching a keyword "
            "or concept. Call this BEFORE search_products when you want to "
            "constrain a search to a section, to discover the exact full path "
            "strings to pass as `categories`."
        ),
        FindCategoriesArgs,
    ),
    _tool_schema(
        "search_products",
        (
            "Search the Decathlon catalog and return up to 10 matching products. "
            "Use this when the user is asking for, comparing, or shopping for "
            "sports gear."
        ),
        SearchProductsArgs,
    ),
    _tool_schema(
        "get_product",
        (
            "Get full characteristics of one product (description, composition, "
            "technical specs, benefits, available sizes/colours) for detailed "
            "exploration or comparison. Use the product `id` from a "
            "search_products result."
        ),
        GetProductArgs,
    ),
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


def _exec_tool(name: str, args: dict) -> object:
    if name == "get_facets":
        a = GetFacetsArgs(**args)
        return search_mod.get_facets(a.query, categories=a.categories, gender=a.gender)

    if name == "find_categories":
        a = FindCategoriesArgs(**args)
        cats = _find_categories(a.query)
        return {"categories": cats} if cats else {
            "categories": [],
            "note": "No matching categories; search without a category filter.",
        }

    if name == "search_products":
        a = SearchProductsArgs(**args)
        products = search_mod.search_products(
            a.query,
            categories=a.categories,
            n=PRODUCT_SEARCH_N,
            gender=a.gender,
            brand=a.brand,
            size=a.size,
            color=a.color,
        )
        return [
            {
                "id": p["id"],
                "title": p["title"],
                "category": p.get("section_path", "").replace("\n", " | "),
                "price": p.get("price"),
                "brand": p.get("brand"),
            }
            for p in products
        ] or {"note": "No products found for that query/filters."}

    if name == "get_product":
        a = GetProductArgs(**args)
        details = get_product_details(a.product_id)
        if not details:
            return {"error": "No product with that id."}
        for key in ("handle", "image_url", "model_code"):
            details.pop(key, None)
        return details

    return {"error": f"Unknown tool {name!r}."}


async def run_agent(messages: list[dict]) -> str:
    """Drive the tool-use loop. Returns the model's final reply.

    `messages` is the full chat history including the leading system prompt.
    Raises on model/transport failure (the endpoint maps that to a 502); tool
    failures are fed back to the model instead of raising.
    """
    convo = list(messages)

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
            return (msg.content or "").strip()

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
                result = _exec_tool(tc.function.name, args)
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
    return (resp.choices[0].message.content or "").strip()
