"""Web UI + chat API for the Decathlon product expert bot.

The chat model runs an agentic tool-use loop (`decathlon.app.agent`): it
decides, turn by turn, whether to look up categories, search the catalog, or
pull a product's full characteristics, then answers grounded in those results.
There is no separate router stage — the system prompt constrains tool use to
genuine product questions.
"""

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError
from pydantic import BaseModel

# Load a local .env if present (real env vars / Replit Secrets take precedence).
load_dotenv()

from decathlon.app.agent import (  # noqa: E402 - must follow load_dotenv()
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    run_agent,
)
from decathlon.app.catalog import PRODUCT_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language the bot replies in. Empty / "auto" = match the user's language.
CHAT_LANGUAGE = os.getenv("CHAT_LANGUAGE", "").strip()

SYSTEM_PROMPT = (
    "You are a knowledgeable Decathlon sports-equipment product expert. "
    "Help the user choose gear, explain differences between products, and "
    "give practical, friendly advice. Be concise; if you are unsure, say so "
    "rather than inventing product details.\n\n"
    "You have catalog tools. Use them ONLY when the user is actually asking "
    "about, shopping for, or comparing products or gear — never for "
    "greetings, thanks, or general chit-chat; just reply normally then.\n"
    "When a product question does need the catalog, prefer this flow:\n"
    "1. If you want to narrow to a section, call `find_categories` to get the "
    "exact valid category paths.\n"
    "2. Call `search_products` (pass `gender`/`categories`/`brand` only when "
    "the user stated or clearly implied them).\n"
    "3. Call `get_product` for a product's full specs before a detailed "
    "comparison or when the user asks about characteristics.\n"
    "Base recommendations only on products returned by the tools; do not "
    "invent products. Cite each recommended product by its exact name and "
    "include its link."
)

if CHAT_LANGUAGE and CHAT_LANGUAGE.lower() != "auto":
    SYSTEM_PROMPT += (
        f" Always reply in {CHAT_LANGUAGE}, regardless of the language the "
        "user writes in."
    )

app = FastAPI(title="Decathlon Product Expert")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "base_url": OPENAI_BASE_URL,
        "language": CHAT_LANGUAGE or "auto",
    }


def cited_products(reply: str, products: list[dict]) -> list[dict]:
    """Trim the products seen during the run to the ones the reply references.

    Primary signal: handles extracted from /products/{handle} URLs in markdown
    links. Fallback: handle or title appearing as plain text in the reply.
    Products without an image are dropped since they can't render a card.
    """
    reply_lc = reply.lower()
    linked_handles: set[str] = {
        m.group(1) for m in re.finditer(r"/products/([a-z0-9][a-z0-9-]*)", reply_lc)
    }
    cards = []
    seen: set[str] = set()
    for p in products:
        image_url = p.get("image_url")
        if not image_url:
            continue
        handle = (p.get("handle") or "").lower()
        title = p.get("title") or ""
        if handle in seen:
            continue
        if (
            (handle and handle in linked_handles)
            or (handle and handle in reply_lc)
            or (title and title.lower() in reply_lc)
        ):
            seen.add(handle)
            cards.append(
                {
                    "title": title,
                    "brand": p.get("brand"),
                    "price": p.get("price"),
                    "available": p.get("available"),
                    "image_url": image_url,
                    "url": PRODUCT_URL.format(handle=handle) if handle else None,
                }
            )
    return cards


@app.post("/api/chat")
async def chat(req: ChatRequest):
    conversation = [{"role": m.role, "content": m.content} for m in req.messages]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]

    try:
        reply, pool = await run_agent(messages)
    except APIError as e:
        logger.error("LLM API error: %s", e)
        return JSONResponse(
            status_code=502,
            content={"error": f"LLM API error: {e}"},
        )
    except Exception as e:  # noqa: BLE001 - transport/connection failures
        logger.error("Could not reach LLM API: %s", e)
        return JSONResponse(
            status_code=502,
            content={"error": f"Could not reach LLM API at {OPENAI_BASE_URL}."},
        )

    if not reply:
        return JSONResponse(
            status_code=502,
            content={"error": "LLM API returned an empty response."},
        )

    return {"reply": reply, "products": cited_products(reply, pool)}


# Serve the chat UI at "/". Path resolved relative to this file so it works
# regardless of the working directory. Kept last so API routes take precedence.
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
